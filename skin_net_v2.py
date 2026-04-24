"""
skin_net_v2.py
==============
SkinAnalyzerV2 : H1 + H2 동시 수정 버전

H1 수정 — Brown/Red 색조 구분 강화
------------------------------------
기존:
  brown_head, red_head 모두 OD chroma feature만 사용.
  OD = -log(RGB) → luminance(밝기) 기반이라 hue(색조) 정보 손실.
  결과: 두 head가 동일한 feature space에서 출발 → 클래스 구분 근거 없음.

V2:
  HueBranch: rgb_cross의 정규화 색비율을 인코딩 → 조명 불변 hue feature.
    chroma_rgb = rgb / (R+G+B+eps)  → 밝기 무관한 색조 표현
    brown(melanin) : R/total > G/total > B/total  (따뜻한 갈색 비율)
    red(hemoglobin): R/total >> G/total ≈ B/total  (순적 비율)
  SpotHeadV2: 각 스케일에서 OD chroma + hue를 1×1 conv로 병합 후 디코딩.

H2 수정 — FrequencyGate 물리 모델 정확성
------------------------------------------
기존:
  ConvBlock 이후 feature space에서 FFT → Beer-Lambert additive 가정 위반.
  (OD_chromophore = OD_mid - OD_low 은 OD 원시값에서만 성립)

V2:
  CrossPolEncoderV2 (ir_encoder_v2.py):
    OD 원시값에 직접 ODFrequencyFilter 적용
    od_chroma = od_mid - gate_low * od_low  ← Beer-Lambert 정확 준수

기존 SkinAnalyzer와 인터페이스 동일 (drop-in replacement).
체크포인트는 호환 안 됨 (HueBranch, ODFrequencyFilter 파라미터 추가).

사용 예
-------
    from skin_net_v2 import build_analyzer_v2
    model = build_analyzer_v2()
    result = model(rgb_cross, rgb_parallel, skin_mask)
"""

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ir_encoder_v2    import CrossPolEncoderV2
from parallel_encoder import ParallelPolEncoder
from task_heads       import WrinkleHead, UpBlock


SkinResult = namedtuple('SkinResult', [
    'brown_mask',   'brown_score',
    'red_mask',     'red_score',
    'wrinkle_mask', 'wrinkle_score',
])


# ══════════════════════════════════════════════════════════════════════════════
# 공통 빌딩블록
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════════════
# HueBranch — H1 핵심: 조명 불변 색조 feature 추출
# ══════════════════════════════════════════════════════════════════════════════
class HueBranch(nn.Module):
    """
    RGB 정규화 색비율 기반 색조(Hue) feature 추출기

    입력: rgb_cross [B, 3, H, W]
    출력: [h1, h2, h3]  multi-scale hue features
          h1: [B, ch,   H,   W]
          h2: [B, ch*2, H/2, H/2]
          h3: [B, ch*4, H/4, H/4]

    원리 — 정규화 색비율(Chromaticity)
    ------------------------------------
      chroma_rgb = rgb / (R + G + B + eps)
      - 조명 강도(밝기)에 무관한 색조 정보 보존
      - brown(melanin) : R비율 > G비율 > B비율 (따뜻한 갈색)
      - red(hemoglobin): R비율 >> G비율 ≈ B비율 (선명한 적색)
      OD가 잃어버린 이 색조 차이를 SpotHeadV2에 전달해 두 클래스 구분 가능하게 함.

    OD branch와 독립적으로 학습되므로 두 정보가 상호 보완적으로 작동.
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        ch = base_ch
        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)

    def forward(self, rgb: torch.Tensor) -> list:
        denom      = rgb.sum(dim=1, keepdim=True).clamp(min=1e-6)
        chroma_rgb = rgb / denom                      # [B, 3, H, W] 조명 불변 색비율

        h1 = self.enc1(chroma_rgb)                    # [B, ch,   H,   W]
        h2 = self.enc2(self.pool(h1))                 # [B, ch*2, H/2, H/2]
        h3 = self.enc3(self.pool(h2))                 # [B, ch*4, H/4, H/4]
        return [h1, h2, h3]


# ══════════════════════════════════════════════════════════════════════════════
# 스코어 계산 유틸 (task_heads.py와 동일)
# ══════════════════════════════════════════════════════════════════════════════
def _compute_score(mask: torch.Tensor,
                   skin_mask: torch.Tensor | None) -> torch.Tensor:
    if skin_mask is None:
        skin_mask = torch.ones_like(mask)
    denom      = skin_mask.sum(dim=[1, 2, 3]).clamp(min=1.0)
    area_ratio = (mask * skin_mask).sum(dim=[1, 2, 3]) / denom
    density    = (mask * skin_mask).flatten(1).max(dim=1).values
    return (area_ratio * 0.6 + density * 0.4).clamp(0.0, 1.0) * 100.0


# ══════════════════════════════════════════════════════════════════════════════
# SpotHeadV2 — H1 핵심: Chroma(OD) + Hue(색조) 통합 예측
# ══════════════════════════════════════════════════════════════════════════════
class SpotHeadV2(nn.Module):
    """
    갈색 반점 / 적색 반점 예측 V2

    기존 SpotHead 대비 변경
    -----------------------
    기존: chroma_feats만 입력 → brown/red가 동일 feature space 공유
    V2:  chroma_feats(OD 발색단) + hue_feats(RGB 색조) 통합
         각 스케일에서 두 feature를 1×1 conv로 먼저 병합 후 기존과 동일한 U-Net 디코더로 예측

    merge_* (1×1 conv): chroma + hue concat → 원래 채널 수로 압축
    → brown head: OD 강도는 같아도 색비율(R-dominant vs R>>others)로 구분
    → red head:   동일 방식으로 hemoglobin 색조에 특화된 feature 학습

    입력
    ----
    chroma_feats : [c1, c2, c3]  (from CrossPolEncoderV2)
    hue_feats    : [h1, h2, h3]  (from HueBranch)
    bottleneck   : [B, ch*8, H/8, H/8]
    skin_mask    : [B, 1, H, W]  optional
    """

    def __init__(self, base_ch: int = 64):
        super().__init__()
        ch = base_ch

        # 각 scale: chroma(ch*k) + hue(ch*k) → ch*k (1×1 병합)
        self.merge3 = nn.Conv2d(ch*4 + ch*4, ch*4, kernel_size=1, bias=False)
        self.merge2 = nn.Conv2d(ch*2 + ch*2, ch*2, kernel_size=1, bias=False)
        self.merge1 = nn.Conv2d(ch   + ch,   ch,   kernel_size=1, bias=False)

        # Decoder (기존 SpotHead와 동일)
        self.up3 = UpBlock(ch * 8, ch * 4, ch * 4)
        self.up2 = UpBlock(ch * 4, ch * 2, ch * 2)
        self.up1 = UpBlock(ch * 2, ch,     ch)

        self.head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, kernel_size=1),
        )

    def forward(self,
                chroma_feats: list,
                hue_feats:    list,
                bottleneck:   torch.Tensor,
                skin_mask:    torch.Tensor | None = None):
        c1, c2, c3 = chroma_feats
        h1, h2, h3 = hue_feats

        # 각 스케일에서 발색단(OD) + 색조(Hue) 병합
        s3 = self.merge3(torch.cat([c3, h3], dim=1))   # [B, ch*4, H/4, H/4]
        s2 = self.merge2(torch.cat([c2, h2], dim=1))   # [B, ch*2, H/2, H/2]
        s1 = self.merge1(torch.cat([c1, h1], dim=1))   # [B, ch,   H,   W]

        d3 = self.up3(bottleneck, s3)
        d2 = self.up2(d3,         s2)
        d1 = self.up1(d2,         s1)

        logit = self.head(d1)
        prob  = torch.sigmoid(logit)
        score = _compute_score(prob, skin_mask)
        return logit, score


# ══════════════════════════════════════════════════════════════════════════════
# SkinAnalyzerV2
# ══════════════════════════════════════════════════════════════════════════════
class SkinAnalyzerV2(nn.Module):
    """
    듀얼 편광 피부 분석 모델 V2 — H1 + H2 동시 수정

    구성 요소
    ---------
    cross_encoder    : CrossPolEncoderV2  — OD 공간 직접 주파수 분리 (H2)
    hue_branch       : HueBranch          — RGB 색조 feature (H1)
    parallel_encoder : ParallelPolEncoder — 주름용 고주파 feature (변경 없음)
    brown_head       : SpotHeadV2         — OD chroma + hue 통합 예측 (H1)
    red_head         : SpotHeadV2         — OD chroma + hue 통합 예측 (H1)
    wrinkle_head     : WrinkleHead        — 변경 없음

    Parameters
    ----------
    base_ch : U-Net 기본 채널 수 (기본 64)
    low_r   : ODFrequencyFilter 저주파 반경 (기본 0.1)
    high_r  : ODFrequencyFilter 중주파 경계 반경 (기본 0.4)
    """

    def __init__(self,
                 base_ch: int   = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4):
        super().__init__()

        self.cross_encoder    = CrossPolEncoderV2(base_ch, low_r, high_r)
        # Separate HueBranch per task: brown(R>G>B) and red(R>>G≈B) have
        # distinct chromaticity patterns.  A shared branch receives opposing
        # gradients from brown and red steps under task-sampled training,
        # which slows red convergence.  Independent branches eliminate this
        # interference at the cost of ~2.4 M extra parameters (base_ch=64).
        self.hue_branch_brown = HueBranch(base_ch)
        self.hue_branch_red   = HueBranch(base_ch)
        self.parallel_encoder = ParallelPolEncoder(base_ch, low_r, high_r)

        self.brown_head   = SpotHeadV2(base_ch)
        self.red_head     = SpotHeadV2(base_ch)
        self.wrinkle_head = WrinkleHead(base_ch)

    def forward(self,
                rgb_cross:    torch.Tensor,
                rgb_parallel: torch.Tensor,
                skin_mask:    torch.Tensor | None = None) -> SkinResult:

        cross_out    = self.cross_encoder(rgb_cross)
        hue_brown    = self.hue_branch_brown(rgb_cross)
        hue_red      = self.hue_branch_red(rgb_cross)
        parallel_out = self.parallel_encoder(rgb_parallel)

        brown_mask,   brown_score   = self.brown_head(
            cross_out.chroma, hue_brown, cross_out.bottleneck, skin_mask)

        red_mask,     red_score     = self.red_head(
            cross_out.chroma, hue_red, cross_out.bottleneck, skin_mask)

        wrinkle_mask, wrinkle_score = self.wrinkle_head(
            parallel_out.texture, parallel_out.bottleneck, skin_mask)

        return SkinResult(
            brown_mask    = brown_mask,
            brown_score   = brown_score,
            red_mask      = red_mask,
            red_score     = red_score,
            wrinkle_mask  = wrinkle_mask,
            wrinkle_score = wrinkle_score,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 팩토리
# ══════════════════════════════════════════════════════════════════════════════
def build_analyzer_v2(base_ch: int   = 64,
                      low_r:   float = 0.1,
                      high_r:  float = 0.4) -> SkinAnalyzerV2:
    """SkinAnalyzerV2 인스턴스 생성"""
    return SkinAnalyzerV2(base_ch=base_ch, low_r=low_r, high_r=high_r)
