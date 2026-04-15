"""
skin_net.py
===========
SkinAnalyzer: 듀얼 편광 입력 피부 분석 모델

편광 분리 원칙
--------------
  교차 편광 (rgb_cross)    → CrossPolEncoder  → brown_head, red_head
  평행 편광 (rgb_parallel) → ParallelPolEncoder → wrinkle_head

물리적 근거:
  - 교차 편광: 표면 반사 제거 → 발색단(melanin/hemoglobin) 포착 → OD 전처리
  - 평행 편광: 표면 텍스처 포착 → 주름 → InstanceNorm + high-freq

사용 예
-------
    from skin_net import build_analyzer
    import torch

    model = build_analyzer()
    rgb_cross    = torch.randn(2, 3, 256, 256)
    rgb_parallel = torch.randn(2, 3, 256, 256)
    result = model(rgb_cross, rgb_parallel)

    print(result.brown_mask.shape)    # [2, 1, 256, 256]
    print(result.brown_score)         # tensor([...])  0~100
    print(result.wrinkle_score)       # tensor([...])  0~100
"""

from collections import namedtuple
import torch
import torch.nn as nn

from ir_encoder       import CrossPolEncoder
from parallel_encoder import ParallelPolEncoder
from task_heads       import SpotHead, WrinkleHead


# ── 출력 타입 ──────────────────────────────────────────────────────────────────
SkinResult = namedtuple('SkinResult', [
    'brown_mask',   'brown_score',
    'red_mask',     'red_score',
    'wrinkle_mask', 'wrinkle_score',
])


# ══════════════════════════════════════════════════════════════════════════════
# SkinAnalyzer
# ══════════════════════════════════════════════════════════════════════════════
class SkinAnalyzer(nn.Module):
    """
    듀얼 편광 피부 분석 모델

    Parameters
    ----------
    base_ch : U-Net 기본 채널 수 (기본 64)
    low_r   : FrequencyGate 저주파(조명) 반경 (기본 0.1)
    high_r  : FrequencyGate 고주파(주름) 시작 반경 (기본 0.4)

    Forward 입력
    ------------
    rgb_cross    : [B, 3, H, W]  교차 편광 이미지 (0~1)
    rgb_parallel : [B, 3, H, W]  평행 편광 이미지 (0~1)
    skin_mask    : [B, 1, H, W]  피부 영역 마스크 (optional)

    Forward 출력 (SkinResult namedtuple)
    -------------------------------------
    brown_mask    [B,1,H,W]  갈색 반점 확률 맵
    brown_score   [B]        갈색 반점 스코어 0~100
    red_mask      [B,1,H,W]  적색 반점 확률 맵
    red_score     [B]        적색 반점 스코어 0~100
    wrinkle_mask  [B,1,H,W]  주름 확률 맵
    wrinkle_score [B]        주름 스코어 0~100
    """

    def __init__(self,
                 base_ch: int   = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4):
        super().__init__()

        # Stage 1-A: 교차 편광 인코더 (OD 전처리 + mid-freq FreqGate)
        self.cross_encoder = CrossPolEncoder(
            base_ch = base_ch,
            low_r   = low_r,
            high_r  = high_r,
        )

        # Stage 1-B: 평행 편광 인코더 (InstanceNorm + high-freq Gate)
        self.parallel_encoder = ParallelPolEncoder(
            base_ch = base_ch,
            low_r   = low_r,
            high_r  = high_r,
        )

        # Stage 2: task-specific heads
        self.brown_head   = SpotHead(base_ch)   # 교차 편광 chroma 사용
        self.red_head     = SpotHead(base_ch)   # 교차 편광 chroma 사용
        self.wrinkle_head = WrinkleHead(base_ch) # 평행 편광 texture 사용

    def forward(self,
                rgb_cross:    torch.Tensor,
                rgb_parallel: torch.Tensor,
                skin_mask:    torch.Tensor | None = None) -> SkinResult:

        # Stage 1-A: 교차 편광 → 발색단 features
        cross_out = self.cross_encoder(rgb_cross)

        # Stage 1-B: 평행 편광 → 주름 features
        parallel_out = self.parallel_encoder(rgb_parallel)

        # Stage 2: 각 head로 예측
        brown_mask,   brown_score   = self.brown_head(
            cross_out.chroma, cross_out.bottleneck, skin_mask)

        red_mask,     red_score     = self.red_head(
            cross_out.chroma, cross_out.bottleneck, skin_mask)

        wrinkle_mask, wrinkle_score = self.wrinkle_head(
            parallel_out.texture, skin_mask)

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
def build_analyzer(base_ch: int   = 64,
                   low_r:   float = 0.1,
                   high_r:  float = 0.4) -> SkinAnalyzer:
    """SkinAnalyzer 인스턴스 생성"""
    return SkinAnalyzer(base_ch=base_ch, low_r=low_r, high_r=high_r)
