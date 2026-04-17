"""
skin_net_enhance.py
===================
SkinAnalyzerEnhanced : bottleneck(enc4)에도 FrequencyGate 적용

기존 모델(skin_net.py)과의 차이
---------------------------------
  enc1/enc2/enc3 : 기존과 동일 (FrequencyGate/HighFreqGate 적용)
  enc4(bottleneck): ★ Enhanced에서 추가
    CrossPolEncoderEnhanced  → FrequencyGate(ch*8)  → chroma 성분만 bottleneck으로
    ParallelPolEncoderEnhanced → HighFreqGate(ch*8) → high-freq 성분만 bottleneck으로

설계 의도
---------
bottleneck은 모든 head(SpotHead, WrinkleHead)에 global context로 전달된다.
기존 모델에서 enc4는 raw feature이므로 주변광(저주파)이 전역 컨텍스트에 잔류한다.
Enhanced는 bottleneck 단계에서도 조명 성분을 억제하여 head가 받는
global context를 더 조명 강인하게 만든다.

사용 예
-------
    from skin_net_enhance import build_analyzer_enhanced
    model = build_analyzer_enhanced()
    result = model(rgb_cross, rgb_parallel, skin_mask)
"""

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ir_encoder       import (FrequencyGate, ODPreprocess,
                               ConvBlock, EncoderOutput)
from parallel_encoder import (HighFreqGate, ParallelPreprocess,
                               ParallelOutput)
from task_heads       import SpotHead, WrinkleHead


# ── 출력 타입 (skin_net.py의 SkinResult와 동일) ───────────────────────────────
SkinResult = namedtuple('SkinResult', [
    'brown_mask',   'brown_score',
    'red_mask',     'red_score',
    'wrinkle_mask', 'wrinkle_score',
])


# ══════════════════════════════════════════════════════════════════════════════
# CrossPolEncoderEnhanced
# ══════════════════════════════════════════════════════════════════════════════
class CrossPolEncoderEnhanced(nn.Module):
    """
    교차 편광 인코더 — enc4(bottleneck)에 FrequencyGate 추가

    Pipeline
    --------
    RGB
     → ODPreprocess          [B, 3, H, W]
     → enc1                  [B, ch,   H,   W]   → freq1 → chroma1, texture1
     → pool → enc2           [B, ch*2, H/2, H/2] → freq2 → chroma2, texture2
     → pool → enc3           [B, ch*4, H/4, H/4] → freq3 → chroma3, texture3
     → pool → enc4           [B, ch*8, H/8, H/8] → freq4 → ★ chroma4 (bottleneck)

    출력: EncoderOutput(
        chroma    = [chroma1, chroma2, chroma3],
        texture   = [texture1, texture2, texture3],
        bottleneck= chroma4   ← 조명 억제된 bottleneck
    )
    """

    def __init__(self, base_ch: int = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4):
        super().__init__()
        ch = base_ch

        self.od_prep = ODPreprocess()
        self.pool    = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)
        self.enc4 = ConvBlock(ch*4, ch * 8)

        # FrequencyGate: enc1~enc3 (기존), enc4 (신규)
        self.freq1 = FrequencyGate(ch,    low_r, high_r)
        self.freq2 = FrequencyGate(ch*2,  low_r, high_r)
        self.freq3 = FrequencyGate(ch*4,  low_r, high_r)
        self.freq4 = FrequencyGate(ch*8,  low_r, high_r)   # ★ bottleneck

    def forward(self, rgb: torch.Tensor) -> EncoderOutput:
        x = self.od_prep(rgb)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        c1, t1 = self.freq1(e1)
        c2, t2 = self.freq2(e2)
        c3, t3 = self.freq3(e3)

        # bottleneck: chroma 성분만 사용 (저주파=조명 억제)
        c4, _  = self.freq4(e4)

        return EncoderOutput(
            chroma    = [c1, c2, c3],
            texture   = [t1, t2, t3],
            bottleneck= c4,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ParallelPolEncoderEnhanced
# ══════════════════════════════════════════════════════════════════════════════
class ParallelPolEncoderEnhanced(nn.Module):
    """
    평행 편광 인코더 — enc4(bottleneck)에 HighFreqGate 추가

    Pipeline
    --------
    rgb_parallel
     → ParallelPreprocess (InstanceNorm + 1×1 Conv)
     → enc1 [B, ch,   H,   W]   → hgate1 → texture1
     → enc2 [B, ch*2, H/2, H/2] → hgate2 → texture2
     → enc3 [B, ch*4, H/4, H/4] → hgate3 → texture3
     → enc4 [B, ch*8, H/8, H/8] → hgate4 → ★ texture4 (bottleneck)

    출력: ParallelOutput(
        texture    = [texture1, texture2, texture3],
        bottleneck = texture4  ← 고주파만 남긴 bottleneck
    )
    """

    def __init__(self, base_ch: int = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4):
        super().__init__()
        ch = base_ch

        self.preprocess = ParallelPreprocess()
        self.pool       = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)
        self.enc4 = ConvBlock(ch*4, ch * 8)

        # HighFreqGate: enc1~enc3 (기존), enc4 (신규)
        self.hgate1 = HighFreqGate(ch,    low_r, high_r)
        self.hgate2 = HighFreqGate(ch*2,  low_r, high_r)
        self.hgate3 = HighFreqGate(ch*4,  low_r, high_r)
        self.hgate4 = HighFreqGate(ch*8,  low_r, high_r)   # ★ bottleneck

    def forward(self, rgb_parallel: torch.Tensor) -> ParallelOutput:
        x = self.preprocess(rgb_parallel)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        t1 = self.hgate1(e1)
        t2 = self.hgate2(e2)
        t3 = self.hgate3(e3)
        t4 = self.hgate4(e4)   # ★ 고주파만 남긴 bottleneck

        return ParallelOutput(
            texture    = [t1, t2, t3],
            bottleneck = t4,
        )


# ══════════════════════════════════════════════════════════════════════════════
# SkinAnalyzerEnhanced
# ══════════════════════════════════════════════════════════════════════════════
class SkinAnalyzerEnhanced(nn.Module):
    """
    듀얼 편광 피부 분석 모델 — bottleneck 주파수 필터링 추가

    기존 SkinAnalyzer와 인터페이스 동일 (drop-in replacement).
    체크포인트는 호환되지 않음 (freq4/hgate4 파라미터 추가).

    Parameters
    ----------
    base_ch : U-Net 기본 채널 수 (기본 64)
    low_r   : FrequencyGate 저주파 반경 (기본 0.1)
    high_r  : FrequencyGate 고주파 시작 반경 (기본 0.4)
    """

    def __init__(self,
                 base_ch: int   = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4):
        super().__init__()

        self.cross_encoder = CrossPolEncoderEnhanced(
            base_ch=base_ch, low_r=low_r, high_r=high_r)

        self.parallel_encoder = ParallelPolEncoderEnhanced(
            base_ch=base_ch, low_r=low_r, high_r=high_r)

        self.brown_head   = SpotHead(base_ch)
        self.red_head     = SpotHead(base_ch)
        self.wrinkle_head = WrinkleHead(base_ch)

    def forward(self,
                rgb_cross:    torch.Tensor,
                rgb_parallel: torch.Tensor,
                skin_mask:    torch.Tensor | None = None) -> SkinResult:

        cross_out    = self.cross_encoder(rgb_cross)
        parallel_out = self.parallel_encoder(rgb_parallel)

        brown_mask,   brown_score   = self.brown_head(
            cross_out.chroma, cross_out.bottleneck, skin_mask)

        red_mask,     red_score     = self.red_head(
            cross_out.chroma, cross_out.bottleneck, skin_mask)

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
def build_analyzer_enhanced(base_ch: int   = 64,
                            low_r:   float = 0.1,
                            high_r:  float = 0.4) -> SkinAnalyzerEnhanced:
    """SkinAnalyzerEnhanced 인스턴스 생성"""
    return SkinAnalyzerEnhanced(base_ch=base_ch, low_r=low_r, high_r=high_r)
