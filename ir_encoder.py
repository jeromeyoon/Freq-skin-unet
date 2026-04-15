"""
ir_encoder.py
=============
Stage 1 : Cross-Polarization Encoder (교차 편광 전용)

교차 편광(cross polarization):
  - 표면 반사 제거 → 피부 하부 발색단(melanin, hemoglobin) 포착
  - Beer-Lambert: OD 전처리가 물리적으로 적합
  - mid-freq feature → brown_head, red_head 입력

구성
----
ODPreprocess       : RGB → OD + log-ratio → 1×1 Conv(5→3)
FrequencyGate      : FFT 기반 주파수 대역 분리 (low=조명 억제, mid=발색단 보존)
CrossPolEncoder    : 위 두 모듈 + 4-level U-Net Encoder
  (IlluminationRobustEncoder = CrossPolEncoder alias)

출력 (EncoderOutput namedtuple)
--------------------------------
chroma     : [f1, f2, f3]  mid-freq skip features  (brown/red head용)
texture    : [t1, t2, t3]  high-freq skip features (미사용, 평행 편광이 담당)
bottleneck : enc4 출력 [B, base_ch*8, H/8, W/8]
"""

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 출력 타입 ──────────────────────────────────────────────────────────────────
EncoderOutput = namedtuple('EncoderOutput', ['chroma', 'texture', 'bottleneck'])


# ══════════════════════════════════════════════════════════════════════════════
# ODPreprocess
# ══════════════════════════════════════════════════════════════════════════════
class ODPreprocess(nn.Module):
    """
    RGB → Optical Density 공간으로 변환

    Beer-Lambert: RGB = I × exp(−OD_tissue)
    → OD = −log(RGB) = −log(I) + OD_tissue
    → log(R/G) = OD_R − OD_G  (조명 중립 가정 시 조명 불변)

    5채널 [OD_R, OD_G, OD_B, log(R/G), log(G/B)] → 1×1 Conv → 3채널
    초기화: OD 3채널은 identity (Beer-Lambert 물리 보장), ratio는 near-zero
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.proj = nn.Conv2d(5, 3, kernel_size=1, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            # OD 채널: identity 초기화
            self.proj.weight[:3, :3, 0, 0] = torch.eye(3)
            # log-ratio 채널: 0 초기화 (학습으로 발견)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B, 3, H, W]  0~1
        od = -torch.log(rgb.clamp(min=self.eps))        # [B, 3, H, W]
        rg = od[:, 0] - od[:, 1]                        # log(R/G)
        gb = od[:, 1] - od[:, 2]                        # log(G/B)
        x5 = torch.cat([od, rg[:, None], gb[:, None]], dim=1)  # [B, 5, H, W]
        return self.proj(x5)                             # [B, 3, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# FrequencyGate
# ══════════════════════════════════════════════════════════════════════════════
class FrequencyGate(nn.Module):
    """
    FFT 기반 주파수 대역 분리

    주파수 대역 (정규화 반경 기준):
      low  [0,    low_r)  : 조명 성분 → 억제 (low_logit 초기값 −2)
      mid  [low_r, high_r): 발색단 성분 → 보존 (mid_logit 초기값 +2)
      high [high_r, 1.0)  : 주름/텍스처 → 보존 (high_logit 초기값 +2)

    forward(x) → (x_chroma, x_texture)
      x_chroma  : mid 대역 필터링 결과
      x_texture : high 대역 필터링 결과
    """

    def __init__(self, channels: int, low_r: float = 0.1, high_r: float = 0.4):
        super().__init__()
        self.low_r   = low_r
        self.high_r  = high_r
        self.low_logit  = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))
        self.mid_logit  = nn.Parameter(torch.full((1, channels, 1, 1), +2.0))
        self.high_logit = nn.Parameter(torch.full((1, channels, 1, 1), +2.0))

    def _radial_mask(self, H: int, W: int,
                     r_lo: float, r_hi: float,
                     device: torch.device) -> torch.Tensor:
        """[H, W] bool 마스크: 정규화 반경 r_lo ≤ r < r_hi"""
        cy, cx = H / 2.0, W / 2.0
        ys = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1) - cy
        xs = torch.arange(W, device=device, dtype=torch.float32).view(1, -1) - cx
        r  = torch.sqrt(ys ** 2 + xs ** 2) / min(H, W)
        return (r >= r_lo) & (r < r_hi)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        F_shift = torch.fft.fftshift(torch.fft.fft2(x))  # [B, C, H, W] complex

        low_mask  = self._radial_mask(H, W, 0.0,        self.low_r,  x.device)
        mid_mask  = self._radial_mask(H, W, self.low_r,  self.high_r, x.device)
        high_mask = self._radial_mask(H, W, self.high_r, 1.0,         x.device)

        # gate_low → 0에 가까울수록 조명(저주파) 성분 억제
        gate_low  = torch.sigmoid(self.low_logit)   # [1, C, 1, 1]
        gate_mid  = torch.sigmoid(self.mid_logit)
        gate_high = torch.sigmoid(self.high_logit)

        x_low = torch.fft.ifft2(
            torch.fft.ifftshift(F_shift * low_mask)
        ).real

        x_mid = torch.fft.ifft2(
            torch.fft.ifftshift(F_shift * mid_mask)
        ).real

        x_high = torch.fft.ifft2(
            torch.fft.ifftshift(F_shift * high_mask)
        ).real

        # 발색단(chroma): 중간 주파수 보존 + 저주파(조명) 억제
        x_chroma  = x_mid * gate_mid - x_low * gate_low

        # 텍스처: 고주파 보존
        x_texture = x_high * gate_high

        return x_chroma, x_texture


# ══════════════════════════════════════════════════════════════════════════════
# ConvBlock
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv3×3 → BN → ReLU × 2"""

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
# IlluminationRobustEncoder
# ══════════════════════════════════════════════════════════════════════════════
class IlluminationRobustEncoder(nn.Module):
    """
    Stage 1: 조명 강인 특징 추출기

    Pipeline
    --------
    RGB
     → ODPreprocess          [B, 3, H, W]
     → enc1 (ConvBlock)      [B, ch,   H,   W]   → FrequencyGate → chroma1, texture1
     → pool → enc2           [B, ch*2, H/2, H/2] → FrequencyGate → chroma2, texture2
     → pool → enc3           [B, ch*4, H/4, H/4] → FrequencyGate → chroma3, texture3
     → pool → enc4(bottleneck)[B,ch*8, H/8, H/8]

    출력: EncoderOutput(
        chroma   = [chroma1, chroma2, chroma3],   # mid-freq skips
        texture  = [texture1, texture2, texture3], # high-freq skips
        bottleneck                                 # enc4
    )
    """

    def __init__(self, base_ch: int = 64,
                 low_r: float = 0.1,
                 high_r: float = 0.4):
        super().__init__()
        ch = base_ch

        self.od_prep = ODPreprocess()
        self.pool    = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)
        self.enc4 = ConvBlock(ch*4, ch * 8)   # bottleneck

        # FrequencyGate (3 levels)
        self.freq1 = FrequencyGate(ch,    low_r, high_r)
        self.freq2 = FrequencyGate(ch*2,  low_r, high_r)
        self.freq3 = FrequencyGate(ch*4,  low_r, high_r)

    def forward(self, rgb: torch.Tensor) -> EncoderOutput:
        x = self.od_prep(rgb)           # [B, 3, H, W]

        e1 = self.enc1(x)               # [B, ch,   H,   W]
        e2 = self.enc2(self.pool(e1))   # [B, ch*2, H/2, H/2]
        e3 = self.enc3(self.pool(e2))   # [B, ch*4, H/4, H/4]
        e4 = self.enc4(self.pool(e3))   # [B, ch*8, H/8, H/8]  bottleneck

        c1, t1 = self.freq1(e1)
        c2, t2 = self.freq2(e2)
        c3, t3 = self.freq3(e3)

        return EncoderOutput(
            chroma    = [c1, c2, c3],
            texture   = [t1, t2, t3],
            bottleneck= e4,
        )


# alias: 교차 편광 전용임을 명시
CrossPolEncoder = IlluminationRobustEncoder
