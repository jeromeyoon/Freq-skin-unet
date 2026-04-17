"""
ir_encoder_stable.py
====================
CrossPolEncoder 안정화 버전 (ir_encoder.py 대체)

변경사항 (vs ir_encoder.py):
  1. ODPreprocess  : eps 1e-6 → 1e-3, OD 출력값 clamp(-6, 6)
                     극단적 픽셀(매우 어두운 영역)에서 log 폭주 방지
  2. FrequencyGate : low_logit 초기값 -2.0 → -3.0
                     초기 학습에서 저주파(조명) 억제를 더 강하게 유지

사용법:
  from ir_encoder_stable import CrossPolEncoder, IlluminationRobustEncoder
  (skin_net.py에서 import 경로만 변경하면 됨)
"""

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 출력 타입 ──────────────────────────────────────────────────────────────────
EncoderOutput = namedtuple('EncoderOutput', ['chroma', 'texture', 'bottleneck'])


# ══════════════════════════════════════════════════════════════════════════════
# ODPreprocess (안정화 버전)
# ══════════════════════════════════════════════════════════════════════════════
class ODPreprocess(nn.Module):
    """
    RGB → Optical Density 공간으로 변환 (수치 안정화)

    변경:
      - eps: 1e-6 → 1e-3  (-log(1e-6)=13.8 → -log(1e-3)=6.9)
        실제 피부 이미지에서 완전 검은 픽셀은 노이즈이므로 1e-3 클리핑이 물리적으로 타당
      - OD 출력을 clamp(-6, 6)으로 제한
        이상치 픽셀이 Focal/Dice loss를 통해 폭발적 gradient를 만드는 것을 차단
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.proj = nn.Conv2d(5, 3, kernel_size=1, bias=False)

        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            self.proj.weight[:3, :3, 0, 0] = torch.eye(3)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb: [B, 3, H, W]  0~1
        rgb_safe = rgb.clamp(min=self.eps, max=1.0)          # 양쪽 클리핑
        od = -torch.log(rgb_safe)                             # [B, 3, H, W]
        od = od.clamp(-6.0, 6.0)                             # 폭주 방지
        rg = od[:, 0] - od[:, 1]
        gb = od[:, 1] - od[:, 2]
        x5 = torch.cat([od, rg[:, None], gb[:, None]], dim=1)
        return self.proj(x5)                                  # [B, 3, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# FrequencyGate (초기 억제 강화)
# ══════════════════════════════════════════════════════════════════════════════
class FrequencyGate(nn.Module):
    """
    FFT 기반 주파수 대역 분리

    변경:
      - low_logit 초기값: -2.0 → -3.0
        sigmoid(-2)=0.12, sigmoid(-3)=0.047
        초기 학습에서 조명 성분이 발색단 feature에 섞이지 않도록 더 강하게 억제
        (w_freq_reg가 Phase3에서 활성화될 때까지 게이트가 열리는 것을 지연)
    """

    def __init__(self, channels: int, low_r: float = 0.1, high_r: float = 0.4):
        super().__init__()
        self.low_r   = low_r
        self.high_r  = high_r
        # -3.0: sigmoid(-3)≈0.047 → 초기 저주파 억제를 더 강하게
        self.low_logit  = nn.Parameter(torch.full((1, channels, 1, 1), -3.0))
        self.mid_logit  = nn.Parameter(torch.full((1, channels, 1, 1), +2.0))
        self.high_logit = nn.Parameter(torch.full((1, channels, 1, 1), +2.0))

    def _radial_mask(self, H: int, W: int,
                     r_lo: float, r_hi: float,
                     device: torch.device) -> torch.Tensor:
        cy, cx = H / 2.0, W / 2.0
        ys = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1) - cy
        xs = torch.arange(W, device=device, dtype=torch.float32).view(1, -1) - cx
        r  = torch.sqrt(ys ** 2 + xs ** 2) / min(H, W)
        return (r >= r_lo) & (r < r_hi)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        F_shift = torch.fft.fftshift(torch.fft.fft2(x))

        low_mask  = self._radial_mask(H, W, 0.0,        self.low_r,  x.device)
        mid_mask  = self._radial_mask(H, W, self.low_r,  self.high_r, x.device)
        high_mask = self._radial_mask(H, W, self.high_r, 1.0,         x.device)

        gate_low  = torch.sigmoid(self.low_logit)
        gate_mid  = torch.sigmoid(self.mid_logit)
        gate_high = torch.sigmoid(self.high_logit)

        x_low = torch.fft.ifft2(torch.fft.ifftshift(F_shift * low_mask)).real
        x_mid = torch.fft.ifft2(torch.fft.ifftshift(F_shift * mid_mask)).real
        x_high = torch.fft.ifft2(torch.fft.ifftshift(F_shift * high_mask)).real

        x_chroma  = x_mid * gate_mid - x_low * gate_low
        x_texture = x_high * gate_high

        return x_chroma, x_texture


# ══════════════════════════════════════════════════════════════════════════════
# ConvBlock (변경 없음)
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
# IlluminationRobustEncoder (변경 없음 — 위 모듈 교체로 자동 반영)
# ══════════════════════════════════════════════════════════════════════════════
class IlluminationRobustEncoder(nn.Module):
    def __init__(self, base_ch: int = 64,
                 low_r: float = 0.1,
                 high_r: float = 0.4):
        super().__init__()
        ch = base_ch

        self.od_prep = ODPreprocess()
        self.pool    = nn.MaxPool2d(2)

        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)
        self.enc4 = ConvBlock(ch*4, ch * 8)

        self.freq1 = FrequencyGate(ch,    low_r, high_r)
        self.freq2 = FrequencyGate(ch*2,  low_r, high_r)
        self.freq3 = FrequencyGate(ch*4,  low_r, high_r)

    def forward(self, rgb: torch.Tensor) -> EncoderOutput:
        x = self.od_prep(rgb)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        c1, t1 = self.freq1(e1)
        c2, t2 = self.freq2(e2)
        c3, t3 = self.freq3(e3)

        return EncoderOutput(
            chroma    = [c1, c2, c3],
            texture   = [t1, t2, t3],
            bottleneck= e4,
        )


CrossPolEncoder = IlluminationRobustEncoder
