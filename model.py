"""
model.py
========
FreqAwareUNet : Frequency-aware Skip Connection U-Net

핵심 아이디어
-------------
조명(illumination)은 공간적으로 완만하게 변하므로 저주파(low freq)에 집중.
Beer-Lambert 법칙에서 log 도메인 조명은 additive offset → 저주파 제거 = 조명 제거.

주파수 대역별 역할:
  Low  freq → 조명 (억제)
  Mid  freq → melanin / hemoglobin (chromophore decoder로 전달)
  High freq → 주름 / 모공 (wrinkle head로 전달)

Skip connection에서 FFT → 대역 분리 → iFFT 를 수행하므로
전처리가 아닌 네트워크 내부에서 동적으로 조명 분리.

Architecture
------------
  ODPreprocess    : RGB → OD + channel ratio (Beer-Lambert 기반)
  Encoder         : 4-level ConvBlock + MaxPool
  FrequencyGate   : Learnable frequency band mask (per skip level)
  ChromDecoder    : mid-freq skip으로 melanin/hemoglobin 복원
  WrinkleHead     : 모든 레벨의 high-freq concat → 주름 예측
  Output          : brown [B,1,H,W], red [B,1,H,W], wrinkle [B,1,H,W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# OD 전처리
# ══════════════════════════════════════════════════════════════════════════════
class ODPreprocess(nn.Module):
    """
    RGB linear → Optical Density + channel log-ratio

    Beer-Lambert:
        OD = -log(RGB)
        log(RGB) = log(illuminant) - OD  → 조명이 additive offset

    channel ratio log(R/G), log(G/B):
        중성 조명 가정 시 조명에 무관한 표현

    5ch = [OD_R, OD_G, OD_B, log(R/G), log(G/B)]
        → 학습 가능 Conv1x1 으로 3ch 로 압축 (DINOv2 입력 크기 맞춤 불필요)
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps  = eps
        self.proj = nn.Conv2d(5, 3, kernel_size=1, bias=True)
        self._init_proj()

    def _init_proj(self):
        with torch.no_grad():
            nn.init.zeros_(self.proj.weight)
            self.proj.weight[0, 0, 0, 0] = 1.0   # OD_R → ch0
            self.proj.weight[1, 1, 0, 0] = 1.0   # OD_G → ch1
            self.proj.weight[2, 2, 0, 0] = 1.0   # OD_B → ch2
            nn.init.zeros_(self.proj.bias)

    def forward(self, rgb: torch.Tensor):
        """
        rgb     : [B, 3, H, W] linear RGB 0~1
        returns : [B, 3, H, W] OD-based representation
        """
        rgb_safe = rgb.clamp(min=self.eps)
        log_rgb  = torch.log(rgb_safe)          # [B, 3, H, W]
        od       = -log_rgb                      # optical density

        rg = log_rgb[:, 0:1] - log_rgb[:, 1:2]  # log(R/G)
        gb = log_rgb[:, 1:2] - log_rgb[:, 2:3]  # log(G/B)

        feat5 = torch.cat([od, rg, gb], dim=1)  # [B, 5, H, W]
        return self.proj(feat5)                  # [B, 3, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# Frequency-aware Gate (핵심 모듈)
# ══════════════════════════════════════════════════════════════════════════════
class FrequencyGate(nn.Module):
    """
    Learnable Frequency Band Mask for Skip Connections

    동작 원리
    ---------
    1. Encoder feature → FFT2D → 주파수 도메인
    2. fftshift → 저주파를 중앙으로 이동
    3. 반경 r 기반으로 low / mid / high 대역 구분
    4. 각 대역에 학습 가능한 gate (sigmoid) 곱셈
       - low_gate  : 조명 성분 억제 (초기값 0 → sigmoid ≈ 0.5, 학습으로 0에 수렴 유도)
       - mid_gate  : chromophore 대역 보존 (초기값 1)
       - high_gate : texture/wrinkle 대역 보존 (초기값 1)
    5. iFFT → 공간 도메인 복원
    6. 두 출력:
       x_chroma  = low(억제) + mid(보존)   → chromophore decoder 전달
       x_texture = high(보존)              → wrinkle head 전달

    Parameters
    ----------
    channels : feature 채널 수
    low_r    : 저주파 반경 (정규화, 0~1). default 0.1
    high_r   : 고주파 시작 반경 (정규화, 0~1). default 0.4
    """

    def __init__(self,
                 channels: int,
                 low_r   : float = 0.1,
                 high_r  : float = 0.4):
        super().__init__()
        self.low_r  = low_r
        self.high_r = high_r

        # 채널별 learnable gate (logit 공간, sigmoid 적용)
        # low: 조명 억제 → 초기 logit 음수 (-2 → sigmoid ≈ 0.12)
        self.low_logit  = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))
        # mid: chromophore 보존 → 초기 logit 양수 (+2 → sigmoid ≈ 0.88)
        self.mid_logit  = nn.Parameter(torch.full((1, channels, 1, 1),  2.0))
        # high: texture 보존
        self.high_logit = nn.Parameter(torch.full((1, channels, 1, 1),  2.0))

    def _radial_masks(self, H: int, W: int, device: torch.device):
        """반경 기반 주파수 마스크 생성 [H, W]"""
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=torch.float32) - cy
        x = torch.arange(W, device=device, dtype=torch.float32) - cx
        Y, X   = torch.meshgrid(y, x, indexing='ij')
        r      = (Y**2 + X**2).sqrt() / (min(H, W) / 2.0)

        low_m  = (r <  self.low_r).float()
        high_m = (r >= self.high_r).float()
        mid_m  = 1.0 - low_m - high_m

        return low_m[None, None], mid_m[None, None], high_m[None, None]  # [1,1,H,W]

    def forward(self, x: torch.Tensor):
        """
        x : [B, C, H, W]
        returns
          x_chroma  : [B, C, H, W]  low(억제)+mid → chromophore decoder
          x_texture : [B, C, H, W]  high          → wrinkle head
        """
        B, C, H, W = x.shape

        # FFT2D + 저주파 중앙 정렬
        X_f = torch.fft.fft2(x, norm='ortho')
        X_f = torch.fft.fftshift(X_f, dim=(-2, -1))

        # 마스크 [1, 1, H, W]
        low_m, mid_m, high_m = self._radial_masks(H, W, x.device)

        # 학습 가능 gate (0~1)
        g_low  = torch.sigmoid(self.low_logit)   # [1, C, 1, 1]
        g_mid  = torch.sigmoid(self.mid_logit)
        g_high = torch.sigmoid(self.high_logit)

        # 대역별 가중치 합산
        weight_chroma  = low_m * g_low + mid_m * g_mid   # [1, C, H, W]
        weight_texture = high_m * g_high

        X_chroma  = X_f * weight_chroma
        X_texture = X_f * weight_texture

        # iFFT
        x_chroma  = torch.fft.ifft2(
            torch.fft.ifftshift(X_chroma,  dim=(-2, -1)), norm='ortho'
        ).real
        x_texture = torch.fft.ifft2(
            torch.fft.ifftshift(X_texture, dim=(-2, -1)), norm='ortho'
        ).real

        return x_chroma, x_texture


# ══════════════════════════════════════════════════════════════════════════════
# 기본 빌딩 블록
# ══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv3x3 → BN → ReLU × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """ConvTranspose2d → concat(skip) → ConvBlock"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode='bilinear', align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


# ══════════════════════════════════════════════════════════════════════════════
# Main Model
# ══════════════════════════════════════════════════════════════════════════════
class FreqAwareUNet(nn.Module):
    """
    Frequency-aware Skip Connection U-Net for Skin Analysis

    Parameters
    ----------
    base_ch : 기본 채널 수. [base, base*2, base*4, base*8] 사용
    low_r   : 저주파 반경 (조명 억제 대역). default 0.1
    high_r  : 고주파 시작 반경 (주름 대역). default 0.4

    Outputs
    -------
    brown   : [B, 1, H, W]  melanin (갈색 색소) map
    red     : [B, 1, H, W]  hemoglobin (붉은 혈색) map
    wrinkle : [B, 1, H, W]  wrinkle (주름) map

    학습 파라미터
    ----------
    ODPreprocess.proj      : OD+ratio → 3ch 압축
    FrequencyGate × 3      : 각 스케일별 주파수 gate 학습
    Encoder / Decoder      : 표준 U-Net 파라미터
    """

    def __init__(self,
                 base_ch: int   = 64,
                 low_r  : float = 0.1,
                 high_r : float = 0.4):
        super().__init__()

        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]

        # ── 전처리 ────────────────────────────────────────────────────────────
        self.od_prep = ODPreprocess()

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(3,     ch[0])   # H    × W
        self.enc2 = ConvBlock(ch[0], ch[1])   # H/2  × W/2
        self.enc3 = ConvBlock(ch[1], ch[2])   # H/4  × W/4
        self.enc4 = ConvBlock(ch[2], ch[3])   # H/8  × W/8  (bottleneck)
        self.pool = nn.MaxPool2d(2)

        # ── Frequency-aware Skip Connections (3 레벨) ─────────────────────────
        self.freq1 = FrequencyGate(ch[0], low_r, high_r)   # enc1 스케일
        self.freq2 = FrequencyGate(ch[1], low_r, high_r)   # enc2 스케일
        self.freq3 = FrequencyGate(ch[2], low_r, high_r)   # enc3 스케일

        # ── Chromophore Decoder (mid-freq skip 사용) ──────────────────────────
        self.up3 = UpBlock(ch[3], ch[2], ch[2])
        self.up2 = UpBlock(ch[2], ch[1], ch[1])
        self.up1 = UpBlock(ch[1], ch[0], ch[0])

        # ── Chromophore Heads ─────────────────────────────────────────────────
        self.brown_head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )
        self.red_head = nn.Sequential(
            nn.Conv2d(ch[0], ch[0] // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

        # ── Wrinkle Head (high-freq 3 레벨 fusion) ────────────────────────────
        # 각 레벨의 high-freq를 H×W 로 upsample 후 concat
        wrinkle_in = ch[0] + ch[1] + ch[2]
        self.wrinkle_head = nn.Sequential(
            nn.Conv2d(wrinkle_in, ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0], ch[0] // 2, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[0] // 2, 1, 1),
        )

    def forward(self, rgb: torch.Tensor):
        """
        rgb : [B, 3, H, W] linear RGB

        Returns
        -------
        brown, red, wrinkle : 각 [B, 1, H, W], 값 범위 0~1
        """
        # ── OD 전처리 ─────────────────────────────────────────────────────────
        x = self.od_prep(rgb)                    # [B, 3, H, W]

        # ── Encoder ───────────────────────────────────────────────────────────
        e1 = self.enc1(x)                        # [B, C1, H,   W  ]
        e2 = self.enc2(self.pool(e1))            # [B, C2, H/2, W/2]
        e3 = self.enc3(self.pool(e2))            # [B, C3, H/4, W/4]
        e4 = self.enc4(self.pool(e3))            # [B, C4, H/8, W/8]

        # ── Frequency-aware Skip ──────────────────────────────────────────────
        s1_chroma, s1_texture = self.freq1(e1)  # C1
        s2_chroma, s2_texture = self.freq2(e2)  # C2
        s3_chroma, s3_texture = self.freq3(e3)  # C3

        # ── Chromophore Decoder ───────────────────────────────────────────────
        d3 = self.up3(e4, s3_chroma)             # [B, C3, H/4, W/4]
        d2 = self.up2(d3, s2_chroma)             # [B, C2, H/2, W/2]
        d1 = self.up1(d2, s1_chroma)             # [B, C1, H,   W  ]

        brown   = torch.sigmoid(self.brown_head(d1))
        red     = torch.sigmoid(self.red_head(d1))

        # ── Wrinkle Head ─────────────────────────────────────────────────────
        H, W = e1.shape[2], e1.shape[3]
        t2 = F.interpolate(s2_texture, size=(H, W), mode='bilinear', align_corners=False)
        t3 = F.interpolate(s3_texture, size=(H, W), mode='bilinear', align_corners=False)

        texture_all = torch.cat([s1_texture, t2, t3], dim=1)  # [B, C1+C2+C3, H, W]
        wrinkle = torch.sigmoid(self.wrinkle_head(texture_all))

        return brown, red, wrinkle


# ══════════════════════════════════════════════════════════════════════════════
# 파라미터 통계
# ══════════════════════════════════════════════════════════════════════════════
def build_model(base_ch: int = 64,
                low_r  : float = 0.1,
                high_r : float = 0.4) -> FreqAwareUNet:
    model = FreqAwareUNet(base_ch=base_ch, low_r=low_r, high_r=high_r)

    total = sum(p.numel() for p in model.parameters())
    freq_params = sum(
        p.numel()
        for m in [model.freq1, model.freq2, model.freq3]
        for p in m.parameters()
    )
    od_params = sum(p.numel() for p in model.od_prep.parameters())

    print(f"\n=== FreqAwareUNet 파라미터 ===")
    print(f"  ODPreprocess   : {od_params / 1e3:.1f}K")
    print(f"  FrequencyGate  : {freq_params / 1e3:.1f}K  ← 조명/chromophore/wrinkle 대역 학습")
    print(f"  전체           : {total / 1e6:.2f}M")
    print(f"  low_r={low_r}, high_r={high_r}\n")

    return model


if __name__ == '__main__':
    model = build_model()
    dummy = torch.randn(2, 3, 256, 256)
    brown, red, wrinkle = model(dummy)
    print(f"brown  : {brown.shape}")
    print(f"red    : {red.shape}")
    print(f"wrinkle: {wrinkle.shape}")
