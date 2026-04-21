"""
parallel_encoder.py
===================
ParallelPolEncoder : 평행 편광 이미지 → 주름/텍스처 feature 추출기

물리적 배경
-----------
평행 편광(parallel polarization):
  - 표면 반사 + 피부 하부 산란 모두 포착
  - 주름은 표면 기하학적 구조 → 고주파 텍스처로 표현
  - OD 전처리 불필요 (Beer-Lambert는 투과 발색단용)

조명 강인성
-----------
  - Instance Normalization으로 이미지별 밝기/채널 평균 제거
  - FrequencyGate: low-freq(조명) 억제, high-freq(주름) 보존

출력 (ParallelOutput namedtuple)
---------------------------------
  texture    : [t1, t2, t3]  high-freq features (WrinkleHead 입력)
  bottleneck : enc4 출력 [B, base_ch*8, H/8, H/8]
"""

from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 출력 타입 ──────────────────────────────────────────────────────────────────
ParallelOutput = namedtuple('ParallelOutput', ['texture', 'bottleneck'])


# ══════════════════════════════════════════════════════════════════════════════
# 공통 빌딩블록
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
# 평행 편광 전처리
# ══════════════════════════════════════════════════════════════════════════════
class ParallelPreprocess(nn.Module):
    """
    평행 편광 이미지 전처리

    Instance Normalization → 각 이미지의 조명 평균/분산 제거
    이후 1×1 Conv로 채널 혼합 (학습 가능)

    OD(-log)를 쓰지 않는 이유:
      - OD는 투과 발색단(Beer-Lambert) 모델 기반
      - 평행 편광은 표면 반사 포함 → OD 가정 위반
      - 대신 IN으로 조명 의존 통계를 직접 제거
    """

    def __init__(self):
        super().__init__()
        # 채널별 독립 Instance Normalization (affine=True: 학습 가능 scale/bias)
        self.inst_norm = nn.InstanceNorm2d(3, affine=True)
        self.proj      = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        nn.init.eye_(self.proj.weight.squeeze(-1).squeeze(-1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        x = self.inst_norm(rgb)   # [B, 3, H, W] — 조명 평균 제거
        return self.proj(x)       # [B, 3, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# FrequencyGate (고주파 특화)
# ══════════════════════════════════════════════════════════════════════════════
class HighFreqGate(nn.Module):
    """
    평행 편광용 FrequencyGate

    주름은 고주파 → high_logit 초기값 +2 (강하게 보존)
    조명은 저주파 → low_logit 초기값 −2 (강하게 억제)
    mid_logit = +1 (중간 보존, 주름 주변 구조 참고)

    forward(x) → x_texture  [B, C, H, W]  high-freq 성분만 반환
    """

    def __init__(self, channels: int, low_r: float = 0.1, high_r: float = 0.4):
        super().__init__()
        self.low_r  = low_r
        self.high_r = high_r
        self.low_logit  = nn.Parameter(torch.full((1, channels, 1, 1), -2.0))
        self.high_logit = nn.Parameter(torch.full((1, channels, 1, 1), +2.0))

    def _radial_mask(self, H, W, r_lo, r_hi, device):
        cy, cx = H / 2.0, W / 2.0
        ys = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1) - cy
        xs = torch.arange(W, device=device, dtype=torch.float32).view(1, -1) - cx
        r  = torch.sqrt(ys ** 2 + xs ** 2) / min(H, W)
        return (r >= r_lo) & (r < r_hi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # FFT 경로는 mixed precision에서 complex half 경고 및 NaN을 유발할 수 있어
        # autocast 밖에서 float32로 강제 계산한다.
        if x.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                x_fp32 = x.float()
                F_shift = torch.fft.fftshift(torch.fft.fft2(x_fp32))

                high_mask = self._radial_mask(H, W, self.high_r, 1.0, x.device)
                high_mask = high_mask.to(dtype=F_shift.dtype)
                gate_high = torch.sigmoid(self.high_logit).float()

                x_texture = torch.fft.ifft2(
                    torch.fft.ifftshift(F_shift * high_mask)
                ).real * gate_high
        else:
            x_fp32 = x.float()
            F_shift = torch.fft.fftshift(torch.fft.fft2(x_fp32))

            high_mask = self._radial_mask(H, W, self.high_r, 1.0, x.device)
            high_mask = high_mask.to(dtype=F_shift.dtype)
            gate_high = torch.sigmoid(self.high_logit).float()

            x_texture = torch.fft.ifft2(
                torch.fft.ifftshift(F_shift * high_mask)
            ).real * gate_high

        return x_texture.to(dtype=x.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# ParallelPolEncoder
# ══════════════════════════════════════════════════════════════════════════════
class ParallelPolEncoder(nn.Module):
    """
    평행 편광 전용 인코더

    Pipeline
    --------
    rgb_parallel
     → ParallelPreprocess (InstanceNorm + 1×1 Conv)
     → enc1 [B, ch,   H,   W]   → HighFreqGate → texture1
     → enc2 [B, ch*2, H/2, H/2] → HighFreqGate → texture2
     → enc3 [B, ch*4, H/4, H/4] → HighFreqGate → texture3
     → enc4 [B, ch*8, H/8, H/8] (bottleneck)

    출력: ParallelOutput(
        texture    = [texture1, texture2, texture3],
        bottleneck = enc4
    )
    """

    def __init__(self, base_ch: int = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4):
        super().__init__()
        ch = base_ch

        self.preprocess = ParallelPreprocess()
        self.pool       = nn.MaxPool2d(2)

        # Encoder (교차 편광 인코더와 동일 구조, 독립 가중치)
        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)
        self.enc4 = ConvBlock(ch*4, ch * 8)   # bottleneck

        # High-freq gate (3 levels)
        self.hgate1 = HighFreqGate(ch,   low_r, high_r)
        self.hgate2 = HighFreqGate(ch*2, low_r, high_r)
        self.hgate3 = HighFreqGate(ch*4, low_r, high_r)

    def forward(self, rgb_parallel: torch.Tensor) -> ParallelOutput:
        x = self.preprocess(rgb_parallel)

        e1 = self.enc1(x)               # [B, ch,   H,   W]
        e2 = self.enc2(self.pool(e1))   # [B, ch*2, H/2, H/2]
        e3 = self.enc3(self.pool(e2))   # [B, ch*4, H/4, H/4]
        e4 = self.enc4(self.pool(e3))   # [B, ch*8, H/8, H/8]

        t1 = self.hgate1(e1)
        t2 = self.hgate2(e2)
        t3 = self.hgate3(e3)

        return ParallelOutput(
            texture    = [t1, t2, t3],
            bottleneck = e4,
        )
