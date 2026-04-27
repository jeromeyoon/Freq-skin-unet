"""
ir_encoder_v2.py
================
CrossPolEncoderV2 : H2 수정 — OD 공간에서 직접 주파수 분리

기존 FrequencyGate(ir_encoder_stable.py) 문제
----------------------------------------------
기존 파이프라인:
  RGB → ODPreprocess → ConvBlocks → FrequencyGate(feature space)

문제: ConvBlock을 거친 feature space에서 FFT 대역 분리는 Beer-Lambert 가정과 불일치.
  Beer-Lambert는 OD 공간에서 덧셈 모델:
    OD_observed = OD_illumination(저주파) + OD_chromophore(중주파)
  따라서 조명 제거는 OD 원시값에 직접 적용해야 수학적으로 성립.
  ConvBlock 이후에는 저주파/중주파가 비선형적으로 혼합되어 있어 단순 FFT 분리로
  OD_illumination을 정확히 추출할 수 없음.

V2 파이프라인
-------------
  RGB → OD 계산 → ODFrequencyFilter(OD space) → od_chroma → ConvBlocks

  ODFrequencyFilter:
    od_low   = FFT_low(od)         (조명 — 저주파 성분)
    od_mid   = FFT_mid(od)         (발색단 — 중주파 성분)
    od_chroma = od_mid - gate_low × od_low   ← Beer-Lambert 정확 준수

  gate_low = sigmoid(low_logit), 초기값 sigmoid(-3) ≈ 0.047
    → 초기: 저주파 억제 미약 (od_mid 위주 사용)
    → 학습 진행: w_freq_reg 스케줄에 따라 조명 제거 강도 조절

사용 예
-------
    from ir_encoder_v2 import CrossPolEncoderV2, EncoderOutput
    enc = CrossPolEncoderV2(base_ch=64)
    out = enc(rgb_cross)
    # out.chroma    = [e1, e2, e3]   발색단 multi-scale features
    # out.bottleneck= e4             global context
"""

from collections import namedtuple
import torch
import torch.nn as nn


EncoderOutput = namedtuple('EncoderOutput', ['chroma', 'texture', 'bottleneck'])


# ══════════════════════════════════════════════════════════════════════════════
# ODFrequencyFilter — OD 공간 직접 주파수 분리 (H2 핵심)
# ══════════════════════════════════════════════════════════════════════════════
class ODFrequencyFilter(nn.Module):
    """
    OD 공간에서 Beer-Lambert 가정에 부합하는 조명 제거

    수식
    ----
      od_low    = IFFT(FFT(od) * low_mask)      [저주파 = 조명]
      od_mid    = IFFT(FFT(od) * mid_mask)      [중주파 = 발색단]
      od_chroma = od_mid - gate_low * od_low    [조명 제거된 발색단 OD]

    gate_low = sigmoid(low_logit)  — 채널(R, G, B)별 독립 학습
      초기: sigmoid(-3) ≈ 0.047  → 억제 미약 (안전하게 시작)
      학습: w_freq_reg가 증가할수록 gate_low가 열리며 조명 제거 강화

    기존 feature-space FrequencyGate와 비교
    ----------------------------------------
    기존: x_chroma = x_mid * gate_mid - x_low * gate_low  (feature space)
      - ConvBlock 이후라 저/중주파 성분이 비선형 혼합 → Beer-Lambert 위반
    V2:  od_chroma = od_mid - gate_low * od_low           (OD space)
      - 원시 OD에 직접 적용 → additive Beer-Lambert 정확 준수
    """

    def __init__(self, low_r: float = 0.1, high_r: float = 0.4):
        super().__init__()
        self.low_r  = low_r
        self.high_r = high_r
        # R, G, B 채널별 독립 억제 강도 (초기: sigmoid(-1) ≈ 0.27)
        # -3(≈0.05)에서 -1(≈0.27)로 올려 초기부터 조명 제거 동작
        self.low_logit = nn.Parameter(torch.full((1, 3, 1, 1), -1.0))

    def _radial_mask(self, H: int, W: int,
                     r_lo: float, r_hi: float,
                     device: torch.device) -> torch.Tensor:
        cy, cx = H / 2.0, W / 2.0
        ys = torch.arange(H, device=device, dtype=torch.float32).view(-1, 1) - cy
        xs = torch.arange(W, device=device, dtype=torch.float32).view(1, -1) - cx
        r  = torch.sqrt(ys ** 2 + xs ** 2) / min(H, W)
        return (r >= r_lo) & (r < r_hi)

    def forward(self, od: torch.Tensor) -> torch.Tensor:
        B, C, H, W = od.shape
        compute_dtype = od.dtype

        # AMP 환경에서 complex half 오류 및 NaN 방지 — float32 강제
        if od.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                od_fp32 = od.float()
                F_shift = torch.fft.fftshift(torch.fft.fft2(od_fp32))
                low_mask = self._radial_mask(H, W, 0.0, self.low_r, od.device)
                mid_mask = self._radial_mask(H, W, self.low_r, self.high_r, od.device)
                od_low = torch.fft.ifft2(torch.fft.ifftshift(F_shift * low_mask)).real
                od_mid = torch.fft.ifft2(torch.fft.ifftshift(F_shift * mid_mask)).real
                gate_low = torch.sigmoid(self.low_logit).float()
                od_chroma = od_mid - gate_low * od_low
        else:
            F_shift = torch.fft.fftshift(torch.fft.fft2(od))
            low_mask = self._radial_mask(H, W, 0.0, self.low_r, od.device)
            mid_mask = self._radial_mask(H, W, self.low_r, self.high_r, od.device)
            od_low = torch.fft.ifft2(torch.fft.ifftshift(F_shift * low_mask)).real
            od_mid = torch.fft.ifft2(torch.fft.ifftshift(F_shift * mid_mask)).real
            gate_low = torch.sigmoid(self.low_logit)
            od_chroma = od_mid - gate_low * od_low

        return od_chroma.to(dtype=compute_dtype)


# ══════════════════════════════════════════════════════════════════════════════
# ConvBlock
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
# CrossPolEncoderV2
# ══════════════════════════════════════════════════════════════════════════════
class CrossPolEncoderV2(nn.Module):
    """
    교차 편광 인코더 V2 — OD 공간 직접 주파수 분리

    Pipeline
    --------
    1. RGB → raw OD  (-log(rgb.clamp(eps)).clamp(-6, 6))
    2. ODFrequencyFilter → od_chroma  [B, 3, H, W]
       (Beer-Lambert 준수: od_chroma = od_mid - gate_low * od_low)
    3. 색차 채널 추가: [od_chroma(3ch) | rg(1ch) | gb(1ch)] → 5ch
       input_proj: 5ch → 3ch  (학습 가능 혼합)
    4. ConvBlock × 4  (enc1~enc4)

    출력: EncoderOutput(
        chroma    = [e1, e2, e3],   발색단 features (SpotHeadV2 입력)
        texture   = [],             cross-pol에서는 미사용
        bottleneck= e4              global context
    )
    """

    def __init__(self, base_ch: int = 64,
                 low_r:   float = 0.1,
                 high_r:  float = 0.4,
                 eps:     float = 1e-3):
        super().__init__()
        ch = base_ch
        self.eps = eps

        self.od_filter  = ODFrequencyFilter(low_r, high_r)

        # rg/gb 색차 포함 5ch → 3ch 입력 프로젝션 (항등 초기화)
        self.input_proj = nn.Conv2d(5, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            nn.init.zeros_(self.input_proj.weight)
            self.input_proj.weight[:3, :3, 0, 0] = torch.eye(3)

        self.pool = nn.MaxPool2d(2)
        self.enc1 = ConvBlock(3,    ch)
        self.enc2 = ConvBlock(ch,   ch * 2)
        self.enc3 = ConvBlock(ch*2, ch * 4)
        self.enc4 = ConvBlock(ch*4, ch * 8)

    def forward(self, rgb: torch.Tensor) -> EncoderOutput:
        # 1. OD 계산 (수치 안정화)
        od = -torch.log(rgb.clamp(min=self.eps, max=1.0)).clamp(-6.0, 6.0)

        # 2. Gray-world 색온도 정규화 (색온도 불변 OD)
        # 곱셈 조명 변화(warm/cool light)는 OD_R, OD_G, OD_B를 동일하게 이동시킴.
        # 채널별 공간 평균을 빼면 이 이동이 제거되고 발색단 비율 정보만 남음.
        # 예: warm light → OD_R 감소, OD_G 감소 → 평균 제거 후 상대 비율 보존.
        od_mean = od.flatten(2).mean(dim=2).unsqueeze(-1).unsqueeze(-1)  # [B,3,1,1]
        od = od - od_mean

        # 3. OD 공간에서 조명 제거 (Beer-Lambert additive 성질 활용)
        od_chroma = self.od_filter(od)                # [B, 3, H, W]

        # 4. 색차 채널 추가 후 프로젝션
        rg = od_chroma[:, 0:1] - od_chroma[:, 1:2]   # R-G 색차
        gb = od_chroma[:, 1:2] - od_chroma[:, 2:3]   # G-B 색차
        x5 = torch.cat([od_chroma, rg, gb], dim=1)   # [B, 5, H, W]
        x  = self.input_proj(x5)                      # [B, 3, H, W]

        # 4. 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        return EncoderOutput(
            chroma    = [e1, e2, e3],
            texture   = [],
            bottleneck= e4,
        )
