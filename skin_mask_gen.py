"""
skin_mask_gen.py
================
입력 이미지에서 피부 영역 이진 마스크 자동 생성

방법
----
1차: MediaPipe Selfie Segmentation  (설치된 경우 자동 사용, 더 정확)
2차: YCbCr + HSV 색상 기반 검출    (추가 의존성 없음, 폴백)

YCbCr 기준 (ITU-R BT.601 기반)
  Cb : 77 ~ 127
  Cr : 133 ~ 173

HSV 기준
  H : 0~17 또는 330~360 (붉은~노란 계열)
  S : 0.15 ~ 0.85
  V : 0.2 ~ 1.0

편광 이미지 특성
  교차 편광(cross)은 표면 반사 제거 → 깊은 피부 발색단 색이 강조됨
  색상 범위를 약간 넓게 설정해 편광 이미지에도 대응

사용 예
-------
    from skin_mask_gen import generate_skin_mask

    rgb = load_rgb_full(cross_path)          # [3, H, W] float 0~1
    mask = generate_skin_mask(rgb)           # [1, H, W] binary float 0/1
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# 형태학적 연산 (torch 기반, GPU/CPU 모두 동작)
# ══════════════════════════════════════════════════════════════════════════════

def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """이진 팽창 (dilation). mask: [1, 1, H, W] float 0/1"""
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius).clamp(0.0, 1.0)


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """이진 침식 (erosion). mask: [1, 1, H, W] float 0/1"""
    k = 2 * radius + 1
    return 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=radius).clamp(0.0, 1.0)


def _morphological_close(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Closing = 팽창 → 침식 (구멍 메우기)"""
    return _erode(_dilate(mask, radius), radius)


def _morphological_open(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Opening = 침식 → 팽창 (노이즈 제거)"""
    return _dilate(_erode(mask, radius), radius)


# ══════════════════════════════════════════════════════════════════════════════
# 색상 기반 피부 검출 (numpy/PIL 전용, 추가 의존성 없음)
# ══════════════════════════════════════════════════════════════════════════════

def _skin_by_color(rgb: torch.Tensor) -> torch.Tensor:
    """
    YCbCr + HSV 조합으로 피부 마스크 생성.

    Parameters
    ----------
    rgb : [3, H, W] float 0~1

    Returns
    -------
    mask : [1, 1, H, W] float 0/1  (형태학적 처리 전 raw mask)
    """
    # [H, W, 3] uint8 변환
    img_np = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np, 'RGB')

    # ── YCbCr 기반 ───────────────────────────────────────────────────────────
    ycbcr = np.array(img_pil.convert('YCbCr'), dtype=np.uint8)
    Cb = ycbcr[:, :, 1].astype(np.int32)
    Cr = ycbcr[:, :, 2].astype(np.int32)
    # 편광 이미지 대응: 범위를 약간 넓게
    skin_ycbcr = (
        (Cb >= 70) & (Cb <= 135) &
        (Cr >= 128) & (Cr <= 180)
    )

    # ── HSV 기반 ─────────────────────────────────────────────────────────────
    # PIL은 HSV를 직접 지원하지 않으므로 numpy로 계산
    rgb_f = img_np.astype(np.float32) / 255.0
    R, G, B = rgb_f[:, :, 0], rgb_f[:, :, 1], rgb_f[:, :, 2]
    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin + 1e-8

    # Hue 계산 [0, 360)
    H = np.zeros_like(Cmax)
    mask_r = (Cmax == R) & (delta > 1e-8)
    mask_g = (Cmax == G) & (delta > 1e-8)
    mask_b = (Cmax == B) & (delta > 1e-8)
    H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / delta[mask_r]) % 360)
    H[mask_g] = (60.0 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 120)
    H[mask_b] = (60.0 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 240)

    S = np.where(Cmax > 1e-8, delta / (Cmax + 1e-8), 0.0)
    V = Cmax

    # 피부색: 붉은~노란 계열 H[0,25] ∪ [330,360], S와 V 범위 제한
    skin_hsv = (
        ((H <= 25) | (H >= 330)) &
        (S >= 0.12) & (S <= 0.90) &
        (V >= 0.20)
    )

    # ── 합산 ──────────────────────────────────────────────────────────────────
    skin = (skin_ycbcr | skin_hsv).astype(np.float32)   # [H, W]

    # torch tensor로 변환: [1, 1, H, W]
    mask = torch.from_numpy(skin).unsqueeze(0).unsqueeze(0)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe (선택적 의존성)
# ══════════════════════════════════════════════════════════════════════════════

def _skin_by_mediapipe(rgb: torch.Tensor) -> torch.Tensor | None:
    """
    MediaPipe Selfie Segmentation으로 피부(전경) 마스크 생성.
    패키지 미설치 시 None 반환 → 색상 기반으로 폴백.
    """
    try:
        import mediapipe as mp
    except ImportError:
        return None

    img_np = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)

    seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    result = seg.process(img_np)
    seg.close()

    if result.segmentation_mask is None:
        return None

    mask_np = (result.segmentation_mask > 0.5).astype(np.float32)   # [H, W]
    return torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)       # [1, 1, H, W]


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════════════════════════════════════

def generate_skin_mask(
    rgb:           torch.Tensor,
    close_radius:  int   = 12,
    open_radius:   int   = 4,
    use_mediapipe: bool  = True,
) -> torch.Tensor:
    """
    교차 편광 RGB 이미지에서 피부 영역 이진 마스크 생성.

    Parameters
    ----------
    rgb           : [3, H, W] float 0~1  (교차 편광 이미지 권장)
    close_radius  : Closing 연산 반경 (구멍 메우기, px 단위)
    open_radius   : Opening 연산 반경 (작은 노이즈 제거, px 단위)
    use_mediapipe : True면 MediaPipe 우선 시도, 실패 시 색상 기반으로 폴백

    Returns
    -------
    mask : [1, H, W] binary float (0.0 or 1.0)
    """
    raw: torch.Tensor | None = None

    # 1차: MediaPipe
    if use_mediapipe:
        raw = _skin_by_mediapipe(rgb)

    # 2차: 색상 기반 (폴백)
    if raw is None:
        raw = _skin_by_color(rgb)   # [1, 1, H, W]

    # 형태학적 정제
    closed = _morphological_close(raw,    close_radius)
    opened = _morphological_open(closed,  open_radius)

    # 이진화 + [1, H, W] 반환
    mask = (opened.squeeze(0) >= 0.5).float()   # [1, H, W]
    return mask
