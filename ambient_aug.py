"""
ambient_aug.py
==============
조명 변형 Augmentation (Illumination Augmentation)

Beer-Lambert 기반:
  RGB = illuminant × reflectance
  조명 변형 = illuminant 를 곱/나누기

방법
----
1. Color temperature shift (색온도 변화)
   → R/G/B 채널별 다른 배율 적용
2. Intensity change (밝기 변화)
   → 전체 스케일 변경
3. Vignetting (주변부 어두움)
   → 가우시안 마스크로 공간적 조명 변화
4. Tint shift (색상 편향)
   → 특정 채널 강조
"""

import numpy as np
import torch


def _make_vignette(H: int, W: int, strength: float) -> np.ndarray:
    """가우시안 vignette 마스크 [H, W]"""
    cy, cx  = H / 2, W / 2
    Y, X    = np.mgrid[0:H, 0:W]
    dist    = np.sqrt((Y - cy)**2 + (X - cx)**2)
    sigma   = min(H, W) / (2.0 * strength)
    mask    = np.exp(-dist**2 / (2 * sigma**2))
    # strength=0 → 균일, strength 클수록 주변부 어두움
    vignette = 1.0 - strength * (1.0 - mask)
    return vignette.clip(0.1, 1.0)


def apply_illumination_aug(rgb: torch.Tensor,
                            intensity_range : tuple = (0.5, 1.8),
                            color_temp_range: tuple = (0.7, 1.3),
                            tint_range      : tuple = (0.8, 1.2),
                            vignette_prob   : float = 0.3,
                            vignette_strength: float = 0.4,
                            ) -> torch.Tensor:
    """
    단일 이미지에 랜덤 조명 변형 적용

    Parameters
    ----------
    rgb              : [3, H, W] float tensor 0~1
    intensity_range  : 전체 밝기 배율 범위
    color_temp_range : 색온도 배율 범위 (채널별 독립)
    tint_range       : 틴트 배율 범위
    vignette_prob    : vignette 적용 확률
    vignette_strength: vignette 강도

    Returns
    -------
    rgb_aug : [3, H, W] float tensor 0~1
    """
    device = rgb.device
    rgb_np  = rgb.cpu().numpy()   # [3, H, W]
    H, W    = rgb_np.shape[1], rgb_np.shape[2]

    # 1. 전체 밝기 변화
    intensity = np.random.uniform(*intensity_range)
    rgb_aug   = rgb_np * intensity

    # 2. 색온도 변화 (채널별 독립 배율)
    for c in range(3):
        scale = np.random.uniform(*color_temp_range)
        rgb_aug[c] *= scale

    # 3. 틴트 변화 (두 채널 반대 방향)
    tint_r = np.random.uniform(*tint_range)
    tint_b = np.random.uniform(*tint_range)
    rgb_aug[0] *= tint_r
    rgb_aug[2] *= tint_b

    # 4. Vignetting (공간적 조명 변화)
    if np.random.random() < vignette_prob:
        strength = np.random.uniform(0.1, vignette_strength)
        vignette = _make_vignette(H, W, strength)   # [H, W]
        rgb_aug  = rgb_aug * vignette[None]          # broadcast [3, H, W]

    rgb_aug = np.clip(rgb_aug, 0.0, 1.0)
    return torch.from_numpy(rgb_aug.astype(np.float32)).to(device)


def apply_batch_illumination_aug(rgb_batch: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    배치 단위 조명 변형

    rgb_batch : [B, 3, H, W]
    returns   : [B, 3, H, W]
    """
    return torch.stack([
        apply_illumination_aug(rgb_batch[i], **kwargs)
        for i in range(rgb_batch.shape[0])
    ])
