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

순수 PyTorch 구현: CPU/numpy 왕복 없이 GPU에서 전부 처리
"""

import torch


def _make_vignette_torch(H: int, W: int, strength: float,
                         device: torch.device) -> torch.Tensor:
    """가우시안 vignette 마스크 [H, W] — GPU 텐서 반환"""
    cy, cx  = H / 2.0, W / 2.0
    y = torch.arange(H, dtype=torch.float32, device=device)
    x = torch.arange(W, dtype=torch.float32, device=device)
    Y, X    = torch.meshgrid(y, x, indexing='ij')
    dist    = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    sigma   = min(H, W) / (2.0 * strength)
    mask    = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    vignette = 1.0 - strength * (1.0 - mask)
    return vignette.clamp(0.1, 1.0)                     # [H, W]


def apply_illumination_aug(rgb: torch.Tensor,
                            intensity_range  : tuple = (0.5, 1.8),
                            color_temp_range : tuple = (0.7, 1.3),
                            tint_range       : tuple = (0.8, 1.2),
                            vignette_prob    : float = 0.3,
                            vignette_strength: float = 0.4,
                            ) -> torch.Tensor:
    """
    단일 이미지에 랜덤 조명 변형 적용 (GPU 텐서에서 직접 처리)

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
    H, W   = rgb.shape[1], rgb.shape[2]

    # 1. 전체 밝기 변화
    intensity = torch.empty(1, device=device).uniform_(*intensity_range).item()
    rgb_aug   = rgb * intensity

    # 2. 색온도 변화 (채널별 독립 배율)
    ch_scales = torch.empty(3, device=device).uniform_(*color_temp_range)
    rgb_aug   = rgb_aug * ch_scales.view(3, 1, 1)

    # 3. 틴트 변화 (R, B 채널 독립 배율)
    tint = torch.empty(3, device=device).fill_(1.0)
    tint[0] = torch.empty(1, device=device).uniform_(*tint_range).item()
    tint[2] = torch.empty(1, device=device).uniform_(*tint_range).item()
    rgb_aug  = rgb_aug * tint.view(3, 1, 1)

    # 4. Vignetting (공간적 조명 변화)
    if torch.rand(1, device=device).item() < vignette_prob:
        strength = torch.empty(1, device=device).uniform_(0.1, vignette_strength).item()
        vignette = _make_vignette_torch(H, W, strength, device)   # [H, W]
        rgb_aug  = rgb_aug * vignette.unsqueeze(0)                 # [3, H, W]

    return rgb_aug.clamp(0.0, 1.0)


def apply_batch_illumination_aug(rgb_batch: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    배치 단위 조명 변형 (이미지별 독립 랜덤 파라미터)

    rgb_batch : [B, 3, H, W]
    returns   : [B, 3, H, W]
    """
    return torch.stack([
        apply_illumination_aug(rgb_batch[i], **kwargs)
        for i in range(rgb_batch.shape[0])
    ])
