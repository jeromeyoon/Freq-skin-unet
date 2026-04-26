"""
ambient_aug.py
==============
조명 변형 Augmentation — 매장 천장 조명 특화

배포 환경 가정
--------------
- 매장 천장 조명 (형광등 / LED)
- 기기 자체 LED 광원이 주광원 → 주변광은 부가 오염 수준
- 비네팅, 스탠드, 창문 패턴은 고려하지 않음

시뮬레이션 패턴
---------------
1. Intensity change      : 전체 밝기 균일 스케일 (기기 LED 강도 변동)
2. Color temperature     : 채널별 배율 — 천장 형광등(cool) vs LED(warm) 차이
3. Additive ambient fog  : 편광 차단 후 남은 천장 조명 leakage (물리적 오염)
4. Ceiling gradient      : 위→아래 방향 부드러운 기울기 (천장 조명 위치)

제거된 패턴
-----------
- Vignetting (비네팅): 매장 환경에서 해당 없음
- Tint shift (R/B 독립 변화): 천장 조명에서 미미함
- Directional random tilt: 천장은 거의 정면이므로 약한 범위로 제한
"""

import torch
import torch.nn.functional as F


# ── 공간 조명 패턴 생성 ──────────────────────────────────────────────────────


def _make_ceiling_gradient(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    천장 조명 그라데이션 [H, W]

    천장이 위에 있으므로 위쪽이 밝고 아래쪽이 약간 어두운 패턴.
    매장 천장은 비교적 균일 → 강도 변화 범위를 좁게 제한.
    좌우 tilt는 천장 조명 위치 편차를 시뮬레이션 (작은 범위).

    반환 범위: [0.88, 1.0]
    """
    y = torch.linspace(1.0, 0.0, H, device=device)   # 위=1.0, 아래=0.0
    x = torch.linspace(-1.0, 1.0, W, device=device)
    # 천장 조명은 정면에 가까우므로 tilt를 작게 제한
    tilt = torch.empty(1, device=device).uniform_(-0.1, 0.1).item()
    Y, X = torch.meshgrid(y, x, indexing='ij')

    grad = Y + tilt * X
    g_min, g_max = grad.min(), grad.max()
    grad = (grad - g_min) / (g_max - g_min + 1e-6)
    # [0.88, 1.0]: 매장 천장 조명은 부드러운 변화 (12% 이내)
    return (0.88 + 0.12 * grad).clamp(0.85, 1.0)


def _make_ambient_fog(
    H: int,
    W: int,
    alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """
    천장 조명 leakage 시뮬레이션 — additive ambient fog [3, H, W]

    편광 시스템이 대부분의 주변광을 차단하지만 완전 차단은 불가능.
    남은 leakage는 공간적으로 부드럽고 색온도가 일정한 패턴.

    물리: I_measured = I_cross_pol + alpha * I_ambient
    OD_measured = -log(I_cross_pol + alpha * I_ambient)  [비선형 오염]

    구현: 저해상도 랜덤 텍스처 → bilinear upscale → 부드러운 fog
    """
    # 저해상도 ambient 패턴 (공간적으로 smooth)
    fog_lr = torch.rand(1, 3, max(H // 16, 2), max(W // 16, 2), device=device)
    fog = F.interpolate(fog_lr, size=(H, W), mode='bilinear', align_corners=False)
    fog = fog.squeeze(0)  # [3, H, W]

    # 천장 조명 색온도는 neutral~slightly cool → 채널 편차 작게
    ch_tint = torch.empty(3, device=device).uniform_(0.9, 1.1)
    fog = fog * ch_tint.view(3, 1, 1)

    return (alpha * fog).clamp(0.0, 1.0)


# ── 단일 이미지 augmentation ────────────────────────────────────────────────


def apply_illumination_aug(
    rgb: torch.Tensor,
    intensity_range: tuple = (0.85, 1.15),
    color_temp_range: tuple = (0.85, 1.15),
    ambient_fog_alpha_range: tuple = (0.0, 0.05),
    ceiling_gradient_prob: float = 0.5,
) -> torch.Tensor:
    """
    단일 이미지에 천장 조명 기반 augmentation 적용

    Parameters
    ----------
    rgb                     : [3, H, W] float 0~1
    intensity_range         : 기기 LED 강도 변동 (±15%)
    color_temp_range        : 천장 조명 색온도 (형광 cool ~ LED warm)
    ambient_fog_alpha_range : 편광 차단 후 남은 leakage 비율 (0~5%)
    ceiling_gradient_prob   : 천장 기울기 적용 확률

    Returns
    -------
    rgb_aug : [3, H, W] float 0~1
    """
    device = rgb.device
    H, W = rgb.shape[1], rgb.shape[2]

    # 1. 전체 밝기 변화 (기기 LED 출력 변동)
    intensity = torch.empty(1, device=device).uniform_(*intensity_range).item()
    rgb_aug = rgb * intensity

    # 2. 색온도 변화 (천장 조명 채널별 영향 — 범위 좁게)
    ch_scales = torch.empty(3, device=device).uniform_(*color_temp_range)
    rgb_aug = rgb_aug * ch_scales.view(3, 1, 1)

    # 3. 천장 그라데이션 (위→아래, 작은 좌우 기울기)
    if torch.rand(1, device=device).item() < ceiling_gradient_prob:
        grad = _make_ceiling_gradient(H, W, device)
        rgb_aug = rgb_aug * grad.unsqueeze(0)

    # 4. Additive ambient fog (편광 leakage — 물리적 오염)
    alpha = torch.empty(1, device=device).uniform_(*ambient_fog_alpha_range).item()
    if alpha > 1e-4:
        fog = _make_ambient_fog(H, W, alpha, device)
        rgb_aug = rgb_aug + fog

    return rgb_aug.clamp(0.0, 1.0)


# ── 배치 단위 augmentation ───────────────────────────────────────────────────


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
