"""
ambient_aug.py
==============
조명 변형 Augmentation — 실내 환경 특화

배포 환경: 실내 (천장 조명, 창문, 스탠드 등)

시뮬레이션 패턴
---------------
1. Intensity change      : 전체 밝기 균일 스케일
2. Color temperature     : 채널별 독립 배율 (백열 warm ↔ 형광 cool)
3. Tint shift            : R/B 채널 편향
4. Vignetting            : 원형 국소 조명 (스탠드, 국소 광원)
5. Directional gradient  : 위→아래 선형 그라데이션 (천장 조명/창문) ← 실내 핵심 추가

순수 PyTorch 구현: CPU/numpy 왕복 없이 GPU에서 전부 처리
"""

import torch


# ── 공간 조명 패턴 생성 ──────────────────────────────────────────────────────


def _make_vignette_torch(H: int, W: int, strength: float,
                         device: torch.device) -> torch.Tensor:
    """가우시안 vignette 마스크 [H, W] — 원형 국소 광원 시뮬레이션"""
    cy, cx   = H / 2.0, W / 2.0
    y        = torch.arange(H, dtype=torch.float32, device=device)
    x        = torch.arange(W, dtype=torch.float32, device=device)
    Y, X     = torch.meshgrid(y, x, indexing='ij')
    dist     = torch.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    sigma    = min(H, W) / (2.0 * strength)
    mask     = torch.exp(-dist ** 2 / (2 * sigma ** 2))
    vignette = 1.0 - strength * (1.0 - mask)
    return vignette.clamp(0.1, 1.0)                      # [H, W]


def _make_directional_gradient(H: int, W: int,
                                device: torch.device) -> torch.Tensor:
    """
    방향성 조명 그라데이션 [H, W] — 천장 조명 / 창문 광원 시뮬레이션

    기본 방향: 위(밝음) → 아래(어두움)  (천장 조명의 가장 흔한 패턴)
    tilt: 좌우 기울기를 랜덤으로 추가 → 창문이 한쪽에 있는 상황 시뮬레이션

    반환 범위: [0.75, 1.0] — 실내 조명은 부드러운 변화
    """
    y    = torch.linspace(1.0, 0.0, H, device=device)    # 위=1.0, 아래=0.0
    x    = torch.linspace(-1.0, 1.0, W, device=device)
    tilt = torch.empty(1, device=device).uniform_(-0.3, 0.3).item()
    Y, X = torch.meshgrid(y, x, indexing='ij')

    grad = Y + tilt * X                                   # 기울어진 그라데이션
    # [0, 1] 정규화 후 [0.75, 1.0] 범위로 압축 (실내는 극단적 변화 없음)
    g_min, g_max = grad.min(), grad.max()
    grad = (grad - g_min) / (g_max - g_min + 1e-6)
    return (0.75 + 0.25 * grad).clamp(0.6, 1.0)          # [H, W]


# ── 단일 이미지 augmentation ────────────────────────────────────────────────


def apply_illumination_aug(rgb: torch.Tensor,
                            intensity_range   : tuple = (0.6, 1.4),
                            color_temp_range  : tuple = (0.7, 1.3),
                            tint_range        : tuple = (0.8, 1.2),
                            vignette_prob     : float = 0.5,
                            vignette_strength : float = 0.4,
                            gradient_prob     : float = 0.5,
                            ) -> torch.Tensor:
    """
    단일 이미지에 랜덤 실내 조명 변형 적용 (GPU 텐서에서 직접 처리)

    Parameters
    ----------
    rgb               : [3, H, W] float tensor 0~1
    intensity_range   : 전체 밝기 배율 (실내: 0.6~1.4)
    color_temp_range  : 색온도 배율 — 채널별 독립 (백열 warm ↔ 형광 cool)
    tint_range        : R/B 채널 틴트 배율
    vignette_prob     : 원형 국소 조명 적용 확률
    vignette_strength : vignette 강도
    gradient_prob     : 방향성 그라데이션 적용 확률 (천장/창문 조명)

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
    tint      = torch.ones(3, device=device)
    tint[0]   = torch.empty(1, device=device).uniform_(*tint_range).item()
    tint[2]   = torch.empty(1, device=device).uniform_(*tint_range).item()
    rgb_aug   = rgb_aug * tint.view(3, 1, 1)

    # 4. Vignetting — 원형 국소 조명 (스탠드, 포인트 광원)
    if torch.rand(1, device=device).item() < vignette_prob:
        strength = torch.empty(1, device=device).uniform_(0.1, vignette_strength).item()
        vignette = _make_vignette_torch(H, W, strength, device)
        rgb_aug  = rgb_aug * vignette.unsqueeze(0)

    # 5. 방향성 그라데이션 — 천장 조명 / 창문 (실내 핵심)
    if torch.rand(1, device=device).item() < gradient_prob:
        grad    = _make_directional_gradient(H, W, device)
        rgb_aug = rgb_aug * grad.unsqueeze(0)

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
