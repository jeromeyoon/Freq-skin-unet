"""
skin_mask_gen.py
================
inference용 피부 영역 마스크 자동 생성

data_prep.py 와 동일한 MediaPipe Face Mesh 랜드마크 기반 방법 사용.
  - 얼굴 윤곽선 폴리곤 채움
  - 귀·목 경계 침식 제거
  - 눈·눈썹 영역 팽창 후 제거
  - MediaPipe 미설치 시 → 이미지 중앙 타원형 fallback

의존성
------
    pip install mediapipe opencv-python

사용 예
-------
    from skin_mask_gen import generate_skin_mask

    rgb = load_rgb_full(cross_path)       # [3, H, W] float 0~1
    mask = generate_skin_mask(rgb)        # [1, H, W] binary float 0/1
"""

from __future__ import annotations

import numpy as np
import torch
import cv2


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe Face Mesh 랜드마크 인덱스 (data_prep.py 와 동일)
# ══════════════════════════════════════════════════════════════════════════════
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

_LEFT_EYE      = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE     = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_EYEBROW  = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

_EXCLUDE_REGIONS = [_LEFT_EYE, _RIGHT_EYE, _LEFT_EYEBROW, _RIGHT_EYEBROW]


# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe 임포트 헬퍼
# ══════════════════════════════════════════════════════════════════════════════
def _try_import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 마스크 생성 (data_prep.py 와 동일한 로직)
# ══════════════════════════════════════════════════════════════════════════════
def _mask_from_mediapipe(img_bgr: np.ndarray, mp_face_mesh) -> np.ndarray | None:
    """
    MediaPipe Face Mesh로 피부 마스크 생성.

    - 얼굴 윤곽선 폴리곤 채움 → 귀·목 침식 제거
    - 눈·눈썹 영역 팽창 후 제거

    Returns
    -------
    uint8 [H, W]  (유효 피부=255, 나머지=0)
    얼굴 미검출 시 None
    """
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode        = True,
        max_num_faces            = 1,
        refine_landmarks         = False,
        min_detection_confidence = 0.4,
    ) as fm:
        results = fm.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    def lm_pts(indices: list[int]) -> np.ndarray:
        return np.array(
            [[int(lm[i].x * W), int(lm[i].y * H)] for i in indices],
            dtype=np.int32,
        )

    # 1. 얼굴 윤곽선 폴리곤 채움
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [lm_pts(_FACE_OVAL)], 255)

    # 침식: 귀·목 경계 제거
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask    = cv2.erode(mask, k_erode, iterations=2)

    # 2. 눈·눈썹 영역 제거
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    for indices in _EXCLUDE_REGIONS:
        excl = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(excl, [lm_pts(indices)], 255)
        excl = cv2.dilate(excl, k_dilate, iterations=1)   # 여유 패딩
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl))

    return mask


def _mask_ellipse_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """
    MediaPipe 미설치 또는 얼굴 미검출 시 fallback.
    이미지 중앙 타원형 마스크 (data_prep.py 와 동일).
    """
    H, W = img_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    cx, cy = W // 2, int(H * 0.45)
    axes   = (int(W * 0.35), int(H * 0.40))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════════════════════════════════════
def generate_skin_mask(
    rgb:           torch.Tensor,
    use_mediapipe: bool = True,
) -> torch.Tensor:
    """
    교차 편광 RGB 이미지에서 피부 영역 이진 마스크 생성.
    data_prep.py 와 동일한 Face Mesh 랜드마크 기반 방법 사용.

    Parameters
    ----------
    rgb           : [3, H, W] float 0~1  (교차 편광 이미지)
    use_mediapipe : False면 타원형 fallback만 사용

    Returns
    -------
    mask : [1, H, W] binary float (0.0 or 1.0)
    """
    # torch tensor → BGR numpy (OpenCV/MediaPipe 입력 포맷)
    img_np  = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mask_np: np.ndarray | None = None

    if use_mediapipe:
        mp = _try_import_mediapipe()
        if mp is not None:
            mask_np = _mask_from_mediapipe(img_bgr, mp.solutions.face_mesh)
            if mask_np is None:
                print("  [WARN] MediaPipe 얼굴 미검출 → 타원형 fallback")

    if mask_np is None:
        mask_np = _mask_ellipse_fallback(img_bgr)

    # uint8 [H, W] (0 or 255) → float tensor [1, H, W] (0.0 or 1.0)
    mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
    return mask_tensor
