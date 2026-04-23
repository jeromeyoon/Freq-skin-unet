"""
skin_mask_gen.py
================
Inference-time facial skin mask generation.

This module now supports the same MediaPipe Tasks FaceLandmarker path used by
``new_data_prep.py``.  When ``face_landmarker_task_path`` is supplied,
``generate_skin_mask`` uses the new-data-prep mask rules:

- face oval fill
- erode face boundary
- exclude eyes / eyebrows / lips / inner mouth
- remove neck region below jawline
- use the same ellipse fallback shape as ``new_data_prep.py``

If no ``face_landmarker_task_path`` is supplied, the legacy
``mp.solutions.face_mesh`` path is kept for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

_OUTER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 78,
]

_INNER_MOUTH = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178,
]

_JAWLINE = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 148,
    152,
    377, 400, 378, 379, 365, 397, 288, 361, 323, 454,
]

_EXCLUDE_REGIONS_LEGACY = [_LEFT_EYE, _RIGHT_EYE, _LEFT_EYEBROW, _RIGHT_EYEBROW]
_EXCLUDE_REGIONS_NEW_PREP = [
    _LEFT_EYE,
    _RIGHT_EYE,
    _LEFT_EYEBROW,
    _RIGHT_EYEBROW,
    _OUTER_LIPS,
    _INNER_MOUTH,
]


def _try_import_mediapipe():
    try:
        import mediapipe as mp

        return mp
    except ImportError:
        return None


def _create_face_landmarker(mp, task_model_path: str | None):
    """Create the MediaPipe Tasks FaceLandmarker used by new_data_prep.py."""
    if not task_model_path:
        return None

    model_path = Path(task_model_path)
    if not model_path.exists():
        print(f"  [WARN] face_landmarker.task not found: {model_path} -> fallback")
        return None

    try:
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"  [WARN] FaceLandmarker init failed ({e}) -> fallback")
        return None


def _fill_landmark_regions(
    h: int,
    w: int,
    lm_pts,
    exclude_regions: list[list[int]],
    remove_neck: bool,
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lm_pts(_FACE_OVAL)], 255)

    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, k_erode, iterations=2)

    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    for indices in exclude_regions:
        excl = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(excl, [lm_pts(indices)], 255)
        excl = cv2.dilate(excl, k_dilate, iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl))

    if remove_neck:
        jaw_pts = lm_pts(_JAWLINE)
        jaw_cut = np.vstack([
            jaw_pts,
            np.array([[w - 1, h - 1], [0, h - 1]], dtype=np.int32),
        ])
        neck_region = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(neck_region, [jaw_cut], 255)
        jaw_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        neck_region = cv2.dilate(neck_region, jaw_dilate, iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(neck_region))

    return mask


def _mask_from_legacy_face_mesh(img_bgr: np.ndarray, mp_face_mesh) -> np.ndarray | None:
    """Backward-compatible mp.solutions.face_mesh skin mask."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.4,
    ) as fm:
        results = fm.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    def lm_pts(indices: list[int]) -> np.ndarray:
        return np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in indices],
            dtype=np.int32,
        )

    return _fill_landmark_regions(
        h=h,
        w=w,
        lm_pts=lm_pts,
        exclude_regions=_EXCLUDE_REGIONS_LEGACY,
        remove_neck=False,
    )


def _mask_from_face_landmarker_tasks(img_bgr: np.ndarray, mp, landmarker) -> np.ndarray | None:
    """new_data_prep.py-compatible MediaPipe Tasks skin mask."""
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    results = landmarker.detect(mp_image)
    if not results.face_landmarks:
        return None

    lm = results.face_landmarks[0]

    def lm_pts(indices: list[int]) -> np.ndarray:
        return np.array(
            [[int(lm[i].x * w), int(lm[i].y * h)] for i in indices],
            dtype=np.int32,
        )

    return _fill_landmark_regions(
        h=h,
        w=w,
        lm_pts=lm_pts,
        exclude_regions=_EXCLUDE_REGIONS_NEW_PREP,
        remove_neck=True,
    )


def _mask_ellipse_fallback(img_bgr: np.ndarray) -> np.ndarray:
    """new_data_prep.py-compatible ellipse fallback."""
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = w // 2, int(h * 0.43)
    axes = (int(w * 0.33), int(h * 0.36))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    neck_cut_y = min(h, cy + int(axes[1] * 0.78))
    mask[neck_cut_y:, :] = 0
    return mask


def generate_skin_mask(
    rgb: torch.Tensor,
    use_mediapipe: bool = True,
    face_landmarker_task_path: str | None = None,
) -> torch.Tensor:
    """
    Generate a binary facial skin mask from an RGB tensor.

    Parameters
    ----------
    rgb:
        [3, H, W] float tensor in 0..1.
    use_mediapipe:
        If False, use ellipse fallback only.
    face_landmarker_task_path:
        If supplied, use the same MediaPipe Tasks FaceLandmarker logic as
        new_data_prep.py.  If omitted, keep the legacy FaceMesh behavior.

    Returns
    -------
    torch.Tensor
        [1, H, W] float binary mask.
    """
    img_np = (rgb.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mask_np: np.ndarray | None = None

    if use_mediapipe:
        mp = _try_import_mediapipe()
        if mp is not None:
            landmarker = _create_face_landmarker(mp, face_landmarker_task_path)
            if landmarker is not None:
                try:
                    mask_np = _mask_from_face_landmarker_tasks(img_bgr, mp, landmarker)
                    if mask_np is None:
                        print("  [WARN] FaceLandmarker face not found -> fallback")
                except Exception as e:
                    print(f"  [WARN] FaceLandmarker runtime failure ({e}) -> fallback")
                finally:
                    landmarker.close()
            else:
                mask_np = _mask_from_legacy_face_mesh(img_bgr, mp.solutions.face_mesh)
                if mask_np is None:
                    print("  [WARN] MediaPipe FaceMesh face not found -> fallback")

    if mask_np is None:
        mask_np = _mask_ellipse_fallback(img_bgr)

    return torch.from_numpy((mask_np > 127).astype(np.float32)).unsqueeze(0)
