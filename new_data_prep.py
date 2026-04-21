"""
new_data_prep.py
================
VISIA 얼굴 이미지용 패치 생성 스크립트.

기존 data_prep.py를 기반으로 다음을 보강했다.
1. 얼굴 마스크에서 눈/눈썹뿐 아니라 입술/입 영역도 제외
2. 저장 전 RGB/GT에 얼굴 마스크를 적용해 불필요 영역을 0으로 제거

입력 예시
---------
python new_data_prep.py \
    --input_path ./raw_data/images \
    --gt_path    ./raw_data/gt \
    --patch_dir  ./patches_v2 \
    --patch_size 256 --stride 256
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_prep import (
    GT_SUBDIRS,
    _FACE_OVAL,
    _LEFT_EYE,
    _LEFT_EYEBROW,
    _RIGHT_EYE,
    _RIGHT_EYEBROW,
    _try_import_mediapipe,
    bgr2rgb_pil,
    build_gt_id_map,
    crop_patch,
    find_matching_gt,
    find_valid_positions,
    load_gray,
    load_rgb_bgr,
)


# MediaPipe Face Mesh indices
_OUTER_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 78,
]
_INNER_MOUTH = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178,
]

_EXCLUDE_REGIONS_V2 = [
    _LEFT_EYE,
    _RIGHT_EYE,
    _LEFT_EYEBROW,
    _RIGHT_EYEBROW,
    _OUTER_LIPS,
    _INNER_MOUTH,
]


def generate_face_mask_mediapipe_v2(
    img_bgr: np.ndarray,
    mp_face_mesh,
) -> np.ndarray | None:
    """
    얼굴 타원 내부를 채운 뒤, 눈/눈썹/입술/입 내부를 제외한 얼굴 마스크를 생성한다.
    반환값: uint8 [H, W], 얼굴=255 / 배경=0
    """
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
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

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lm_pts(_FACE_OVAL)], 255)

    # 타원 경계를 조금 안쪽으로 줄여 머리카락/배경 혼입을 완화
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.erode(mask, k_erode, iterations=2)

    # 눈/눈썹/입술/입 내부 제거
    k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    for indices in _EXCLUDE_REGIONS_V2:
        excl = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(excl, [lm_pts(indices)], 255)
        excl = cv2.dilate(excl, k_dilate, iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(excl))

    return mask


def generate_face_mask_v2(img_bgr: np.ndarray) -> np.ndarray:
    """
    MediaPipe를 우선 사용하고, 실패하면 기존 ellipse fallback을 사용한다.
    """
    mp = _try_import_mediapipe()
    if mp is not None:
        mask = generate_face_mask_mediapipe_v2(img_bgr, mp.solutions.face_mesh)
        if mask is not None:
            return mask
        print("    [WARN] MediaPipe 얼굴 검출 실패 -> ellipse fallback")

    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, int(h * 0.45)
    axes = (int(w * 0.35), int(h * 0.40))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


def apply_mask_to_rgb(rgb_bgr_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
    """얼굴 영역만 남기고 배경을 0으로 만든다."""
    return cv2.bitwise_and(rgb_bgr_patch, rgb_bgr_patch, mask=mask_patch)


def apply_mask_to_gray(gray_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
    """GT에서도 얼굴 바깥 영역을 0으로 만든다."""
    return cv2.bitwise_and(gray_patch, gray_patch, mask=mask_patch)


def prepare(
    input_path: str,
    gt_path: str,
    patch_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    min_mask_coverage: float = 0.7,
    apply_mask: bool = True,
) -> None:
    input_root = Path(input_path)
    gt_root = Path(gt_path)
    out_root = Path(patch_dir)

    for sub in ["rgb_cross", "rgb_parallel", "brown", "red", "wrinkle", "mask"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    gt_maps = {task: build_gt_id_map(gt_root, task) for task in GT_SUBDIRS}
    print(
        f"GT ID 수: brown={len(gt_maps['brown'])}, "
        f"red={len(gt_maps['red'])}, wrinkle={len(gt_maps['wrinkle'])}"
    )

    subject_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"input ID 수: {len(subject_dirs)}")

    manifest: dict[str, dict] = {}
    total_patches = 0
    skipped_ids: list[str] = []

    for subj_dir in subject_dirs:
        subject_name = subj_dir.name

        cross_path = subj_dir / "F_10.jpg"
        parallel_path = subj_dir / "F_11.jpg"

        if not cross_path.exists() or not parallel_path.exists():
            print(f"  [SKIP] {subject_name}: F_10.jpg / F_11.jpg 없음")
            skipped_ids.append(subject_name)
            continue

        gt_paths = {
            task: find_matching_gt(subject_name, gt_maps[task])
            for task in GT_SUBDIRS
        }
        has_gt = {task: p is not None for task, p in gt_paths.items()}
        matched = [t for t, h in has_gt.items() if h]
        print(
            f"  {subject_name}: GT={matched}"
            if matched
            else f"  [INFO] {subject_name}: 매칭된 GT 없음 -> 입력 only"
        )

        cross_bgr = load_rgb_bgr(cross_path)
        h, w = cross_bgr.shape[:2]
        parallel_bgr = load_rgb_bgr(parallel_path, target_hw=(h, w))

        face_mask = generate_face_mask_v2(cross_bgr)

        gt_arrays = {}
        for task, gt_p in gt_paths.items():
            if gt_p is not None:
                gt_arrays[task] = load_gray(gt_p, target_hw=(h, w))

        valid_positions = find_valid_positions(
            face_mask, patch_size, stride, min_mask_coverage
        )

        saved = 0
        for idx, (y, x) in enumerate(valid_positions):
            c_p = crop_patch(cross_bgr, y, x, patch_size)
            p_p = crop_patch(parallel_bgr, y, x, patch_size)
            m_p = crop_patch(face_mask, y, x, patch_size)

            if apply_mask:
                c_p = apply_mask_to_rgb(c_p, m_p)
                p_p = apply_mask_to_rgb(p_p, m_p)

            stem = f"{subject_name}_{idx:04d}"

            bgr2rgb_pil(c_p).save(out_root / "rgb_cross" / f"{stem}.png")
            bgr2rgb_pil(p_p).save(out_root / "rgb_parallel" / f"{stem}.png")
            Image.fromarray(m_p, mode="L").save(out_root / "mask" / f"{stem}.png")

            gt_pos_ratios = {}
            for task in ["brown", "red", "wrinkle"]:
                if task in gt_arrays:
                    gt_patch = crop_patch(gt_arrays[task], y, x, patch_size)
                    if apply_mask:
                        gt_patch = apply_mask_to_gray(gt_patch, m_p)
                    Image.fromarray(gt_patch, mode="L").save(
                        out_root / task / f"{stem}.png"
                    )
                    pos_ratio = float((gt_patch > 127).sum()) / gt_patch.size
                    gt_pos_ratios[f"{task}_pos_ratio"] = round(pos_ratio, 6)

            manifest[stem] = {
                "subject_name": subject_name,
                "patch_idx": idx,
                "patch_pos": [y, x],
                "has_brown": has_gt["brown"],
                "has_red": has_gt["red"],
                "has_wrinkle": has_gt["wrinkle"],
                "mask_applied": apply_mask,
                **gt_pos_ratios,
            }
            saved += 1
            total_patches += 1

        print(f"    -> {saved}개 패치 저장")

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n완료: 총 {total_patches}개 패치 생성 -> {patch_dir}")
    if skipped_ids:
        print(f"원본 이미지가 없어 건너뛴 ID: {skipped_ids}")

    stats = {
        t: sum(1 for v in manifest.values() if v[f"has_{t}"])
        for t in ["brown", "red", "wrinkle"]
    }
    print(f"  brown GT:   {stats['brown']:,} 패치")
    print(f"  red GT:     {stats['red']:,} 패치")
    print(f"  wrinkle GT: {stats['wrinkle']:,} 패치")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VISIA 얼굴 패치 생성기 (입술 제외 + 마스킹 적용 버전)"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--patch_dir", default="./patches_v2")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min_mask_coverage", type=float, default=0.7)
    parser.add_argument(
        "--no_apply_mask",
        action="store_true",
        help="저장 시 RGB/GT에 얼굴 마스크를 적용하지 않음",
    )
    args = parser.parse_args()

    prepare(
        input_path=args.input_path,
        gt_path=args.gt_path,
        patch_dir=args.patch_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        min_mask_coverage=args.min_mask_coverage,
        apply_mask=not args.no_apply_mask,
    )
