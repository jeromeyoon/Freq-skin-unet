"""
new_data_prep.py
================
VISIA 얼굴 이미지에서 학습용 패치를 생성한다.

주요 기능
---------
1. 최신 MediaPipe Tasks API(FaceLandmarker) 지원
2. 얼굴 타원 + jawline 기준 목 제거
3. 눈 / 눈썹 / 입술 / 입 내부 제외
4. RGB / GT 저장 시 얼굴 마스크 적용 가능
5. negative patch 수 제한

예시
----
python new_data_prep.py \
    --input_path ./raw_data/images \
    --gt_path ./raw_data/gt \
    --patch_dir ./patches_v2 \
    --face_landmarker_task_path ./models/face_landmarker.task
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    bgr2rgb_pil,
    build_gt_id_map,
    crop_patch,
    find_matching_gt,
    find_valid_positions,
    load_gray,
    load_rgb_bgr,
)


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

_EXCLUDE_REGIONS_V2 = [
    _LEFT_EYE,
    _RIGHT_EYE,
    _LEFT_EYEBROW,
    _RIGHT_EYEBROW,
    _OUTER_LIPS,
    _INNER_MOUTH,
]


def create_face_landmarker(task_model_path: str | None):
    """
    최신 MediaPipe Tasks API용 FaceLandmarker를 생성한다.
    실패 시 (None, None)을 반환한다.
    """
    if not task_model_path:
        print("    [WARN] face_landmarker.task 경로가 없어 fallback 사용")
        return None, None

    model_path = Path(task_model_path)
    if not model_path.exists():
        print(f"    [WARN] 모델 파일이 없습니다: {model_path} -> fallback 사용")
        return None, None

    try:
        import mediapipe as mp

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
        landmarker = FaceLandmarker.create_from_options(options)
        return mp, landmarker
    except Exception as e:
        print(f"    [WARN] FaceLandmarker 초기화 실패 ({e}) -> fallback 사용")
        return None, None


_THREAD_LOCAL = threading.local()


def get_thread_face_landmarker(disable_mediapipe: bool, task_model_path: str | None):
    """
    Return a thread-local MediaPipe FaceLandmarker.

    FaceLandmarker instances are not shared across worker threads. This avoids
    concurrent access to the same native object while still reusing one
    landmarker per worker thread.
    """
    if disable_mediapipe:
        return None, None

    key = str(task_model_path or "")
    cached_key = getattr(_THREAD_LOCAL, "face_landmarker_key", None)
    cached_mp = getattr(_THREAD_LOCAL, "face_landmarker_mp", None)
    cached_landmarker = getattr(_THREAD_LOCAL, "face_landmarker", None)

    if cached_key == key and cached_mp is not None and cached_landmarker is not None:
        return cached_mp, cached_landmarker

    if cached_landmarker is not None:
        try:
            cached_landmarker.close()
        except Exception:
            pass

    mp, landmarker = create_face_landmarker(task_model_path)
    _THREAD_LOCAL.face_landmarker_key = key
    _THREAD_LOCAL.face_landmarker_mp = mp
    _THREAD_LOCAL.face_landmarker = landmarker
    return mp, landmarker


def close_thread_face_landmarker() -> None:
    landmarker = getattr(_THREAD_LOCAL, "face_landmarker", None)
    if landmarker is not None:
        try:
            landmarker.close()
        except Exception:
            pass
    _THREAD_LOCAL.face_landmarker = None
    _THREAD_LOCAL.face_landmarker_mp = None
    _THREAD_LOCAL.face_landmarker_key = None


def generate_face_mask_tasks(
    img_bgr: np.ndarray,
    mp,
    landmarker,
) -> np.ndarray | None:
    """
    MediaPipe Tasks FaceLandmarker 결과로 얼굴 마스크를 만든다.
    """
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

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [lm_pts(_FACE_OVAL)], 255)

    # hair / background 혼입 완화
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

    # jawline 아래를 제거해서 목 제외
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


def generate_face_mask_fallback_only(img_bgr: np.ndarray) -> np.ndarray:
    """
    MediaPipe 없이 중앙 타원형 얼굴 ROI를 생성한다.
    정밀 jawline은 아니지만 VISIA 정면 사진에서 안정적으로 동작하는 fallback이다.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, int(h * 0.43)
    axes = (int(w * 0.33), int(h * 0.36))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)

    # 목/하단 배경 제거
    neck_cut_y = min(h, cy + int(axes[1] * 0.78))
    mask[neck_cut_y:, :] = 0
    return mask


def generate_face_mask_v2(
    img_bgr: np.ndarray,
    mp=None,
    landmarker=None,
) -> np.ndarray:
    """
    MediaPipe Tasks를 우선 사용하고, 실패하면 fallback 마스크를 사용한다.
    """
    if mp is not None and landmarker is not None:
        try:
            mask = generate_face_mask_tasks(img_bgr, mp, landmarker)
            if mask is not None:
                return mask
            print("    [WARN] FaceLandmarker가 얼굴을 찾지 못해 fallback 사용")
        except Exception as e:
            print(f"    [WARN] FaceLandmarker runtime failure ({e}) -> fallback 사용")

    return generate_face_mask_fallback_only(img_bgr)


def apply_mask_to_rgb(rgb_bgr_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(rgb_bgr_patch, rgb_bgr_patch, mask=mask_patch)


def apply_mask_to_gray(gray_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
    return cv2.bitwise_and(gray_patch, gray_patch, mask=mask_patch)


def select_negative_indices(num_negative: int, keep_count: int) -> set[int]:
    """
    negative patch를 균등 간격으로 일부만 선택한다.
    """
    if keep_count <= 0 or num_negative <= 0:
        return set()
    if keep_count >= num_negative:
        return set(range(num_negative))

    idxs = np.linspace(0, num_negative - 1, num=keep_count, dtype=int)
    return set(int(i) for i in idxs.tolist())


def add_red_curr_aliases(red_gt_map: dict[str, Path]) -> dict[str, Path]:
    """
    Add aliases for red GT files whose filename ends with curr.

    Examples
    --------
    ID_curr.png  -> ID
    ID-curr.png  -> ID
    IDcurr.png   -> ID
    """
    out = dict(red_gt_map)
    for stem, path in red_gt_map.items():
        lower = stem.lower()
        for suffix in ("_curr", "-curr", " curr", "curr"):
            if lower.endswith(suffix):
                alias = stem[: -len(suffix)].rstrip("_- ")
                if alias:
                    out.setdefault(alias, path)
    return out


def generate_task_centered_positions(
    gt_array: np.ndarray,
    face_mask: np.ndarray,
    patch_size: int,
    max_patches: int,
    jitter: int,
    min_mask_coverage: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    """
    Generate extra patch positions centered around sparse task-positive pixels.

    This augments the regular sliding-window candidates and directly addresses
    sparse labels such as red curr masks and wrinkle lines.
    """
    if max_patches <= 0:
        return []

    h, w = gt_array.shape[:2]
    if h < patch_size or w < patch_size:
        return []

    positive = np.argwhere((gt_array > 127) & (face_mask > 127))
    if len(positive) == 0:
        return []

    replace = len(positive) < max_patches
    sample_count = max_patches if replace else min(max_patches, len(positive))
    chosen = rng.choice(len(positive), size=sample_count, replace=replace)

    positions: set[tuple[int, int]] = set()
    face_binary = face_mask > 127

    for idx in chosen:
        cy, cx = positive[int(idx)]
        if jitter > 0:
            cy = int(cy + rng.integers(-jitter, jitter + 1))
            cx = int(cx + rng.integers(-jitter, jitter + 1))

        y = int(np.clip(cy - patch_size // 2, 0, h - patch_size))
        x = int(np.clip(cx - patch_size // 2, 0, w - patch_size))

        patch_mask = face_binary[y:y + patch_size, x:x + patch_size]
        if patch_mask.mean() >= min_mask_coverage:
            positions.add((y, x))

    return sorted(positions)


def process_subject(
    subject_index: int,
    subj_dir: Path,
    gt_maps: dict,
    out_root: Path,
    patch_size: int,
    stride: int,
    min_mask_coverage: float,
    apply_mask: bool,
    neg_pos_ratio: float,
    max_negative_if_no_positive: int,
    wrinkle_min_pos_ratio: float,
    wrinkle_neg_pos_ratio: float,
    max_wrinkle_negative_if_no_positive: int,
    centered_tasks: tuple[str, ...],
    centered_patches_per_task: int,
    centered_jitter: int,
    centered_min_mask_coverage: float,
    centered_seed: int,
    disable_mediapipe: bool,
    face_landmarker_task_path: str | None,
) -> dict:
    """
    Process one subject directory and save its patch files.

    The returned manifest is subject-local and is merged by the main thread.
    """
    subject_name = subj_dir.name
    logs: list[str] = []
    subject_manifest: dict[str, dict] = {}

    cross_path = subj_dir / "F_10.jpg"
    parallel_path = subj_dir / "F_11.jpg"

    if not cross_path.exists() or not parallel_path.exists():
        logs.append(f"  [SKIP] {subject_name}: F_10.jpg / F_11.jpg ?놁쓬")
        return {
            "index": subject_index,
            "subject_name": subject_name,
            "skipped": True,
            "saved": 0,
            "manifest": subject_manifest,
            "logs": logs,
        }

    gt_paths = {
        task: find_matching_gt(subject_name, gt_maps[task])
        for task in GT_SUBDIRS
    }
    has_gt = {task: p is not None for task, p in gt_paths.items()}
    matched = [t for t, h in has_gt.items() if h]
    if matched:
        logs.append(f"  {subject_name}: GT={matched}")
    else:
        logs.append(f"  [INFO] {subject_name}: 留ㅼ묶??GT ?놁쓬 -> ?낅젰 only")

    cross_bgr = load_rgb_bgr(cross_path)
    h, w = cross_bgr.shape[:2]
    parallel_bgr = load_rgb_bgr(parallel_path, target_hw=(h, w))

    if disable_mediapipe:
        face_mask = generate_face_mask_fallback_only(cross_bgr)
    else:
        mp, landmarker = get_thread_face_landmarker(
            disable_mediapipe=disable_mediapipe,
            task_model_path=face_landmarker_task_path,
        )
        face_mask = generate_face_mask_v2(cross_bgr, mp=mp, landmarker=landmarker)

    gt_arrays = {}
    for task, gt_p in gt_paths.items():
        if gt_p is not None:
            gt_arrays[task] = load_gray(gt_p, target_hw=(h, w))

    rng = np.random.default_rng(centered_seed + subject_index)
    valid_positions = find_valid_positions(
        face_mask, patch_size, stride, min_mask_coverage
    )
    position_set = set(valid_positions)
    centered_counts: dict[str, int] = {}
    for task in centered_tasks:
        if task not in gt_arrays:
            centered_counts[task] = 0
            continue
        extra_positions = generate_task_centered_positions(
            gt_array=gt_arrays[task],
            face_mask=face_mask,
            patch_size=patch_size,
            max_patches=centered_patches_per_task,
            jitter=centered_jitter,
            min_mask_coverage=centered_min_mask_coverage,
            rng=rng,
        )
        added = 0
        for pos in extra_positions:
            if pos not in position_set:
                valid_positions.append(pos)
                position_set.add(pos)
                added += 1
        centered_counts[task] = added

    patch_candidates = []
    for y, x in valid_positions:
        c_p = crop_patch(cross_bgr, y, x, patch_size)
        p_p = crop_patch(parallel_bgr, y, x, patch_size)
        m_p = crop_patch(face_mask, y, x, patch_size)

        if apply_mask:
            c_p = apply_mask_to_rgb(c_p, m_p)
            p_p = apply_mask_to_rgb(p_p, m_p)

        gt_pos_ratios = {}
        task_positive = {}
        is_positive = False
        gt_patch_map = {}
        for task in ["brown", "red", "wrinkle"]:
            if task in gt_arrays:
                gt_patch = crop_patch(gt_arrays[task], y, x, patch_size)
                if apply_mask:
                    gt_patch = apply_mask_to_gray(gt_patch, m_p)
                gt_patch_map[task] = gt_patch

                pos_ratio = float((gt_patch > 127).sum()) / gt_patch.size
                gt_pos_ratios[f"{task}_pos_ratio"] = round(pos_ratio, 6)
                task_positive[task] = pos_ratio > 0.0
                if pos_ratio > 0.0:
                    is_positive = True

        patch_candidates.append(
            {
                "y": y,
                "x": x,
                "cross": c_p,
                "parallel": p_p,
                "mask": m_p,
                "gt_patch_map": gt_patch_map,
                "gt_pos_ratios": gt_pos_ratios,
                "task_positive": task_positive,
                "is_positive": is_positive,
            }
        )

    positives = [p for p in patch_candidates if p["is_positive"]]
    negatives = [p for p in patch_candidates if not p["is_positive"]]

    wrinkle_positives = [
        p for p in patch_candidates
        if p["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0) >= wrinkle_min_pos_ratio
    ]
    wrinkle_negatives = [
        p for p in patch_candidates
        if "wrinkle" in p["gt_patch_map"]
        and p["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0) < wrinkle_min_pos_ratio
    ]

    if positives:
        neg_keep_count = min(
            len(negatives),
            int(np.ceil(len(positives) * neg_pos_ratio)),
        )
    else:
        neg_keep_count = min(len(negatives), max_negative_if_no_positive)

    neg_keep_indices = select_negative_indices(len(negatives), neg_keep_count)
    selected_patches = positives + [
        neg for i, neg in enumerate(negatives) if i in neg_keep_indices
    ]

    if wrinkle_positives:
        wrinkle_neg_keep_count = min(
            len(wrinkle_negatives),
            int(np.ceil(len(wrinkle_positives) * wrinkle_neg_pos_ratio)),
        )
    else:
        wrinkle_neg_keep_count = min(
            len(wrinkle_negatives),
            max_wrinkle_negative_if_no_positive,
        )

    wrinkle_neg_keep_indices = select_negative_indices(
        len(wrinkle_negatives),
        wrinkle_neg_keep_count,
    )
    selected_ids = {id(p) for p in selected_patches}
    for p in wrinkle_positives:
        if id(p) not in selected_ids:
            selected_patches.append(p)
            selected_ids.add(id(p))
    for i, p in enumerate(wrinkle_negatives):
        if i in wrinkle_neg_keep_indices and id(p) not in selected_ids:
            selected_patches.append(p)
            selected_ids.add(id(p))

    saved = 0
    for idx, patch in enumerate(selected_patches):
        y = patch["y"]
        x = patch["x"]
        stem = f"{subject_name}_{idx:04d}"

        bgr2rgb_pil(patch["cross"]).save(out_root / "rgb_cross" / f"{stem}.png")
        bgr2rgb_pil(patch["parallel"]).save(out_root / "rgb_parallel" / f"{stem}.png")
        Image.fromarray(patch["mask"], mode="L").save(out_root / "mask" / f"{stem}.png")

        for task, gt_patch in patch["gt_patch_map"].items():
            Image.fromarray(gt_patch, mode="L").save(out_root / task / f"{stem}.png")

        wrinkle_pos_ratio = patch["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0)
        has_wrinkle_patch = has_gt["wrinkle"] and wrinkle_pos_ratio >= wrinkle_min_pos_ratio

        subject_manifest[stem] = {
            "subject_name": subject_name,
            "patch_idx": idx,
            "patch_pos": [y, x],
            "has_brown": has_gt["brown"],
            "has_red": has_gt["red"],
            "has_wrinkle": has_wrinkle_patch,
            "mask_applied": apply_mask,
            "is_positive": patch["is_positive"],
            "is_wrinkle_positive": has_wrinkle_patch,
            "is_wrinkle_negative": has_gt["wrinkle"] and not has_wrinkle_patch,
            **patch["gt_pos_ratios"],
        }
        saved += 1

    logs.append(
        f"    -> {saved}媛??⑥튂 ???"
        f"(positive={len(positives)}, negative_kept={neg_keep_count}/{len(negatives)}, "
        f"centered={centered_counts}, "
        f"wrinkle_pos={len(wrinkle_positives)}, "
        f"wrinkle_neg_kept={wrinkle_neg_keep_count}/{len(wrinkle_negatives)})"
    )

    return {
        "index": subject_index,
        "subject_name": subject_name,
        "skipped": False,
        "saved": saved,
        "manifest": subject_manifest,
        "logs": logs,
    }


def prepare(
    input_path: str,
    gt_path: str,
    patch_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    min_mask_coverage: float = 0.7,
    apply_mask: bool = True,
    neg_pos_ratio: float = 2.0,
    max_negative_if_no_positive: int = 0,
    wrinkle_min_pos_ratio: float = 1e-5,
    wrinkle_neg_pos_ratio: float = 2.0,
    max_wrinkle_negative_if_no_positive: int = 0,
    centered_tasks: tuple[str, ...] = ("red", "wrinkle"),
    centered_patches_per_task: int = 128,
    centered_jitter: int = 64,
    centered_min_mask_coverage: float = 0.5,
    centered_seed: int = 42,
    disable_mediapipe: bool = False,
    face_landmarker_task_path: str | None = None,
    num_workers: int = 1,
) -> None:
    input_root = Path(input_path)
    gt_root = Path(gt_path)
    out_root = Path(patch_dir)

    for sub in ["rgb_cross", "rgb_parallel", "brown", "red", "wrinkle", "mask"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    gt_maps = {task: build_gt_id_map(gt_root, task) for task in GT_SUBDIRS}
    gt_maps["red"] = add_red_curr_aliases(gt_maps["red"])
    print(
        f"GT ID 수: brown={len(gt_maps['brown'])}, "
        f"red={len(gt_maps['red'])}, wrinkle={len(gt_maps['wrinkle'])}"
    )

    subject_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"input ID 수: {len(subject_dirs)}")

    manifest: dict[str, dict] = {}
    total_patches = 0
    skipped_ids: list[str] = []

    num_workers = max(1, int(num_workers))
    print(f"workers: {num_workers}")

    worker_kwargs = dict(
        gt_maps=gt_maps,
        out_root=out_root,
        patch_size=patch_size,
        stride=stride,
        min_mask_coverage=min_mask_coverage,
        apply_mask=apply_mask,
        neg_pos_ratio=neg_pos_ratio,
        max_negative_if_no_positive=max_negative_if_no_positive,
        wrinkle_min_pos_ratio=wrinkle_min_pos_ratio,
        wrinkle_neg_pos_ratio=wrinkle_neg_pos_ratio,
        max_wrinkle_negative_if_no_positive=max_wrinkle_negative_if_no_positive,
        centered_tasks=centered_tasks,
        centered_patches_per_task=centered_patches_per_task,
        centered_jitter=centered_jitter,
        centered_min_mask_coverage=centered_min_mask_coverage,
        centered_seed=centered_seed,
        disable_mediapipe=disable_mediapipe,
        face_landmarker_task_path=face_landmarker_task_path,
    )

    results: list[dict | None] = [None] * len(subject_dirs)
    try:
        if num_workers == 1:
            for i, subj_dir in enumerate(subject_dirs):
                results[i] = process_subject(i, subj_dir, **worker_kwargs)
                for line in results[i]["logs"]:
                    print(line)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_subject, i, subj_dir, **worker_kwargs): i
                    for i, subj_dir in enumerate(subject_dirs)
                }
                done = 0
                for future in as_completed(futures):
                    i = futures[future]
                    results[i] = future.result()
                    done += 1
                    result = results[i]
                    print(
                        f"  [DONE {done}/{len(subject_dirs)}] "
                        f"{result['subject_name']}: saved={result['saved']}"
                    )
    finally:
        # Closes the main-thread landmarker when num_workers=1. Worker-thread
        # landmarker instances are released when the process exits.
        close_thread_face_landmarker()

    for result in results:
        if result is None:
            continue
        if num_workers > 1:
            for line in result["logs"]:
                print(line)
        if result["skipped"]:
            skipped_ids.append(result["subject_name"])
            continue
        manifest.update(result["manifest"])
        total_patches += int(result["saved"])

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n?꾨즺: 珥?{total_patches}媛??⑥튂 ?앹꽦 -> {patch_dir}")
    if skipped_ids:
        print(f"?먮낯 ?대?吏媛 ?놁뼱 嫄대꼫??ID: {skipped_ids}")

    stats = {
        t: sum(1 for v in manifest.values() if v[f"has_{t}"])
        for t in ["brown", "red", "wrinkle"]
    }
    print(f"  brown GT:   {stats['brown']:,} ?⑥튂")
    print(f"  red GT:     {stats['red']:,} ?⑥튂")
    print(f"  wrinkle GT: {stats['wrinkle']:,} ?⑥튂")
    return

    rng = np.random.default_rng(centered_seed)

    mp = None
    landmarker = None
    if not disable_mediapipe:
        mp, landmarker = create_face_landmarker(face_landmarker_task_path)

    try:
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
            if matched:
                print(f"  {subject_name}: GT={matched}")
            else:
                print(f"  [INFO] {subject_name}: 매칭된 GT 없음 -> 입력 only")

            cross_bgr = load_rgb_bgr(cross_path)
            h, w = cross_bgr.shape[:2]
            parallel_bgr = load_rgb_bgr(parallel_path, target_hw=(h, w))

            if disable_mediapipe:
                face_mask = generate_face_mask_fallback_only(cross_bgr)
            else:
                face_mask = generate_face_mask_v2(cross_bgr, mp=mp, landmarker=landmarker)

            gt_arrays = {}
            for task, gt_p in gt_paths.items():
                if gt_p is not None:
                    gt_arrays[task] = load_gray(gt_p, target_hw=(h, w))

            valid_positions = find_valid_positions(
                face_mask, patch_size, stride, min_mask_coverage
            )
            position_set = set(valid_positions)
            centered_counts: dict[str, int] = {}
            for task in centered_tasks:
                if task not in gt_arrays:
                    centered_counts[task] = 0
                    continue
                extra_positions = generate_task_centered_positions(
                    gt_array=gt_arrays[task],
                    face_mask=face_mask,
                    patch_size=patch_size,
                    max_patches=centered_patches_per_task,
                    jitter=centered_jitter,
                    min_mask_coverage=centered_min_mask_coverage,
                    rng=rng,
                )
                added = 0
                for pos in extra_positions:
                    if pos not in position_set:
                        valid_positions.append(pos)
                        position_set.add(pos)
                        added += 1
                centered_counts[task] = added

            patch_candidates = []
            for y, x in valid_positions:
                c_p = crop_patch(cross_bgr, y, x, patch_size)
                p_p = crop_patch(parallel_bgr, y, x, patch_size)
                m_p = crop_patch(face_mask, y, x, patch_size)

                if apply_mask:
                    c_p = apply_mask_to_rgb(c_p, m_p)
                    p_p = apply_mask_to_rgb(p_p, m_p)

                gt_pos_ratios = {}
                task_positive = {}
                is_positive = False
                gt_patch_map = {}
                for task in ["brown", "red", "wrinkle"]:
                    if task in gt_arrays:
                        gt_patch = crop_patch(gt_arrays[task], y, x, patch_size)
                        if apply_mask:
                            gt_patch = apply_mask_to_gray(gt_patch, m_p)
                        gt_patch_map[task] = gt_patch

                        pos_ratio = float((gt_patch > 127).sum()) / gt_patch.size
                        gt_pos_ratios[f"{task}_pos_ratio"] = round(pos_ratio, 6)
                        task_positive[task] = pos_ratio > 0.0
                        if pos_ratio > 0.0:
                            is_positive = True

                patch_candidates.append(
                    {
                        "y": y,
                        "x": x,
                        "cross": c_p,
                        "parallel": p_p,
                        "mask": m_p,
                        "gt_patch_map": gt_patch_map,
                        "gt_pos_ratios": gt_pos_ratios,
                        "task_positive": task_positive,
                        "is_positive": is_positive,
                    }
                )

            positives = [p for p in patch_candidates if p["is_positive"]]
            negatives = [p for p in patch_candidates if not p["is_positive"]]

            wrinkle_positives = [
                p for p in patch_candidates
                if p["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0) >= wrinkle_min_pos_ratio
            ]
            wrinkle_negatives = [
                p for p in patch_candidates
                if "wrinkle" in p["gt_patch_map"]
                and p["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0) < wrinkle_min_pos_ratio
            ]

            if positives:
                neg_keep_count = min(
                    len(negatives),
                    int(np.ceil(len(positives) * neg_pos_ratio)),
                )
            else:
                neg_keep_count = min(len(negatives), max_negative_if_no_positive)

            neg_keep_indices = select_negative_indices(len(negatives), neg_keep_count)
            selected_patches = positives + [
                neg for i, neg in enumerate(negatives) if i in neg_keep_indices
            ]

            if wrinkle_positives:
                wrinkle_neg_keep_count = min(
                    len(wrinkle_negatives),
                    int(np.ceil(len(wrinkle_positives) * wrinkle_neg_pos_ratio)),
                )
            else:
                wrinkle_neg_keep_count = min(
                    len(wrinkle_negatives),
                    max_wrinkle_negative_if_no_positive,
                )

            wrinkle_neg_keep_indices = select_negative_indices(
                len(wrinkle_negatives),
                wrinkle_neg_keep_count,
            )
            selected_ids = {id(p) for p in selected_patches}
            for p in wrinkle_positives:
                if id(p) not in selected_ids:
                    selected_patches.append(p)
                    selected_ids.add(id(p))
            for i, p in enumerate(wrinkle_negatives):
                if i in wrinkle_neg_keep_indices and id(p) not in selected_ids:
                    selected_patches.append(p)
                    selected_ids.add(id(p))

            saved = 0
            for idx, patch in enumerate(selected_patches):
                y = patch["y"]
                x = patch["x"]
                stem = f"{subject_name}_{idx:04d}"

                bgr2rgb_pil(patch["cross"]).save(out_root / "rgb_cross" / f"{stem}.png")
                bgr2rgb_pil(patch["parallel"]).save(out_root / "rgb_parallel" / f"{stem}.png")
                Image.fromarray(patch["mask"], mode="L").save(out_root / "mask" / f"{stem}.png")

                for task, gt_patch in patch["gt_patch_map"].items():
                    Image.fromarray(gt_patch, mode="L").save(out_root / task / f"{stem}.png")

                wrinkle_pos_ratio = patch["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0)
                has_wrinkle_patch = has_gt["wrinkle"] and wrinkle_pos_ratio >= wrinkle_min_pos_ratio

                manifest[stem] = {
                    "subject_name": subject_name,
                    "patch_idx": idx,
                    "patch_pos": [y, x],
                    "has_brown": has_gt["brown"],
                    "has_red": has_gt["red"],
                    "has_wrinkle": has_wrinkle_patch,
                    "mask_applied": apply_mask,
                    "is_positive": patch["is_positive"],
                    "is_wrinkle_positive": has_wrinkle_patch,
                    "is_wrinkle_negative": has_gt["wrinkle"] and not has_wrinkle_patch,
                    **patch["gt_pos_ratios"],
                }
                saved += 1
                total_patches += 1

            print(
                f"    -> {saved}개 패치 저장 "
                f"(positive={len(positives)}, negative_kept={neg_keep_count}/{len(negatives)}, "
                f"centered={centered_counts}, "
                f"wrinkle_pos={len(wrinkle_positives)}, "
                f"wrinkle_neg_kept={wrinkle_neg_keep_count}/{len(wrinkle_negatives)})"
            )
    finally:
        if landmarker is not None:
            landmarker.close()

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
        description="VISIA 얼굴 패치 생성기 (최신 MediaPipe Tasks API 지원)"
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
    parser.add_argument(
        "--neg_pos_ratio",
        type=float,
        default=2.0,
        help="negative patch를 positive patch 수의 몇 배까지 유지할지 지정 (기본 2.0)",
    )
    parser.add_argument(
        "--max_negative_if_no_positive",
        type=int,
        default=0,
        help="positive가 하나도 없을 때 유지할 negative patch 최대 개수 (기본 0)",
    )
    parser.add_argument(
        "--wrinkle_min_pos_ratio",
        type=float,
        default=1e-5,
        help="wrinkle positive patch로 인정할 최소 양성 픽셀 비율 (기본 1e-5)",
    )
    parser.add_argument(
        "--wrinkle_neg_pos_ratio",
        type=float,
        default=2.0,
        help="wrinkle positive 수 대비 유지할 wrinkle negative 최대 비율 (기본 2.0)",
    )
    parser.add_argument(
        "--max_wrinkle_negative_if_no_positive",
        type=int,
        default=0,
        help="wrinkle positive가 없을 때 유지할 wrinkle negative 최대 개수 (기본 0)",
    )
    parser.add_argument(
        "--centered_tasks",
        default="red,wrinkle",
        help="희소 GT 중심 patch를 추가할 task 목록 (기본: red,wrinkle)",
    )
    parser.add_argument(
        "--centered_patches_per_task",
        type=int,
        default=128,
        help="ID당 task별 positive-centered patch 후보 최대 개수 (기본 128)",
    )
    parser.add_argument(
        "--centered_jitter",
        type=int,
        default=64,
        help="positive-centered crop 중심 jitter 픽셀 범위 (기본 64)",
    )
    parser.add_argument(
        "--centered_min_mask_coverage",
        type=float,
        default=0.5,
        help="centered patch의 최소 얼굴 mask coverage (기본 0.5)",
    )
    parser.add_argument(
        "--centered_seed",
        type=int,
        default=42,
        help="positive-centered patch sampling seed (기본 42)",
    )
    parser.add_argument(
        "--disable_mediapipe",
        action="store_true",
        help="MediaPipe를 사용하지 않고 fallback 얼굴 마스크만 사용",
    )
    parser.add_argument(
        "--face_landmarker_task_path",
        type=str,
        default=None,
        help="최신 MediaPipe Tasks API용 face_landmarker.task 모델 경로",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="ID 단위 병렬 처리 worker 수 (기본 1, 예: 4)",
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
        neg_pos_ratio=args.neg_pos_ratio,
        max_negative_if_no_positive=args.max_negative_if_no_positive,
        wrinkle_min_pos_ratio=args.wrinkle_min_pos_ratio,
        wrinkle_neg_pos_ratio=args.wrinkle_neg_pos_ratio,
        max_wrinkle_negative_if_no_positive=args.max_wrinkle_negative_if_no_positive,
        centered_tasks=tuple(
            task.strip() for task in args.centered_tasks.split(",") if task.strip()
        ),
        centered_patches_per_task=args.centered_patches_per_task,
        centered_jitter=args.centered_jitter,
        centered_min_mask_coverage=args.centered_min_mask_coverage,
        centered_seed=args.centered_seed,
        disable_mediapipe=args.disable_mediapipe,
        face_landmarker_task_path=args.face_landmarker_task_path,
        num_workers=args.num_workers,
    )
