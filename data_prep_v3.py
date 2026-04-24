"""
data_prep_v3.py
===============
Patch generator for mixed-input multi-task training.

Design
------
This version keeps the current partial-label training style:

- one patch sample -> one input pair (rgb_cross, rgb_parallel)
- tasks without aligned supervision remain absent for that sample

Streams
-------
1. main stream
   - inputs : input_root/{ID}/F_10.jpg, F_11.jpg
   - labels : brown / red

2. wrinkle stream
   - inputs : gt_root/wrinkle-deep/images/sr-proto/{ID}-F_9.jpg, {ID}-F_11.jpg
   - labels : wrinkle

Rationale
---------
Wrinkle GT may live in a different image coordinate system from the main
F_10/F_11 pair. Instead of forcing resize-based alignment across unrelated
images, generate wrinkle patches from the wrinkle-specific image pair and keep
brown/red patches on the original pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_prep import (
    GT_SUBDIRS,
    bgr2rgb_pil,
    build_gt_id_map,
    crop_patch,
    find_matching_gt,
    find_valid_positions,
    load_gray,
    load_rgb_bgr,
)
from new_data_prep import (
    add_red_curr_aliases,
    add_wrinkle_f11_aliases,
    apply_mask_to_gray,
    apply_mask_to_rgb,
    close_thread_face_landmarker,
    generate_face_mask_v2,
    generate_task_centered_positions,
    get_thread_face_landmarker,
    select_negative_indices,
)


def _strip_suffix_once(stem: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if stem.endswith(suffix):
            alias = stem[: -len(suffix)].rstrip("_- ")
            return alias or None
    return None


def build_wrinkle_image_maps(gt_root: Path) -> dict[str, dict[str, Path]]:
    """
    Build subject-id -> wrinkle image maps.

    Expected files under:
      gt_root/wrinkle-deep/images/sr-proto/
        {ID}-F_9.jpg
        {ID}-F_11.jpg
    """
    folder = gt_root / "wrinkle-deep" / "images" / "sr-proto"
    maps = {"cross": {}, "parallel": {}}
    if not folder.exists():
        return maps

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files: list[Path] = []
    for pattern in exts:
        files.extend(folder.glob(pattern))

    cross_suffixes = ("-F_9", "_F_9", "-F9", "_F9", "-f_9", "_f_9", "-f9", "_f9")
    parallel_suffixes = ("-F_11", "_F_11", "-F11", "_F11", "-f_11", "_f_11", "-f11", "_f11")

    for path in files:
        stem = path.stem
        alias = _strip_suffix_once(stem, cross_suffixes)
        if alias:
            maps["cross"].setdefault(alias, path)
            continue
        alias = _strip_suffix_once(stem, parallel_suffixes)
        if alias:
            maps["parallel"].setdefault(alias, path)

    return maps


def _collect_patch_candidates(
    *,
    cross_bgr,
    parallel_bgr,
    face_mask,
    gt_arrays: dict[str, np.ndarray],
    patch_size: int,
    stride: int,
    min_mask_coverage: float,
    apply_mask: bool,
    mask_gt_on_save: bool,
    centered_tasks: tuple[str, ...],
    centered_patches_per_task: int,
    centered_jitter: int,
    centered_min_mask_coverage: float,
    centered_seed: int,
    subject_index: int,
) -> list[dict]:
    h, w = cross_bgr.shape[:2]
    rng = np.random.default_rng(centered_seed + subject_index)

    valid_positions = find_valid_positions(
        face_mask, patch_size, stride, min_mask_coverage
    )
    position_set = set(valid_positions)

    for task in centered_tasks:
        if task not in gt_arrays:
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
        for pos in extra_positions:
            if pos not in position_set:
                valid_positions.append(pos)
                position_set.add(pos)

    patch_candidates = []
    for y, x in valid_positions:
        c_p = crop_patch(cross_bgr, y, x, patch_size)
        p_p = crop_patch(parallel_bgr, y, x, patch_size)
        m_p = crop_patch(face_mask, y, x, patch_size)

        if apply_mask:
            c_p = apply_mask_to_rgb(c_p, m_p)
            p_p = apply_mask_to_rgb(p_p, m_p)

        gt_patch_map = {}
        gt_pos_ratios = {}
        is_positive = False
        for task, gt_arr in gt_arrays.items():
            gt_patch = crop_patch(gt_arr, y, x, patch_size)
            gt_patch_for_ratio = (
                apply_mask_to_gray(gt_patch, m_p) if apply_mask else gt_patch
            )
            gt_patch_save = (
                apply_mask_to_gray(gt_patch, m_p)
                if (apply_mask and mask_gt_on_save)
                else gt_patch
            )
            gt_patch_map[task] = gt_patch_save

            pos_ratio = float((gt_patch_for_ratio > 127).sum()) / max(float(gt_patch_for_ratio.size), 1.0)
            gt_pos_ratios[f"{task}_pos_ratio"] = round(pos_ratio, 6)
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
                "is_positive": is_positive,
            }
        )
    return patch_candidates


def _select_candidates(
    patch_candidates: list[dict],
    *,
    neg_pos_ratio: float,
    max_negative_if_no_positive: int,
) -> list[dict]:
    positives = [p for p in patch_candidates if p["is_positive"]]
    negatives = [p for p in patch_candidates if not p["is_positive"]]

    if positives:
        neg_keep_count = min(len(negatives), int(np.ceil(len(positives) * neg_pos_ratio)))
    else:
        neg_keep_count = min(len(negatives), max_negative_if_no_positive)

    neg_keep_indices = select_negative_indices(len(negatives), neg_keep_count)
    return positives + [neg for i, neg in enumerate(negatives) if i in neg_keep_indices]


def save_stream_patches(
    *,
    out_root: Path,
    subject_name: str,
    stream_name: str,
    selected_patches: list[dict],
    has_brown: bool,
    has_red: bool,
    has_wrinkle: bool,
    wrinkle_min_pos_ratio: float,
    apply_mask: bool,
    mask_gt_on_save: bool,
    manifest: dict[str, dict],
) -> int:
    saved = 0
    for idx, patch in enumerate(selected_patches):
        y = patch["y"]
        x = patch["x"]
        stem = f"{subject_name}_{stream_name}_{idx:04d}"

        bgr2rgb_pil(patch["cross"]).save(out_root / "rgb_cross" / f"{stem}.png")
        bgr2rgb_pil(patch["parallel"]).save(out_root / "rgb_parallel" / f"{stem}.png")
        Image.fromarray(patch["mask"], mode="L").save(out_root / "mask" / f"{stem}.png")

        for task, gt_patch in patch["gt_patch_map"].items():
            Image.fromarray(gt_patch, mode="L").save(out_root / task / f"{stem}.png")

        wrinkle_pos_ratio = patch["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0)
        has_wrinkle_patch = has_wrinkle and (wrinkle_pos_ratio >= wrinkle_min_pos_ratio)

        manifest[stem] = {
            "subject_name": subject_name,
            "stream": stream_name,
            "patch_idx": idx,
            "patch_pos": [y, x],
            "has_brown": has_brown,
            "has_red": has_red,
            "has_wrinkle": has_wrinkle_patch,
            "mask_applied": apply_mask,
            "gt_mask_applied": bool(apply_mask and mask_gt_on_save),
            "is_positive": patch["is_positive"],
            "is_wrinkle_positive": has_wrinkle_patch,
            "is_wrinkle_negative": has_wrinkle and not has_wrinkle_patch,
            **patch["gt_pos_ratios"],
        }
        saved += 1
    return saved


def prepare_v3(
    input_path: str,
    gt_path: str,
    patch_dir: str,
    patch_size: int = 256,
    stride: int = 256,
    min_mask_coverage: float = 0.7,
    apply_mask: bool = True,
    mask_gt_on_save: bool = True,
    neg_pos_ratio: float = 2.0,
    max_negative_if_no_positive: int = 0,
    wrinkle_min_pos_ratio: float = 1e-5,
    wrinkle_neg_pos_ratio: float = 2.0,
    max_wrinkle_negative_if_no_positive: int = 0,
    centered_red_patches: int = 128,
    centered_wrinkle_patches: int = 128,
    centered_jitter: int = 64,
    centered_min_mask_coverage: float = 0.5,
    centered_seed: int = 42,
    disable_mediapipe: bool = False,
    face_landmarker_task_path: str | None = None,
) -> None:
    input_root = Path(input_path)
    gt_root = Path(gt_path)
    out_root = Path(patch_dir)

    for sub in ["rgb_cross", "rgb_parallel", "brown", "red", "wrinkle", "mask"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    gt_maps = {task: build_gt_id_map(gt_root, task) for task in GT_SUBDIRS}
    gt_maps["red"] = add_red_curr_aliases(gt_maps["red"])
    gt_maps["wrinkle"] = add_wrinkle_f11_aliases(gt_maps["wrinkle"])
    wrinkle_img_maps = build_wrinkle_image_maps(gt_root)

    print(
        f"GT ID 수: brown={len(gt_maps['brown'])}, "
        f"red={len(gt_maps['red'])}, wrinkle={len(gt_maps['wrinkle'])}"
    )
    print(
        f"Wrinkle image ID 수: cross(F_9)={len(wrinkle_img_maps['cross'])}, "
        f"parallel(F_11)={len(wrinkle_img_maps['parallel'])}"
    )

    subject_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"input ID 수: {len(subject_dirs)}")

    manifest: dict[str, dict] = {}
    skipped = []
    total_saved = 0

    try:
        for subject_index, subj_dir in enumerate(subject_dirs):
            subject_name = subj_dir.name
            print(f"\n[{subject_index + 1}/{len(subject_dirs)}] {subject_name}")

            main_cross_path = subj_dir / "F_10.jpg"
            main_parallel_path = subj_dir / "F_11.jpg"
            wrinkle_cross_path = find_matching_gt(subject_name, wrinkle_img_maps["cross"])
            wrinkle_parallel_path = find_matching_gt(subject_name, wrinkle_img_maps["parallel"])

            brown_gt = find_matching_gt(subject_name, gt_maps["brown"])
            red_gt = find_matching_gt(subject_name, gt_maps["red"])
            wrinkle_gt = find_matching_gt(subject_name, gt_maps["wrinkle"])

            saved_this_subject = 0

            # Main stream: brown / red on original F_10/F_11
            if main_cross_path.exists() and main_parallel_path.exists() and (brown_gt or red_gt):
                main_cross = load_rgb_bgr(main_cross_path)
                h, w = main_cross.shape[:2]
                main_parallel = load_rgb_bgr(main_parallel_path, target_hw=(h, w))

                if disable_mediapipe:
                    mp = landmarker = None
                else:
                    mp, landmarker = get_thread_face_landmarker(
                        disable_mediapipe=False,
                        task_model_path=face_landmarker_task_path,
                    )
                face_mask_main = generate_face_mask_v2(main_cross, mp=mp, landmarker=landmarker)

                gt_arrays_main = {}
                if brown_gt is not None:
                    gt_arrays_main["brown"] = load_gray(brown_gt, target_hw=(h, w))
                if red_gt is not None:
                    gt_arrays_main["red"] = load_gray(red_gt, target_hw=(h, w))

                main_candidates = _collect_patch_candidates(
                    cross_bgr=main_cross,
                    parallel_bgr=main_parallel,
                    face_mask=face_mask_main,
                    gt_arrays=gt_arrays_main,
                    patch_size=patch_size,
                    stride=stride,
                    min_mask_coverage=min_mask_coverage,
                    apply_mask=apply_mask,
                    mask_gt_on_save=mask_gt_on_save,
                    centered_tasks=tuple(t for t in ("red",) if t in gt_arrays_main),
                    centered_patches_per_task=centered_red_patches,
                    centered_jitter=centered_jitter,
                    centered_min_mask_coverage=centered_min_mask_coverage,
                    centered_seed=centered_seed,
                    subject_index=subject_index,
                )
                main_selected = _select_candidates(
                    main_candidates,
                    neg_pos_ratio=neg_pos_ratio,
                    max_negative_if_no_positive=max_negative_if_no_positive,
                )
                n_saved = save_stream_patches(
                    out_root=out_root,
                    subject_name=subject_name,
                    stream_name="main",
                    selected_patches=main_selected,
                    has_brown=("brown" in gt_arrays_main),
                    has_red=("red" in gt_arrays_main),
                    has_wrinkle=False,
                    wrinkle_min_pos_ratio=wrinkle_min_pos_ratio,
                    apply_mask=apply_mask,
                    mask_gt_on_save=mask_gt_on_save,
                    manifest=manifest,
                )
                saved_this_subject += n_saved
                print(f"  main stream: saved={n_saved}")
            else:
                print("  main stream: skip")

            # Wrinkle stream: wrinkle-specific F_9/F_11 pair
            if wrinkle_gt is not None and wrinkle_cross_path is not None and wrinkle_parallel_path is not None:
                wrinkle_cross = load_rgb_bgr(wrinkle_cross_path)
                h, w = wrinkle_cross.shape[:2]
                wrinkle_parallel = load_rgb_bgr(wrinkle_parallel_path, target_hw=(h, w))

                if disable_mediapipe:
                    mp = landmarker = None
                else:
                    mp, landmarker = get_thread_face_landmarker(
                        disable_mediapipe=False,
                        task_model_path=face_landmarker_task_path,
                    )
                face_mask_wrinkle = generate_face_mask_v2(wrinkle_cross, mp=mp, landmarker=landmarker)

                gt_arrays_wrinkle = {
                    "wrinkle": load_gray(wrinkle_gt, target_hw=(h, w)),
                }
                wrinkle_candidates = _collect_patch_candidates(
                    cross_bgr=wrinkle_cross,
                    parallel_bgr=wrinkle_parallel,
                    face_mask=face_mask_wrinkle,
                    gt_arrays=gt_arrays_wrinkle,
                    patch_size=patch_size,
                    stride=stride,
                    min_mask_coverage=min_mask_coverage,
                    apply_mask=apply_mask,
                    mask_gt_on_save=mask_gt_on_save,
                    centered_tasks=("wrinkle",),
                    centered_patches_per_task=centered_wrinkle_patches,
                    centered_jitter=centered_jitter,
                    centered_min_mask_coverage=centered_min_mask_coverage,
                    centered_seed=centered_seed + 100000,
                    subject_index=subject_index,
                )
                wrinkle_selected = _select_candidates(
                    wrinkle_candidates,
                    neg_pos_ratio=wrinkle_neg_pos_ratio,
                    max_negative_if_no_positive=max_wrinkle_negative_if_no_positive,
                )
                n_saved = save_stream_patches(
                    out_root=out_root,
                    subject_name=subject_name,
                    stream_name="wrinkle",
                    selected_patches=wrinkle_selected,
                    has_brown=False,
                    has_red=False,
                    has_wrinkle=True,
                    wrinkle_min_pos_ratio=wrinkle_min_pos_ratio,
                    apply_mask=apply_mask,
                    mask_gt_on_save=mask_gt_on_save,
                    manifest=manifest,
                )
                saved_this_subject += n_saved
                print(f"  wrinkle stream: saved={n_saved}")
            else:
                print("  wrinkle stream: skip")

            if saved_this_subject == 0:
                skipped.append(subject_name)
            total_saved += saved_this_subject
    finally:
        close_thread_face_landmarker()

    with (out_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    stats = {
        t: sum(1 for v in manifest.values() if v.get(f"has_{t}", False))
        for t in ("brown", "red", "wrinkle")
    }
    print(f"\n완료: 총 {total_saved}개 patch 생성 -> {out_root}")
    print(f"  brown GT:   {stats['brown']:,} 패치")
    print(f"  red GT:     {stats['red']:,} 패치")
    print(f"  wrinkle GT: {stats['wrinkle']:,} 패치")
    if skipped:
        print(f"  저장된 patch가 없는 subject: {len(skipped)}개")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mixed-input patch generator for main(brown/red) + wrinkle-specific streams"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--patch_dir", default="./patches_v3")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min_mask_coverage", type=float, default=0.7)
    parser.add_argument("--no_apply_mask", action="store_true")
    parser.add_argument("--no_mask_gt_on_save", action="store_true")
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0)
    parser.add_argument("--max_negative_if_no_positive", type=int, default=0)
    parser.add_argument("--wrinkle_min_pos_ratio", type=float, default=1e-5)
    parser.add_argument("--wrinkle_neg_pos_ratio", type=float, default=2.0)
    parser.add_argument("--max_wrinkle_negative_if_no_positive", type=int, default=0)
    parser.add_argument("--centered_red_patches", type=int, default=128)
    parser.add_argument("--centered_wrinkle_patches", type=int, default=128)
    parser.add_argument("--centered_jitter", type=int, default=64)
    parser.add_argument("--centered_min_mask_coverage", type=float, default=0.5)
    parser.add_argument("--centered_seed", type=int, default=42)
    parser.add_argument("--disable_mediapipe", action="store_true")
    parser.add_argument("--face_landmarker_task_path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_v3(
        input_path=args.input_path,
        gt_path=args.gt_path,
        patch_dir=args.patch_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        min_mask_coverage=args.min_mask_coverage,
        apply_mask=not args.no_apply_mask,
        mask_gt_on_save=not args.no_mask_gt_on_save,
        neg_pos_ratio=args.neg_pos_ratio,
        max_negative_if_no_positive=args.max_negative_if_no_positive,
        wrinkle_min_pos_ratio=args.wrinkle_min_pos_ratio,
        wrinkle_neg_pos_ratio=args.wrinkle_neg_pos_ratio,
        max_wrinkle_negative_if_no_positive=args.max_wrinkle_negative_if_no_positive,
        centered_red_patches=args.centered_red_patches,
        centered_wrinkle_patches=args.centered_wrinkle_patches,
        centered_jitter=args.centered_jitter,
        centered_min_mask_coverage=args.centered_min_mask_coverage,
        centered_seed=args.centered_seed,
        disable_mediapipe=args.disable_mediapipe,
        face_landmarker_task_path=args.face_landmarker_task_path,
    )


if __name__ == "__main__":
    main()
