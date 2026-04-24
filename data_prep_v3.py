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
            # GT는 apply_mask 여부와 무관하게 항상 face mask를 적용한다.
            # 피부 영역 밖의 GT 픽셀이 학습 신호에 포함되면 noise가 되기 때문.
            gt_patch_masked = apply_mask_to_gray(gt_patch, m_p)
            gt_patch_map[task] = gt_patch_masked

            pos_ratio = float((gt_patch_masked > 127).sum()) / max(float(gt_patch_masked.size), 1.0)
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


def _same_size_patch_overlap(a: dict, b: dict, patch_size: int) -> float:
    """두 패치의 겹치는 면적을 단일 패치 면적 대비 비율로 반환 (0~1).

    IoU(Jaccard) 대신 단순 overlap ratio를 사용한다.
    "25% 이상 겹치지 않게" 라는 조건은 inter/area < 0.25 이며,
    이것이 IoU (inter/union) 보다 직관적으로 중복 한계를 표현한다.
    """
    ay1, ax1 = int(a["y"]), int(a["x"])
    by1, bx1 = int(b["y"]), int(b["x"])
    ay2, ax2 = ay1 + patch_size, ax1 + patch_size
    by2, bx2 = by1 + patch_size, bx1 + patch_size

    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter = inter_h * inter_w
    if inter <= 0:
        return 0.0

    return float(inter) / float(patch_size * patch_size)


def _annotate_stream_candidates(
    selected_patches: list[dict],
    *,
    stream_name: str,
    has_brown: bool,
    has_red: bool,
    has_wrinkle: bool,
) -> list[dict]:
    out = []
    for patch in selected_patches:
        item = dict(patch)
        item["_stream_name"] = stream_name
        item["_has_brown"] = has_brown
        item["_has_red"] = has_red
        item["_has_wrinkle"] = has_wrinkle
        out.append(item)
    return out


def _stream_patch_quality(patch: dict) -> tuple:
    wrinkle_ratio = float(patch["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0))
    red_ratio = float(patch["gt_pos_ratios"].get("red_pos_ratio", 0.0))
    brown_ratio = float(patch["gt_pos_ratios"].get("brown_pos_ratio", 0.0))

    wrinkle_pos = int(wrinkle_ratio > 0.0)
    red_pos = int(red_ratio > 0.0)
    brown_pos = int(brown_ratio > 0.0)
    positive_task_count = wrinkle_pos + red_pos + brown_pos
    is_positive = int(patch.get("is_positive", False))

    return (
        wrinkle_pos,
        positive_task_count,
        red_pos,
        brown_pos,
        min(wrinkle_ratio, 1.0),
        min(red_ratio, 1.0),
        min(brown_ratio, 1.0),
        is_positive,
    )


def dedup_selected_patches(
    selected_patches: list[dict],
    patch_size: int,
    max_overlap_fraction: float,
) -> tuple[list[dict], int]:
    """중복 패치 제거: 이미 선택된 패치와의 overlap ratio >= max_overlap_fraction 이면 드롭.

    high-quality 패치를 먼저 유지하기 위해 _stream_patch_quality로 내림차순 정렬 후
    탐욕적(greedy) 선택을 수행한다.
    """
    if max_overlap_fraction <= 0.0 or len(selected_patches) <= 1:
        return selected_patches, 0

    ranked = sorted(
        selected_patches,
        key=_stream_patch_quality,
        reverse=True,
    )
    kept: list[dict] = []
    dropped = 0
    for cand in ranked:
        if any(
            _same_size_patch_overlap(cand, prev, patch_size) >= max_overlap_fraction
            for prev in kept
        ):
            dropped += 1
            continue
        kept.append(cand)
    return kept, dropped


def select_subjects(
    subject_dirs: list[Path],
    gt_maps: dict[str, dict],
    max_subjects: int | None,
    unused_path: Path | None,
) -> tuple[list[Path], list[Path]]:
    """GT 우선순위 기반으로 max_subjects명을 선택하고 나머지를 unused_path에 저장.

    선택 기준 (내림차순):
      wrinkle GT 보유 → 4점
      red     GT 보유 → 2점
      brown   GT 보유 → 1점
      점수가 같을 경우 subject 이름 오름차순 (재현성 보장)

    모든 GT를 갖춘 사람이 우선 포함되어 학습 데이터 품질이 최대화된다.
    """
    if max_subjects is None or max_subjects <= 0 or max_subjects >= len(subject_dirs):
        return list(subject_dirs), []

    def _score(subj_dir: Path) -> int:
        name = subj_dir.name
        score = 0
        if find_matching_gt(name, gt_maps.get("wrinkle", {})):
            score += 4
        if find_matching_gt(name, gt_maps.get("red", {})):
            score += 2
        if find_matching_gt(name, gt_maps.get("brown", {})):
            score += 1
        return score

    scored = sorted(subject_dirs, key=lambda d: (-_score(d), d.name))
    selected = scored[:max_subjects]
    unused   = scored[max_subjects:]

    print(f"Subject 선택: {len(selected)}명 사용 / {len(unused)}명 미사용 (전체 {len(subject_dirs)}명)")

    if unused_path is not None and unused:
        with open(unused_path, "w", encoding="utf-8") as f:
            for d in unused:
                f.write(d.name + "\n")
        print(f"미사용 subject 목록 → {unused_path}")

    return selected, unused


def _combined_patch_quality(patch: dict) -> tuple:
    wrinkle_ratio = float(patch["gt_pos_ratios"].get("wrinkle_pos_ratio", 0.0))
    red_ratio = float(patch["gt_pos_ratios"].get("red_pos_ratio", 0.0))
    brown_ratio = float(patch["gt_pos_ratios"].get("brown_pos_ratio", 0.0))

    wrinkle_pos = int(wrinkle_ratio > 0.0)
    red_pos = int(red_ratio > 0.0)
    brown_pos = int(brown_ratio > 0.0)
    positive_task_count = wrinkle_pos + red_pos + brown_pos
    is_positive = int(patch.get("is_positive", False))
    wrinkle_stream = int(patch.get("_stream_name") == "wrinkle")

    return (
        wrinkle_pos,
        positive_task_count,
        wrinkle_stream,
        red_pos,
        brown_pos,
        min(wrinkle_ratio, 1.0),
        min(red_ratio, 1.0),
        min(brown_ratio, 1.0),
        is_positive,
    )


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
            "gt_mask_applied": bool(apply_mask),
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
    neg_pos_ratio: float = 2.0,
    max_negative_if_no_positive: int = 0,
    wrinkle_min_pos_ratio: float = 1e-5,
    wrinkle_neg_pos_ratio: float = 2.0,
    max_wrinkle_negative_if_no_positive: int = 0,
    centered_red_patches: int = 48,
    centered_wrinkle_patches: int = 48,
    centered_jitter: int = 32,
    centered_min_mask_coverage: float = 0.5,
    centered_seed: int = 42,
    max_patches_per_subject: int | None = None,
    patch_max_overlap: float = 0.25,
    max_subjects: int | None = None,
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

    all_subject_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"input ID 수 (전체): {len(all_subject_dirs)}")

    unused_subjects_path = out_root / "unused_subjects.txt"
    subject_dirs, unused_dirs = select_subjects(
        all_subject_dirs,
        gt_maps,
        max_subjects=max_subjects,
        unused_path=unused_subjects_path if max_subjects else None,
    )

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
            subject_selected: list[dict] = []

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
                main_selected, main_dedup_dropped = dedup_selected_patches(
                    main_selected,
                    patch_size=patch_size,
                    max_overlap_fraction=patch_max_overlap,
                )
                subject_selected.extend(
                    _annotate_stream_candidates(
                        main_selected,
                        stream_name="main",
                        has_brown=("brown" in gt_arrays_main),
                        has_red=("red" in gt_arrays_main),
                        has_wrinkle=False,
                    )
                )
                print(f"  main stream: selected={len(main_selected)} dedup_dropped={main_dedup_dropped}")
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
                wrinkle_selected, wrinkle_dedup_dropped = dedup_selected_patches(
                    wrinkle_selected,
                    patch_size=patch_size,
                    max_overlap_fraction=patch_max_overlap,
                )
                subject_selected.extend(
                    _annotate_stream_candidates(
                        wrinkle_selected,
                        stream_name="wrinkle",
                        has_brown=False,
                        has_red=False,
                        has_wrinkle=True,
                    )
                )
                print(f"  wrinkle stream: selected={len(wrinkle_selected)} dedup_dropped={wrinkle_dedup_dropped}")
            else:
                print("  wrinkle stream: skip")

            if max_patches_per_subject and max_patches_per_subject > 0:
                if len(subject_selected) > max_patches_per_subject:
                    subject_selected = sorted(
                        subject_selected,
                        key=_combined_patch_quality,
                        reverse=True,
                    )[:max_patches_per_subject]
                    print(f"  subject cap applied: kept={len(subject_selected)}")

            by_stream: dict[str, list[dict]] = {"main": [], "wrinkle": []}
            for patch in subject_selected:
                by_stream[patch["_stream_name"]].append(patch)

            for stream_name, patches in by_stream.items():
                if not patches:
                    continue
                n_saved = save_stream_patches(
                    out_root=out_root,
                    subject_name=subject_name,
                    stream_name=stream_name,
                    selected_patches=patches,
                    has_brown=bool(patches[0]["_has_brown"]),
                    has_red=bool(patches[0]["_has_red"]),
                    has_wrinkle=bool(patches[0]["_has_wrinkle"]),
                    wrinkle_min_pos_ratio=wrinkle_min_pos_ratio,
                    apply_mask=apply_mask,
                    manifest=manifest,
                )
                saved_this_subject += n_saved

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
    parser.add_argument("--neg_pos_ratio", type=float, default=2.0)
    parser.add_argument("--max_negative_if_no_positive", type=int, default=0)
    parser.add_argument("--wrinkle_min_pos_ratio", type=float, default=1e-5)
    parser.add_argument("--wrinkle_neg_pos_ratio", type=float, default=2.0)
    parser.add_argument("--max_wrinkle_negative_if_no_positive", type=int, default=0)
    parser.add_argument("--centered_red_patches", type=int, default=48)
    parser.add_argument("--centered_wrinkle_patches", type=int, default=48)
    parser.add_argument("--centered_jitter", type=int, default=32)
    parser.add_argument("--centered_min_mask_coverage", type=float, default=0.5)
    parser.add_argument("--centered_seed", type=int, default=42)
    parser.add_argument("--max_patches_per_subject", type=int, default=0)
    parser.add_argument(
        "--patch_max_overlap", type=float, default=0.25,
        help="두 패치의 겹치는 면적이 단일 패치 면적의 이 비율 이상이면 중복 제거 (기본 0.25 = 25%%)",
    )
    parser.add_argument(
        "--max_subjects", type=int, default=0,
        help="사용할 최대 subject 수. 0이면 전체 사용. GT 우선순위(wrinkle>red>brown)로 선택.",
    )
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
        max_patches_per_subject=(
            args.max_patches_per_subject if args.max_patches_per_subject > 0 else None
        ),
        patch_max_overlap=args.patch_max_overlap,
        max_subjects=args.max_subjects if args.max_subjects > 0 else None,
        disable_mediapipe=args.disable_mediapipe,
        face_landmarker_task_path=args.face_landmarker_task_path,
    )


if __name__ == "__main__":
    main()
