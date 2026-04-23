"""
new_data_prep_2000.py
=====================
Standalone entrypoint for building a "good 2000 patches" dataset.

This script uses the implementation in new_data_prep.py, but exposes its own
CLI entrypoint with defaults tuned for a smaller, higher-quality dataset:

- max_total_patches      = 2000
- max_patches_per_subject = 12
"""

from __future__ import annotations

import argparse

from new_data_prep import prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VISIA patch generator (2000-patch quality-selected preset)"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--patch_dir", default="./patches_2000")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--min_mask_coverage", type=float, default=0.7)
    parser.add_argument(
        "--no_apply_mask",
        action="store_true",
        help="do not apply face mask to RGB patches",
    )
    parser.add_argument(
        "--neg_pos_ratio",
        type=float,
        default=2.0,
        help="keep at most this many generic negatives per positive",
    )
    parser.add_argument(
        "--max_negative_if_no_positive",
        type=int,
        default=0,
        help="max generic negatives to keep when no positive exists",
    )
    parser.add_argument(
        "--wrinkle_min_pos_ratio",
        type=float,
        default=1e-5,
        help="minimum positive ratio to mark a wrinkle patch as positive",
    )
    parser.add_argument(
        "--wrinkle_neg_pos_ratio",
        type=float,
        default=2.0,
        help="keep at most this many wrinkle negatives per wrinkle positive",
    )
    parser.add_argument(
        "--max_wrinkle_negative_if_no_positive",
        type=int,
        default=0,
        help="max wrinkle negatives to keep when no wrinkle positive exists",
    )
    parser.add_argument(
        "--centered_tasks",
        default="red,wrinkle",
        help="tasks that receive extra positive-centered patch candidates",
    )
    parser.add_argument(
        "--centered_patches_per_task",
        type=int,
        default=128,
        help="max positive-centered patch candidates per task per subject",
    )
    parser.add_argument(
        "--centered_jitter",
        type=int,
        default=64,
        help="jitter radius for positive-centered crop sampling",
    )
    parser.add_argument(
        "--centered_min_mask_coverage",
        type=float,
        default=0.5,
        help="minimum face-mask coverage for centered patch candidates",
    )
    parser.add_argument(
        "--centered_seed",
        type=int,
        default=42,
        help="seed for positive-centered sampling",
    )
    parser.add_argument(
        "--disable_mediapipe",
        action="store_true",
        help="disable MediaPipe and use fallback face mask only",
    )
    parser.add_argument(
        "--face_landmarker_task_path",
        type=str,
        default=None,
        help="path to MediaPipe face_landmarker.task",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of subject-level workers",
    )
    parser.add_argument(
        "--max_total_patches",
        type=int,
        default=2000,
        help="final total patch limit (default: 2000)",
    )
    parser.add_argument(
        "--max_patches_per_subject",
        type=int,
        default=12,
        help="max patches kept per subject before final global prune",
    )
    parser.add_argument(
        "--patch_dedup_iou",
        type=float,
        default=0.80,
        help="IoU threshold for removing near-duplicate overlapping patches",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        max_total_patches=args.max_total_patches,
        max_patches_per_subject=args.max_patches_per_subject,
        patch_dedup_iou=args.patch_dedup_iou,
    )


if __name__ == "__main__":
    main()
