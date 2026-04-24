"""
new_data_prep_2000.py
=====================
3510장 입력 이미지에서 퀄리티 기준 상위 2000장(subject)을 선별한 뒤
패치를 생성하는 엔트리포인트.

Subject 선별 기준 (이미지 로딩 없이 GT 보유 여부만 확인)
---------------------------------------------------------
  wrinkle GT 보유 : +4점  (가장 희소하고 학습에 중요)
  red GT 보유     : +2점
  brown GT 보유   : +1점
  동점이면 subject 이름 오름차순

패치 생성 전략
--------------
  stride=256 : 256×256 패치 간 overlap 없음 (중복 패치 방지)
  centered    : wrinkle/red 레이블 중심 패치 추가 생성 (희소 레이블 보강)
  dedup       : IoU ≥ 0.80 인접 패치 제거

패치 수 제한 없음
-----------------
  max_subjects=2000 으로 입력 subject 수를 제어.
  패치 자체에는 인위적 상한을 두지 않는다.
  Subject당 평균 12~20장이므로 총 24,000~40,000장 예상.
  너무 많다면 --max_patches_per_subject 으로 조절.
"""

from __future__ import annotations

import argparse

from new_data_prep import prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VISIA patch generator — quality-based subject selection"
    )
    parser.add_argument("--input_path", required=True,
                        help="root directory containing all subject folders")
    parser.add_argument("--gt_path", required=True,
                        help="root directory of GT annotations")
    parser.add_argument("--patch_dir", default="./patches",
                        help="output patch directory")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256,
                        help="sliding window stride; 256 = no overlap between grid patches")
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
        help="max generic negatives to keep when subject has no positive patch",
    )
    parser.add_argument(
        "--wrinkle_min_pos_ratio",
        type=float,
        default=1e-5,
        help="minimum positive pixel ratio to mark a wrinkle patch as positive",
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
        help="max wrinkle negatives when subject has no wrinkle positive",
    )
    parser.add_argument(
        "--centered_tasks",
        default="red,wrinkle",
        help="tasks that get extra positive-centered patch candidates",
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
        help="jitter radius (px) for positive-centered crop sampling",
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
        help="path to MediaPipe face_landmarker.task model",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of parallel subject-level workers",
    )
    # ── Subject selection ──────────────────────────────────────────────────
    parser.add_argument(
        "--max_subjects",
        type=int,
        default=2000,
        help="select this many top-quality subjects from all inputs "
             "(ranked by GT richness: wrinkle=4pt, red=2pt, brown=1pt); "
             "set 0 to disable and process all subjects",
    )
    # ── Per-subject patch cap ──────────────────────────────────────────────
    parser.add_argument(
        "--max_patches_per_subject",
        type=int,
        default=0,
        help="hard cap on patches saved per subject (0 = no limit); "
             "use only if disk space or training time is a concern",
    )
    # ── Dedup ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--patch_dedup_iou",
        type=float,
        default=0.80,
        help="IoU threshold for removing near-duplicate overlapping patches "
             "(default 0.80; lower = more aggressive dedup)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    max_subjects = args.max_subjects if args.max_subjects > 0 else None
    max_patches_per_subject = (
        args.max_patches_per_subject if args.max_patches_per_subject > 0 else None
    )

    print("=== new_data_prep_2000: quality-based subject selection ===")
    print(f"  max_subjects          : {max_subjects or 'all'}")
    print(f"  stride                : {args.stride} px (no overlap)")
    print(f"  patch_dedup_iou       : {args.patch_dedup_iou}")
    print(f"  max_patches_per_subject: {max_patches_per_subject or 'no limit'}")
    print(f"  max_total_patches     : no limit (controlled by subject count)")

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
            t.strip() for t in args.centered_tasks.split(",") if t.strip()
        ),
        centered_patches_per_task=args.centered_patches_per_task,
        centered_jitter=args.centered_jitter,
        centered_min_mask_coverage=args.centered_min_mask_coverage,
        centered_seed=args.centered_seed,
        disable_mediapipe=args.disable_mediapipe,
        face_landmarker_task_path=args.face_landmarker_task_path,
        num_workers=args.num_workers,
        max_subjects=max_subjects,
        max_total_patches=None,          # subject 수로 제어, 패치 수 무제한
        max_patches_per_subject=max_patches_per_subject,
        patch_dedup_iou=args.patch_dedup_iou,
    )


if __name__ == "__main__":
    main()
