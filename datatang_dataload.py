"""
datatang_dataload.py
====================
DataTang-style inference file-list builder.

Expected directory structure
----------------------------
Input images:
    input_root/
        ID001/
            F10.jpg
            F_11.jpg
        ID002/
            F10.jpg
            F_11.jpg

GT masks:
    gt_root/
        brownspots/
            images/
                sr-proto/
                    ID001.jpg
                    ID002.jpg
        red/
            images/
                sr-proto/
                    ID001.jpg
        wrinkle-deep/
            images/
                sr-proto/
                    ID001.jpg

Returned record format
----------------------
[
    {
        'stem'        : 'ID001',
        'rgb_cross'   : Path('.../ID001/F10.jpg'),
        'rgb_parallel': Path('.../ID001/F_11.jpg'),
        'brown'       : Path('.../brownspots/images/sr-proto/ID001.jpg') or None,
        'red'         : Path('.../red/images/sr-proto/ID001.jpg') or None,
        'wrinkle'     : Path('.../wrinkle-deep/images/sr-proto/ID001.jpg') or None,
        'has_brown'   : bool,
        'has_red'     : bool,
        'has_wrinkle' : bool,
    },
    ...
]
"""

from __future__ import annotations

from pathlib import Path


def _resolve_task_gt_path(gt_root: Path | None, task_dir: str, stem: str, suffix: str) -> Path | None:
    if gt_root is None:
        return None
    path = gt_root / task_dir / 'images' / 'sr-proto' / f'{stem}{suffix}'
    return path if path.exists() else None


def build_file_list(
    input_root: str | Path,
    gt_root: str | Path | None = None,
    *,
    cross_name: str = 'F10.jpg',
    parallel_name: str = 'F_11.jpg',
    gt_suffix: str = '.jpg',
    brown_dir: str = 'brownspots',
    red_dir: str = 'red',
    wrinkle_dir: str = 'wrinkle-deep',
    verbose: bool = True,
) -> list[dict]:
    """
    Build a file list for DataTang-style inference.

    Parameters
    ----------
    input_root:
        Root containing subject folders directly under it.
    gt_root:
        Root containing brownspots/red/wrinkle-deep GT folders.
        If None, GT paths are omitted.
    cross_name:
        Cross-polarization filename. Default: F10.jpg
    parallel_name:
        Parallel-polarization filename. Default: F_11.jpg
    gt_suffix:
        GT filename suffix. Default: .jpg
    brown_dir / red_dir / wrinkle_dir:
        Task directory names under gt_root.
    """
    input_root = Path(input_root)
    gt_root = Path(gt_root) if gt_root is not None else None

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    records: list[dict] = []
    skipped = 0

    for subject_dir in sorted(input_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        stem = subject_dir.name
        cross_path = subject_dir / cross_name
        parallel_path = subject_dir / parallel_name

        if not cross_path.exists():
            if verbose:
                print(f"[skip] cross missing: {cross_path}")
            skipped += 1
            continue

        if not parallel_path.exists():
            if verbose:
                print(f"[skip] parallel missing: {parallel_path}")
            skipped += 1
            continue

        brown_path = _resolve_task_gt_path(gt_root, brown_dir, stem, gt_suffix)
        red_path = _resolve_task_gt_path(gt_root, red_dir, stem, gt_suffix)
        wrinkle_path = _resolve_task_gt_path(gt_root, wrinkle_dir, stem, gt_suffix)

        records.append({
            'stem': stem,
            'rgb_cross': cross_path,
            'rgb_parallel': parallel_path,
            'brown': brown_path,
            'red': red_path,
            'wrinkle': wrinkle_path,
            'has_brown': brown_path is not None,
            'has_red': red_path is not None,
            'has_wrinkle': wrinkle_path is not None,
        })

    if verbose:
        total = len(records)
        n_b = sum(r['has_brown'] for r in records)
        n_r = sum(r['has_red'] for r in records)
        n_w = sum(r['has_wrinkle'] for r in records)
        print(
            f"[build_file_list] total={total} "
            f"(brown={n_b}, red={n_r}, wrinkle={n_w}, skip={skipped})"
        )

    return records


def filter_has_gt(records: list[dict], task: str) -> list[dict]:
    """Filter records with GT for a specific task: brown | red | wrinkle."""
    return [r for r in records if r[f'has_{task}']]


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Inspect DataTang-style inference file list')
    parser.add_argument('--input_root', required=True, help='input root path')
    parser.add_argument('--gt_root', default=None, help='GT root path')
    parser.add_argument('--cross_name', default='F10.jpg')
    parser.add_argument('--parallel_name', default='F_11.jpg')
    parser.add_argument('--gt_suffix', default='.jpg')
    parser.add_argument('--brown_dir', default='brownspots')
    parser.add_argument('--red_dir', default='red')
    parser.add_argument('--wrinkle_dir', default='wrinkle-deep')
    parser.add_argument('--dump_json', default=None, help='optional JSON output path')
    args = parser.parse_args()

    records = build_file_list(
        input_root=args.input_root,
        gt_root=args.gt_root,
        cross_name=args.cross_name,
        parallel_name=args.parallel_name,
        gt_suffix=args.gt_suffix,
        brown_dir=args.brown_dir,
        red_dir=args.red_dir,
        wrinkle_dir=args.wrinkle_dir,
    )

    if args.dump_json:
        serializable = [
            {k: str(v) if isinstance(v, Path) else v for k, v in r.items()}
            for r in records
        ]
        Path(args.dump_json).write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
        print(f"JSON saved: {args.dump_json}")
