"""
in_house_dataload.py
====================
사내 데이터셋 파일 리스트 빌더 (inference용)

디렉토리 구조
-------------
입력 이미지:
    input_root/
        D0531/
            007_1973_7_Male/
                F_10.jpg   ← 교차 편광 (cross)
                F_11.jpg   ← 평행 편광 (parallel)
            012_2048_3_Female/
                F_10.jpg
                F_11.jpg
        D0532/
            ...

GT 마스크 (선택):
    brown_root/  D05310007.png
    red_root/    D05310007.png
    wrinkle_root/D05310007.png

GT 파일명 규칙
--------------
    stem = D폴더명 + 피험자 폴더 첫 세자리를 네자리로 변환
    예시: D0531 / 007_1973_7_Male  →  D05310007
         D0104 / 023_0456_2_Male   →  D01040023

반환 구조 (build_file_list)
---------------------------
    [
        {
            'stem'        : 'D05310007',
            'rgb_cross'   : Path('.../F_10.jpg'),
            'rgb_parallel': Path('.../F_11.jpg'),
            'brown'       : Path('.../D05310007.png') or None,
            'red'         : Path('.../D05310007.png') or None,
            'wrinkle'     : Path('.../D05310007.png') or None,
            'has_brown'   : bool,
            'has_red'     : bool,
            'has_wrinkle' : bool,
        },
        ...
    ]

사용 예
-------
    from in_house_dataload import build_file_list

    records = build_file_list(
        input_root   = '/data/raw',
        brown_root   = '/data/gt/brown',
        red_root     = '/data/gt/red',
        wrinkle_root = '/data/gt/wrinkle',
    )
    print(f"총 {len(records)}개 샘플")
    for r in records[:3]:
        print(r['stem'], r['has_brown'], r['has_red'])
"""

from __future__ import annotations

import re
from pathlib import Path


# ── GT stem 유도 ──────────────────────────────────────────────────────────────

_D_FOLDER_RE = re.compile(r'^D\d{4}$', re.IGNORECASE)  # D0531 형식 검증


def _derive_stem(d_folder: str, subject_folder: str) -> str:
    """
    폴더명 조합으로 GT 파일 stem 생성.

    Parameters
    ----------
    d_folder       : 'D0531'          (입력 루트 바로 아래 폴더)
    subject_folder : '007_1973_7_Male' (피험자 하위 폴더)

    Returns
    -------
    'D05310007'
        - d_folder 그대로 + 피험자 폴더 첫 토큰을 4자리 0-패딩으로 붙임

    Raises
    ------
    ValueError
        subject_folder 첫 토큰이 숫자가 아닌 경우
    """
    first_token = subject_folder.split('_')[0]   # '007'
    if not first_token.isdigit():
        raise ValueError(
            f"subject_folder '{subject_folder}' 첫 토큰이 숫자가 아닙니다: '{first_token}'"
        )
    subject_id = int(first_token)                 # 7
    return f"{d_folder}{subject_id:04d}"          # 'D05310007'


# ── 파일 리스트 빌더 ──────────────────────────────────────────────────────────

def build_file_list(
    input_root   : str | Path,
    brown_root   : str | Path | None = None,
    red_root     : str | Path | None = None,
    wrinkle_root : str | Path | None = None,
    cross_name   : str = 'F_10.jpg',
    parallel_name: str = 'F_11.jpg',
    verbose      : bool = True,
) -> list[dict]:
    """
    입력 루트를 재귀 탐색해 (cross, parallel) 쌍과 GT 경로를 매핑한 리스트 반환.

    Parameters
    ----------
    input_root    : 입력 이미지 루트 경로 (D*/subject*/F_10.jpg 구조)
    brown_root    : brownspot GT 마스크 폴더 (없으면 None)
    red_root      : red spot GT 마스크 폴더  (없으면 None)
    wrinkle_root  : wrinkle GT 마스크 폴더  (없으면 None)
    cross_name    : 교차 편광 파일명 (기본 'F_10.jpg')
    parallel_name : 평행 편광 파일명 (기본 'F_11.jpg')
    verbose       : 탐색 요약 출력 여부

    Returns
    -------
    records : list[dict]
        stem, rgb_cross, rgb_parallel, brown/red/wrinkle (Path or None),
        has_brown/red/wrinkle (bool)
    """
    input_root = Path(input_root)

    gt_roots: dict[str, Path | None] = {
        'brown'  : Path(brown_root)   if brown_root   else None,
        'red'    : Path(red_root)     if red_root     else None,
        'wrinkle': Path(wrinkle_root) if wrinkle_root else None,
    }

    records: list[dict] = []
    skipped = 0

    # D* 폴더 순회
    for d_dir in sorted(input_root.iterdir()):
        if not d_dir.is_dir():
            continue
        if not _D_FOLDER_RE.match(d_dir.name):
            continue

        d_folder = d_dir.name.upper()   # 대소문자 통일

        # 피험자 하위 폴더 순회
        for subj_dir in sorted(d_dir.iterdir()):
            if not subj_dir.is_dir():
                continue

            cross_path    = subj_dir / cross_name
            parallel_path = subj_dir / parallel_name

            # 입력 이미지 쌍이 모두 존재해야 유효한 샘플
            if not cross_path.exists():
                if verbose:
                    print(f"[skip] cross 없음: {cross_path}")
                skipped += 1
                continue
            if not parallel_path.exists():
                if verbose:
                    print(f"[skip] parallel 없음: {parallel_path}")
                skipped += 1
                continue

            # stem 유도
            try:
                stem = _derive_stem(d_folder, subj_dir.name)
            except ValueError as e:
                if verbose:
                    print(f"[skip] stem 유도 실패 ({subj_dir}): {e}")
                skipped += 1
                continue

            # GT 경로 매핑
            gt_paths: dict[str, Path | None] = {}
            has: dict[str, bool] = {}
            for task, gt_root in gt_roots.items():
                if gt_root is not None:
                    p = gt_root / f"{stem}.png"
                    gt_paths[task] = p if p.exists() else None
                else:
                    gt_paths[task] = None
                has[task] = gt_paths[task] is not None

            records.append({
                'stem'        : stem,
                'rgb_cross'   : cross_path,
                'rgb_parallel': parallel_path,
                'brown'       : gt_paths['brown'],
                'red'         : gt_paths['red'],
                'wrinkle'     : gt_paths['wrinkle'],
                'has_brown'   : has['brown'],
                'has_red'     : has['red'],
                'has_wrinkle' : has['wrinkle'],
            })

    if verbose:
        total = len(records)
        n_b = sum(r['has_brown']   for r in records)
        n_r = sum(r['has_red']     for r in records)
        n_w = sum(r['has_wrinkle'] for r in records)
        print(
            f"[build_file_list] 총 {total}개 샘플 "
            f"(brown={n_b}, red={n_r}, wrinkle={n_w}, skip={skipped})"
        )

    return records


# ── 간단한 필터 유틸 ──────────────────────────────────────────────────────────

def filter_has_gt(records: list[dict], task: str) -> list[dict]:
    """GT가 있는 샘플만 필터링. task = 'brown' | 'red' | 'wrinkle'"""
    return [r for r in records if r[f'has_{task}']]


def filter_d_folder(records: list[dict], d_folder: str) -> list[dict]:
    """특정 D 폴더에 속한 샘플만 필터링. 예: filter_d_folder(records, 'D0531')"""
    prefix = d_folder.upper()
    return [r for r in records if r['stem'].startswith(prefix)]


# ── CLI (파일 리스트 확인용) ──────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='사내 데이터셋 파일 리스트 확인'
    )
    parser.add_argument('--input_root',   required=True, help='입력 이미지 루트')
    parser.add_argument('--brown_root',   default=None,  help='brownspot GT 폴더')
    parser.add_argument('--red_root',     default=None,  help='red spot GT 폴더')
    parser.add_argument('--wrinkle_root', default=None,  help='wrinkle GT 폴더')
    parser.add_argument('--cross_name',   default='F_10.jpg')
    parser.add_argument('--parallel_name',default='F_11.jpg')
    parser.add_argument('--dump_json',    default=None,
                        help='파일 리스트를 JSON으로 저장할 경로 (선택)')
    args = parser.parse_args()

    records = build_file_list(
        input_root    = args.input_root,
        brown_root    = args.brown_root,
        red_root      = args.red_root,
        wrinkle_root  = args.wrinkle_root,
        cross_name    = args.cross_name,
        parallel_name = args.parallel_name,
    )

    if args.dump_json:
        # Path는 JSON 직렬화 불가 → str로 변환
        serializable = [
            {k: str(v) if isinstance(v, Path) else v for k, v in r.items()}
            for r in records
        ]
        Path(args.dump_json).write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding='utf-8'
        )
        print(f"JSON 저장 완료: {args.dump_json}")

    # 처음 5개 샘플 출력
    for r in records[:5]:
        print(
            f"  {r['stem']}  "
            f"brown={'O' if r['has_brown'] else '-'}  "
            f"red={'O' if r['has_red'] else '-'}  "
            f"wrinkle={'O' if r['has_wrinkle'] else '-'}"
        )
