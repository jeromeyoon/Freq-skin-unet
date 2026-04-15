"""
data_prep.py
============
원본 데이터 → 패치 추출 스크립트

입력 구조
---------
input_path/
  {ID}/
    F_10.jpg   ← 교차 편광 (cross polarization)
    F_11.jpg   ← 평행 편광 (parallel polarization)

gt_path/
  brownspots/annotations/sr-proto/{ID}.png
  redspots/annotations/sr-proto/{ID}.png
  wrinkle-deep/annotations/sr-proto/{ID}.png

출력 구조
---------
patch_dir/
  rgb_cross/      {ID}_{idx:04d}.png
  rgb_parallel/   {ID}_{idx:04d}.png
  brown/          {ID}_{idx:04d}.png   (GT가 있는 ID만)
  red/            {ID}_{idx:04d}.png   (GT가 있는 ID만)
  wrinkle/        {ID}_{idx:04d}.png   (GT가 있는 ID만)
  mask/           {ID}_{idx:04d}.png   (전체 영역 = 255)
  manifest.json                         (패치별 GT 가용성 정보)

실행
----
  python data_prep.py \
      --input_path ./raw_data/images \
      --gt_path    ./raw_data/gt \
      --patch_dir  ./patches \
      --patch_size 256 \
      --stride     256
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# GT 경로 탐색
# ══════════════════════════════════════════════════════════════════════════════
GT_SUBDIRS = {
    'brown'  : 'brownspots/annotations/sr-proto',
    'red'    : 'redspots/annotations/sr-proto',
    'wrinkle': 'wrinkle-deep/annotations/sr-proto',
}


def find_gt(gt_root: Path, task: str, subject_id: str) -> Path | None:
    """GT 파일 경로 반환. 없으면 None."""
    folder = gt_root / GT_SUBDIRS[task]
    # ID가 폴더명과 정확히 일치하거나 숫자 부분만 일치하는 경우 모두 탐색
    candidates = list(folder.glob(f'{subject_id}.png')) + \
                 list(folder.glob(f'{subject_id}*.png'))
    return candidates[0] if candidates else None


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 로드 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_rgb(path: Path, target_size: tuple | None = None) -> np.ndarray:
    """RGB 이미지 로드 → uint8 [H, W, 3]"""
    img = Image.open(path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)


def load_gray(path: Path, target_size: tuple | None = None) -> np.ndarray:
    """Gray 이미지 로드 → uint8 [H, W]"""
    img = Image.open(path).convert('L')
    if target_size:
        img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)


# ══════════════════════════════════════════════════════════════════════════════
# 패치 추출
# ══════════════════════════════════════════════════════════════════════════════
def extract_patches(arr: np.ndarray,
                    patch_size: int,
                    stride: int) -> list[tuple[np.ndarray, tuple]]:
    """
    2D 또는 3D 배열에서 슬라이딩 윈도우 패치 추출

    Returns
    -------
    list of (patch_array, (y, x))
    """
    H = arr.shape[0]
    W = arr.shape[1]
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = arr[y:y+patch_size, x:x+patch_size]
            patches.append((patch, (y, x)))
    return patches


def save_rgb_patch(arr: np.ndarray, path: Path):
    Image.fromarray(arr.astype(np.uint8), mode='RGB').save(path)


def save_gray_patch(arr: np.ndarray, path: Path):
    Image.fromarray(arr.astype(np.uint8), mode='L').save(path)


# ══════════════════════════════════════════════════════════════════════════════
# 메인 준비 함수
# ══════════════════════════════════════════════════════════════════════════════
def prepare(input_path: str,
            gt_path:    str,
            patch_dir:  str,
            patch_size: int = 256,
            stride:     int = 256,
            min_foreground: float = 0.0):
    """
    Parameters
    ----------
    input_path     : 원본 이미지 루트
    gt_path        : GT 루트
    patch_dir      : 출력 패치 폴더
    patch_size     : 패치 크기 (정사각형)
    stride         : 슬라이딩 보폭
    min_foreground : GT가 있는 경우, 패치 내 양성 픽셀 비율 최소값
                     (0이면 모든 패치 저장, 0.01이면 1% 이상만)
    """
    input_root = Path(input_path)
    gt_root    = Path(gt_path)
    out_root   = Path(patch_dir)

    # 출력 폴더 생성
    for sub in ['rgb_cross', 'rgb_parallel', 'brown', 'red', 'wrinkle', 'mask']:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    manifest = {}   # {patch_stem: {has_brown, has_red, has_wrinkle}}
    total_patches = 0

    # 모든 ID 폴더 탐색
    subject_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"발견된 ID: {len(subject_dirs)}개")

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        # 교차/평행 편광 파일 확인
        cross_path    = subj_dir / 'F_10.jpg'
        parallel_path = subj_dir / 'F_11.jpg'

        if not cross_path.exists() or not parallel_path.exists():
            print(f"  [SKIP] {subject_id}: F_10.jpg 또는 F_11.jpg 없음")
            continue

        # GT 파일 탐색 (없으면 None)
        gt_paths = {
            task: find_gt(gt_root, task, subject_id)
            for task in ['brown', 'red', 'wrinkle']
        }
        has_gt = {task: (p is not None) for task, p in gt_paths.items()}

        # 이미지 로드
        cross_img    = load_rgb(cross_path)
        parallel_img = load_rgb(parallel_path)
        H, W = cross_img.shape[:2]

        # 해상도가 다르면 교차 편광 기준으로 평행 편광 리사이즈
        if parallel_img.shape[:2] != (H, W):
            parallel_img = load_rgb(parallel_path, target_size=(W, H))

        # GT 로드 (교차 편광 해상도에 맞춤)
        gt_imgs = {}
        for task, gt_p in gt_paths.items():
            if gt_p is not None:
                gt_imgs[task] = load_gray(gt_p, target_size=(W, H))
            else:
                gt_imgs[task] = None

        # 패치 추출
        cross_patches    = extract_patches(cross_img,    patch_size, stride)
        parallel_patches = extract_patches(parallel_img, patch_size, stride)

        if len(cross_patches) != len(parallel_patches):
            print(f"  [WARN] {subject_id}: 교차/평행 패치 수 불일치")

        # GT 패치 추출
        gt_patch_lists = {}
        for task, gt_arr in gt_imgs.items():
            if gt_arr is not None:
                gt_patch_lists[task] = extract_patches(gt_arr, patch_size, stride)

        for idx, ((c_patch, pos), (p_patch, _)) in enumerate(
                zip(cross_patches, parallel_patches)):

            stem = f'{subject_id}_{idx:04d}'

            # GT가 있는 경우 foreground 비율 확인
            skip = False
            if min_foreground > 0:
                for task in ['brown', 'red', 'wrinkle']:
                    if has_gt[task]:
                        gt_p = gt_patch_lists[task][idx][0]
                        fg_ratio = (gt_p > 127).sum() / (patch_size * patch_size)
                        # 모든 task가 foreground 없으면 skip
                        # (하나라도 있으면 포함)
                        break

            if skip:
                continue

            # 저장
            save_rgb_patch(c_patch, out_root / 'rgb_cross'    / f'{stem}.png')
            save_rgb_patch(p_patch, out_root / 'rgb_parallel' / f'{stem}.png')

            # 마스크 (전 영역 255)
            mask = np.full((patch_size, patch_size), 255, dtype=np.uint8)
            save_gray_patch(mask, out_root / 'mask' / f'{stem}.png')

            # GT 패치 저장
            for task in ['brown', 'red', 'wrinkle']:
                if has_gt[task]:
                    gt_p = gt_patch_lists[task][idx][0]
                    save_gray_patch(gt_p, out_root / task / f'{stem}.png')

            manifest[stem] = {
                'subject_id'  : subject_id,
                'patch_idx'   : idx,
                'patch_pos'   : list(pos),
                'has_brown'   : has_gt['brown'],
                'has_red'     : has_gt['red'],
                'has_wrinkle' : has_gt['wrinkle'],
            }
            total_patches += 1

    # manifest 저장
    manifest_path = out_root / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n완료: 총 {total_patches}개 패치 → {patch_dir}")
    print(f"manifest: {manifest_path}")

    # GT별 통계 출력
    stats = {t: sum(1 for v in manifest.values() if v[f'has_{t}'])
             for t in ['brown', 'red', 'wrinkle']}
    print(f"  brown GT 있는 패치:   {stats['brown']:,}")
    print(f"  red GT 있는 패치:     {stats['red']:,}")
    print(f"  wrinkle GT 있는 패치: {stats['wrinkle']:,}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='피부 분석 패치 준비')
    parser.add_argument('--input_path', required=True,
                        help='원본 이미지 루트 (ID 폴더 포함)')
    parser.add_argument('--gt_path',    required=True,
                        help='GT 루트 (brownspots/redspots/wrinkle-deep)')
    parser.add_argument('--patch_dir',  default='./patches',
                        help='패치 출력 폴더')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--stride',     type=int, default=256,
                        help='슬라이딩 보폭 (stride<patch_size이면 겹침)')
    parser.add_argument('--min_foreground', type=float, default=0.0,
                        help='GT 양성 픽셀 최소 비율 (0=모든 패치 저장)')
    args = parser.parse_args()

    prepare(
        input_path    = args.input_path,
        gt_path       = args.gt_path,
        patch_dir     = args.patch_dir,
        patch_size    = args.patch_size,
        stride        = args.stride,
        min_foreground= args.min_foreground,
    )
