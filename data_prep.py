"""
data_prep.py
============
원본 데이터 → 패치 추출 스크립트

입력 구조
---------
input_path/
  {ID_name}/          ← 예: PATIENT_001, HONG_123  (이름+숫자 혼합)
    F_10.jpg          교차 편광
    F_11.jpg          평행 편광

gt_path/
  brownspots/annotations/sr-proto/{ID}.png    ← ID만 (숫자)
  redspots/annotations/sr-proto/{ID}.png
  wrinkle-deep/annotations/sr-proto/{ID}.png

ID 매칭 전략 (우선순위 순)
--------------------------
1. 정확히 일치 (input_name == GT_ID)
2. GT_ID가 input_name 안에 포함 ("001" in "HONG_001")
3. input_name의 숫자 추출 후 매핑 ("HONG001" → "001")

얼굴 마스크 생성
-----------------
MediaPipe Face Mesh로 얼굴 윤곽선 폴리곤 → 귀·목 자동 제외
MediaPipe 미설치 시 → 이미지 중앙 타원형 fallback

출력 구조
---------
patch_dir/
  rgb_cross/      {ID}_{idx:04d}.png
  rgb_parallel/   {ID}_{idx:04d}.png
  brown/          {ID}_{idx:04d}.png   (GT 있는 경우만)
  red/            {ID}_{idx:04d}.png
  wrinkle/        {ID}_{idx:04d}.png
  mask/           {ID}_{idx:04d}.png   얼굴 마스크
  manifest.json

설치
----
  pip install mediapipe opencv-python Pillow numpy

실행
----
  python data_prep.py \\
      --input_path ./raw_data/images \\
      --gt_path    ./raw_data/gt \\
      --patch_dir  ./patches \\
      --patch_size 256 --stride 256
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ══════════════════════════════════════════════════════════════════════════════
# GT 경로 설정
# ══════════════════════════════════════════════════════════════════════════════
GT_SUBDIRS = {
    'brown'  : 'brownspots/annotations/sr-proto',
    'red'    : 'redspots/annotations/sr-proto',
    'wrinkle': 'wrinkle-deep/annotations/sr-proto',
}


# ══════════════════════════════════════════════════════════════════════════════
# ID 매칭
# ══════════════════════════════════════════════════════════════════════════════
def _extract_numeric(name: str) -> list[str]:
    """문자열에서 숫자 덩어리 추출. 예: 'HONG_001' → ['001']"""
    return re.findall(r'\d+', name)


def build_gt_id_map(gt_root: Path, task: str) -> dict[str, Path]:
    """GT 폴더의 ID → Path 매핑 딕셔너리 반환"""
    folder = gt_root / GT_SUBDIRS[task]
    if not folder.exists():
        return {}
    return {p.stem: p for p in folder.glob('*.png')}


def find_matching_gt(input_name: str,
                     gt_map: dict[str, Path]) -> Path | None:
    """
    input 폴더명에서 GT ID를 찾아 Path 반환.
    없으면 None.

    매칭 순서:
    1. 정확히 일치
    2. GT ID가 input_name에 포함 (예: "001" in "HONG_001")
    3. input_name의 숫자 부분이 GT ID와 일치 (zero-padding 무시)
    """
    if not gt_map:
        return None

    # 1. 정확히 일치
    if input_name in gt_map:
        return gt_map[input_name]

    # 2. GT ID ⊂ input_name
    for gt_id, path in gt_map.items():
        if gt_id in input_name:
            return path

    # 3. 숫자 추출 후 zero-padding 무시 비교
    input_nums = _extract_numeric(input_name)
    for gt_id, path in gt_map.items():
        gt_nums = _extract_numeric(gt_id)
        # 한 쪽이라도 숫자가 lstrip('0')으로 일치하면 매핑
        for inum in input_nums:
            for gnum in gt_nums:
                if inum.lstrip('0') == gnum.lstrip('0') and inum.lstrip('0') != '':
                    return path

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 얼굴 마스크 생성
# ══════════════════════════════════════════════════════════════════════════════
# MediaPipe Face Mesh의 얼굴 윤곽선 랜드마크 인덱스
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]


def _try_import_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        return None


def generate_face_mask_mediapipe(img_bgr: np.ndarray,
                                 mp_face_mesh) -> np.ndarray | None:
    """
    MediaPipe Face Mesh로 얼굴 윤곽선 마스크 생성.
    성공 시 uint8 [H,W] (얼굴=255, 배경=0), 실패 시 None.
    """
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode        = True,
        max_num_faces            = 1,
        refine_landmarks         = False,
        min_detection_confidence = 0.4,
    ) as fm:
        results = fm.process(img_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark

    # 얼굴 윤곽선 좌표 (픽셀 단위)
    pts = np.array([
        [int(lm[i].x * W), int(lm[i].y * H)]
        for i in _FACE_OVAL
    ], dtype=np.int32)

    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    # 형태학적 침식으로 경계 정리 (귀 근처 제거)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask   = cv2.erode(mask, kernel, iterations=2)

    return mask


def generate_face_mask_ellipse(img_bgr: np.ndarray) -> np.ndarray:
    """
    MediaPipe 없을 때 fallback: 이미지 중앙 타원형 마스크.
    일반적인 얼굴 사진에서 귀·목·배경을 어느 정도 제외.
    """
    H, W = img_bgr.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    # 중앙 타원: 가로 70%, 세로 80% (상단 쪽에 치우침)
    cx, cy   = W // 2, int(H * 0.45)
    axes     = (int(W * 0.35), int(H * 0.40))
    cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 255, -1)
    return mask


def generate_face_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    얼굴 마스크 생성 (MediaPipe 우선, fallback은 타원형).
    반환: uint8 [H, W]  (얼굴=255)
    """
    mp = _try_import_mediapipe()
    if mp is not None:
        mask = generate_face_mask_mediapipe(img_bgr, mp.solutions.face_mesh)
        if mask is not None:
            return mask
        print("    [WARN] MediaPipe 얼굴 미검출 → 타원형 fallback")

    return generate_face_mask_ellipse(img_bgr)


# ══════════════════════════════════════════════════════════════════════════════
# 이미지 로드 유틸
# ══════════════════════════════════════════════════════════════════════════════
def load_rgb_bgr(path: Path, target_hw: tuple | None = None) -> np.ndarray:
    """BGR numpy [H,W,3]  (OpenCV 포맷, 마스크 생성용)"""
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"이미지 로드 실패: {path}")
    if target_hw:
        img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
    return img


def bgr2rgb_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_gray(path: Path, target_hw: tuple | None = None) -> np.ndarray:
    """uint8 [H,W]"""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"GT 로드 실패: {path}")
    if target_hw:
        img = cv2.resize(img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 패치 추출
# ══════════════════════════════════════════════════════════════════════════════
def extract_patches(arr: np.ndarray,
                    patch_size: int,
                    stride: int) -> list[tuple[np.ndarray, tuple]]:
    """슬라이딩 윈도우 패치. 반환: [(patch, (y, x)), ...]"""
    H, W = arr.shape[:2]
    patches = []
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patches.append((arr[y:y+patch_size, x:x+patch_size], (y, x)))
    return patches


def mask_coverage(mask_patch: np.ndarray, threshold: float) -> bool:
    """패치 내 마스크 커버리지가 threshold 이상이면 True"""
    return (mask_patch > 127).mean() >= threshold


# ══════════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════════
def prepare(input_path:       str,
            gt_path:          str,
            patch_dir:        str,
            patch_size:       int   = 256,
            stride:           int   = 256,
            min_mask_coverage: float = 0.3):
    """
    Parameters
    ----------
    min_mask_coverage : 패치 저장 최소 얼굴 마스크 비율 (0~1)
                        0.3 → 패치의 30% 이상이 얼굴 영역이어야 저장
    """
    input_root = Path(input_path)
    gt_root    = Path(gt_path)
    out_root   = Path(patch_dir)

    for sub in ['rgb_cross', 'rgb_parallel', 'brown', 'red', 'wrinkle', 'mask']:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    # GT ID 맵 사전 구축
    gt_maps = {task: build_gt_id_map(gt_root, task) for task in GT_SUBDIRS}
    print(f"GT ID 수: brown={len(gt_maps['brown'])}, "
          f"red={len(gt_maps['red'])}, wrinkle={len(gt_maps['wrinkle'])}")

    subject_dirs = sorted([d for d in input_root.iterdir() if d.is_dir()])
    print(f"input ID 수: {len(subject_dirs)}")

    manifest      = {}
    total_patches = 0
    skipped_ids   = []

    for subj_dir in subject_dirs:
        subject_name = subj_dir.name

        cross_path    = subj_dir / 'F_10.jpg'
        parallel_path = subj_dir / 'F_11.jpg'

        if not cross_path.exists() or not parallel_path.exists():
            print(f"  [SKIP] {subject_name}: F_10.jpg / F_11.jpg 없음")
            skipped_ids.append(subject_name)
            continue

        # GT 매칭
        gt_paths = {
            task: find_matching_gt(subject_name, gt_maps[task])
            for task in GT_SUBDIRS
        }
        has_gt = {task: p is not None for task, p in gt_paths.items()}

        matched = [t for t, h in has_gt.items() if h]
        if not matched:
            print(f"  [INFO] {subject_name}: 매칭된 GT 없음 → 저장 (입력 only)")
        else:
            print(f"  {subject_name}: GT={matched}")

        # 이미지 로드 (BGR, OpenCV)
        cross_bgr    = load_rgb_bgr(cross_path)
        H, W         = cross_bgr.shape[:2]
        parallel_bgr = load_rgb_bgr(parallel_path, target_hw=(H, W))

        # 얼굴 마스크 생성 (교차 편광 이미지 기준)
        face_mask = generate_face_mask(cross_bgr)   # [H, W] uint8

        # GT 로드
        gt_arrays = {}
        for task, gt_p in gt_paths.items():
            if gt_p is not None:
                gt_arrays[task] = load_gray(gt_p, target_hw=(H, W))

        # 패치 추출
        cross_patches    = extract_patches(cross_bgr,    patch_size, stride)
        parallel_patches = extract_patches(parallel_bgr, patch_size, stride)
        mask_patches     = extract_patches(face_mask,    patch_size, stride)

        gt_patch_lists = {
            task: extract_patches(arr, patch_size, stride)
            for task, arr in gt_arrays.items()
        }

        saved = 0
        for idx, ((c_p, pos), (p_p, _), (m_p, _)) in enumerate(
                zip(cross_patches, parallel_patches, mask_patches)):

            # 얼굴 마스크 커버리지 필터
            if not mask_coverage(m_p, min_mask_coverage):
                continue

            stem = f'{subject_name}_{idx:04d}'

            # 저장
            bgr2rgb_pil(c_p).save(out_root / 'rgb_cross'    / f'{stem}.png')
            bgr2rgb_pil(p_p).save(out_root / 'rgb_parallel' / f'{stem}.png')
            Image.fromarray(m_p, mode='L').save(out_root / 'mask' / f'{stem}.png')

            for task in ['brown', 'red', 'wrinkle']:
                if task in gt_patch_lists:
                    gt_p_arr = gt_patch_lists[task][idx][0]
                    Image.fromarray(gt_p_arr, mode='L').save(
                        out_root / task / f'{stem}.png')

            manifest[stem] = {
                'subject_name': subject_name,
                'patch_idx'   : idx,
                'patch_pos'   : list(pos),
                'has_brown'   : has_gt['brown'],
                'has_red'     : has_gt['red'],
                'has_wrinkle' : has_gt['wrinkle'],
            }
            saved         += 1
            total_patches += 1

        print(f"    → {saved}개 패치 저장")

    # manifest 저장
    manifest_path = out_root / 'manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 최종 통계
    print(f"\n완료: 총 {total_patches}개 패치 → {patch_dir}")
    if skipped_ids:
        print(f"편광 이미지 없어 스킵된 ID: {skipped_ids}")
    stats = {t: sum(1 for v in manifest.values() if v[f'has_{t}'])
             for t in ['brown', 'red', 'wrinkle']}
    print(f"  brown GT:   {stats['brown']:,} 패치")
    print(f"  red GT:     {stats['red']:,} 패치")
    print(f"  wrinkle GT: {stats['wrinkle']:,} 패치")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='피부 분석 패치 준비')
    parser.add_argument('--input_path',  required=True)
    parser.add_argument('--gt_path',     required=True)
    parser.add_argument('--patch_dir',   default='./patches')
    parser.add_argument('--patch_size',  type=int,   default=256)
    parser.add_argument('--stride',      type=int,   default=256)
    parser.add_argument('--min_mask_coverage', type=float, default=0.3,
                        help='패치 저장 최소 얼굴 마스크 비율 (기본 0.3)')
    args = parser.parse_args()

    prepare(
        input_path        = args.input_path,
        gt_path           = args.gt_path,
        patch_dir         = args.patch_dir,
        patch_size        = args.patch_size,
        stride            = args.stride,
        min_mask_coverage = args.min_mask_coverage,
    )
