"""
skin_dataset.py
===============
SkinDataset : 부분 GT(Partial Label) 지원 데이터셋

manifest.json을 읽어 패치별로 어떤 GT가 있는지 파악.
GT가 없는 task는 None 반환 → skin_loss.py에서 해당 loss 스킵.

디렉토리 구조 (data_prep.py 출력)
-----------------------------------
patch_dir/
  rgb_cross/      {stem}.png   교차 편광
  rgb_parallel/   {stem}.png   평행 편광
  brown/          {stem}.png   (있는 경우만)
  red/            {stem}.png   (있는 경우만)
  wrinkle/        {stem}.png   (있는 경우만)
  mask/           {stem}.png   피부 영역 (없으면 전체)
  manifest.json                패치별 GT 가용성

__getitem__ 반환
----------------
{
  'rgb_cross'    : [3,H,W],
  'rgb_parallel' : [3,H,W],
  'brown'        : [1,H,W] or None,
  'red'          : [1,H,W] or None,
  'wrinkle'      : [1,H,W] or None,
  'mask'         : [1,H,W],
  'has_brown'    : bool,
  'has_red'      : bool,
  'has_wrinkle'  : bool,
  'stem'         : str,
}
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image


class SkinDataset(Dataset):

    def __init__(self,
                 patch_dir: str,
                 img_size:  int  = 256,
                 augment:   bool = False):
        self.patch_dir = Path(patch_dir)
        self.img_size  = img_size
        self.augment   = augment

        # manifest 로드
        manifest_path = self.patch_dir / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                self.manifest = json.load(f)
            self.stems = list(self.manifest.keys())
        else:
            # manifest 없으면 rgb_cross 기준으로 자동 탐색 (GT 없음으로 간주)
            self.stems = sorted(
                p.stem for p in (self.patch_dir / 'rgb_cross').glob('*.png')
            )
            self.manifest = {
                stem: {'has_brown': False, 'has_red': False, 'has_wrinkle': False}
                for stem in self.stems
            }

        # brown / red / wrinkle 이 전부 False인 패치 제외
        # → 어떤 GT도 없는 샘플은 지도학습 신호가 없으므로 학습 불필요
        before = len(self.stems)
        self.stems = [
            stem for stem in self.stems
            if any(self.manifest[stem].get(f'has_{t}', False)
                   for t in ('brown', 'red', 'wrinkle'))
        ]
        excluded = before - len(self.stems)
        if excluded:
            print(f"[SkinDataset] GT 없는 패치 {excluded}개 제외 "
                  f"({before} → {len(self.stems)})")

        assert len(self.stems) > 0, f"패치를 찾을 수 없습니다: {patch_dir}"

    def __len__(self):
        return len(self.stems)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('RGB').resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)   # [3, H, W]  0~1

    def _load_gray(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert('L').resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)   # [1, H, W]  0~1

    def _apply_augment(self, *tensors):
        """동일한 geometric augment를 모든 텐서에 적용 (None 안전)"""
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5
        result = []
        for t in tensors:
            if t is None:
                result.append(None)
                continue
            if do_hflip:
                t = TF.hflip(t)
            if do_vflip:
                t = TF.vflip(t)
            result.append(t)
        return tuple(result)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        info = self.manifest[stem]

        # 필수 입력 (항상 존재)
        rgb_cross    = self._load_rgb(self.patch_dir / 'rgb_cross'    / f'{stem}.png')
        rgb_parallel = self._load_rgb(self.patch_dir / 'rgb_parallel' / f'{stem}.png')

        # 마스크 (없으면 전체 영역)
        mask_path = self.patch_dir / 'mask' / f'{stem}.png'
        mask = self._load_gray(mask_path) if mask_path.exists() \
               else torch.ones(1, self.img_size, self.img_size)

        # GT (부분 로드)
        def load_gt(task: str) -> torch.Tensor | None:
            if not info.get(f'has_{task}', False):
                return None
            gt_path = self.patch_dir / task / f'{stem}.png'
            return self._load_gray(gt_path) if gt_path.exists() else None

        brown   = load_gt('brown')
        red     = load_gt('red')
        wrinkle = load_gt('wrinkle')

        # Augmentation (GT가 None인 경우 그대로 통과)
        if self.augment:
            rgb_cross, rgb_parallel, mask, brown, red, wrinkle = self._apply_augment(
                rgb_cross, rgb_parallel, mask, brown, red, wrinkle)

        return {
            'rgb_cross'   : rgb_cross,
            'rgb_parallel': rgb_parallel,
            'brown'       : brown,
            'red'         : red,
            'wrinkle'     : wrinkle,
            'mask'        : mask,
            'has_brown'   : brown   is not None,
            'has_red'     : red     is not None,
            'has_wrinkle' : wrinkle is not None,
            'stem'        : stem,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Collate : None GT 처리 (DataLoader용)
# ══════════════════════════════════════════════════════════════════════════════
def skin_collate_fn(batch: list) -> dict:
    """
    None GT가 섞인 배치를 처리하는 커스텀 collate.
    GT가 하나도 없는 task는 None을 반환, 있는 경우만 stack.
    """
    out = {}

    # 항상 존재하는 필드
    for key in ['rgb_cross', 'rgb_parallel', 'mask']:
        out[key] = torch.stack([b[key] for b in batch])

    # bool 필드
    for key in ['has_brown', 'has_red', 'has_wrinkle']:
        out[key] = [b[key] for b in batch]   # list[bool]

    # GT: None이 섞여 있을 수 있음 → None은 zeros_like로 대체, has_* 로 마스킹
    for task in ['brown', 'red', 'wrinkle']:
        has_key = f'has_{task}'
        tensors = []
        flags   = []
        for b in batch:
            t = b[task]
            flags.append(t is not None)
            tensors.append(t if t is not None else
                           torch.zeros(1, b['mask'].shape[1], b['mask'].shape[2]))
        out[task]   = torch.stack(tensors)          # [B,1,H,W]
        out[has_key] = flags                         # list[bool]

    out['stem'] = [b['stem'] for b in batch]
    return out
