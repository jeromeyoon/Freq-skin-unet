"""
skin_dataset.py
===============
SkinDataset : 듀얼 편광 피부 분석 데이터셋

디렉토리 구조
-------------
patch_dir/
  ├── rgb_cross/    *.png   교차 편광 RGB (발색단용)
  ├── rgb_parallel/ *.png   평행 편광 RGB (주름용)
  ├── brown/        *.png   melanin GT       (grayscale)
  ├── red/          *.png   hemoglobin GT    (grayscale)
  ├── wrinkle/      *.png   wrinkle GT       (grayscale)
  └── mask/         *.png   피부 영역 마스크  (optional)

주의: rgb_cross 와 rgb_parallel은 동일한 stem 이름으로 쌍을 이뤄야 함.
"""

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

        # 교차 편광 이미지 목록을 기준으로 인덱싱
        self.stems = sorted(
            p.stem for p in (self.patch_dir / 'rgb_cross').glob('*.png')
        )
        assert len(self.stems) > 0, \
            f"이미지를 찾을 수 없습니다: {patch_dir}/rgb_cross"

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
        """동일한 geometric augment를 모든 텐서에 적용"""
        if random.random() > 0.5:
            tensors = tuple(TF.hflip(t) for t in tensors)
        if random.random() > 0.5:
            tensors = tuple(TF.vflip(t) for t in tensors)
        return tensors

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]

        rgb_cross    = self._load_rgb(self.patch_dir / 'rgb_cross'    / f'{stem}.png')
        rgb_parallel = self._load_rgb(self.patch_dir / 'rgb_parallel' / f'{stem}.png')
        brown        = self._load_gray(self.patch_dir / 'brown'       / f'{stem}.png')
        red          = self._load_gray(self.patch_dir / 'red'         / f'{stem}.png')
        wrinkle      = self._load_gray(self.patch_dir / 'wrinkle'     / f'{stem}.png')

        mask_path = self.patch_dir / 'mask' / f'{stem}.png'
        mask = self._load_gray(mask_path) if mask_path.exists() \
               else torch.ones(1, self.img_size, self.img_size)

        if self.augment:
            rgb_cross, rgb_parallel, brown, red, wrinkle, mask = self._apply_augment(
                rgb_cross, rgb_parallel, brown, red, wrinkle, mask)

        return {
            'rgb_cross'   : rgb_cross,
            'rgb_parallel': rgb_parallel,
            'brown'       : brown,
            'red'         : red,
            'wrinkle'     : wrinkle,
            'mask'        : mask,
            'stem'        : stem,
        }
