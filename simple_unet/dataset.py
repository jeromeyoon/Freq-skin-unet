import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}


def _collect_pairs(input_root: Path, gt_root: Path):
    pairs = []
    for img_path in sorted(input_root.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        gt_path = gt_root / (img_path.stem + '.png')
        if not gt_path.exists():
            # try same extension
            gt_path = gt_root / img_path.name
        if gt_path.exists():
            pairs.append((img_path, gt_path))
    return pairs


class SimpleDataset(Dataset):
    def __init__(self, pairs, img_size=512, augment=False):
        self.pairs   = pairs
        self.img_size = img_size
        self.augment  = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, gt_path = self.pairs[idx]

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(gt_path).convert('L')

        img  = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                img  = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                img  = TF.vflip(img)
                mask = TF.vflip(mask)
            k = random.randint(0, 3)
            if k:
                img  = TF.rotate(img,  90 * k)
                mask = TF.rotate(mask, 90 * k)

        img_t  = TF.to_tensor(img)                              # [3,H,W] float 0-1
        img_t  = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        mask_t = torch.from_numpy(np.array(mask)).float() / 255.0
        mask_t = (mask_t > 0.5).float().unsqueeze(0)            # [1,H,W] binary

        return img_t, mask_t


def build_datasets(input_root, gt_root, img_size=512, val_ratio=0.1, seed=42):
    pairs = _collect_pairs(Path(input_root), Path(gt_root))
    assert len(pairs) > 0, f"No matched image-mask pairs found in {input_root} / {gt_root}"

    rng = random.Random(seed)
    rng.shuffle(pairs)

    n_val   = max(1, int(len(pairs) * val_ratio))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]

    train_ds = SimpleDataset(train_p, img_size=img_size, augment=True)
    val_ds   = SimpleDataset(val_p,   img_size=img_size, augment=False)

    print(f"Dataset: {len(train_p)} train / {len(val_p)} val  (total {len(pairs)} pairs)")
    return train_ds, val_ds
