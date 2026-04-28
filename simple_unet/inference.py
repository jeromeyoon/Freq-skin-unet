"""
Simple UNet inference script.

Usage (GT 없음 - 예측만):
    python inference.py --checkpoint checkpoints/brown/best_brown.pth \
                        --input_root /path/to/rgb \
                        --save_dir /path/to/results

Usage (GT 있음 - 예측 + loss/metric 계산):
    python inference.py --checkpoint checkpoints/brown/best_brown.pth \
                        --input_root /path/to/rgb \
                        --gt_root /path/to/brown_masks \
                        --save_dir /path/to/results
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from tqdm import tqdm

from model import EfficientNetUNet
from loss  import SegLoss

# ───────────────────────── constants ─────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_EXTS      = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
OVERLAY_COLOR = (255, 64, 64)   # 예측 마스크 오버레이 색 (R,G,B)
OVERLAY_ALPHA = 0.45


# ───────────────────────── dataset ───────────────────────────

class InferDataset(Dataset):
    def __init__(self, input_root: Path, gt_root: Path | None, img_size: int):
        self.img_size = img_size
        # items[i] = (img_path, gt_path or None)
        self.items: list[tuple[Path, Path | None]] = []

        for p in sorted(input_root.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            gt = None
            if gt_root is not None:
                for ext in ['.png', p.suffix]:
                    c = gt_root / (p.stem + ext)
                    if c.exists():
                        gt = c
                        break
            self.items.append((p, gt))

        assert self.items, f"No images found in {input_root}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, gt_path = self.items[idx]

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        img_r = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_t = TF.normalize(TF.to_tensor(img_r), IMAGENET_MEAN, IMAGENET_STD)

        if gt_path is not None:
            mask = Image.open(gt_path).convert('L')
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_t = (torch.from_numpy(np.array(mask)).float() / 255.0 > 0.5).float().unsqueeze(0)
            has_gt = True
        else:
            mask_t = torch.zeros(1, self.img_size, self.img_size)
            has_gt = False

        return {
            'img':    img_t,
            'mask':   mask_t,
            'has_gt': has_gt,
            'orig_h': orig_h,
            'orig_w': orig_w,
            'idx':    idx,          # 원본 경로 조회용
        }


# ───────────────────────── metrics ───────────────────────────

def compute_metrics(pred_bin: torch.Tensor, gt: torch.Tensor) -> tuple[float, float]:
    """pred_bin, gt: [1,1,H,W] binary float"""
    p = pred_bin.view(-1)
    g = gt.view(-1)
    inter = (p * g).sum()
    dice  = ((2 * inter + 1) / (p.sum() + g.sum() + 1)).item()
    iou   = ((inter + 1) / (p.sum() + g.sum() - inter + 1)).item()
    return dice, iou


# ───────────────────────── visualisation ─────────────────────

def save_pred_mask(pred_np: np.ndarray, out_path: Path):
    Image.fromarray((pred_np * 255).astype(np.uint8)).save(out_path)


def save_overlay(orig_img: Image.Image, pred_np: np.ndarray, out_path: Path):
    """원본 이미지 위에 예측 마스크를 반투명 오버레이로 저장."""
    mask_img    = Image.fromarray((pred_np * 255).astype(np.uint8), 'L')
    mask_img    = mask_img.resize(orig_img.size, Image.NEAREST)
    color_layer = Image.new('RGBA', orig_img.size,
                            OVERLAY_COLOR + (int(255 * OVERLAY_ALPHA),))
    overlay     = Image.new('RGBA', orig_img.size, (0, 0, 0, 0))
    overlay.paste(color_layer, mask=mask_img)
    result = Image.alpha_composite(orig_img.convert('RGBA'), overlay).convert('RGB')
    result.save(out_path)


def save_compare(orig_img: Image.Image,
                 gt_pil: Image.Image | None,
                 pred_np: np.ndarray,
                 out_path: Path,
                 dice: float | None = None):
    """
    GT 있음: [input | GT | pred]
    GT 없음: [input | pred]
    """
    W, H   = orig_img.size
    cols   = 3 if gt_pil is not None else 2
    canvas = Image.new('RGB', (W * cols, H), (30, 30, 30))

    # 패널 1: 원본
    canvas.paste(orig_img, (0, 0))

    # 패널 2: GT (있을 때)
    if gt_pil is not None:
        gt_arr = np.array(gt_pil.resize((W, H), Image.NEAREST))
        canvas.paste(Image.fromarray(np.stack([gt_arr] * 3, axis=-1)), (W, 0))

    # 패널 마지막: 예측
    pred_arr = (pred_np * 255).astype(np.uint8)
    pred_img = Image.fromarray(np.stack([pred_arr] * 3, axis=-1)).resize((W, H), Image.NEAREST)
    canvas.paste(pred_img, (W * (cols - 1), 0))

    if dice is not None:
        draw = ImageDraw.Draw(canvas)
        draw.text((W * (cols - 1) + 6, 6), f"Dice {dice:.3f}", fill=(255, 220, 0))

    canvas.save(out_path)


# ───────────────────────── main ──────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',  required=True,  help='checkpoint .pth 경로')
    p.add_argument('--input_root',  required=True,  help='RGB 이미지 폴더')
    p.add_argument('--gt_root',     default=None,   help='GT 마스크 폴더 (선택)')
    p.add_argument('--save_dir',    default='./inference_results')
    p.add_argument('--img_size',    type=int,   default=512)
    p.add_argument('--batch_size',  type=int,   default=8)
    p.add_argument('--threshold',   type=float, default=0.5)
    p.add_argument('--num_workers', type=int,   default=4)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── 출력 폴더 ──
    save_dir = Path(args.save_dir)
    pred_dir = save_dir / 'pred'
    over_dir = save_dir / 'overlay'
    comp_dir = save_dir / 'compare'
    for d in [pred_dir, over_dir, comp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ──
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EfficientNetUNet(pretrained=False).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Checkpoint : {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    criterion = SegLoss()

    # ── 데이터셋 ──
    gt_root = Path(args.gt_root) if args.gt_root else None
    ds      = InferDataset(Path(args.input_root), gt_root, args.img_size)
    loader  = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True,
                         collate_fn=_collate)
    print(f"Images     : {len(ds)}  |  GT: {'yes' if gt_root else 'no'}")

    # ── 집계 ──
    all_dice, all_iou, all_loss = [], [], []
    csv_rows: list[dict] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc='Inference'):
            imgs    = batch['img'].to(device)
            masks   = batch['mask'].to(device)
            preds   = model(imgs)               # [B,1,H,W]

            for i in range(imgs.size(0)):
                item_idx = batch['idx'][i]
                stem     = ds.items[item_idx][0].stem
                gt_path  = ds.items[item_idx][1]
                has_gt   = batch['has_gt'][i]
                orig_h   = batch['orig_h'][i]
                orig_w   = batch['orig_w'][i]

                # 원본 해상도로 복원
                pred_up = F.interpolate(preds[i:i+1], size=(orig_h, orig_w),
                                        mode='bilinear', align_corners=False)
                pred_np = (pred_up[0, 0].cpu().numpy() > args.threshold).astype(np.uint8)

                # 원본 이미지 로드 (overlay/compare용)
                orig_img = Image.open(ds.items[item_idx][0]).convert('RGB')

                # ① 예측 마스크
                save_pred_mask(pred_np, pred_dir / f"{stem}.png")

                # ② 오버레이
                save_overlay(orig_img, pred_np, over_dir / f"{stem}.png")

                # ③ 비교 + metric
                dice_val = None
                if has_gt:
                    pred_bin = (preds[i:i+1] > args.threshold).float()
                    dice_val, iou_val = compute_metrics(pred_bin, masks[i:i+1])
                    loss_val = criterion(preds[i:i+1], masks[i:i+1]).item()
                    all_dice.append(dice_val)
                    all_iou.append(iou_val)
                    all_loss.append(loss_val)
                    csv_rows.append({'stem': stem,
                                     'dice': f'{dice_val:.4f}',
                                     'iou':  f'{iou_val:.4f}',
                                     'loss': f'{loss_val:.4f}'})
                    gt_pil = Image.open(gt_path).convert('L')
                else:
                    gt_pil = None

                save_compare(orig_img, gt_pil, pred_np,
                             comp_dir / f"{stem}.png", dice=dice_val)

    # ── 결과 요약 ──
    print(f"\n{'='*52}")
    print(f"Results → {save_dir}")
    print(f"  pred/     binary masks")
    print(f"  overlay/  prediction overlay on original")
    print(f"  compare/  side-by-side (input | GT | pred)")

    if all_dice:
        print(f"\n── Metrics (n={len(all_dice)}) ──────────────────────")
        print(f"  Dice  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
        print(f"  IoU   {np.mean(all_iou):.4f} ± {np.std(all_iou):.4f}")
        print(f"  Loss  {np.mean(all_loss):.4f}")

        csv_path = save_dir / 'metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['stem', 'dice', 'iou', 'loss'])
            writer.writeheader()
            writer.writerows(csv_rows)
            writer.writerow({'stem': 'MEAN',
                             'dice': f'{np.mean(all_dice):.4f}',
                             'iou':  f'{np.mean(all_iou):.4f}',
                             'loss': f'{np.mean(all_loss):.4f}'})
        print(f"  CSV   {csv_path}")


def _collate(batch: list[dict]) -> dict:
    return {
        'img':    torch.stack([b['img']  for b in batch]),
        'mask':   torch.stack([b['mask'] for b in batch]),
        'has_gt': [b['has_gt'] for b in batch],
        'orig_h': [b['orig_h'] for b in batch],
        'orig_w': [b['orig_w'] for b in batch],
        'idx':    [b['idx']    for b in batch],
    }


if __name__ == '__main__':
    main()
