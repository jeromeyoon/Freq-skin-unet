"""
Simple EfficientNet-UNet for skin segmentation (binary mask).

Usage:
    python train.py --task brown  --input_root /path/rgb  --gt_root /path/brown
    python train.py --task red    --input_root /path/rgb  --gt_root /path/red
    python train.py --task wrinkle --input_root /path/rgb --gt_root /path/wrinkle
"""

import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model   import EfficientNetUNet
from dataset import build_datasets
from loss    import SegLoss


# ───────────────────────── args ──────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Simple UNet skin segmentation')
    p.add_argument('--task',       type=str, required=True, choices=['brown', 'red', 'wrinkle'])
    p.add_argument('--input_root', type=str, required=True, help='Folder of RGB images')
    p.add_argument('--gt_root',    type=str, required=True, help='Folder of binary GT masks')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--img_size',   type=int, default=512)
    p.add_argument('--val_ratio',  type=float, default=0.1)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--save_dir',   type=str, default='./checkpoints')
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--no_pretrained', dest='pretrained', action='store_false')
    return p.parse_args()


# ───────────────────────── metric ────────────────────────

def dice_score(pred, gt, threshold=0.5):
    pred_bin = (pred > threshold).float()
    gt_flat  = gt.view(gt.size(0), -1)
    p_flat   = pred_bin.view(pred_bin.size(0), -1)
    inter    = (p_flat * gt_flat).sum(dim=1)
    return ((2 * inter + 1) / (p_flat.sum(dim=1) + gt_flat.sum(dim=1) + 1)).mean().item()


# ───────────────────────── train / val loops ─────────────

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, masks).item() * imgs.size(0)
        total_dice += dice_score(preds, masks) * imgs.size(0)
    n = len(loader.dataset)
    return total_loss / n, total_dice / n


# ───────────────────────── main ──────────────────────────

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Task: {args.task}  |  Device: {device}")

    save_dir = Path(args.save_dir) / args.task
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── data ──
    train_ds, val_ds = build_datasets(
        args.input_root, args.gt_root,
        img_size=args.img_size, val_ratio=args.val_ratio,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── model ──
    model     = EfficientNetUNet(pretrained=args.pretrained).to(device)
    criterion = SegLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        train_loss             = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice     = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_dice={val_dice:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = save_dir / f"best_{args.task}.pth"
            torch.save({
                'epoch':      epoch,
                'model':      model.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'val_loss':   val_loss,
                'val_dice':   val_dice,
                'args':       vars(args),
            }, ckpt_path)
            print(f"  → saved best checkpoint  (val_loss={val_loss:.4f})")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}  Checkpoint: {save_dir}/best_{args.task}.pth")


if __name__ == '__main__':
    main()
