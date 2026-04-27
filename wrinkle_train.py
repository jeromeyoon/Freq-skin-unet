"""
wrinkle_train.py
================
Wrinkle-only training entrypoint with simplified loss options.

Design goal:
- Keep the same model/dataset/augmentation environment as V9 where possible.
- Remove multi-task loss coupling and train wrinkle segmentation only.
- Start from simple, reproducible losses (Dice or BCE+Dice).
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import skin_train_v9 as v9
from skin_dataset import skin_collate_fn
from skin_net_v2 import build_analyzer_v2


def _filter_wrinkle_only(dataset: v9.WrinklePositiveSkinDataset) -> None:
    """Keep only stems with wrinkle GT enabled after wrinkle_pos gate."""
    kept = []
    dropped = 0
    for stem in dataset.stems:
        if dataset.manifest[stem].get('has_wrinkle', False):
            kept.append(stem)
        else:
            dropped += 1
    dataset.stems = kept
    print(f"[wrinkle_only] filtered dataset: kept={len(kept)}, dropped_non_wrinkle={dropped}")


class SimpleWrinkleLoss(torch.nn.Module):
    """Simple wrinkle segmentation loss: dice or bce+dice."""

    def __init__(self, mode: str = 'bce_dice', bce_weight: float = 0.5):
        super().__init__()
        if mode not in {'dice', 'bce_dice'}:
            raise ValueError(f'unsupported loss mode: {mode}')
        self.mode = mode
        self.bce_weight = float(bce_weight)

    @staticmethod
    def _masked_dice_loss(logits, gt, face_mask, valid, eps=1e-6):
        prob = torch.sigmoid(logits) * face_mask
        gt = gt * face_mask
        prob = prob[valid].flatten(1)
        gt = gt[valid].flatten(1)

        inter = (prob * gt).sum(dim=1)
        denom = prob.sum(dim=1) + gt.sum(dim=1)
        dice = (2.0 * inter + eps) / (denom + eps)
        return 1.0 - dice.mean()

    def forward(self, logits, gt, face_mask, has_wrinkle):
        valid = torch.tensor(has_wrinkle, dtype=torch.bool, device=logits.device)
        if not valid.any():
            return logits.new_tensor(0.0)

        dice_loss = self._masked_dice_loss(logits, gt, face_mask, valid)
        if self.mode == 'dice':
            return dice_loss

        # BCE is computed only on valid samples and masked by face region.
        bce_map = F.binary_cross_entropy_with_logits(logits, gt, reduction='none')
        bce_map = bce_map * face_mask
        bce = bce_map[valid].sum() / face_mask[valid].sum().clamp(min=1.0)
        return self.bce_weight * bce + (1.0 - self.bce_weight) * dice_loss


@torch.no_grad()
def _dice_metric(logits, gt, face_mask, has_wrinkle, threshold=0.5, eps=1e-6):
    valid = torch.tensor(has_wrinkle, dtype=torch.bool, device=logits.device)
    if not valid.any():
        return 0.0, 0
    pred = (torch.sigmoid(logits) >= threshold).float() * face_mask
    gt = (gt >= 0.5).float() * face_mask

    pred = pred[valid].flatten(1)
    gt = gt[valid].flatten(1)
    inter = (pred * gt).sum(dim=1)
    denom = pred.sum(dim=1) + gt.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.sum().item()), int(valid.sum().item())


def _build_aug_kwargs(cfg, epoch):
    warmup = cfg.get('aug_warmup_epochs', 0)
    aug_scale = min(1.0, epoch / max(warmup, 1)) if warmup > 0 else 1.0

    def _scale_range(rng, scale):
        mid = (rng[0] + rng[1]) / 2.0
        half = (rng[1] - rng[0]) / 2.0 * scale
        return (mid - half, mid + half)

    kwargs = dict(
        intensity_range=_scale_range(cfg['intensity_range'], aug_scale),
        color_temp_range=_scale_range(cfg['color_temp_range'], aug_scale),
        ambient_fog_alpha_range=_scale_range(
            cfg.get('ambient_fog_alpha_range', (0.0, 0.05)), aug_scale
        ),
        ceiling_gradient_prob=cfg.get('ceiling_gradient_prob', 0.5) * aug_scale,
    )

    # For simple OOD illumination setup, keep only global brightness + color temp jitter.
    if cfg.get('illumination_simple_mode', True):
        kwargs['ambient_fog_alpha_range'] = (0.0, 0.0)
        kwargs['ceiling_gradient_prob'] = 0.0
    return kwargs


def train_one_epoch_wrinkle(model, loader, criterion, optimizer, scaler, device, cfg, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    processed = 0
    skipped = 0

    aug_kwargs = _build_aug_kwargs(cfg, epoch)
    amp_enabled = device.type == 'cuda'

    pbar = tqdm(loader, desc=f"Train {epoch:03d}/{total_epochs}", unit='batch', leave=False)
    for batch in pbar:
        rgb_cross = batch['rgb_cross'].to(device, non_blocking=True)
        rgb_parallel = batch['rgb_parallel'].to(device, non_blocking=True)
        wrinkle_gt = batch['wrinkle'].to(device, non_blocking=True)
        face_mask = batch['mask'].to(device, non_blocking=True)
        has_wrinkle = batch['has_wrinkle']

        if not any(has_wrinkle):
            skipped += 1
            continue

        rgb_cross_aug, rgb_parallel_aug = v9.apply_paired_batch_illumination_aug_fast(
            rgb_cross, rgb_parallel, **aug_kwargs
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            result = model(rgb_cross_aug, rgb_parallel_aug, face_mask)
            loss = criterion(result.wrinkle_mask.float(), wrinkle_gt.float(), face_mask.float(), has_wrinkle)

        if not loss.requires_grad:
            skipped += 1
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        processed += 1
        pbar.set_postfix(loss=f"{(total_loss / max(processed, 1)):.4f}", skip=skipped)

    return {
        'total': total_loss / max(processed, 1),
        'processed_steps': processed,
        'skipped_steps': skipped,
    }


@torch.no_grad()
def validate_wrinkle(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    processed = 0
    dice_sum = 0.0
    dice_count = 0

    for batch in loader:
        rgb_cross = batch['rgb_cross'].to(device, non_blocking=True)
        rgb_parallel = batch['rgb_parallel'].to(device, non_blocking=True)
        wrinkle_gt = batch['wrinkle'].to(device, non_blocking=True)
        face_mask = batch['mask'].to(device, non_blocking=True)
        has_wrinkle = batch['has_wrinkle']

        if not any(has_wrinkle):
            continue

        result = model(rgb_cross, rgb_parallel, face_mask)
        loss = criterion(result.wrinkle_mask.float(), wrinkle_gt.float(), face_mask.float(), has_wrinkle)
        total_loss += float(loss.item())
        processed += 1

        dsum, dcnt = _dice_metric(result.wrinkle_mask.float(), wrinkle_gt.float(), face_mask.float(), has_wrinkle)
        dice_sum += dsum
        dice_count += dcnt

    return {
        'total': total_loss / max(processed, 1),
        'wrinkle_dice': dice_sum / max(dice_count, 1),
    }


@torch.no_grad()
def save_validation_previews(model, loader, device, out_dir: Path, max_items: int = 10):
    """
    Save validation preview images:
      input(rgb_cross), gt(wrinkle), face_mask, pred(wrinkle prob)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def _gray_to_rgb_uint8(t: torch.Tensor) -> np.ndarray:
        arr = (t.clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
        return np.stack([arr, arr, arr], axis=-1)

    saved = 0
    for batch in loader:
        if saved >= max_items:
            break

        rgb_cross = batch['rgb_cross'].to(device, non_blocking=True)
        rgb_parallel = batch['rgb_parallel'].to(device, non_blocking=True)
        wrinkle_gt = batch['wrinkle'].to(device, non_blocking=True)
        face_mask = batch['mask'].to(device, non_blocking=True)

        result = model(rgb_cross, rgb_parallel, face_mask)
        pred_prob = torch.sigmoid(result.wrinkle_mask)

        bsz = rgb_cross.shape[0]
        for i in range(bsz):
            if saved >= max_items:
                break

            inp = (rgb_cross[i].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
            gt = _gray_to_rgb_uint8(wrinkle_gt[i, 0])
            msk = _gray_to_rgb_uint8(face_mask[i, 0])
            prd = _gray_to_rgb_uint8(pred_prob[i, 0])

            grid = np.concatenate([inp, gt, msk, prd], axis=1)
            Image.fromarray(grid).save(out_dir / f'val_{saved:02d}.png')
            saved += 1

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description='Wrinkle-only training (simple loss)')
    parser.add_argument('--patch_dir', type=str, default=v9.CFG['patch_dir'])
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_wrinkle')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wrinkle_min_pos_ratio', type=float, default=v9.CFG['wrinkle_min_pos_ratio'])
    parser.add_argument('--epochs', type=int, default=v9.CFG['epochs'])
    parser.add_argument('--batch_size', type=int, default=v9.CFG['batch_size'])
    parser.add_argument('--loss_mode', type=str, choices=['dice', 'bce_dice'], default='bce_dice')
    parser.add_argument('--bce_weight', type=float, default=0.5)
    parser.add_argument('--val_preview_count', type=int, default=10)
    args = parser.parse_args()

    cfg = copy.deepcopy(v9.CFG)
    cfg.update(
        patch_dir=args.patch_dir,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        wrinkle_min_pos_ratio=args.wrinkle_min_pos_ratio,
        illumination_simple_mode=True,
    )

    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Seed: {seed}")
    print(f"Loss mode: {args.loss_mode} (bce_weight={args.bce_weight:.2f})")

    ckpt_dir = Path(cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_analyzer_v2(base_ch=cfg['base_ch'], low_r=cfg['low_r'], high_r=cfg['high_r']).to(device)

    full_ds = v9.WrinklePositiveSkinDataset(
        cfg['patch_dir'],
        cfg['img_size'],
        augment=True,
        wrinkle_min_pos_ratio=cfg['wrinkle_min_pos_ratio'],
        compute_missing_pos_ratio=cfg.get('wrinkle_compute_missing_pos_ratio', True),
    )
    _filter_wrinkle_only(full_ds)
    if len(full_ds) <= 1:
        raise RuntimeError('Need at least 2 wrinkle-positive samples for train/val split.')

    val_size = max(1, int(len(full_ds) * cfg['val_ratio']))
    if val_size >= len(full_ds):
        val_size = len(full_ds) - 1
    train_size = len(full_ds) - val_size

    train_ds, val_ds = random_split(
        full_ds,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg['seed']),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=device.type == 'cuda',
        prefetch_factor=cfg.get('prefetch_factor', 2) if cfg['num_workers'] > 0 else None,
        collate_fn=skin_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg['num_workers'],
        pin_memory=device.type == 'cuda',
        prefetch_factor=cfg.get('prefetch_factor', 2) if cfg['num_workers'] > 0 else None,
        collate_fn=skin_collate_fn,
    )

    criterion = SimpleWrinkleLoss(mode=args.loss_mode, bce_weight=args.bce_weight).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    start_epoch = 1
    best_wrinkle = float('-inf')

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_wrinkle = ckpt.get('best_wrinkle_dice', float('-inf'))
        print(f"Resume: epoch {ckpt['epoch']} -> start at {start_epoch}")

    print('\nStart wrinkle-only training...')
    for epoch in range(start_epoch, cfg['epochs'] + 1):
        train_log = train_one_epoch_wrinkle(
            model, train_loader, criterion, optimizer, scaler, device, cfg, epoch, cfg['epochs']
        )
        val_log = validate_wrinkle(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_log['wrinkle_dice'] > best_wrinkle
        if is_best:
            best_wrinkle = val_log['wrinkle_dice']

        ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_wrinkle_dice': best_wrinkle,
            'val_loss': val_log,
            'train_loss': train_log,
            'cfg': cfg,
            'loss_mode': args.loss_mode,
            'bce_weight': args.bce_weight,
        }
        torch.save(ckpt, ckpt_dir / 'last_checkpoint.pth')
        if is_best:
            torch.save(ckpt, ckpt_dir / 'best_wrinkle_analyzer.pth')

        print(
            f"[{epoch:03d}/{cfg['epochs']}]"
            f" trn={train_log['total']:.4f}"
            f" val={val_log['total']:.4f}"
            f" wrinkle_dice={val_log['wrinkle_dice']:.3f}"
            f" skipped={train_log['skipped_steps']}"
            + (' *' if is_best else '')
        )

        preview_dir = ckpt_dir / 'val_previews' / f'epoch_{epoch:03d}'
        n_saved = save_validation_previews(
            model,
            val_loader,
            device,
            preview_dir,
            max_items=max(int(args.val_preview_count), 0),
        )
        print(f"  saved validation previews: {n_saved} -> {preview_dir}")

    print('\nWrinkle-only training complete!')


if __name__ == '__main__':
    main()
