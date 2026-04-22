"""
skin_train_v5.py
================
SkinAnalyzerV2 training script V5

Applied speed-focused changes from v4:
1. DataLoader prefetch_factor enabled
2. cudnn.benchmark enabled for fixed-size inputs
3. optimizer.zero_grad(set_to_none=True)
4. consistency clean pass uses torch.inference_mode()
5. validation uses non_blocking transfer + AMP + inference_mode
6. paired illumination augmentation partially vectorized
7. consistency target computed every N steps instead of every step
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import skin_train_v2 as base
from ambient_aug import _make_directional_gradient, _make_vignette_torch


CFG = copy.deepcopy(base.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v5',
    preview_dir='./previews_v5',
    use_weighted_sampler=False,
    prefetch_factor=4,
    consistency_every_n_steps=2,
)


def _skin_result_to_fp32(result):
    return result._replace(
        brown_mask=result.brown_mask.float(),
        brown_score=result.brown_score.float(),
        red_mask=result.red_mask.float(),
        red_score=result.red_score.float(),
        wrinkle_mask=result.wrinkle_mask.float(),
        wrinkle_score=result.wrinkle_score.float(),
    )


def apply_paired_batch_illumination_aug_fast(
    rgb_cross_batch: torch.Tensor,
    rgb_parallel_batch: torch.Tensor,
    *,
    intensity_range: tuple = (0.6, 1.4),
    color_temp_range: tuple = (0.7, 1.3),
    tint_range: tuple = (0.8, 1.2),
    vignette_prob: float = 0.5,
    vignette_strength: float = 0.4,
    gradient_prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the same illumination augmentation to cross/parallel images
    from the same sample. Scalar color transforms are vectorized across
    the batch, while spatial masks remain per-sample.
    """
    assert rgb_cross_batch.shape == rgb_parallel_batch.shape, (
        "rgb_cross_batch and rgb_parallel_batch must have the same shape"
    )

    b, _, h, w = rgb_cross_batch.shape
    device = rgb_cross_batch.device

    intensity = torch.empty(b, 1, 1, 1, device=device).uniform_(*intensity_range)
    ch_scales = torch.empty(b, 3, 1, 1, device=device).uniform_(*color_temp_range)
    tint = torch.ones(b, 3, 1, 1, device=device)
    tint[:, 0] = torch.empty(b, 1, 1, device=device).uniform_(*tint_range)
    tint[:, 2] = torch.empty(b, 1, 1, device=device).uniform_(*tint_range)

    cross_aug = rgb_cross_batch * intensity * ch_scales * tint
    parallel_aug = rgb_parallel_batch * intensity * ch_scales * tint

    spatial = torch.ones(b, 1, h, w, device=device)
    r = torch.rand(b, device=device)
    use_vignette = r < vignette_prob
    use_gradient = (~use_vignette) & (r < vignette_prob + gradient_prob)

    vignette_indices = torch.where(use_vignette)[0]
    for idx in vignette_indices.tolist():
        strength = torch.empty(1, device=device).uniform_(0.1, vignette_strength).item()
        spatial[idx, 0] = _make_vignette_torch(h, w, strength, device)

    gradient_indices = torch.where(use_gradient)[0]
    for idx in gradient_indices.tolist():
        spatial[idx, 0] = _make_directional_gradient(h, w, device)

    cross_aug = (cross_aug * spatial).clamp(0.0, 1.0)
    parallel_aug = (parallel_aug * spatial).clamp(0.0, 1.0)
    return cross_aug, parallel_aug


@torch.inference_mode()
def validate_fast(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon']}
    dice_sum = dict(brown=0.0, red=0.0, wrinkle=0.0)
    dice_cnt = dict(brown=0, red=0, wrinkle=0)

    amp_enabled = device.type == 'cuda'
    pbar = tqdm(loader, desc="  Val  ", unit='batch', leave=False, dynamic_ncols=True)

    for step, batch in enumerate(pbar, 1):
        rgb_cross = batch['rgb_cross'].to(device, non_blocking=True)
        rgb_parallel = batch['rgb_parallel'].to(device, non_blocking=True)
        brown_gt = batch['brown'].to(device, non_blocking=True)
        red_gt = batch['red'].to(device, non_blocking=True)
        wrinkle_gt = batch['wrinkle'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            result = model(rgb_cross, rgb_parallel, mask)

        result_for_loss = _skin_result_to_fp32(result)
        loss, detail = criterion(
            result_for_loss,
            brown_gt.float(),
            red_gt.float(),
            wrinkle_gt.float(),
            rgb_cross=rgb_cross.float(),
            face_mask=mask.float(),
            has_brown=batch['has_brown'],
            has_red=batch['has_red'],
            has_wrinkle=batch['has_wrinkle'],
            model=None,
        )

        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

        brown_sum, brown_cnt = base._dice_sum_and_count(
            result_for_loss.brown_mask,
            brown_gt.float(),
            mask.float(),
            batch['has_brown'],
            threshold=CFG.get('dice_threshold', 0.5),
        )
        red_sum, red_cnt = base._dice_sum_and_count(
            result_for_loss.red_mask,
            red_gt.float(),
            mask.float(),
            batch['has_red'],
            threshold=CFG.get('dice_threshold', 0.5),
        )
        wrinkle_sum, wrinkle_cnt = base._dice_sum_and_count(
            result_for_loss.wrinkle_mask,
            wrinkle_gt.float(),
            mask.float(),
            batch['has_wrinkle'],
            threshold=CFG.get('dice_threshold', 0.5),
        )

        dice_sum['brown'] += brown_sum
        dice_sum['red'] += red_sum
        dice_sum['wrinkle'] += wrinkle_sum
        dice_cnt['brown'] += brown_cnt
        dice_cnt['red'] += red_cnt
        dice_cnt['wrinkle'] += wrinkle_cnt

        pbar.set_postfix(loss=f"{total_loss / step:.4f}")

    pbar.close()
    n = max(len(loader), 1)
    out = {'total': total_loss / n, **{k: v / n for k, v in log.items()}}
    out['brown_dice'] = dice_sum['brown'] / max(dice_cnt['brown'], 1)
    out['red_dice'] = dice_sum['red'] / max(dice_cnt['red'], 1)
    out['wrinkle_dice'] = dice_sum['wrinkle'] / max(dice_cnt['wrinkle'], 1)
    out['score'] = base._weighted_best_score(out, CFG)
    return out


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon', 'consist', 'freq_reg']}

    warmup = cfg.get('aug_warmup_epochs', 0)
    aug_scale = min(1.0, epoch / max(warmup, 1)) if warmup > 0 else 1.0

    def _scale_range(rng, scale):
        mid = (rng[0] + rng[1]) / 2.0
        half = (rng[1] - rng[0]) / 2.0 * scale
        return (mid - half, mid + half)

    aug_kwargs = dict(
        intensity_range=_scale_range(cfg['intensity_range'], aug_scale),
        color_temp_range=_scale_range(cfg['color_temp_range'], aug_scale),
        tint_range=_scale_range(cfg['tint_range'], aug_scale),
        vignette_prob=cfg['vignette_prob'] * aug_scale,
        gradient_prob=cfg['gradient_prob'] * aug_scale,
    )

    amp_enabled = device.type == 'cuda'
    consistency_every = max(int(cfg.get('consistency_every_n_steps', 1)), 1)

    pbar = tqdm(
        loader,
        desc=f"Train {epoch:03d}/{total_epochs}",
        unit='batch',
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar, 1):
        rgb_cross = batch['rgb_cross'].to(device, non_blocking=True)
        rgb_parallel = batch['rgb_parallel'].to(device, non_blocking=True)
        brown_gt = batch['brown'].to(device, non_blocking=True)
        red_gt = batch['red'].to(device, non_blocking=True)
        wrinkle_gt = batch['wrinkle'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        has_brown = batch['has_brown']
        has_red = batch['has_red']
        has_wrinkle = batch['has_wrinkle']

        rgb_cross_aug, rgb_parallel_aug = apply_paired_batch_illumination_aug_fast(
            rgb_cross,
            rgb_parallel,
            **aug_kwargs,
        )

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            result = model(rgb_cross_aug, rgb_parallel_aug, mask)

        result_clean = None
        if criterion.w_consist > 0 and step % consistency_every == 0:
            was_training = model.training
            model.eval()
            with torch.inference_mode():
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    result_clean = model(rgb_cross, rgb_parallel, mask)
            if was_training:
                model.train()

        result_for_loss = _skin_result_to_fp32(result)
        result_clean_for_loss = _skin_result_to_fp32(result_clean) if result_clean is not None else None

        loss, detail = criterion(
            result_for_loss,
            brown_gt.float(),
            red_gt.float(),
            wrinkle_gt.float(),
            rgb_cross=rgb_cross.float(),
            face_mask=mask.float(),
            has_brown=has_brown,
            has_red=has_red,
            has_wrinkle=has_wrinkle,
            result_aug=result_clean_for_loss,
            model=None,
        )

        if criterion.w_freq_reg > 0:
            l_freq = base._v2_freq_reg(model).float()
            loss = loss + criterion.w_freq_reg * l_freq
            detail['freq_reg'] = l_freq.item()

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

        pbar.set_postfix(
            loss=f"{total_loss / step:.4f}",
            brown=f"{log['brown'] / step:.4f}",
            red=f"{log['red'] / step:.4f}",
        )

    pbar.close()
    n = max(len(loader), 1)
    return {'total': total_loss / n, **{k: v / n for k, v in log.items()}}


def main():
    parser = argparse.ArgumentParser(description='SkinAnalyzerV2 training V5 (speed-focused)')
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    args = parser.parse_args()

    seed = CFG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seed: {seed}")
    print("Model   : SkinAnalyzerV2")
    print("Version : V5 speed-focused")
    print("  - paired illumination augmentation (partially vectorized)")
    print("  - consistency target computed every N steps")
    print("  - validation fast path (AMP + inference_mode + non_blocking)")
    print("  - DataLoader prefetch_factor enabled")
    print(f"  - weighted sampler default: {CFG['use_weighted_sampler']}")
    print(f"  - consistency_every_n_steps: {CFG['consistency_every_n_steps']}")
    print("Loss    : skin_loss_smooth (same objective as V4)")

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = base.build_analyzer_v2(
        base_ch=CFG['base_ch'],
        low_r=CFG['low_r'],
        high_r=CFG['high_r'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    full_ds = base.SkinDataset(CFG['patch_dir'], CFG['img_size'], augment=True)
    val_size = max(1, int(len(full_ds) * CFG['val_ratio']))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    val_ds.dataset = copy.copy(full_ds)
    val_ds.dataset.augment = False

    prefetch_factor = CFG['prefetch_factor'] if CFG['num_workers'] > 0 else None

    train_loader = DataLoader(
        train_ds,
        CFG['batch_size'],
        shuffle=not CFG.get('use_weighted_sampler', False),
        num_workers=CFG['num_workers'],
        pin_memory=True,
        persistent_workers=CFG['num_workers'] > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=base.skin_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=True,
        persistent_workers=CFG['num_workers'] > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=base.skin_collate_fn,
    )
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    criterion = base.SkinAnalyzerLoss(
        w_brown=CFG['w_brown'],
        w_red=CFG['w_red'],
        w_wrinkle=CFG['w_wrinkle'],
        w_recon=CFG['w_recon'],
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CFG['lr'],
        weight_decay=CFG['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    start_epoch = 1
    best_score = float('-inf')

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_score = ckpt.get('best_score', float('-inf'))
        print(f"Resume: epoch {ckpt['epoch']} -> start at {start_epoch}")

    print("\nStart training...")

    base.CFG = CFG

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        weights = base.get_loss_weights(
            epoch,
            CFG['epochs'],
            w_brown=CFG['w_brown'],
            w_red=CFG['w_red'],
            w_wrinkle=CFG['w_wrinkle'],
            w_recon=CFG['w_recon'],
        )
        for k, v in weights.items():
            setattr(criterion, k, v)

        train_log = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, CFG,
            epoch, CFG['epochs'],
        )
        val_log = validate_fast(model, val_loader, criterion, device)
        scheduler.step()

        is_best = val_log['score'] > best_score
        if is_best:
            best_score = val_log['score']

        last_ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_score': best_score,
            'val_loss': val_log,
            'cfg': CFG,
        }
        torch.save(last_ckpt, ckpt_dir / 'last_checkpoint.pth')
        if is_best:
            torch.save(last_ckpt, ckpt_dir / 'best_skin_analyzer.pth')

        tqdm.write(
            f"[{epoch:03d}/{CFG['epochs']}]"
            f"  trn={train_log['total']:.4f}"
            f"  val={val_log['total']:.4f}"
            f"  B_dice={val_log['brown_dice']:.3f}"
            f"  R_dice={val_log['red_dice']:.3f}"
            f"  score={val_log['score']:.3f}"
            + (' *' if is_best else '')
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
