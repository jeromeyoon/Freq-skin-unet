"""
skin_train_v9.py
================
V8 + task-sampled multi-task training.

Intent
------
Brown / red / wrinkle are still predicted together, but each training step
selects one active task and backpropagates only that task's supervised loss.

Why
---
The tasks have different label density and morphology:

- brown: spot-like and relatively stable
- red: sparse specialist "curr" masks can be scarce / mixed with area labels
- wrinkle: very sparse line structure, easily dominated by black patches

Task-sampled training reduces gradient interference and prevents dense/easier
tasks from dominating sparse wrinkle/red signals.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

import skin_train_v2 as base
import skin_train_v5 as v5
import skin_train_v7 as v7
import skin_train_v8 as v8


CFG = copy.deepcopy(v8.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v9',
    preview_dir='./previews_v9',
    use_weighted_sampler=True,
    sampler_neg_weight=0.02,
    task_sampling_enabled=True,
    # Sparse tasks get higher probability.  Selection is restricted to tasks
    # that are actually available in the current batch.
    task_probs=dict(
        brown=0.20,
        red=0.25,
        wrinkle=0.55,
    ),
)


def _parse_task_probs(text: str) -> dict[str, float]:
    """Parse brown=0.20,red=0.25,wrinkle=0.55."""
    out = {}
    for part in text.split(','):
        if not part.strip():
            continue
        key, value = part.split('=', 1)
        key = key.strip()
        if key not in ('brown', 'red', 'wrinkle'):
            raise ValueError(f"unknown task in --task_probs: {key}")
        out[key] = float(value)
    if not out:
        raise ValueError('--task_probs must not be empty')
    for task in ('brown', 'red', 'wrinkle'):
        out.setdefault(task, 0.0)
    return out


def choose_active_task(batch: dict, cfg: dict) -> str | None:
    """
    Choose one task for this step from tasks with available GT in the batch.

    Restricting to available GT avoids zero-loss steps when, for example,
    wrinkle is selected but the batch has no wrinkle-positive patch.
    """
    if not cfg.get('task_sampling_enabled', True):
        return None

    available = [
        task for task in ('brown', 'red', 'wrinkle')
        if any(batch[f'has_{task}'])
    ]
    if not available:
        return None

    probs = cfg.get('task_probs', {})
    weights = [max(float(probs.get(task, 0.0)), 0.0) for task in available]
    if sum(weights) <= 0:
        weights = [1.0] * len(available)

    return random.choices(available, weights=weights, k=1)[0]


class TaskAwareSkinAnalyzerLossV9(v7.TaskAwareSkinAnalyzerLossV7):
    """
    V7 task-aware objective with optional active_task gating.

    active_task=None keeps legacy all-task behavior for validation.
    active_task in {"brown", "red", "wrinkle"} backpropagates only that task.
    """

    def forward(
        self,
        result,
        brown_gt,
        red_gt,
        wrinkle_gt,
        rgb_cross,
        face_mask,
        has_brown,
        has_red,
        has_wrinkle,
        result_aug=None,
        model=None,
        active_task: str | None = None,
    ):
        detail = {'active_task': active_task or 'all'}

        l_brown = self._focal_dice_spot(
            result.brown_mask, brown_gt, face_mask, has_brown, alpha=0.5
        )

        l_red_base = self._generalized_dice_loss(
            result.red_mask, red_gt, face_mask, has_red
        )
        l_red_area = self._area_ratio_penalty(
            result.red_mask, red_gt, face_mask, has_red
        )
        l_red_outside = self._outside_gt_penalty(
            result.red_mask, red_gt, face_mask, has_red
        )
        l_red = (
            l_red_base
            + self.red_area_weight * l_red_area
            + self.red_outside_weight * l_red_outside
        )

        l_wrinkle_base = self._wrinkle_weighted_dice(
            result.wrinkle_mask, wrinkle_gt, face_mask, has_wrinkle
        )
        l_wrinkle_distance = self._wrinkle_distance_penalty(
            result.wrinkle_mask, wrinkle_gt, face_mask, has_wrinkle
        )
        l_wrinkle = (
            l_wrinkle_base
            + self.wrinkle_distance_weight * l_wrinkle_distance
        )

        detail.update(
            brown=l_brown.item(),
            red=l_red.item(),
            wrinkle=l_wrinkle.item(),
            red_base=l_red_base.item(),
            red_area=l_red_area.item(),
            red_outside=l_red_outside.item(),
            wrinkle_base=l_wrinkle_base.item(),
            wrinkle_distance=l_wrinkle_distance.item(),
        )

        loss = torch.tensor(0.0, device=rgb_cross.device)

        def use(task: str) -> bool:
            return active_task is None or active_task == task

        if use('brown') and any(has_brown):
            loss = loss + self.w_brown * l_brown
        if use('red') and any(has_red):
            loss = loss + self.w_red * l_red
        if use('wrinkle') and any(has_wrinkle):
            loss = loss + self.w_wrinkle * l_wrinkle

        # Recon is brown/red chemistry-driven, so skip it on wrinkle-only steps.
        if self.w_recon > 0 and (active_task is None or active_task in ('brown', 'red')):
            l_recon = self._beer_lambert_recon(
                result.brown_mask, result.red_mask, rgb_cross, face_mask, has_brown, has_red
            )
            detail['recon'] = l_recon.item()
            loss = loss + self.w_recon * l_recon
        else:
            detail['recon'] = 0.0

        if self.w_consist > 0 and result_aug is not None:
            if active_task is None:
                l_consist = (
                    self._partial_l1(result.brown_mask, result_aug.brown_mask.detach(), face_mask, has_brown)
                    + self._partial_l1(result.red_mask, result_aug.red_mask.detach(), face_mask, has_red)
                    + self._partial_l1(result.wrinkle_mask, result_aug.wrinkle_mask.detach(), face_mask, has_wrinkle)
                ) / 3.0
            elif active_task == 'brown':
                l_consist = self._partial_l1(
                    result.brown_mask, result_aug.brown_mask.detach(), face_mask, has_brown
                )
            elif active_task == 'red':
                l_consist = self._partial_l1(
                    result.red_mask, result_aug.red_mask.detach(), face_mask, has_red
                )
            else:
                l_consist = self._partial_l1(
                    result.wrinkle_mask, result_aug.wrinkle_mask.detach(), face_mask, has_wrinkle
                )
            detail['consist'] = l_consist.item()
            loss = loss + self.w_consist * l_consist
        else:
            detail['consist'] = 0.0

        if self.w_freq_reg > 0 and model is not None and (
            active_task is None or active_task in ('brown', 'red')
        ):
            l_freq = self._freq_reg(model)
            detail['freq_reg'] = l_freq.item()
            loss = loss + self.w_freq_reg * l_freq
        else:
            detail['freq_reg'] = 0.0

        return loss, detail


def train_one_epoch_task_sampled(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    cfg,
    epoch,
    total_epochs,
):
    model.train()
    total_loss = 0.0
    processed_steps = 0
    skipped_steps = 0
    log = {k: 0.0 for k in ['brown', 'red', 'wrinkle', 'recon', 'consist', 'freq_reg']}
    task_counts = {k: 0 for k in ['brown', 'red', 'wrinkle', 'all', 'none']}

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
        active_task = choose_active_task(batch, cfg)
        task_counts[active_task or 'all'] += 1

        rgb_cross = batch['rgb_cross'].to(device, non_blocking=True)
        rgb_parallel = batch['rgb_parallel'].to(device, non_blocking=True)
        brown_gt = batch['brown'].to(device, non_blocking=True)
        red_gt = batch['red'].to(device, non_blocking=True)
        wrinkle_gt = batch['wrinkle'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)

        has_brown = batch['has_brown']
        has_red = batch['has_red']
        has_wrinkle = batch['has_wrinkle']

        rgb_cross_aug, rgb_parallel_aug = v5.apply_paired_batch_illumination_aug_fast(
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

        result_for_loss = v5._skin_result_to_fp32(result)
        result_clean_for_loss = (
            v5._skin_result_to_fp32(result_clean) if result_clean is not None else None
        )

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
            active_task=active_task,
        )

        if (
            criterion.w_freq_reg > 0
            and (active_task is None or active_task in ('brown', 'red'))
        ):
            l_freq = base._v2_freq_reg(model).float()
            loss = loss + criterion.w_freq_reg * l_freq
            detail['freq_reg'] = l_freq.item()

        if not loss.requires_grad:
            skipped_steps += 1
            continue

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        processed_steps += 1
        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

        denom = max(processed_steps, 1)
        pbar.set_postfix(
            loss=f"{total_loss / denom:.4f}",
            task=active_task or 'all',
            brown=f"{log['brown'] / denom:.4f}",
            red=f"{log['red'] / denom:.4f}",
            wrinkle=f"{log['wrinkle'] / denom:.4f}",
        )

    pbar.close()
    n = max(processed_steps, 1)
    out = {'total': total_loss / n, **{k: v / n for k, v in log.items()}}
    out.update({f'task_{k}_steps': v for k, v in task_counts.items()})
    out['skipped_steps'] = skipped_steps
    return out


def main():
    parser = argparse.ArgumentParser(
        description='SkinAnalyzerV2 training V9 (task-sampled multi-task)'
    )
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument(
        '--wrinkle_min_pos_ratio',
        type=float,
        default=CFG['wrinkle_min_pos_ratio'],
        help='minimum wrinkle_pos_ratio required to enable wrinkle loss for a patch',
    )
    parser.add_argument(
        '--task_probs',
        type=str,
        default='brown=0.20,red=0.25,wrinkle=0.55',
        help='task sampling probabilities, e.g. brown=0.20,red=0.25,wrinkle=0.55',
    )
    parser.add_argument(
        '--disable_task_sampling',
        action='store_true',
        help='disable task sampling and train all tasks jointly',
    )
    args = parser.parse_args()
    CFG['wrinkle_min_pos_ratio'] = args.wrinkle_min_pos_ratio
    CFG['task_probs'] = _parse_task_probs(args.task_probs)
    CFG['task_sampling_enabled'] = not args.disable_task_sampling

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
    print("Version : V9 task-sampled multi-task")
    print(f"  - task_sampling_enabled: {CFG['task_sampling_enabled']}")
    print(f"  - task_probs: {CFG['task_probs']}")
    print(f"  - wrinkle_min_pos_ratio: {CFG['wrinkle_min_pos_ratio']}")

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = base.build_analyzer_v2(
        base_ch=CFG['base_ch'],
        low_r=CFG['low_r'],
        high_r=CFG['high_r'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    full_ds = v8.WrinklePositiveSkinDataset(
        CFG['patch_dir'],
        CFG['img_size'],
        augment=True,
        wrinkle_min_pos_ratio=CFG['wrinkle_min_pos_ratio'],
        compute_missing_pos_ratio=CFG['wrinkle_compute_missing_pos_ratio'],
    )
    val_size = max(1, int(len(full_ds) * CFG['val_ratio']))
    train_ds, val_ds = random_split(full_ds, [len(full_ds) - val_size, val_size])

    val_ds.dataset = copy.copy(full_ds)
    val_ds.dataset.augment = False

    prefetch_factor = CFG['prefetch_factor'] if CFG['num_workers'] > 0 else None
    train_sampler = None
    if CFG.get('use_weighted_sampler', False):
        all_weights = full_ds.get_sample_weights(
            neg_weight=CFG.get('sampler_neg_weight', 0.02)
        )
        train_weights = [all_weights[i] for i in train_ds.indices]
        train_sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )
        print(
            "Weighted sampler: ON "
            f"(train_samples={len(train_weights)}, "
            f"neg_weight={CFG.get('sampler_neg_weight', 0.02)})"
        )
    else:
        print("Weighted sampler: OFF")

    train_loader = DataLoader(
        train_ds,
        CFG['batch_size'],
        shuffle=train_sampler is None,
        sampler=train_sampler,
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

    criterion = TaskAwareSkinAnalyzerLossV9(
        w_brown=CFG['w_brown'],
        w_red=CFG['w_red'],
        w_wrinkle=CFG['w_wrinkle'],
        w_recon=CFG['w_recon'],
        red_area_weight=CFG['red_area_weight'],
        red_outside_weight=CFG['red_outside_weight'],
        wrinkle_distance_weight=CFG['wrinkle_distance_weight'],
        wrinkle_pos_boost=CFG['wrinkle_pos_boost'],
        wrinkle_band_boost=CFG['wrinkle_band_boost'],
        wrinkle_band_sigma=CFG['wrinkle_band_sigma'],
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

    v5.CFG = CFG
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

        train_log = train_one_epoch_task_sampled(
            model, train_loader, criterion, optimizer, scaler, device, CFG,
            epoch, CFG['epochs'],
        )
        # Validation remains all-task, so val metrics are comparable to V7/V8.
        val_log = v5.validate_fast(model, val_loader, criterion, device)
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
            'train_loss': train_log,
            'cfg': CFG,
        }
        torch.save(last_ckpt, ckpt_dir / 'last_checkpoint.pth')
        if is_best:
            torch.save(last_ckpt, ckpt_dir / 'best_skin_analyzer.pth')

        print(
            f"[{epoch:03d}/{CFG['epochs']}]"
            f"  trn={train_log['total']:.4f}"
            f"  val={val_log['total']:.4f}"
            f"  B_dice={val_log['brown_dice']:.3f}"
            f"  R_dice={val_log['red_dice']:.3f}"
            f"  W_dice={val_log['wrinkle_dice']:.3f}"
            f"  steps(B/R/W)="
            f"{train_log['task_brown_steps']}/"
            f"{train_log['task_red_steps']}/"
            f"{train_log['task_wrinkle_steps']}"
            f"  skipped={train_log['skipped_steps']}"
            f"  score={val_log['score']:.3f}"
            + (' *' if is_best else '')
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
