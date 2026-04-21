"""
skin_train_v3.py
================
SkinAnalyzerV2 학습 스크립트 V3

skin_train_v2 대비 주요 변경점
------------------------------
1. rgb_cross / rgb_parallel 에 동일한 illumination augmentation 적용
   - 편광 쌍 입력의 상대적 관계가 augmentation noise로 깨지지 않도록 보정
2. consistency target(result_clean) 계산 시 model.eval() 사용
   - train mode 상태의 BatchNorm / Dropout 노이즈가 target에 섞이지 않도록 보정
3. weighted sampler 기본 비활성화
   - batch 분포 왜곡과 replacement sampling으로 인한 학습 진동 완화

실행
----
  python skin_train_v3.py

  # 이어서 학습
  python skin_train_v3.py --resume ./checkpoints_v3/last_checkpoint.pth
"""

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
from ambient_aug import _make_directional_gradient, _make_vignette_torch


CFG = copy.deepcopy(base.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v3',
    preview_dir='./previews_v3',
    use_weighted_sampler=False,
)


def _apply_illumination_aug_with_params(
    rgb: torch.Tensor,
    *,
    intensity: float,
    ch_scales: torch.Tensor,
    tint: torch.Tensor,
    use_vignette: bool,
    vignette_strength: float | None,
    use_gradient: bool,
) -> torch.Tensor:
    """
    ambient_aug.apply_illumination_aug 와 동일 계열 처리이되,
    샘플링된 augmentation 파라미터를 외부에서 주입받아 재사용한다.
    """
    device = rgb.device
    h, w = rgb.shape[1], rgb.shape[2]

    rgb_aug = rgb * intensity
    rgb_aug = rgb_aug * ch_scales.view(3, 1, 1)
    rgb_aug = rgb_aug * tint.view(3, 1, 1)

    if use_vignette:
        vignette = _make_vignette_torch(h, w, vignette_strength, device)
        rgb_aug = rgb_aug * vignette.unsqueeze(0)
    elif use_gradient:
        grad = _make_directional_gradient(h, w, device)
        rgb_aug = rgb_aug * grad.unsqueeze(0)

    return rgb_aug.clamp(0.0, 1.0)


def apply_paired_batch_illumination_aug(
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
    편광 쌍 입력(cross / parallel)에 동일한 illumination augmentation을 적용한다.
    배치 내 각 샘플은 독립 랜덤이지만, 같은 샘플의 두 입력은 동일 파라미터를 공유한다.
    """
    assert rgb_cross_batch.shape == rgb_parallel_batch.shape, (
        "rgb_cross_batch and rgb_parallel_batch must have the same shape"
    )

    cross_out = []
    parallel_out = []

    for i in range(rgb_cross_batch.shape[0]):
        cross = rgb_cross_batch[i]
        parallel = rgb_parallel_batch[i]
        device = cross.device

        intensity = torch.empty(1, device=device).uniform_(*intensity_range).item()
        ch_scales = torch.empty(3, device=device).uniform_(*color_temp_range)
        tint = torch.ones(3, device=device)
        tint[0] = torch.empty(1, device=device).uniform_(*tint_range).item()
        tint[2] = torch.empty(1, device=device).uniform_(*tint_range).item()

        r = torch.rand(1, device=device).item()
        use_vignette = r < vignette_prob
        use_gradient = (not use_vignette) and (r < vignette_prob + gradient_prob)
        sampled_vignette_strength = None
        if use_vignette:
            sampled_vignette_strength = torch.empty(1, device=device).uniform_(0.1, vignette_strength).item()

        cross_aug = _apply_illumination_aug_with_params(
            cross,
            intensity=intensity,
            ch_scales=ch_scales,
            tint=tint,
            use_vignette=use_vignette,
            vignette_strength=sampled_vignette_strength,
            use_gradient=use_gradient,
        )
        parallel_aug = _apply_illumination_aug_with_params(
            parallel,
            intensity=intensity,
            ch_scales=ch_scales,
            tint=tint,
            use_vignette=use_vignette,
            vignette_strength=sampled_vignette_strength,
            use_gradient=use_gradient,
        )

        cross_out.append(cross_aug)
        parallel_out.append(parallel_aug)

    return torch.stack(cross_out), torch.stack(parallel_out)


def train_one_epoch(model, loader, criterion, optimizer, device, cfg, epoch, total_epochs):
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

    pbar = tqdm(
        loader,
        desc=f"Train {epoch:03d}/{total_epochs}",
        unit='batch',
        leave=False,
        dynamic_ncols=True,
    )

    for step, batch in enumerate(pbar, 1):
        rgb_cross = batch['rgb_cross'].to(device)
        rgb_parallel = batch['rgb_parallel'].to(device)
        brown_gt = batch['brown'].to(device)
        red_gt = batch['red'].to(device)
        wrinkle_gt = batch['wrinkle'].to(device)
        mask = batch['mask'].to(device)

        has_brown = batch['has_brown']
        has_red = batch['has_red']
        has_wrinkle = batch['has_wrinkle']

        rgb_cross_aug, rgb_parallel_aug = apply_paired_batch_illumination_aug(
            rgb_cross,
            rgb_parallel,
            **aug_kwargs,
        )

        result = model(rgb_cross_aug, rgb_parallel_aug, mask)

        result_clean = None
        if criterion.w_consist > 0:
            was_training = model.training
            model.eval()
            with torch.no_grad():
                result_clean = model(rgb_cross, rgb_parallel, mask)
            if was_training:
                model.train()

        loss, detail = criterion(
            result,
            brown_gt, red_gt, wrinkle_gt,
            rgb_cross=rgb_cross,
            face_mask=mask,
            has_brown=has_brown,
            has_red=has_red,
            has_wrinkle=has_wrinkle,
            result_aug=result_clean,
            model=None,
        )

        if criterion.w_freq_reg > 0:
            l_freq = base._v2_freq_reg(model)
            loss = loss + criterion.w_freq_reg * l_freq
            detail['freq_reg'] = l_freq.item()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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
    parser = argparse.ArgumentParser(description='SkinAnalyzerV2 학습 V3 (stability fixes)')
    parser.add_argument('--resume', type=str, default=None, help='재개할 체크포인트 경로')
    args = parser.parse_args()

    seed = CFG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Seed: {seed}")
    print("Model   : SkinAnalyzerV2")
    print("Version : V3 stability fixes")
    print("  - paired illumination augmentation")
    print("  - eval-mode consistency target")
    print(f"  - weighted sampler default: {CFG['use_weighted_sampler']}")
    print("Loss    : skin_loss_smooth (same curriculum as V2)")

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

    if CFG.get('use_weighted_sampler', False):
        all_weights = full_ds.get_sample_weights(
            neg_weight=CFG.get('sampler_neg_weight', 0.05)
        )
        train_weights = [all_weights[i] for i in train_ds.indices]
        sampler = WeightedRandomSampler(
            weights=train_weights,
            num_samples=len(train_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds,
            CFG['batch_size'],
            sampler=sampler,
            num_workers=CFG['num_workers'],
            pin_memory=True,
            persistent_workers=CFG['num_workers'] > 0,
            prefetch_factor=4 if CFG['num_workers'] > 0 else None,
            collate_fn=base.skin_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            CFG['batch_size'],
            shuffle=True,
            num_workers=CFG['num_workers'],
            pin_memory=True,
            persistent_workers=CFG['num_workers'] > 0,
            prefetch_factor=4 if CFG['num_workers'] > 0 else None,
            collate_fn=base.skin_collate_fn,
        )

    val_loader = DataLoader(
        val_ds,
        CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=True,
        persistent_workers=CFG['num_workers'] > 0,
        prefetch_factor=4 if CFG['num_workers'] > 0 else None,
        collate_fn=base.skin_collate_fn,
    )
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    preview_loader = None
    if CFG['preview_interval'] > 0:
        excl_ds = base.ExcludedDataset.from_skin_dataset(full_ds)
        if len(excl_ds) > 0:
            preview_loader = DataLoader(
                excl_ds,
                batch_size=CFG['preview_n_images'],
                shuffle=True,
                num_workers=0,
                collate_fn=base.preview_collate_fn,
            )
            print(f"Preview 대상 {len(excl_ds)}개 (매 {CFG['preview_interval']} epoch)")

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

    start_epoch = 1
    best_score = float('-inf')

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print("  [WARN] optimizer 상태 없음 → 초기화")
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        else:
            for _ in range(ckpt['epoch']):
                scheduler.step()
        start_epoch = ckpt['epoch'] + 1
        best_score = ckpt.get('best_score', ckpt.get('val_metric', {}).get('score', float('-inf')))
        print(
            f"Resume: epoch {ckpt['epoch']} → {start_epoch}부터 재개 "
            f"(best_score={best_score:.4f}  lr={scheduler.get_last_lr()[0]:.2e})"
        )

    print("\n학습 시작...")

    base.CFG = CFG

    epoch_bar = tqdm(
        range(start_epoch, CFG['epochs'] + 1),
        desc='Epochs',
        unit='ep',
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
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
            model, train_loader, criterion, optimizer, device, CFG,
            epoch, CFG['epochs'],
        )
        val_log = base.validate(model, val_loader, criterion, device)
        scheduler.step()

        od_gate = torch.sigmoid(model.cross_encoder.od_filter.low_logit).squeeze().tolist()
        is_best = val_log['score'] > best_score

        epoch_bar.set_postfix(
            train=f"{train_log['total']:.4f}",
            val=f"{val_log['total']:.4f}",
            bD=f"{val_log['brown_dice']:.3f}",
            rD=f"{val_log['red_dice']:.3f}",
            wD=f"{val_log['wrinkle_dice']:.3f}",
        )

        tqdm.write(
            f"[{epoch:03d}/{CFG['epochs']}]"
            f"  trn={train_log['total']:.4f}"
            f"  val={val_log['total']:.4f}"
            f"  B={val_log['brown']:.4f}"
            f"  R={val_log['red']:.4f}"
            f"  W={val_log['wrinkle']:.4f}"
            f"  B_dice={val_log['brown_dice']:.3f}"
            f"  R_dice={val_log['red_dice']:.3f}"
            f"  W_dice={val_log['wrinkle_dice']:.3f}"
            f"  score={val_log['score']:.3f}"
            f"  rc={val_log['recon']:.4f}"
            f"  od_gate(R,G,B)=[{od_gate[0]:.3f},{od_gate[1]:.3f},{od_gate[2]:.3f}]"
            f"  w_consist={criterion.w_consist:.3f}"
            f"  w_freq={criterion.w_freq_reg:.3f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
            + (' *' if is_best else '')
        )

        if is_best:
            best_score = val_log['score']

        last_ckpt = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_score': best_score,
            'val_loss': val_log,
            'val_metric': {
                'brown_dice': val_log['brown_dice'],
                'red_dice': val_log['red_dice'],
                'wrinkle_dice': val_log['wrinkle_dice'],
                'score': val_log['score'],
            },
            'cfg': CFG,
        }
        torch.save(last_ckpt, ckpt_dir / 'last_checkpoint.pth')

        if is_best:
            torch.save(last_ckpt, ckpt_dir / 'best_skin_analyzer.pth')
            tqdm.write(
                "  → Best 갱신"
                f" score={best_score:.3f}"
                f" (B={val_log['brown_dice']:.3f}"
                f", R={val_log['red_dice']:.3f}"
                f", W={val_log['wrinkle_dice']:.3f})"
            )

        if (
            preview_loader is not None
            and CFG['preview_interval'] > 0
            and epoch % CFG['preview_interval'] == 0
        ):
            base.save_preview(
                model,
                preview_loader,
                device,
                epoch,
                save_dir=Path(CFG['preview_dir']),
                n_images=CFG['preview_n_images'],
            )

    epoch_bar.close()
    print("\n학습 완료!")


if __name__ == '__main__':
    main()
