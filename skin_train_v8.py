"""
skin_train_v8.py
================
Variant of skin_train_v7 that skips wrinkle supervision on black / near-empty
wrinkle patches using manifest ``wrinkle_pos_ratio``.

Why
---
new_data_prep.py limits global negative patches, but a patch can be positive for
brown/red while still being black for wrinkle.  If such patches keep
``has_wrinkle=True`` because the subject has a wrinkle GT file, the wrinkle loss
receives many "predict zero everywhere" updates and can collapse.

This version keeps the V7 objective, but overrides ``has_wrinkle`` at dataset
load time:

    has_wrinkle = has_wrinkle and wrinkle_pos_ratio >= wrinkle_min_pos_ratio

The on-disk manifest and patch images are not modified.
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, random_split

import skin_train_v2 as base
import skin_train_v5 as v5
import skin_train_v7 as v7


CFG = copy.deepcopy(v7.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v8',
    preview_dir='./previews_v8',
    # Patch-level wrinkle supervision gate.
    # 256x256 기준 0.00001 ~= 0.65 pixels 이므로, 실질적으로 "1픽셀 이상"에 가깝다.
    wrinkle_min_pos_ratio=1e-5,
    # wrinkle_pos_ratio가 없는 구 manifest에서는 GT 파일을 읽어 ratio를 계산한다.
    wrinkle_compute_missing_pos_ratio=True,
)


class WrinklePositiveSkinDataset(base.SkinDataset):
    """
    SkinDataset wrapper that disables wrinkle loss for black wrinkle patches.

    ``new_data_prep.py`` stores per-patch ``wrinkle_pos_ratio`` in manifest.
    The base dataset treats ``has_wrinkle`` as subject-level GT availability.
    This wrapper converts it to patch-level availability in memory.
    """

    def __init__(
        self,
        patch_dir: str,
        img_size: int = 256,
        augment: bool = False,
        wrinkle_min_pos_ratio: float = 1e-5,
        compute_missing_pos_ratio: bool = True,
    ):
        super().__init__(patch_dir, img_size, augment)
        self.wrinkle_min_pos_ratio = float(wrinkle_min_pos_ratio)
        self.compute_missing_pos_ratio = bool(compute_missing_pos_ratio)
        self._apply_wrinkle_patch_gate()

    def _read_wrinkle_pos_ratio(self, stem: str) -> float:
        gt_path = self.patch_dir / 'wrinkle' / f'{stem}.png'
        if not gt_path.exists():
            return 0.0
        arr = np.array(Image.open(gt_path).convert('L'))
        return float((arr > 127).sum()) / max(float(arr.size), 1.0)

    def _apply_wrinkle_patch_gate(self) -> None:
        before_has = 0
        after_has = 0
        disabled_black = 0
        missing_ratio = 0

        for stem in self.stems:
            info = self.manifest[stem]
            if not info.get('has_wrinkle', False):
                continue

            before_has += 1

            ratio = info.get('wrinkle_pos_ratio', None)
            if ratio is None:
                missing_ratio += 1
                if self.compute_missing_pos_ratio:
                    ratio = self._read_wrinkle_pos_ratio(stem)
                    # In-memory only.  Do not rewrite manifest.json.
                    info['wrinkle_pos_ratio'] = round(float(ratio), 6)
                else:
                    # Keep legacy behavior if ratio is unavailable and computing is disabled.
                    ratio = self.wrinkle_min_pos_ratio

            if float(ratio) >= self.wrinkle_min_pos_ratio:
                after_has += 1
            else:
                info['has_wrinkle'] = False
                disabled_black += 1

        print(
            "[WrinklePositiveSkinDataset] wrinkle patch gate: "
            f"threshold={self.wrinkle_min_pos_ratio:g}, "
            f"has_before={before_has}, has_after={after_has}, "
            f"disabled_black={disabled_black}, missing_ratio={missing_ratio}"
        )


def main():
    parser = argparse.ArgumentParser(
        description='SkinAnalyzerV2 training V8 (V7 + patch-level wrinkle gate)'
    )
    parser.add_argument('--resume', type=str, default=None, help='checkpoint path to resume')
    parser.add_argument(
        '--wrinkle_min_pos_ratio',
        type=float,
        default=CFG['wrinkle_min_pos_ratio'],
        help='minimum wrinkle_pos_ratio required to enable wrinkle loss for a patch',
    )
    args = parser.parse_args()
    CFG['wrinkle_min_pos_ratio'] = args.wrinkle_min_pos_ratio

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
    print("Version : V8 patch-level wrinkle gate")
    print("  - base objective: V7")
    print("  - red: generalized dice + area/outside penalties")
    print("  - wrinkle: weighted Dice + distance penalty")
    print(f"  - wrinkle_min_pos_ratio: {CFG['wrinkle_min_pos_ratio']}")

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = base.build_analyzer_v2(
        base_ch=CFG['base_ch'],
        low_r=CFG['low_r'],
        high_r=CFG['high_r'],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    full_ds = WrinklePositiveSkinDataset(
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

    criterion = v7.TaskAwareSkinAnalyzerLossV7(
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

        train_log = v5.train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, CFG,
            epoch, CFG['epochs'],
        )
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
            f"  score={val_log['score']:.3f}"
            + (' *' if is_best else '')
        )

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
