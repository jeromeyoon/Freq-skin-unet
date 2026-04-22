"""
skin_train_v6.py
================
Minimal-loss-update variant based on skin_train_v5.

Goals
-----
- Keep the V5 training/inference-speed structure
- Keep brown loss as-is
- Treat red as a region task rather than a spot task
- Add spread-control penalties for red / wrinkle so they do not
  collapse into face-wide activations
"""

import argparse
import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import skin_train_v5 as v5
import skin_train_v2 as base


CFG = copy.deepcopy(v5.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v6',
    preview_dir='./previews_v6',
    red_area_weight=0.20,
    red_outside_weight=0.12,
    wrinkle_distance_weight=0.30,
    wrinkle_pos_boost=4.0,
    wrinkle_band_boost=2.0,
    wrinkle_band_sigma=6.0,
)


class TaskAwareSkinAnalyzerLoss(base.SkinAnalyzerLoss):
    """
    Brown: keep spot-style supervision from the original loss.
    Red:   region-style focal+dice + area matching + mild outside penalty.
    Wrinkle: focal+dice + stronger spread penalty to suppress face-wide response.
    """

    def __init__(
        self,
        *args,
        red_area_weight: float = 0.20,
        red_outside_weight: float = 0.12,
        wrinkle_distance_weight: float = 0.30,
        wrinkle_pos_boost: float = 4.0,
        wrinkle_band_boost: float = 2.0,
        wrinkle_band_sigma: float = 6.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.red_area_weight = red_area_weight
        self.red_outside_weight = red_outside_weight
        self.wrinkle_distance_weight = wrinkle_distance_weight
        self.wrinkle_pos_boost = wrinkle_pos_boost
        self.wrinkle_band_boost = wrinkle_band_boost
        self.wrinkle_band_sigma = wrinkle_band_sigma

    @staticmethod
    def _avail_tensor(has_gt, device):
        return torch.tensor(has_gt, dtype=torch.float32, device=device)

    def _area_ratio_penalty(self, pred, gt, face_mask, has_gt):
        avail = self._avail_tensor(has_gt, pred.device)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred) * face_mask
        gt = gt * face_mask

        face_area = face_mask.flatten(1).sum(1).clamp(min=1.0)
        pred_ratio = prob.flatten(1).sum(1) / face_area
        gt_ratio = gt.flatten(1).sum(1) / face_area

        return ((pred_ratio - gt_ratio).abs() * avail).sum() / avail.sum().clamp(min=1.0)

    def _outside_gt_penalty(self, pred, gt, face_mask, has_gt):
        avail = self._avail_tensor(has_gt, pred.device).view(-1, 1, 1, 1)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred)
        outside_mask = (1.0 - gt).clamp(0.0, 1.0) * face_mask
        per_pixel = prob * outside_mask * avail
        denom = (outside_mask * avail).sum().clamp(min=1.0)
        return per_pixel.sum() / denom

    def _wrinkle_distance_maps(self, gt, face_mask, has_gt):
        """
        Build distance-to-GT maps on CPU using OpenCV distance transform.
        Returns normalized distance maps in [0, 1], masked to the face region.
        """
        dist_maps = []
        gt_np = gt.detach().cpu().numpy()
        mask_np = face_mask.detach().cpu().numpy()

        for i, has in enumerate(has_gt):
            if not has:
                dist_maps.append(np.zeros_like(gt_np[i, 0], dtype=np.float32))
                continue

            gt_bin = (gt_np[i, 0] >= 0.5).astype(np.uint8)
            face_bin = (mask_np[i, 0] >= 0.5).astype(np.uint8)

            if gt_bin.sum() == 0:
                # If GT is effectively empty, use max penalty over the face ROI.
                dist = face_bin.astype(np.float32)
            else:
                inv_gt = (1 - gt_bin).astype(np.uint8)
                dist = cv2.distanceTransform(inv_gt, cv2.DIST_L2, 3).astype(np.float32)
                dist *= face_bin.astype(np.float32)
                max_val = float(dist[face_bin > 0].max()) if face_bin.sum() > 0 else 0.0
                if max_val > 0:
                    dist /= max_val

            dist_maps.append(dist)

        dist_tensor = torch.from_numpy(np.stack(dist_maps, axis=0)).to(gt.device).unsqueeze(1)
        return dist_tensor * face_mask

    def _weighted_dice_loss(self, pred, gt, face_mask, has_gt, weight_map):
        avail = self._avail_tensor(has_gt, pred.device).view(-1, 1, 1, 1)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred) * face_mask
        gt = gt * face_mask
        w = weight_map * face_mask * avail

        inter = (w * prob * gt).flatten(1).sum(1)
        p_sum = (w * prob).flatten(1).sum(1)
        g_sum = (w * gt).flatten(1).sum(1)
        dice = 1.0 - (2.0 * inter + 1.0) / (p_sum + g_sum + 1.0)

        avail_1d = self._avail_tensor(has_gt, pred.device)
        return (dice * avail_1d).sum() / avail_1d.sum().clamp(min=1.0)

    def _wrinkle_weighted_dice(self, pred, gt, face_mask, has_gt):
        dist_map = self._wrinkle_distance_maps(gt, face_mask, has_gt)
        near_gt = torch.exp(-dist_map / max(self.wrinkle_band_sigma, 1e-6))
        weight_map = (
            1.0
            + self.wrinkle_pos_boost * gt * face_mask
            + self.wrinkle_band_boost * near_gt * face_mask
        )
        return self._weighted_dice_loss(pred, gt, face_mask, has_gt, weight_map)

    def _wrinkle_distance_penalty(self, pred, gt, face_mask, has_gt):
        avail = self._avail_tensor(has_gt, pred.device).view(-1, 1, 1, 1)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred)
        dist_map = self._wrinkle_distance_maps(gt, face_mask, has_gt)
        background = (1.0 - gt).clamp(0.0, 1.0) * face_mask
        per_pixel = prob * dist_map * background * avail
        denom = (background * avail).sum().clamp(min=1.0)
        return per_pixel.sum() / denom

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
    ):
        detail = {}

        # Brown: keep the existing spot-oriented loss.
        l_brown = self._focal_dice_spot(
            result.brown_mask, brown_gt, face_mask, has_brown, alpha=0.5
        )

        # Red: region-style supervision + mild anti-spread penalties.
        l_red_base = self._focal_dice(
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

        # Wrinkle: weighted Dice + distance-aware spread suppression.
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
        if any(has_brown):
            loss = loss + self.w_brown * l_brown
        if any(has_red):
            loss = loss + self.w_red * l_red
        if any(has_wrinkle):
            loss = loss + self.w_wrinkle * l_wrinkle

        if self.w_recon > 0:
            l_recon = self._beer_lambert_recon(
                result.brown_mask, result.red_mask, rgb_cross, face_mask, has_brown, has_red
            )
            detail['recon'] = l_recon.item()
            loss = loss + self.w_recon * l_recon
        else:
            detail['recon'] = 0.0

        if self.w_consist > 0 and result_aug is not None:
            l_consist = (
                self._partial_l1(result.brown_mask, result_aug.brown_mask.detach(), face_mask, has_brown)
                + self._partial_l1(result.red_mask, result_aug.red_mask.detach(), face_mask, has_red)
                + self._partial_l1(result.wrinkle_mask, result_aug.wrinkle_mask.detach(), face_mask, has_wrinkle)
            ) / 3.0
            detail['consist'] = l_consist.item()
            loss = loss + self.w_consist * l_consist
        else:
            detail['consist'] = 0.0

        if self.w_freq_reg > 0 and model is not None:
            l_freq = self._freq_reg(model)
            detail['freq_reg'] = l_freq.item()
            loss = loss + self.w_freq_reg * l_freq
        else:
            detail['freq_reg'] = 0.0

        return loss, detail


def main():
    parser = argparse.ArgumentParser(description='SkinAnalyzerV2 training V6 (task-aware loss)')
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
    print("Version : V6 task-aware loss")
    print("  - brown: original spot loss")
    print("  - red: region-aware loss + area/outside penalties")
    print("  - wrinkle: weighted Dice + distance penalty")
    print(f"  - red_area_weight: {CFG['red_area_weight']}")
    print(f"  - red_outside_weight: {CFG['red_outside_weight']}")
    print(f"  - wrinkle_distance_weight: {CFG['wrinkle_distance_weight']}")
    print(f"  - wrinkle_pos_boost: {CFG['wrinkle_pos_boost']}")
    print(f"  - wrinkle_band_boost: {CFG['wrinkle_band_boost']}")
    print(f"  - wrinkle_band_sigma: {CFG['wrinkle_band_sigma']}")

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

    criterion = TaskAwareSkinAnalyzerLoss(
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

    # Reuse V5 helpers, but point them to this config for metric/threshold handling.
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
