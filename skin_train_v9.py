"""
skin_train_v9.py
================
Standalone V9 training script.

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

This file intentionally does not import skin_train_v2~v8.  Shared project
modules are still used directly:

- skin_net_v2.py      : model factory
- skin_dataset.py     : patch dataset / collate
- skin_loss_smooth.py : base loss primitives and curriculum scheduler
- ambient_aug.py      : vectorized illumination helpers
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from tqdm import tqdm

from ambient_aug import _make_directional_gradient, _make_vignette_torch
from skin_dataset import SkinDataset, skin_collate_fn
from skin_loss_smooth import SkinAnalyzerLoss, get_loss_weights
from skin_net_v2 import build_analyzer_v2


CFG = dict(
    patch_dir='./patches',
    checkpoint_dir='./checkpoints_v9',

    base_ch=64,
    low_r=0.1,
    high_r=0.4,
    img_size=256,

    epochs=50,
    batch_size=16,
    lr=1e-4,
    weight_decay=1e-4,
    num_workers=2,
    val_ratio=0.1,

    w_brown=1.0,
    w_red=1.0,
    w_wrinkle=1.0,
    w_recon=0.3,
    dice_threshold=0.5,
    best_w_brown=1.0,
    best_w_red=1.0,
    best_w_wrinkle=1.2,

    aug_warmup_epochs=15,
    intensity_range=(0.6, 1.4),
    color_temp_range=(0.7, 1.3),
    tint_range=(0.8, 1.2),
    vignette_prob=0.5,
    gradient_prob=0.5,

    use_weighted_sampler=True,
    sampler_neg_weight=0.10,
    prefetch_factor=4,
    consistency_every_n_steps=2,

    red_area_weight=0.20,
    red_outside_weight=0.30,

    # ── Wrinkle loss (Tversky + Focal + Edge) ─────────────────────────────
    # Tversky: alpha=FP weight, beta=FN weight.
    # beta > alpha → miss no wrinkle line (tolerate false positives).
    wrinkle_tversky_alpha=0.3,
    wrinkle_tversky_beta=0.7,
    # Focal BCE: high gamma suppresses easy background, alpha balances
    # positive/negative pixel gradient budget for sparse wrinkle lines.
    # alpha=0.80: negatives contribute 20% of gradient (vs 3% at 0.97),
    # preventing "predict positive everywhere" collapse.
    wrinkle_focal_gamma=3.0,
    wrinkle_focal_alpha=0.80,
    # Sobel edge agreement: encourages precise wrinkle boundary placement.
    wrinkle_edge_weight=0.15,
    # GT soft-label dilation (pixels).  VISIA wrinkle GT is often annotated
    # as a thin center line; dilating by 1–2 px makes supervision forgiving
    # without losing spatial precision.
    wrinkle_gt_dilation=2,

    # Minimum wrinkle-positive pixel ratio to count a patch as supervised.
    # 5e-4 ≈ 33 pixels in a 256×256 patch — avoids training on near-empty GT.
    wrinkle_min_pos_ratio=5e-4,
    wrinkle_compute_missing_pos_ratio=True,

    task_sampling_enabled=True,
    # Increase wrinkle share: brown/red are well-supervised blobs,
    # wrinkle needs more gradient steps to converge.
    # red raised 0.20→0.30 to recover sufficient gradient budget.
    task_probs=dict(
        brown=0.15,
        red=0.30,
        wrinkle=0.55,
    ),
    # Train wrinkle-only for the first N epochs so wrinkle signal is not
    # dominated by the stronger brown/red gradient in early training.
    # Extended 5→15: ParallelPolEncoder high-freq gate starts at sigmoid(-2)≈0.12
    # and needs more epochs to open before brown/red gradients compete.
    wrinkle_solo_epochs=8,
    # Curriculum Tversky ramp length (epochs). Set to ~30% of total epochs.
    wrinkle_tversky_ramp_epochs=15,

    seed=42,
    preview_interval=10,
    preview_n_images=8,
    preview_dir='./previews_v9',
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


class WrinklePositiveSkinDataset(SkinDataset):
    """
    SkinDataset wrapper that disables wrinkle loss for black wrinkle patches.

    new_data_prep.py stores per-patch wrinkle_pos_ratio in manifest.  The base
    dataset treats has_wrinkle as GT file availability.  This wrapper converts it
    to patch-level availability in memory.
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
        self._drop_empty_after_wrinkle_gate()

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
                    # In-memory only. Do not rewrite manifest.json.
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

    def _drop_empty_after_wrinkle_gate(self) -> None:
        """
        Drop samples that become fully unsupervised after disabling black wrinkle.

        The base SkinDataset filters samples with no GT before this wrapper
        changes has_wrinkle. If a sample had only a black wrinkle GT, the wrapper
        can turn all task flags off. Such samples would produce a zero supervised
        loss and can make backward unsafe, so remove them from this dataset.
        """
        before = len(self.stems)
        still_valid = []
        newly_empty = []
        for stem in self.stems:
            info = self.manifest[stem]
            if any(info.get(f'has_{task}', False) for task in ('brown', 'red', 'wrinkle')):
                still_valid.append(stem)
            else:
                newly_empty.append(stem)

        if newly_empty:
            self.stems = still_valid
            self.excluded_stems.extend(newly_empty)

        print(
            "[WrinklePositiveSkinDataset] empty-after-gate filter: "
            f"kept={len(self.stems)}/{before}, newly_excluded={len(newly_empty)}"
        )


def _skin_result_to_fp32(result):
    """Keep loss calculations in fp32 even when model forward used AMP."""
    return result._replace(
        brown_mask=result.brown_mask.float(),
        brown_score=result.brown_score.float(),
        red_mask=result.red_mask.float(),
        red_score=result.red_score.float(),
        wrinkle_mask=result.wrinkle_mask.float(),
        wrinkle_score=result.wrinkle_score.float(),
    )


def _v2_freq_reg(model: torch.nn.Module) -> torch.Tensor:
    """
    ODFrequencyFilter.low_logit regularization for SkinAnalyzerV2.

    gate_low = sigmoid(low_logit). Higher values suppress illumination more.
    """
    gate = torch.sigmoid(model.cross_encoder.od_filter.low_logit)
    return gate.mean()


@torch.no_grad()
def _dice_sum_and_count(
    pred_logit: torch.Tensor,
    gt: torch.Tensor,
    face_mask: torch.Tensor,
    has_gt: list[bool],
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> tuple[float, int]:
    """Partial-GT Dice metric; only samples with has_gt=True are averaged."""
    valid = torch.tensor(has_gt, dtype=torch.bool, device=pred_logit.device)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return 0.0, 0

    pred_bin = (torch.sigmoid(pred_logit) >= threshold).float() * face_mask
    gt_bin = (gt >= 0.5).float() * face_mask

    pred_flat = pred_bin[valid].flatten(1)
    gt_flat = gt_bin[valid].flatten(1)

    inter = (pred_flat * gt_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)
    dice = (2.0 * inter + eps) / (denom + eps)
    return float(dice.sum().item()), n_valid


def _weighted_best_score(val_log: dict, cfg: dict) -> float:
    wb = cfg.get('best_w_brown', 1.0)
    wr = cfg.get('best_w_red', 1.0)
    ww = cfg.get('best_w_wrinkle', 1.0)
    denom = wb + wr + ww
    return (
        wb * val_log.get('brown_dice', 0.0)
        + wr * val_log.get('red_dice', 0.0)
        + ww * val_log.get('wrinkle_dice', 0.0)
    ) / max(denom, 1e-6)


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
    Apply the same illumination augmentation to cross/parallel images from the
    same sample. Scalar color transforms are vectorized across the batch.
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

    for idx in torch.where(use_vignette)[0].tolist():
        strength = torch.empty(1, device=device).uniform_(0.1, vignette_strength).item()
        spatial[idx, 0] = _make_vignette_torch(h, w, strength, device)

    for idx in torch.where(use_gradient)[0].tolist():
        spatial[idx, 0] = _make_directional_gradient(h, w, device)

    cross_aug = (cross_aug * spatial).clamp(0.0, 1.0)
    parallel_aug = (parallel_aug * spatial).clamp(0.0, 1.0)
    return cross_aug, parallel_aug


class TaskAwareSkinAnalyzerLossV9(SkinAnalyzerLoss):
    """
    Task-aware V9 objective with optional active_task gating.

    active_task=None keeps all-task behavior for validation.
    active_task in {"brown", "red", "wrinkle"} backpropagates only that task.
    """

    def __init__(
        self,
        *args,
        red_focal_alpha: float = 0.75,
        red_gt_dilation: int = 0,
        red_area_weight: float = 0.20,
        red_outside_weight: float = 0.12,
        wrinkle_tversky_alpha: float = 0.3,
        wrinkle_tversky_beta: float = 0.7,
        wrinkle_focal_gamma: float = 3.0,
        wrinkle_focal_alpha: float = 0.97,
        wrinkle_edge_weight: float = 0.15,
        wrinkle_gt_dilation: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.red_focal_alpha = red_focal_alpha
        self.red_gt_dilation = red_gt_dilation
        self.red_area_weight = red_area_weight
        self.red_outside_weight = red_outside_weight
        self.wrinkle_tversky_alpha = wrinkle_tversky_alpha
        self.wrinkle_tversky_beta = wrinkle_tversky_beta
        self.wrinkle_focal_gamma = wrinkle_focal_gamma
        self.wrinkle_focal_alpha = wrinkle_focal_alpha
        self.wrinkle_edge_weight = wrinkle_edge_weight
        self.wrinkle_gt_dilation = wrinkle_gt_dilation

        # Pre-built edge kernels (registered as buffers so device moves work).
        self.register_buffer(
            '_gauss_k',
            torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]).view(1, 1, 3, 3) / 16.0,
        )
        self.register_buffer(
            '_sobel_x',
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3),
        )

    @staticmethod
    def _avail_tensor(has_gt, device):
        return torch.tensor(has_gt, dtype=torch.float32, device=device)

    def _generalized_dice_loss(self, pred, gt, face_mask, has_gt):
        avail = self._avail_tensor(has_gt, pred.device)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred) * face_mask
        gt = gt * face_mask

        prob_fg = prob.flatten(1)
        gt_fg = gt.flatten(1)
        prob_bg = ((1.0 - prob) * face_mask).flatten(1)
        gt_bg = ((1.0 - gt) * face_mask).flatten(1)

        vol_fg = gt_fg.sum(1)
        vol_bg = gt_bg.sum(1)

        w_fg = 1.0 / (vol_fg.pow(2) + 1e-6)
        w_bg = 1.0 / (vol_bg.pow(2) + 1e-6)

        inter_fg = (prob_fg * gt_fg).sum(1)
        inter_bg = (prob_bg * gt_bg).sum(1)
        union_fg = prob_fg.sum(1) + gt_fg.sum(1)
        union_bg = prob_bg.sum(1) + gt_bg.sum(1)

        gdice = 1.0 - (
            2.0 * (w_fg * inter_fg + w_bg * inter_bg) + 1e-6
        ) / (
            (w_fg * union_fg + w_bg * union_bg) + 1e-6
        )

        return (gdice * avail).sum() / avail.sum().clamp(min=1.0)

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

    def _dilate_gt(self, gt: torch.Tensor) -> torch.Tensor:
        """Morphological dilation of binary GT mask (max-pool approximation).

        VISIA wrinkle GT is typically a thin center-line trace.  Dilating by
        wrinkle_gt_dilation pixels creates a soft supervision band that is more
        forgiving of 1-2 pixel mis-registration and matches the physical width
        of a visible deep wrinkle groove better than a 1-pixel line.
        """
        if self.wrinkle_gt_dilation <= 0:
            return gt
        k = 2 * self.wrinkle_gt_dilation + 1
        return F.max_pool2d(
            gt, kernel_size=k, stride=1, padding=self.wrinkle_gt_dilation
        ).clamp(0.0, 1.0)

    def _dilate_red_gt(self, gt: torch.Tensor) -> torch.Tensor:
        """Optional soft-label dilation for red spot GT.

        Red GT boundaries annotated by specialists may be 1-2 pixels tight.
        A 1-pixel dilation reduces loss spikes from boundary mis-registration
        without expanding the supervision area as aggressively as wrinkle's 2 px.
        red_gt_dilation=0 (default) leaves GT unchanged for V9 compatibility.
        """
        if self.red_gt_dilation <= 0:
            return gt
        k = 2 * self.red_gt_dilation + 1
        return F.max_pool2d(
            gt, kernel_size=k, stride=1, padding=self.red_gt_dilation
        ).clamp(0.0, 1.0)

    def _tversky_loss(self, pred, gt, face_mask, has_gt) -> torch.Tensor:
        """Tversky loss with asymmetric FP/FN weighting.

        For sparse line structures (deep wrinkles < 1% of face pixels):
          alpha = FP weight (low) → tolerate false positives
          beta  = FN weight (high) → never miss a real wrinkle line

        Tversky generalises Dice (alpha=beta=0.5 → Dice).
        """
        avail = self._avail_tensor(has_gt, pred.device).view(-1, 1, 1, 1)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred) * face_mask
        gt_m = gt * face_mask

        tp = (prob * gt_m * avail).flatten(1).sum(1)
        fp = (prob * (1.0 - gt_m) * avail).flatten(1).sum(1)
        fn = ((1.0 - prob) * gt_m * avail).flatten(1).sum(1)

        tversky = (tp + 1e-6) / (
            tp
            + self.wrinkle_tversky_alpha * fp
            + self.wrinkle_tversky_beta * fn
            + 1e-6
        )
        avail_1d = self._avail_tensor(has_gt, pred.device)
        return 1.0 - (tversky * avail_1d).sum() / avail_1d.sum().clamp(min=1.0)

    def _wrinkle_focal_bce(self, pred, gt, face_mask, has_gt) -> torch.Tensor:
        """Focal BCE for sparse wrinkle pixels.

        wrinkle_focal_alpha ≈ 0.97 → 97 % of gradient budget on positive pixels.
        wrinkle_focal_gamma ≈ 3.0  → down-weights easy background heavily.
        Together they counteract the ~100:1 class imbalance without explicit
        per-sample re-weighting.

        Denominator is alpha-weighted (n_pos, n_neg) rather than total face pixels.
        With ~1% wrinkle density, a face-pixel denominator shrinks the loss to
        ~0.01 of its intended scale, preventing the wrinkle logit from moving in
        early training.  The balanced denominator restores the correct gradient scale.
        """
        avail = self._avail_tensor(has_gt, pred.device).view(-1, 1, 1, 1)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        gt_m = gt * face_mask
        bce = F.binary_cross_entropy_with_logits(pred, gt_m, reduction='none')
        prob = torch.sigmoid(pred)
        p_t = torch.where(gt_m > 0.5, prob, 1.0 - prob)
        alpha_t = torch.where(
            gt_m > 0.5,
            pred.new_full(gt_m.shape, self.wrinkle_focal_alpha),
            pred.new_full(gt_m.shape, 1.0 - self.wrinkle_focal_alpha),
        )
        focal = alpha_t * (1.0 - p_t) ** self.wrinkle_focal_gamma * bce
        focal = focal * face_mask * avail

        # Balanced denominator: weight positive and negative pixel counts by their
        # focal alpha so the loss scale is independent of wrinkle pixel density.
        n_pos = ((gt_m > 0.5) * face_mask * avail).sum().clamp(min=1.0)
        n_neg = ((gt_m <= 0.5) * face_mask * avail).sum().clamp(min=1.0)
        denom = (self.wrinkle_focal_alpha * n_pos
                 + (1.0 - self.wrinkle_focal_alpha) * n_neg).clamp(min=1.0)
        return focal.sum() / denom

    def _wrinkle_edge_loss(self, pred, gt, face_mask, has_gt) -> torch.Tensor:
        """Sobel edge-agreement loss with Gaussian pre-smoothing.

        Skin-expert rationale: deep wrinkles under parallel polarisation appear
        as dark grooves with sharp lateral edges.  Matching the Sobel gradient
        of the prediction to that of the GT encourages the model to place
        wrinkle boundaries precisely rather than predicting diffuse blobs.

        Gaussian blur is applied before Sobel because VISIA wrinkle GT is often
        a 1-pixel-wide center-line trace.  Applying Sobel directly to a 1-px line
        produces impulse-like responses that generate noisy, unstable gradients.
        A 3×3 Gaussian blur widens the center-line to ~3 px, yielding a smooth
        and stable edge map for both prediction and GT comparison.
        """
        avail = self._avail_tensor(has_gt, pred.device).view(-1, 1, 1, 1)
        if avail.sum() <= 0:
            return pred.new_tensor(0.0)

        prob = torch.sigmoid(pred) * face_mask
        gt_m = gt * face_mask

        gauss_k = self._gauss_k.to(dtype=prob.dtype)
        sobel_x = self._sobel_x.to(dtype=prob.dtype)
        sobel_y = sobel_x.transpose(2, 3).contiguous()

        def _smooth_edge(x: torch.Tensor) -> torch.Tensor:
            x_s = F.conv2d(x, gauss_k, padding=1)
            gx = F.conv2d(x_s, sobel_x, padding=1)
            gy = F.conv2d(x_s, sobel_y, padding=1)
            return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        per_pixel = F.l1_loss(_smooth_edge(prob), _smooth_edge(gt_m), reduction='none')
        per_pixel = per_pixel * face_mask * avail
        denom = (face_mask * avail).sum().clamp(min=1.0)
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
        active_task: str | None = None,
    ):
        detail = {'active_task': active_task or 'all'}

        l_brown = self._focal_dice_spot(
            result.brown_mask, brown_gt, face_mask, has_brown, alpha=0.5
        )

        # Red loss: Focal(alpha=self.red_focal_alpha) + GDice blended 50/50.
        # Optional 1-px GT dilation (red_gt_dilation) reduces loss spikes from
        # boundary mis-registration in specialist annotations.
        red_gt_eff = self._dilate_red_gt(red_gt)
        l_red_focal = self._focal_dice_spot(
            result.red_mask, red_gt_eff, face_mask, has_red,
            alpha=self.red_focal_alpha,
        )
        l_red_gdice = self._generalized_dice_loss(
            result.red_mask, red_gt_eff, face_mask, has_red
        )
        l_red_base = 0.5 * l_red_focal + 0.5 * l_red_gdice
        l_red_area = self._area_ratio_penalty(
            result.red_mask, red_gt_eff, face_mask, has_red
        )
        l_red_outside = self._outside_gt_penalty(
            result.red_mask, red_gt_eff, face_mask, has_red
        )
        l_red = (
            l_red_base
            + self.red_area_weight * l_red_area
            + self.red_outside_weight * l_red_outside
        )

        # Soft GT: dilate wrinkle center-line annotation to match visible groove
        # width under parallel polarisation (keeps edge loss on raw GT for sharpness).
        wrinkle_gt_soft = self._dilate_gt(wrinkle_gt)

        l_wrinkle_tversky = self._tversky_loss(
            result.wrinkle_mask, wrinkle_gt_soft, face_mask, has_wrinkle
        )
        l_wrinkle_focal = self._wrinkle_focal_bce(
            result.wrinkle_mask, wrinkle_gt_soft, face_mask, has_wrinkle
        )
        # Edge loss uses raw (un-dilated) GT so boundaries remain crisp.
        l_wrinkle_edge = self._wrinkle_edge_loss(
            result.wrinkle_mask, wrinkle_gt, face_mask, has_wrinkle
        )
        l_wrinkle = (
            l_wrinkle_tversky
            + l_wrinkle_focal
            + self.wrinkle_edge_weight * l_wrinkle_edge
        )

        detail.update(
            brown=l_brown.item(),
            red=l_red.item(),
            wrinkle=l_wrinkle.item(),
            red_base=l_red_base.item(),
            red_area=l_red_area.item(),
            red_outside=l_red_outside.item(),
            wrinkle_tversky=l_wrinkle_tversky.item(),
            wrinkle_focal=l_wrinkle_focal.item(),
            wrinkle_edge=l_wrinkle_edge.item(),
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
                result.brown_mask,
                result.red_mask,
                rgb_cross,
                face_mask,
                has_brown,
                has_red,
            )
            detail['recon'] = l_recon.item()
            loss = loss + self.w_recon * l_recon
        else:
            detail['recon'] = 0.0

        if self.w_consist > 0 and result_aug is not None:
            if active_task is None:
                l_consist = (
                    self._partial_l1(
                        result.brown_mask,
                        result_aug.brown_mask.detach(),
                        face_mask,
                        has_brown,
                    )
                    + self._partial_l1(
                        result.red_mask,
                        result_aug.red_mask.detach(),
                        face_mask,
                        has_red,
                    )
                    + self._partial_l1(
                        result.wrinkle_mask,
                        result_aug.wrinkle_mask.detach(),
                        face_mask,
                        has_wrinkle,
                    )
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
        result_clean_for_loss = (
            _skin_result_to_fp32(result_clean) if result_clean is not None else None
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
            l_freq = _v2_freq_reg(model).float()
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


@torch.inference_mode()
def validate_fast(model, loader, criterion, device, cfg):
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
            active_task=None,
        )

        total_loss += loss.item()
        for k in log:
            log[k] += detail.get(k, 0.0)

        brown_sum, brown_cnt = _dice_sum_and_count(
            result_for_loss.brown_mask,
            brown_gt.float(),
            mask.float(),
            batch['has_brown'],
            threshold=cfg.get('dice_threshold', 0.5),
        )
        red_sum, red_cnt = _dice_sum_and_count(
            result_for_loss.red_mask,
            red_gt.float(),
            mask.float(),
            batch['has_red'],
            threshold=cfg.get('dice_threshold', 0.5),
        )
        wrinkle_sum, wrinkle_cnt = _dice_sum_and_count(
            result_for_loss.wrinkle_mask,
            wrinkle_gt.float(),
            mask.float(),
            batch['has_wrinkle'],
            threshold=cfg.get('dice_threshold', 0.5),
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
    out['score'] = _weighted_best_score(out, cfg)
    return out


def main():
    parser = argparse.ArgumentParser(
        description='SkinAnalyzerV2 training V9 standalone (task-sampled multi-task)'
    )
    parser.add_argument(
        '--patch_dir',
        type=str,
        default=CFG['patch_dir'],
        help='patch dataset directory (e.g. patches_v3)',
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
        default='brown=0.15,red=0.30,wrinkle=0.55',
        help='task sampling probabilities, e.g. brown=0.20,red=0.25,wrinkle=0.55',
    )
    parser.add_argument(
        '--disable_task_sampling',
        action='store_true',
        help='disable task sampling and train all tasks jointly',
    )
    args = parser.parse_args()
    CFG['patch_dir'] = args.patch_dir
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
    print("Version : V9 standalone task-sampled multi-task")
    print(f"  - task_sampling_enabled: {CFG['task_sampling_enabled']}")
    print(f"  - task_probs: {CFG['task_probs']}")
    print(f"  - weighted_sampler: {CFG['use_weighted_sampler']}")
    print(f"  - patch_dir: {CFG['patch_dir']}")
    print(f"  - wrinkle_min_pos_ratio: {CFG['wrinkle_min_pos_ratio']}")
    print(f"  - wrinkle_tversky: alpha={CFG['wrinkle_tversky_alpha']}, beta={CFG['wrinkle_tversky_beta']}")
    print(f"  - wrinkle_focal: gamma={CFG['wrinkle_focal_gamma']}, alpha={CFG['wrinkle_focal_alpha']}")
    print(f"  - wrinkle_gt_dilation: {CFG['wrinkle_gt_dilation']} px")
    print(f"  - wrinkle_solo_epochs: {CFG.get('wrinkle_solo_epochs', 0)}")

    ckpt_dir = Path(CFG['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = build_analyzer_v2(
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
        collate_fn=skin_collate_fn,
    )

    val_loader = DataLoader(
        val_ds,
        CFG['batch_size'],
        shuffle=False,
        num_workers=CFG['num_workers'],
        pin_memory=True,
        persistent_workers=CFG['num_workers'] > 0,
        prefetch_factor=prefetch_factor,
        collate_fn=skin_collate_fn,
    )
    print(f"Train: {len(train_ds)} / Val: {len(val_ds)}")

    criterion = TaskAwareSkinAnalyzerLossV9(
        w_brown=CFG['w_brown'],
        w_red=CFG['w_red'],
        w_wrinkle=CFG['w_wrinkle'],
        w_recon=CFG['w_recon'],
        red_focal_alpha=CFG.get('red_focal_alpha', 0.75),
        red_gt_dilation=CFG.get('red_gt_dilation', 0),
        red_area_weight=CFG['red_area_weight'],
        red_outside_weight=CFG['red_outside_weight'],
        wrinkle_tversky_alpha=CFG['wrinkle_tversky_alpha'],
        wrinkle_tversky_beta=CFG['wrinkle_tversky_beta'],
        wrinkle_focal_gamma=CFG['wrinkle_focal_gamma'],
        wrinkle_focal_alpha=CFG['wrinkle_focal_alpha'],
        wrinkle_edge_weight=CFG['wrinkle_edge_weight'],
        wrinkle_gt_dilation=CFG['wrinkle_gt_dilation'],
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

    solo_epochs = CFG.get('wrinkle_solo_epochs', 0)

    for epoch in range(start_epoch, CFG['epochs'] + 1):
        weights = get_loss_weights(
            epoch,
            CFG['epochs'],
            w_brown=CFG['w_brown'],
            w_red=CFG['w_red'],
            w_wrinkle=CFG['w_wrinkle'],
            w_recon=CFG['w_recon'],
        )
        for k, v in weights.items():
            setattr(criterion, k, v)

        # Curriculum Tversky: start with symmetric Dice (beta=0.5) and linearly
        # ramp to the target beta=0.7 over the first 30 epochs.  Jumping straight
        # to high beta causes excessive FP in early training when the model hasn't
        # yet learned what a wrinkle looks like.
        tversky_beta_target = CFG['wrinkle_tversky_beta']
        tversky_ramp = CFG.get('wrinkle_tversky_ramp_epochs', max(CFG['epochs'] // 3, 1))
        criterion.wrinkle_tversky_beta = 0.5 + (tversky_beta_target - 0.5) * min(
            epoch / tversky_ramp, 1.0
        )

        # Wrinkle-solo warmup: suppress brown/red loss for the first N epochs so
        # the wrinkle head receives undiluted gradient before competing tasks
        # dominate the shared encoder.
        in_solo = 0 < epoch <= solo_epochs
        if in_solo:
            criterion.w_brown = 0.0
            criterion.w_red = 0.0
            criterion.w_recon = 0.0
            epoch_cfg = {**CFG, 'task_probs': {'brown': 0.0, 'red': 0.0, 'wrinkle': 1.0}}
        else:
            epoch_cfg = CFG

        train_log = train_one_epoch_task_sampled(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            epoch_cfg,
            epoch,
            CFG['epochs'],
        )

        # Restore full task weights before validation so val metrics are always
        # comparable across all epochs (solo epoch training zeros w_brown/w_red).
        if in_solo:
            for k, v in weights.items():
                setattr(criterion, k, v)

        # Validation remains all-task, so val metrics are comparable to V7/V8.
        val_log = validate_fast(model, val_loader, criterion, device, CFG)
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
