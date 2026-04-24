"""
skin_train_v11.py
=================
Red task fix based on skin_train_v9.

Diagnosis (from V9 intermediate results: R_dice=0.414, red spreading face-wide)
------------------------------------------------------------------------
Three compounding issues caused red predictions to spread across the whole face:

  1. red_focal_alpha=0.75 → negative pixels received only 25% of gradient budget.
     With background contributing so little gradient, the model had almost no
     incentive to suppress false positives outside the red lesion.

  2. red_gt_dilation=0 → raw GT used directly.
     Specialist-drawn (curr) boundaries can be 1-2 px tight.  Every pixel outside
     the GT polygon generates a loss spike, making boundary regions noisy and
     preventing stable convergence near lesion edges.

  3. Red curriculum suppression → w_red × 0.8 for the first 20 % of epochs.
     During the most plastic phase of training, red competed at 80 % weight while
     brown ran at 140 %.  Brown won the feature-space competition early and red
     never fully recovered.

Changes vs V9
-------------
  red_focal_alpha : 0.75 → 0.60   negative gradient budget 25 % → 40 %
  red_gt_dilation : 0   → 1 px    soft label reduces boundary loss spikes
  red curriculum  : 0.8 × removed  red at full w_red from epoch 1

Not changed (V9 already fixed these)
-------------------------------------
  wrinkle_focal_alpha = 0.80  (was 0.97, fixed in V9 commit)
  sampler_neg_weight  = 0.10  (was 0.02, fixed in V9 commit)
  red_outside_weight  = 0.30  (was 0.12, fixed in V9 commit)
  Task-sampled training loop, wrinkle Tversky ramp, wrinkle solo warmup — all V9.

Ablation limitation
-------------------
V11 changes three variables simultaneously.  To isolate causes:
  V11a: only red_focal_alpha changed
  V11b: only red_gt_dilation changed
  V11c: only curriculum changed
If R_dice improves, a follow-up single-variable run can identify the dominant fix.
"""
from __future__ import annotations

import copy

import skin_train_v9 as v9
from skin_loss_smooth import get_loss_weights as _base_get_loss_weights


# ── V11 configuration ──────────────────────────────────────────────────────────
CFG = copy.deepcopy(v9.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v11',
    preview_dir='./previews_v11',
    experiment='v11_red_fix',

    # Red-specific fixes
    red_focal_alpha=0.60,       # V9: 0.75  — less positive bias, more FP penalty
    red_gt_dilation=1,          # V9: 0     — 1-px soft label for spot boundaries
)


# ── Curriculum override ────────────────────────────────────────────────────────
def _get_loss_weights_v11(
    epoch: int,
    total_epochs: int,
    w_brown: float = 1.0,
    w_red: float = 1.0,
    w_wrinkle: float = 1.0,
    w_recon: float = 0.3,
) -> dict:
    """V9 curriculum with early red suppression removed.

    V9 base scheduler:
      epoch  0-20 %: w_red × 0.80   ← REMOVED in V11
      epoch 20-50 %: w_red 0.80→1.0 ← REMOVED; V11 uses w_red × 1.0 throughout

    Rationale: red lost the feature-space competition to brown in early epochs
    because brown ran at 1.4× while red was suppressed to 0.8×.  Equalising
    weights from epoch 1 gives red a fair start.
    """
    weights = _base_get_loss_weights(epoch, total_epochs, w_brown, w_red, w_wrinkle, w_recon)
    progress = epoch / max(total_epochs, 1)
    if progress < 0.50:
        # Override: keep red at full base weight (no 0.8× or 0.8→1.0 ramp)
        weights['w_red'] = w_red
    return weights


# ── Entry point ────────────────────────────────────────────────────────────────
def main() -> None:
    print("Version : V11 red-task fix")
    print(f"  red_focal_alpha : {CFG['red_focal_alpha']}  (V9: 0.75)")
    print(f"  red_gt_dilation : {CFG['red_gt_dilation']} px  (V9: 0 px)")
    print(f"  red curriculum  : no early suppression  (V9: 0.8× for first 20 %)")
    print(f"  everything else : identical to V9")

    # Monkey-patch V9 module symbols so the existing training loop uses V11 settings.
    # NOTE: these patches persist for the lifetime of this process.
    v9.CFG = CFG
    v9.get_loss_weights = _get_loss_weights_v11
    v9.main()


if __name__ == '__main__':
    main()
