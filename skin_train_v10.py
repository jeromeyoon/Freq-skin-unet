"""
skin_train_v10.py
=================
Experimental wrinkle ablation based on skin_train_v9.

Hypothesis
----------
Wrinkle learning may be weakened by the parallel branch preprocessing:

    InstanceNorm2d -> 1x1 Conv

V10 keeps the V9 sparse/task-sampled training loop and loss, but swaps only the
model builder to SkinAnalyzerV10:

    parallel RGB -> Identity preprocess -> encoder -> HighFreqGate

The parallel high-frequency cutoff is also lowered to parallel_high_r=0.30 so
mid/high-frequency wrinkle structures are less likely to be discarded.

Ablation limitation
-------------------
V10 changes two variables simultaneously relative to V9:
  1. parallel preprocess: InstanceNorm+1x1 → Identity
  2. parallel high_r: 0.40 → 0.30

If wrinkle performance changes, isolate the cause by re-running with only one
variable changed (see parallel_high_r in CFG below).

Notes
-----
This wrapper intentionally reuses skin_train_v9's standalone training loop to
avoid duplicating 1000+ lines. Checkpoints are written to ./checkpoints_v10.
"""

from __future__ import annotations

import copy

import skin_train_v9 as v9
from skin_net_v10 import build_analyzer_v10


CFG = copy.deepcopy(v9.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v10',
    preview_dir='./previews_v10',
    experiment='v10_parallel_identity_preprocess_parallel_high_r_0.30',
    # V10-specific: parallel encoder high-r cutoff (lower = keeps more mid-freq).
    # V9 uses high_r=0.40 from CFG['high_r']. Store here for checkpoint traceability.
    parallel_high_r=0.30,
)


def _build_v10(base_ch: int, low_r: float, high_r: float):
    """Build V10 model, reading parallel_high_r from V10 CFG."""
    return build_analyzer_v10(
        base_ch=base_ch,
        low_r=low_r,
        high_r=high_r,
        parallel_high_r=CFG['parallel_high_r'],
    )


def main() -> None:
    print("Version : V10 wrinkle ablation")
    print(f"  - parallel preprocess: identity (V9 uses InstanceNorm+1x1)")
    print(f"  - parallel high_r: {CFG['parallel_high_r']} (V9 uses {CFG['high_r']})")
    print( "  - train loop/loss/sampler: V9 (all V9 fixes inherited)")

    # skin_train_v9 imports build_analyzer_v2 into its module namespace and uses
    # that symbol inside main(). Rebind it here so the V9 loop constructs V10.
    # NOTE: this monkey-patch persists for the lifetime of the process.
    v9.CFG = CFG
    v9.build_analyzer_v2 = _build_v10
    v9.main()


if __name__ == '__main__':
    main()
