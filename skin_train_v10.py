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

The parallel high-frequency cutoff is also lowered to 0.30 inside
SkinAnalyzerV10 so mid/high-frequency wrinkle structures are less likely to be
discarded.

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
)


def main() -> None:
    print("Version : V10 wrinkle ablation")
    print("  - parallel preprocess: identity")
    print("  - parallel high_r: 0.30")
    print("  - train loop/loss/sampler: V9")

    # skin_train_v9 imports build_analyzer_v2 into its module namespace and uses
    # that symbol inside main(). Rebind it here so the V9 loop constructs V10.
    v9.CFG = CFG
    v9.build_analyzer_v2 = build_analyzer_v10
    v9.main()


if __name__ == '__main__':
    main()
