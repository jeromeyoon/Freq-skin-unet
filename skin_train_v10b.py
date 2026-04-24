"""
skin_train_v10b.py
==================
Wrinkle ablation B:

- keep parallel preprocess identical to V9:
    InstanceNorm + 1x1 Conv
- change only parallel high_r:
    0.40 -> 0.30

Goal
----
Isolate whether the V10 regression came from lowering the wrinkle branch
frequency cutoff, while keeping the original V9 preprocess intact.
"""

from __future__ import annotations

import copy

import skin_train_v9 as v9
from skin_net_v10b import build_analyzer_v10b


CFG = copy.deepcopy(v9.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v10b',
    preview_dir='./previews_v10b',
    experiment='v10b_parallel_high_r_only',
    parallel_high_r=0.30,
)


def _build_v10b(base_ch: int, low_r: float, high_r: float):
    return build_analyzer_v10b(
        base_ch=base_ch,
        low_r=low_r,
        high_r=high_r,
        parallel_high_r=CFG['parallel_high_r'],
    )


def main() -> None:
    print("Version : V10B wrinkle ablation")
    print("  - parallel preprocess: V9 original (InstanceNorm+1x1)")
    print(f"  - parallel high_r: {CFG['parallel_high_r']} (V9 uses {CFG['high_r']})")
    print("  - train loop/loss/sampler: V9")

    v9.CFG = CFG
    v9.build_analyzer_v2 = _build_v10b
    v9.main()


if __name__ == '__main__':
    main()
