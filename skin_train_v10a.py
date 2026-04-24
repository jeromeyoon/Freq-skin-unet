"""
skin_train_v10a.py
==================
Wrinkle ablation A:

- change only parallel preprocess:
    InstanceNorm + 1x1 Conv -> Identity
- keep parallel high_r identical to V9:
    0.40

Goal
----
Isolate whether the V10 regression came from removing the preprocess stage
itself, while holding the frequency cutoff constant.
"""

from __future__ import annotations

import copy

import skin_train_v9 as v9
from skin_net_v10 import build_analyzer_v10


CFG = copy.deepcopy(v9.CFG)
CFG.update(
    checkpoint_dir='./checkpoints_v10a',
    preview_dir='./previews_v10a',
    experiment='v10a_parallel_identity_preprocess_only',
    parallel_high_r=0.40,  # keep same as V9 parallel high_r
)


def _build_v10a(base_ch: int, low_r: float, high_r: float):
    return build_analyzer_v10(
        base_ch=base_ch,
        low_r=low_r,
        high_r=high_r,
        parallel_high_r=CFG['parallel_high_r'],
    )


def main() -> None:
    print("Version : V10A wrinkle ablation")
    print("  - parallel preprocess: identity")
    print(f"  - parallel high_r: {CFG['parallel_high_r']} (same as V9)")
    print("  - train loop/loss/sampler: V9")

    v9.CFG = CFG
    v9.build_analyzer_v2 = _build_v10a
    v9.main()


if __name__ == '__main__':
    main()
