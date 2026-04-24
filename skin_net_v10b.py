"""
skin_net_v10b.py
================
SkinAnalyzerV10B experiment.

Only the wrinkle/parallel branch is changed:

- preprocess: same as V9 (InstanceNorm + 1x1 Conv)
- parallel high_r: lowered independently from the cross branch

Purpose
-------
Isolate whether the V10 regression came from lowering the wrinkle branch
high-frequency cutoff, while keeping the original V9 preprocess intact.
"""

from __future__ import annotations

from parallel_encoder import ParallelPolEncoder
from skin_net_v2 import SkinAnalyzerV2


class SkinAnalyzerV10B(SkinAnalyzerV2):
    """SkinAnalyzerV2 with only the parallel branch high_r changed."""

    def __init__(
        self,
        base_ch: int = 64,
        low_r: float = 0.1,
        high_r: float = 0.4,
        parallel_high_r: float = 0.30,
    ):
        super().__init__(base_ch=base_ch, low_r=low_r, high_r=high_r)
        self.parallel_encoder = ParallelPolEncoder(
            base_ch=base_ch,
            low_r=low_r,
            high_r=parallel_high_r,
        )
        self.parallel_high_r = parallel_high_r


def build_analyzer_v10b(
    base_ch: int = 64,
    low_r: float = 0.1,
    high_r: float = 0.4,
    parallel_high_r: float = 0.30,
) -> SkinAnalyzerV10B:
    return SkinAnalyzerV10B(
        base_ch=base_ch,
        low_r=low_r,
        high_r=high_r,
        parallel_high_r=parallel_high_r,
    )
