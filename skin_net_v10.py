"""
skin_net_v10.py
===============
SkinAnalyzerV10 experiment.

Only the wrinkle/parallel branch is changed:

- cross/brown/red path: same as SkinAnalyzerV2
- parallel/wrinkle path: ParallelPolEncoderV10
  - identity parallel preprocess
  - parallel high_r default lowered to 0.30

This is an ablation to test whether InstanceNorm + 1x1 Conv preprocessing in the
parallel branch suppresses wrinkle contrast.
"""

from __future__ import annotations

from parallel_encoder_v10 import ParallelPolEncoderV10
from skin_net_v2 import SkinAnalyzerV2


class SkinAnalyzerV10(SkinAnalyzerV2):
    """SkinAnalyzerV2 with experimental parallel encoder for wrinkle learning."""

    def __init__(
        self,
        base_ch: int = 64,
        low_r: float = 0.1,
        high_r: float = 0.4,
        parallel_high_r: float = 0.30,
    ):
        # Keep cross encoder / hue branch / heads identical to V2.
        super().__init__(base_ch=base_ch, low_r=low_r, high_r=high_r)
        # Replace only the parallel encoder after V2 initialization.
        self.parallel_encoder = ParallelPolEncoderV10(
            base_ch=base_ch,
            low_r=low_r,
            high_r=parallel_high_r,
        )
        self.parallel_high_r = parallel_high_r


def build_analyzer_v10(
    base_ch: int = 64,
    low_r: float = 0.1,
    high_r: float = 0.4,
    parallel_high_r: float = 0.30,
) -> SkinAnalyzerV10:
    return SkinAnalyzerV10(
        base_ch=base_ch,
        low_r=low_r,
        high_r=high_r,
        parallel_high_r=parallel_high_r,
    )
