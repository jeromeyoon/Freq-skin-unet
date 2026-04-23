"""
parallel_encoder_v10.py
=======================
Experimental parallel-polarization encoder for wrinkle ablation.

Purpose
-------
V9 uses:

    RGB parallel -> InstanceNorm2d -> 1x1 Conv -> Conv encoder -> HighFreqGate

This V10 experiment removes the learned normalization/projection stage and feeds
raw parallel RGB directly into the encoder:

    RGB parallel -> Identity -> Conv encoder -> HighFreqGate

Why
---
Wrinkle labels are sparse and line-like. If InstanceNorm or learned 1x1 channel
mixing suppresses subtle wrinkle contrast, the wrinkle head may fail to learn.
This encoder isolates that hypothesis without changing the rest of the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from parallel_encoder import ParallelPolEncoder


class IdentityParallelPreprocess(nn.Module):
    """Pass parallel RGB through unchanged."""

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb


class ParallelPolEncoderV10(ParallelPolEncoder):
    """
    Parallel encoder ablation:

    - preprocess: identity
    - encoder/high-frequency gates: same as ParallelPolEncoder
    - recommended high_r: 0.30 for wrinkle mid/high-frequency preservation
    """

    def __init__(
        self,
        base_ch: int = 64,
        low_r: float = 0.1,
        high_r: float = 0.30,
    ):
        super().__init__(base_ch=base_ch, low_r=low_r, high_r=high_r)
        self.preprocess = IdentityParallelPreprocess()
