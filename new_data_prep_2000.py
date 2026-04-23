"""
new_data_prep_2000.py
=====================
Wrapper for new_data_prep.py with "good 2000 patches" defaults.

Behavior
--------
- Delegates to new_data_prep.py
- If not explicitly provided, injects:
    --max_total_patches 2000
    --max_patches_per_subject 12

All other CLI arguments are passed through unchanged.
"""

from __future__ import annotations

import runpy
import sys


def _ensure_default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _ensure_default("--max_total_patches", "2000")
    _ensure_default("--max_patches_per_subject", "12")
    runpy.run_module("new_data_prep", run_name="__main__")
