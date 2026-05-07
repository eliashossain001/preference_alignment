"""Thin wrapper around train_dpo.py that turns on cDPO-style label smoothing.

This is the noise-aware DPO baseline used as the objective-only competitor in the
paper. cDPO sets a fixed label-smoothing parameter so the loss treats each pair as
"preferred with probability 1-eps", attenuating gradient magnitude on labels the
model is highly uncertain about. It is provably robust to bounded random noise but,
per the paper's structural argument, has no purchase on directional contamination.

Usage:
    python -m src.train_noise_aware_dpo \
        --train_file data/filtered/.../raw.train.jsonl \
        --output_dir results/checkpoints/.../noise_aware \
        --label_smoothing 0.1
"""
from __future__ import annotations

import sys

from . import train_dpo


def main():
    # Force a non-zero default label smoothing if the caller didn't set one.
    argv = sys.argv[1:]
    if "--label_smoothing" not in " ".join(argv):
        argv = argv + ["--label_smoothing", "0.1"]
    sys.argv = [sys.argv[0]] + argv
    train_dpo.main()


if __name__ == "__main__":
    main()
