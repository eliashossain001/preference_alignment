"""Subsample HH-RLHF DPO splits to a fixed size with a fixed seed.

Reads from data/raw/, writes to data/processed/. The processed train file is the
canonical input to create_corruption.py; the dev/test files are held-out.

Usage:
    python -m src.load_datasets --train_size 10000 --dev_size 1000 --test_size 1000 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def load_jsonl(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def to_feedback(rows, dataset: str, split: str, seed: int):
    """DPO triple {prompt, chosen, rejected} -> Feedback {prompt, response_a, response_b, user_choice}.

    HH convention: chosen is the safer/preferred response. We map chosen->A, rejected->B,
    user_choice='A'. Random shuffle of A/B at the row level would obscure structured
    corruption semantics, so we keep the canonical orientation and let the corruption
    module flip user_choice when it wants to inject structure.
    """
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "id": f"{dataset}_{split}_{i:06d}",
                "dataset": dataset,
                "split": split,
                "source_row": i,
                "prompt": r["prompt"],
                "response_a": r["chosen"],
                "response_b": r["rejected"],
                "user_choice": "A",
                "is_clean": True,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_size", type=int, default=10000)
    ap.add_argument("--dev_size", type=int, default=1000)
    ap.add_argument("--test_size", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    train = load_jsonl(RAW / "dpo_hh_train.jsonl")
    val = load_jsonl(RAW / "dpo_hh_val.jsonl")

    rng.shuffle(train)
    train_sample = train[: args.train_size]

    rng.shuffle(val)
    n_dev = min(args.dev_size, max(0, len(val) - args.test_size))
    dev_sample = val[:n_dev]
    test_sample = val[n_dev : n_dev + args.test_size]

    write_jsonl(PROC / "hh_train.jsonl", to_feedback(train_sample, "hh", "train", args.seed))
    write_jsonl(PROC / "hh_dev.jsonl", to_feedback(dev_sample, "hh", "dev", args.seed))
    write_jsonl(PROC / "hh_test.jsonl", to_feedback(test_sample, "hh", "test", args.seed))

    print(f"[hh_train] {len(train_sample)} rows -> {PROC/'hh_train.jsonl'}")
    print(f"[hh_dev]   {len(dev_sample)} rows -> {PROC/'hh_dev.jsonl'}")
    print(f"[hh_test]  {len(test_sample)} rows -> {PROC/'hh_test.jsonl'}")


if __name__ == "__main__":
    main()
