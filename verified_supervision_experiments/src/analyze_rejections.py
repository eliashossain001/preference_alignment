"""Rejection-set analysis (paper Table 3).

Given a rejection-set jsonl produced by consensus_filter.py, classify each rejected
pair into one of:

  - truly_corrupted          : ground truth is_clean=False
  - clean_but_low_consensus  : is_clean=True but consensus rejected (low overall
                                verifier agreement; potential false rejection)
  - clean_with_minority      : is_clean=True AND at least one verifier passed
                                (verifier disagreement; minority opinion got outvoted)
  - clean_unanimous          : is_clean=True AND zero verifiers passed
                                (all verifiers think this is bad; either a real
                                problem the corruption tag missed, or correlated
                                verifier failure)

These map to the paper's rows: truly corrupted vs clean-ambiguous vs
clean-but-valid vs verifier-mistake vs unknown/manual-review.

Usage:
    python -m src.analyze_rejections \
        --rejected results/verifier_outputs/hh_train_struct_eta20/consensus_k3.rejected.jsonl \
        --out_csv results/tables/rejection_breakdown.csv \
        --tag struct_eta20_consensus_k3
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def classify(row: dict) -> str:
    is_clean = row.get("is_clean", True)
    res = row.get("verifier_results", {})
    n_pass = sum(1 for r in res.values() if r.get("passed"))
    n_total = len(res) or 1
    if not is_clean:
        return "truly_corrupted"
    if n_pass == 0:
        # All verifiers failed; either a genuinely bad pair the corruption tag
        # missed or a correlated-failure case.
        return "clean_unanimous_reject"
    if n_pass < n_total / 2:
        return "clean_low_support"
    return "clean_with_minority_support"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejected", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n_examples_per_class", type=int, default=3,
                    help="qualitative examples to dump alongside")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.rejected) if line.strip()]
    counts = {}
    examples = {}
    for r in rows:
        cls = classify(r)
        counts[cls] = counts.get(cls, 0) + 1
        examples.setdefault(cls, []).append(r)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["tag", "class", "count", "fraction"])
        n = max(1, len(rows))
        for cls, c in sorted(counts.items()):
            w.writerow([args.tag, cls, c, f"{c/n:.4f}"])

    # Qualitative dump for manual review
    qual_path = out.with_name(out.stem + f"_{args.tag}_examples.jsonl")
    with open(qual_path, "w") as f:
        for cls, items in examples.items():
            for ex in items[: args.n_examples_per_class]:
                f.write(json.dumps({"class": cls, "tag": args.tag, "row": ex}, ensure_ascii=False) + "\n")

    print(f"[rejected total] {len(rows)}")
    for cls, c in sorted(counts.items()):
        print(f"  {cls:35s}: {c} ({c/max(1,len(rows)):.1%})")
    print(f"[wrote] {out}")
    print(f"[wrote] {qual_path}")


if __name__ == "__main__":
    main()
