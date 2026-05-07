"""Apply k-of-n consensus filtering to a verifier-scored jsonl.

Given a file produced by run_verifiers.py, write four downstream training files:

  raw          - unchanged corrupted data (no filtering)
  single       - filtered by a single verifier (default: safety)
  consensus_kN - filtered by k-of-n consensus
  oracle       - filtered using the ground-truth is_clean flag (upper bound)

For each output the rejection-set is also written under
results/verifier_outputs/<base>.<method>.rejected.jsonl with verifier rationales,
which feeds analyze_rejections.py.

Usage:
    python -m src.consensus_filter \
        --input results/verifier_outputs/hh_train_struct_eta20.scored.jsonl \
        --outdir data/filtered/hh_train_struct_eta20 \
        --rejected_dir results/verifier_outputs/hh_train_struct_eta20 \
        --k 3 --single_verifier safety
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def to_dpo(row: Dict) -> Dict:
    """Strip metadata and emit a {prompt, chosen, rejected} record for DPO training."""
    chosen = row["response_a"] if row["user_choice"] == "A" else row["response_b"]
    rejected = row["response_b"] if row["user_choice"] == "A" else row["response_a"]
    return {"prompt": row["prompt"], "chosen": chosen, "rejected": rejected}


def filter_pass_count(row: Dict) -> int:
    res = row.get("verifier_results", {})
    return sum(1 for r in res.values() if r.get("passed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True, help="where filtered DPO jsonl files land")
    ap.add_argument("--rejected_dir", required=True, help="where rejection-set details land")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument(
        "--single_verifier",
        default="safety",
        help="which verifier acts as the single-verifier baseline filter",
    )
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.input) if line.strip()]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rdir = Path(args.rejected_dir)
    rdir.mkdir(parents=True, exist_ok=True)

    methods: Dict[str, List[Dict]] = {
        "raw": rows,
        "single": [r for r in rows if r["verifier_results"][args.single_verifier]["passed"]],
        f"consensus_k{args.k}": [r for r in rows if filter_pass_count(r) >= args.k],
        "oracle": [r for r in rows if r.get("is_clean", True)],
    }

    summary = {}
    for name, kept in methods.items():
        rejected = [r for r in rows if r not in kept]  # O(n^2) but n small
        # Faster: set of ids
        kept_ids = {id(r) for r in kept}
        rejected = [r for r in rows if id(r) not in kept_ids]

        # Filtered DPO file
        out_path = outdir / f"{name}.train.jsonl"
        with open(out_path, "w") as f:
            for r in kept:
                f.write(json.dumps(to_dpo(r), ensure_ascii=False) + "\n")

        # Full retained-set with metadata (for downstream analysis)
        meta_path = outdir / f"{name}.train.full.jsonl"
        with open(meta_path, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Rejection-set with metadata
        rej_path = rdir / f"{name}.rejected.jsonl"
        with open(rej_path, "w") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Reliability metrics computed against ground-truth is_clean
        n_total = len(rows)
        n_kept = len(kept)
        n_rej = len(rejected)
        kept_clean = sum(1 for r in kept if r.get("is_clean", True))
        kept_corrupt = n_kept - kept_clean
        rej_clean = sum(1 for r in rejected if r.get("is_clean", True))
        rej_corrupt = n_rej - rej_clean
        n_corrupt_total = sum(1 for r in rows if not r.get("is_clean", True))
        n_clean_total = n_total - n_corrupt_total

        # corruption-detection precision = corrupt rejected / total rejected
        det_p = rej_corrupt / n_rej if n_rej else 0.0
        # corruption-detection recall = corrupt rejected / total corrupt
        det_r = rej_corrupt / n_corrupt_total if n_corrupt_total else 0.0
        det_f1 = 2 * det_p * det_r / (det_p + det_r) if (det_p + det_r) > 0 else 0.0

        summary[name] = {
            "kept": n_kept,
            "rejected": n_rej,
            "retention": n_kept / n_total if n_total else 0.0,
            "kept_clean": kept_clean,
            "kept_corrupt": kept_corrupt,
            "harmful_survival_rate": kept_corrupt / n_kept if n_kept else 0.0,
            "clean_retention_rate": kept_clean / n_clean_total if n_clean_total else 0.0,
            "false_rejection_rate": rej_clean / n_clean_total if n_clean_total else 0.0,
            "corruption_detection_precision": det_p,
            "corruption_detection_recall": det_r,
            "corruption_detection_f1": det_f1,
            "out_path": str(out_path),
            "rej_path": str(rej_path),
        }

    with open(outdir / "filter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
