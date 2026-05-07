"""Like consensus_filter.py but reads precomputed both-direction verifier scores.

Given a corrupted file with `is_clean` and `_corruption` metadata, plus a precomputed
file holding `verifier_results_a` and `verifier_results_b` for the SAME row ids,
look up the correct direction per row (using current user_choice) and apply
raw / single / consensus_k3 / oracle filters.

Usage:
    python -m src.consensus_filter_precomputed \
        --corrupted data/corrupted/hh_train_struct_eta20.jsonl \
        --precomputed results/verifier_outputs/hh_train.precomputed.jsonl \
        --outdir data/filtered/hh_train_struct_eta20 \
        --rejected_dir results/verifier_outputs/hh_train_struct_eta20 \
        --k 3 --single_verifier safety
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


def to_dpo(row: Dict) -> Dict:
    chosen = row["response_a"] if row["user_choice"] == "A" else row["response_b"]
    rejected = row["response_b"] if row["user_choice"] == "A" else row["response_a"]
    return {"prompt": row["prompt"], "chosen": chosen, "rejected": rejected}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrupted", required=True)
    ap.add_argument("--precomputed", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--rejected_dir", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--single_verifier", default="safety")
    args = ap.parse_args()

    corrupt_rows = [json.loads(l) for l in open(args.corrupted) if l.strip()]
    pre = {}
    for line in open(args.precomputed):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        pre[r["id"]] = r

    # Attach the right direction's verifier results to each corrupted row
    for r in corrupt_rows:
        prec = pre[r["id"]]
        if r["user_choice"] == "A":
            r["verifier_results"] = prec["verifier_results_a"]
        else:
            r["verifier_results"] = prec["verifier_results_b"]

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    rdir = Path(args.rejected_dir); rdir.mkdir(parents=True, exist_ok=True)

    def n_pass(row):
        return sum(1 for v in row["verifier_results"].values() if v.get("passed"))

    methods = {
        "raw":        corrupt_rows,
        "single":     [r for r in corrupt_rows if r["verifier_results"][args.single_verifier]["passed"]],
        f"consensus_k{args.k}": [r for r in corrupt_rows if n_pass(r) >= args.k],
        "oracle":     [r for r in corrupt_rows if r.get("is_clean", True)],
    }

    summary = {}
    for name, kept in methods.items():
        kept_ids = {id(r) for r in kept}
        rejected = [r for r in corrupt_rows if id(r) not in kept_ids]

        out_path = outdir / f"{name}.train.jsonl"
        with open(out_path, "w") as f:
            for r in kept:
                f.write(json.dumps(to_dpo(r), ensure_ascii=False) + "\n")
        meta_path = outdir / f"{name}.train.full.jsonl"
        with open(meta_path, "w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rej_path = rdir / f"{name}.rejected.jsonl"
        with open(rej_path, "w") as f:
            for r in rejected:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_total = len(corrupt_rows)
        n_kept = len(kept); n_rej = len(rejected)
        kept_clean = sum(1 for r in kept if r.get("is_clean", True))
        kept_corrupt = n_kept - kept_clean
        rej_clean = sum(1 for r in rejected if r.get("is_clean", True))
        rej_corrupt = n_rej - rej_clean
        n_corrupt_total = sum(1 for r in corrupt_rows if not r.get("is_clean", True))
        n_clean_total = n_total - n_corrupt_total

        det_p = rej_corrupt / n_rej if n_rej else 0.0
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
