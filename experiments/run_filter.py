# experiments/run_filter.py

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, List

from agents.base import Feedback
from experiments.io_utils import ensure_dir, write_json, write_jsonl, read_jsonl
from feedback_pipeline.pipeline import FeedbackPipeline


def load_feedback_records(path: str) -> List[Feedback]:
    rows = read_jsonl(path)
    items: List[Feedback] = []

    for row in rows:
        fb = Feedback(
            prompt=row["prompt"],
            response_a=row["response_a"],
            response_b=row["response_b"],
            user_choice=row["user_choice"],
            id=row.get("id"),
            dataset=row.get("dataset"),
            split=row.get("split"),
            source_row=row.get("source_row"),
            is_clean=row.get("is_clean"),
        )
        items.append(fb)

    return items


def make_summary(
    dataset: str,
    split: str,
    mode: str,
    k: int,
    raw_pairs: int,
    kept_pairs: int,
    seed: int,
    proxy_p_correct: float | None,
    output_dir: str,
    start_time: float,
) -> Dict[str, Any]:
    retention_pct = (100.0 * kept_pairs / raw_pairs) if raw_pairs > 0 else 0.0
    trainable = kept_pairs > 0

    return {
        "dataset": dataset,
        "split": split,
        "mode": mode,
        "k": k,
        "raw_pairs": raw_pairs,
        "kept_pairs": kept_pairs,
        "retention_pct": retention_pct,
        "trainable": trainable,
        "seed": seed,
        "proxy_p_correct": proxy_p_correct,
        "output_dir": output_dir,
        "hostname": socket.gethostname(),
        "runtime_sec": round(time.time() - start_time, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input feedback jsonl")
    ap.add_argument("--output_dir", required=True, help="Directory to save outputs")
    ap.add_argument("--dataset", required=True, help="Dataset short name")
    ap.add_argument("--split", default="train", help="Split name")
    ap.add_argument("--k", type=int, required=True, help="Min number of passing verifiers")
    ap.add_argument("--mode", choices=["proxy", "llm"], default="proxy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proxy_p_correct", type=float, default=0.8)

    args = ap.parse_args()
    start_time = time.time()

    ensure_dir(args.output_dir)

    config = {
        "input": args.input,
        "output_dir": args.output_dir,
        "dataset": args.dataset,
        "split": args.split,
        "k": args.k,
        "mode": args.mode,
        "seed": args.seed,
        "proxy_p_correct": args.proxy_p_correct if args.mode == "proxy" else None,
    }
    write_json(str(Path(args.output_dir) / "config.json"), config)

    feedback = load_feedback_records(args.input)

    if args.mode == "proxy":
        pipeline = FeedbackPipeline(
            mode="proxy",
            proxy_p_correct=args.proxy_p_correct,
            seed=args.seed,
        )
    else:
        pipeline = FeedbackPipeline(
            mode="llm",
            seed=args.seed,
        )

    accepted_feedback, full_results = pipeline.run(feedback, min_pass=args.k)

    retained_records = [item for item in full_results if item["accepted"]]
    rejected_records = [item for item in full_results if not item["accepted"]]

    accepted_ids = [item["id"] for item in retained_records if item.get("id") is not None]
    rejected_ids = [item["id"] for item in rejected_records if item.get("id") is not None]

    write_jsonl(str(Path(args.output_dir) / "retained.jsonl"), retained_records)
    write_jsonl(str(Path(args.output_dir) / "rejected.jsonl"), rejected_records)
    write_jsonl(str(Path(args.output_dir) / "full_results.jsonl"), full_results)

    write_json(str(Path(args.output_dir) / "accepted_ids.json"), accepted_ids)
    write_json(str(Path(args.output_dir) / "rejected_ids.json"), rejected_ids)

    summary = make_summary(
        dataset=args.dataset,
        split=args.split,
        mode=args.mode,
        k=args.k,
        raw_pairs=len(feedback),
        kept_pairs=len(accepted_feedback),
        seed=args.seed,
        proxy_p_correct=args.proxy_p_correct if args.mode == "proxy" else None,
        output_dir=args.output_dir,
        start_time=start_time,
    )
    write_json(str(Path(args.output_dir) / "summary.json"), summary)

    print("[DONE] Filtering complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()