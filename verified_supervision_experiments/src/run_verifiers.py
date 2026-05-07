"""Run all four verifiers over a corrupted preference dataset.

Writes one jsonl line per input pair, with the original payload + a
`verifier_results` dict keyed by verifier role:

  {
    "id": ...,
    "prompt": ..., "response_a": ..., "response_b": ..., "user_choice": ...,
    "is_clean": ..., "_corruption": {...},
    "verifier_results": {
       "safety":      {"score": 0.91, "passed": true, "rationale": "..."},
       "helpfulness": {"score": 0.78, "passed": true, "rationale": "..."},
       "factuality":  {"score": 0.65, "passed": false, "rationale": "..."},
       "policy":      {"score": 0.88, "passed": true, "rationale": "..."}
    }
  }

Usage:
    python -m src.run_verifiers \
        --input data/corrupted/hh_train_struct_eta20.jsonl \
        --output results/verifier_outputs/hh_train_struct_eta20.scored.jsonl \
        --model_name Qwen/Qwen2.5-1.5B-Instruct \
        --threshold 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .verifiers import VERIFIER_REGISTRY

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.base import Feedback  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=None, help="cap rows (for smoke tests)")
    ap.add_argument("--log_every", type=int, default=100)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    print(f"[run_verifiers] loading {inp}")
    rows = [json.loads(line) for line in open(inp) if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[run_verifiers] {len(rows)} rows")

    print(f"[run_verifiers] loading verifiers (model={args.model_name}, threshold={args.threshold})")
    # All four verifiers share the same backend instance (singleton-cached in LocalLLMBackend)
    verifiers = {
        name: cls(model_name=args.model_name, threshold=args.threshold)
        for name, cls in VERIFIER_REGISTRY.items()
    }

    t0 = time.time()
    with open(outp, "w") as fout:
        for i, row in enumerate(rows):
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
            results = {}
            for name, v in verifiers.items():
                r = v.verify(fb)
                results[name] = {
                    "score": float(r.score),
                    "passed": bool(r.passed),
                    "rationale": str(r.rationale),
                    "threshold": v.threshold,
                    "model_name": v.model_name,
                }
            row["verifier_results"] = results
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % args.log_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(rows)}] {rate:.2f} rows/s   eta {eta/60:.1f} min")

    print(f"[wrote] {outp}")
    print(f"[done] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
