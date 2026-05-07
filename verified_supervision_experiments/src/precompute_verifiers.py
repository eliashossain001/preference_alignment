"""Score every (prompt, response_a) and (prompt, response_b) ONCE so that
downstream consensus filters at any (corruption, eta) just look up scores.

Uses BatchedLLMBackend so the four role prompts for a single (prompt, response)
are processed in a single forward pass (4x speedup over the sequential backend).

Each output row carries:
  verifier_results_a : {role: {score, passed, rationale, ...}}  (response_a as chosen)
  verifier_results_b : {role: {score, passed, rationale, ...}}  (response_b as chosen)

Sharded by row index so two GPUs can run in parallel.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from .batched_backend import BatchedLLMBackend
from .verifiers import (
    FactualityVerifier,
    HelpfulnessVerifier,
    PolicyVerifier,
    SafetyVerifier,
    SYSTEM_TEMPLATE,
)


VERIFIER_CLASSES = [SafetyVerifier, HelpfulnessVerifier, FactualityVerifier, PolicyVerifier]
ROLE_NAMES = [c.role_name.lower() for c in VERIFIER_CLASSES]


def build_messages_for_chosen(prompt: str, chosen: str):
    """Return four message-lists, one per verifier role, for a single (prompt, chosen) pair."""
    out = []
    for cls in VERIFIER_CLASSES:
        sys_prompt = SYSTEM_TEMPLATE.format(role_instruction=cls.role_instruction)
        user = (
            f"Prompt: {prompt}\n\n"
            f"Chosen response: {chosen}\n\n"
            f"Score the preference along the {cls.role_name} dimension on [0,1].\n"
            "0.0 = clearly unacceptable, 1.0 = clearly acceptable. "
            'Return one JSON object: {"score": <float>, "rationale": "<short>"}'
        )
        out.append([
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ])
    return out


def parse_outputs(generations: List[str], threshold: float) -> Dict[str, Dict]:
    res = {}
    for gen, name in zip(generations, ROLE_NAMES):
        try:
            data = BatchedLLMBackend.extract_json(gen)
            score = max(0.0, min(1.0, float(data.get("score", 0.5))))
            rationale = str(data.get("rationale", ""))[:240]
        except Exception as e:
            score, rationale = 0.5, f"parse_error: {e}"
        res[name] = {
            "score": score,
            "passed": bool(score >= threshold),
            "rationale": rationale,
            "threshold": threshold,
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--shard_total", type=int, default=1)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    if args.shard_total > 1:
        rows = [r for i, r in enumerate(rows) if i % args.shard_total == args.shard_index]
        print(f"[shard {args.shard_index}/{args.shard_total}] keeping {len(rows)} rows")
    if args.limit:
        rows = rows[: args.limit]

    print(f"[precompute] {len(rows)} rows; will batch 4 verifiers x 2 directions per row")

    print(f"[precompute] loading {args.model_name} (max_new_tokens={args.max_new_tokens})")
    backend = BatchedLLMBackend(model_name=args.model_name, max_new_tokens=args.max_new_tokens)

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(outp, "w") as fout:
        for i, row in enumerate(rows):
            # Direction A: response_a is chosen
            msgs_a = build_messages_for_chosen(row["prompt"], row["response_a"])
            outs_a = backend.generate_batch(msgs_a)
            row["verifier_results_a"] = parse_outputs(outs_a, args.threshold)
            for k in row["verifier_results_a"].values():
                k["model_name"] = args.model_name

            # Direction B: response_b is chosen
            msgs_b = build_messages_for_chosen(row["prompt"], row["response_b"])
            outs_b = backend.generate_batch(msgs_b)
            row["verifier_results_b"] = parse_outputs(outs_b, args.threshold)
            for k in row["verifier_results_b"].values():
                k["model_name"] = args.model_name

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % args.log_every == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(rows) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(rows)}] {rate:.2f} rows/s   eta {eta/60:.1f} min")

    print(f"[wrote] {outp}")
    print(f"[done] {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
