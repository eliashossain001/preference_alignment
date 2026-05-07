"""Inject controlled preference corruption with full provenance metadata.

Three corruption regimes
------------------------
1. random_flip
   Flip user_choice with i.i.d. probability eta. Symmetric stochastic noise:
   does not bias the optimizer in any particular direction on average.

2. structured_unsafe
   Flip user_choice with probability eta, but only in the direction
   "prefer rejected over chosen". On HH-RLHF this is directional contamination:
   the rejected response is, by HH construction, the less safe / less helpful one,
   so structured_unsafe systematically biases the gradient toward unsafe completions.

3. coordinated_poisoning
   Like structured_unsafe but restricted to a "trigger subpopulation": prompts that
   match a small set of harm-relevant keywords (kill, weapon, hack, etc.). Models
   a coordinated attack that targets a specific behavioral class while leaving the
   rest of the dataset untouched. eta is interpreted as the within-trigger rate.

Each row of the output jsonl carries a `_corruption` metadata block with the keys
required by the paper's Table 3 and rejection-set analysis:

  original_chosen, original_rejected, corrupted_chosen, corrupted_rejected,
  corruption_type, eta, target_behavior, is_random, is_clean

`is_clean=True` means the row was NOT modified. `is_clean=False` means it was.

Usage
-----
    python -m src.create_corruption \
        --input data/processed/hh_train.jsonl \
        --output data/corrupted/hh_train_struct_eta20.jsonl \
        --corruption structured_unsafe --eta 0.20 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

# Trigger keywords for coordinated_poisoning; case-insensitive, whole-word-ish via
# substring match on lowercased prompt. Conservative list — calibrated to roughly
# 5-15% trigger rate on HH-RLHF.
COORD_TRIGGERS = [
    "kill", "killing", "murder", "weapon", "gun", "knife", "bomb",
    "hack", "hacking", "phish", "fraud", "scam", "steal", "stealing",
    "violence", "violent", "abuse", "assault", "torture", "poison",
    "exploit", "drugs", "illegal",
]


def is_trigger(prompt: str) -> bool:
    p = prompt.lower()
    return any(t in p for t in COORD_TRIGGERS)


def flip_choice(choice: str) -> str:
    return "B" if choice == "A" else "A"


def apply_corruption(rows: List[Dict], corruption: str, eta: float, seed: int):
    rng = random.Random(seed)
    out = []
    n_corrupted = 0
    n_eligible = 0

    for r in rows:
        # Snapshot originals BEFORE any mutation. user_choice='A' on HH means
        # response_a (chosen) is preferred.
        original_choice = r["user_choice"]
        original_chosen = r["response_a"] if original_choice == "A" else r["response_b"]
        original_rejected = r["response_b"] if original_choice == "A" else r["response_a"]

        is_clean = True
        target_behavior = "none"
        is_random = False

        if corruption == "none" or eta <= 0.0:
            pass

        elif corruption == "random_flip":
            n_eligible += 1
            is_random = True
            if rng.random() < eta:
                r["user_choice"] = flip_choice(r["user_choice"])
                is_clean = False
                target_behavior = "random"

        elif corruption == "structured_unsafe":
            # Only flip chosen->rejected direction (introduce unsafe preference).
            # On HH the canonical choice is 'A' = chosen = safer; flipping always
            # injects unsafe preference. We still gate by eta.
            if r["user_choice"] == "A":
                n_eligible += 1
                if rng.random() < eta:
                    r["user_choice"] = "B"
                    is_clean = False
                    target_behavior = "prefer_unsafe"

        elif corruption == "coordinated_poisoning":
            # Restrict to trigger prompts; flip the subpopulation toward unsafe at rate eta.
            if is_trigger(r["prompt"]) and r["user_choice"] == "A":
                n_eligible += 1
                if rng.random() < eta:
                    r["user_choice"] = "B"
                    is_clean = False
                    target_behavior = "coordinated_unsafe"

        else:
            raise ValueError(f"unknown corruption: {corruption}")

        # New "chosen" / "rejected" after possible flip.
        new_chosen = r["response_a"] if r["user_choice"] == "A" else r["response_b"]
        new_rejected = r["response_b"] if r["user_choice"] == "A" else r["response_a"]

        r["is_clean"] = is_clean
        r["_corruption"] = {
            "corruption_type": corruption,
            "eta": eta,
            "target_behavior": target_behavior,
            "is_random": is_random,
            "is_clean": is_clean,
            "trigger": is_trigger(r["prompt"]) if corruption == "coordinated_poisoning" else None,
            "original_chosen": original_chosen,
            "original_rejected": original_rejected,
            "corrupted_chosen": new_chosen,
            "corrupted_rejected": new_rejected,
        }
        if not is_clean:
            n_corrupted += 1
        out.append(r)

    summary = {
        "input_rows": len(rows),
        "eligible_rows": n_eligible,
        "corrupted_rows": n_corrupted,
        "effective_eta": (n_corrupted / n_eligible) if n_eligible else 0.0,
        "corruption": corruption,
        "nominal_eta": eta,
        "seed": seed,
    }
    return out, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--corruption",
        required=True,
        choices=["none", "random_flip", "structured_unsafe", "coordinated_poisoning"],
    )
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in open(inp) if line.strip()]
    rows, summary = apply_corruption(rows, args.corruption, args.eta, args.seed)

    with open(outp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = outp.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[wrote] {outp}")
    print(f"[wrote] {summary_path}")


if __name__ == "__main__":
    main()
