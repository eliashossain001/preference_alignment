#!/usr/bin/env python3

# run_dpo.py

"""
SAVe V.1 – Proxy Verification Experiment Runner

This script runs *data-layer* experiments only (NO LLM API required):
- Simulates k-of-n consensus with proxy verifier accuracy p
- Injects preference corruption (label flip, over-refusal poisoning)
- Measures retention and effective corruption rate (eta')

Output: CSV suitable for paper tables & plots
"""

import argparse
import json
import random
import os
import pandas as pd
from pathlib import Path
from typing import List

# =========================
# Utilities
# =========================

def parse_float_list(s):
    if s is None:
        return None
    return [float(x) for x in s.split(",")]

def parse_int_list(s):
    if s is None:
        return None
    return [int(x) for x in s.split(",")]

def load_feedback(path):
    feedback = []
    with open(path, "r") as f:
        for line in f:
            feedback.append(json.loads(line))
    return feedback

# =========================
# Corruption models
# =========================

def apply_corruption(feedback, eta, corruption_type):
    """
    Mark a fraction eta of preferences as 'bad'
    """
    corrupted = []
    n = len(feedback)
    num_bad = int(eta * n)
    bad_indices = set(random.sample(range(n), num_bad))

    for i, fb in enumerate(feedback):
        fb = fb.copy()
        fb["is_bad"] = False

        if i in bad_indices:
            fb["is_bad"] = True
            if corruption_type == "label_flip":
                fb["user_choice"] = "A" if fb["user_choice"] == "B" else "B"
            elif corruption_type == "poison_overrefusal":
                # treated as bad supervision without changing label
                pass

        corrupted.append(fb)
    return corrupted

# =========================
# Proxy k-of-n verification
# =========================

def proxy_consensus_accept(is_bad, k, n_agents, proxy_p):
    """
    Each agent independently approves with probability:
      - proxy_p if clean
      - (1 - proxy_p) if bad
    """
    approvals = 0
    for _ in range(n_agents):
        if is_bad:
            approve = random.random() < (1 - proxy_p)
        else:
            approve = random.random() < proxy_p
        approvals += int(approve)
    return approvals >= k

# =========================
# Single experiment
# =========================

def run_one_setting(feedback, eta, k, proxy_p, corruption, exp_name):
    n_agents = 4

    corrupted = apply_corruption(feedback, eta, corruption)

    total_retained = 0
    bad_retained = 0

    for fb in corrupted:
        accepted = proxy_consensus_accept(
            is_bad=fb["is_bad"],
            k=k,
            n_agents=n_agents,
            proxy_p=proxy_p
        )
        if accepted:
            total_retained += 1
            if fb["is_bad"]:
                bad_retained += 1

    retained_pct = total_retained / len(corrupted) if corrupted else 0.0
    eta_prime = (bad_retained / total_retained) if total_retained > 0 else 0.0

    return {
        "eta": eta,
        "k": k,
        "proxy_p": proxy_p,
        "corruption": corruption,
        "retained_pct": retained_pct,
        "eta_prime": eta_prime,
        "bad_retained": bad_retained,
        "total_retained": total_retained,
        "dataset": exp_name,
    }

# =========================
# Main
# =========================

def main(args):
    random.seed(42)

    feedback = load_feedback(args.feedback_path)
    print(f"[Loaded] {len(feedback)} feedback items\n")

    etas = parse_float_list(args.etas) or [0.0]
    ks = parse_int_list(args.ks) or [3]
    corruptions = args.corruptions.split(",") if args.corruptions else ["none"]

    # proxy p handling (single or sweep)
    if args.proxy_ps:
        proxy_ps = parse_float_list(args.proxy_ps)
    elif args.proxy_p is not None:
        proxy_ps = [args.proxy_p]
    else:
        proxy_ps = [0.8]

    results = []

    print("=== Running SAVe proxy experiments ===\n")

    for eta in etas:
        for k in ks:
            for proxy_p in proxy_ps:
                for corruption in corruptions:
                    summary = run_one_setting(
                        feedback=feedback,
                        eta=eta,
                        k=k,
                        proxy_p=proxy_p,
                        corruption=corruption,
                        exp_name=args.exp_name
                    )
                    results.append(summary)

    df = pd.DataFrame(results)

    out_dir = Path("experiments/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{args.exp_name}.csv"
    df.to_csv(out_path, index=False)

    print("=== Experiment complete ===")
    print(df.head())
    print(f"\n[Saved] {out_path}")

# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--feedback_path", type=str, required=True)
    parser.add_argument("--etas", type=str, default="0.0")
    parser.add_argument("--ks", type=str, default="3")
    parser.add_argument("--corruptions", type=str, default="none")

    parser.add_argument("--proxy_p", type=float, default=None,
                        help="Single proxy verifier accuracy")
    parser.add_argument("--proxy_ps", type=str, default=None,
                        help="Comma-separated proxy p sweep")

    parser.add_argument("--exp_name", type=str, default="save_proxy")

    args = parser.parse_args()
    main(args)
