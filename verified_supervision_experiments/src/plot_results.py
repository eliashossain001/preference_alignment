"""Generate Figures 1 and 2 from results/tables/main_results.csv.

Figure 1 (corruption-robustness curve): x=eta, y=harmful_survival_rate (filter side),
         curves=method.
Figure 2 (safety-coverage curve): x=retention, y=preference_acc (or unsafe_refusal_rate),
         points labelled by method/eta.

Figure 3 (verifier correlation heatmap) is produced inline by analyze_verifier_correlation.py.

Usage:
    python -m src.plot_results --out_dir results/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load(csv_path):
    p = Path(csv_path)
    if not p.exists():
        print(f"[plot_results] {csv_path} does not exist; skipping plots")
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def numeric(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "results" / "tables" / "main_results.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "figures"))
    args = ap.parse_args()

    rows = load(args.csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    # ---- Figure 1: harmful survival in the *training set after filtering* vs eta ----
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for m in methods:
        pts = [
            (int(r["eta"]) / 100.0, numeric(r["harmful_survival_rate"]))
            for r in rows
            if r["method"] == m and numeric(r["harmful_survival_rate"]) is not None
        ]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, [y * 100 for y in ys], marker="o", label=m, linewidth=2)
    ax.set_xlabel(r"Corruption rate $\eta$")
    ax.set_ylabel("Corrupted-pair survival in train set (%)")
    ax.set_title("Harmful survival vs. corruption rate (HH structured-unsafe)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    p1 = out / "fig1_harmful_survival_vs_eta.pdf"
    p1png = out / "fig1_harmful_survival_vs_eta.png"
    plt.savefig(p1, dpi=200); plt.savefig(p1png, dpi=200)
    plt.close()
    print(f"[wrote] {p1}")

    # ---- Figure 2: preference acc vs retention ----
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    markers = {"raw": "x", "single": "s", "consensus_k3": "o", "oracle": "^", "noise_aware": "d"}
    for m in methods:
        pts = []
        for r in rows:
            if r["method"] != m:
                continue
            ret = numeric(r["retention"])
            acc = numeric(r["preference_acc"])
            if ret is None or acc is None:
                continue
            pts.append((ret * 100, acc, int(r["eta"])))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; etas = [p[2] for p in pts]
        ax.scatter(xs, ys, marker=markers.get(m, "o"), s=60, label=m)
        for x, y, e in zip(xs, ys, etas):
            ax.annotate(f"η={e}", (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Train-set retention (%)")
    ax.set_ylabel("Preference accuracy on HH test")
    ax.set_title("Safety--coverage trade-off")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    p2 = out / "fig2_safety_coverage.pdf"
    p2png = out / "fig2_safety_coverage.png"
    plt.savefig(p2, dpi=200); plt.savefig(p2png, dpi=200)
    plt.close()
    print(f"[wrote] {p2}")


if __name__ == "__main__":
    main()
