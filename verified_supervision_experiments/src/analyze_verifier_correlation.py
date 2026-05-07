"""Verifier-correlation analysis (paper Table 4 + Figure 3 heatmap).

Computes pairwise:
  - agreement rate         (P[B_i = B_j])
  - shared false-positive  (P[B_i=1, B_j=1 | corrupted])
  - shared false-negative  (P[B_i=0, B_j=0 | clean])
  - phi correlation        (binary correlation coefficient)

and an overall effective-correlation proxy rho (closer to N when verifiers fail
together; closer to 1 when they fail independently).

Usage:
    python -m src.analyze_verifier_correlation \
        --scored results/verifier_outputs/hh_train_struct_eta20.scored.jsonl \
        --out_csv results/tables/verifier_correlation.csv \
        --out_heatmap results/figures/verifier_corr_heatmap_struct_eta20.pdf \
        --tag struct_eta20
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def phi(x: np.ndarray, y: np.ndarray) -> float:
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    return 0.0 if denom == 0 else (n11 * n00 - n10 * n01) / float(np.sqrt(denom))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_heatmap", default=None)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.scored) if l.strip()]
    if not rows:
        print("[warn] empty input")
        return

    names = sorted(rows[0]["verifier_results"].keys())
    n = len(rows)
    m = len(names)
    pass_mat = np.zeros((n, m), dtype=int)
    is_clean = np.array([1 if r.get("is_clean", True) else 0 for r in rows])
    is_corrupt = 1 - is_clean

    for i, r in enumerate(rows):
        for j, name in enumerate(names):
            pass_mat[i, j] = int(r["verifier_results"][name]["passed"])

    pair_rows = []
    phi_mat = np.zeros((m, m))
    agree_mat = np.zeros((m, m))
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            x = pass_mat[:, i]
            y = pass_mat[:, j]
            agree = float((x == y).mean())
            sfp = (
                float(((x == 1) & (y == 1) & (is_corrupt == 1)).sum())
                / max(1, int(is_corrupt.sum()))
            )
            sfn = (
                float(((x == 0) & (y == 0) & (is_clean == 1)).sum())
                / max(1, int(is_clean.sum()))
            )
            ph = phi(x, y)
            phi_mat[i, j] = ph
            agree_mat[i, j] = agree
            if i < j:
                pair_rows.append(
                    {
                        "tag": args.tag,
                        "pair": f"{ni}|{nj}",
                        "agreement": f"{agree:.4f}",
                        "shared_fp_rate": f"{sfp:.4f}",
                        "shared_fn_rate": f"{sfn:.4f}",
                        "phi_corr": f"{ph:.4f}",
                    }
                )

    # Effective-correlation proxy:
    # If verifier failures are perfectly independent, mean off-diagonal phi = 0;
    # if perfectly correlated, mean off-diagonal phi -> 1 -> rho_proxy -> N.
    off = []
    for i in range(m):
        for j in range(m):
            if i != j:
                off.append(phi_mat[i, j])
    mean_phi = float(np.mean(off)) if off else 0.0
    # rho proxy: 1 + (N-1) * max(mean_phi, 0)
    rho_proxy = 1.0 + (m - 1) * max(mean_phi, 0.0)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        fields = ["tag", "pair", "agreement", "shared_fp_rate", "shared_fn_rate", "phi_corr"]
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    summary_path = out_csv.with_name(out_csv.stem + f"_{args.tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "n_rows": int(n),
                "n_clean": int(is_clean.sum()),
                "n_corrupt": int(is_corrupt.sum()),
                "verifiers": names,
                "mean_offdiag_phi": mean_phi,
                "rho_proxy": rho_proxy,
            },
            f,
            indent=2,
        )

    if args.out_heatmap:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        sns.heatmap(
            phi_mat,
            xticklabels=names, yticklabels=names,
            annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": r"$\phi$"},
            ax=ax,
        )
        ax.set_title(f"Verifier failure correlation ({args.tag})")
        plt.tight_layout()
        Path(args.out_heatmap).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.out_heatmap, dpi=200)
        plt.close()
        # also PNG sibling for quick previews
        png = str(Path(args.out_heatmap).with_suffix(".png"))
        plt.figure(figsize=(4.4, 3.6))
        sns.heatmap(
            phi_mat,
            xticklabels=names, yticklabels=names,
            annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        )
        plt.title(f"Verifier failure correlation ({args.tag})")
        plt.tight_layout()
        plt.savefig(png, dpi=200)
        plt.close()
        print(f"[wrote] {args.out_heatmap}")
        print(f"[wrote] {png}")

    print(f"[wrote] {out_csv}")
    print(f"[wrote] {summary_path}")
    print(f"mean off-diagonal phi = {mean_phi:.3f}, rho_proxy = {rho_proxy:.3f}")


if __name__ == "__main__":
    main()
