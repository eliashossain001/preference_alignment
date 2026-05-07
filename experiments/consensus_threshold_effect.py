# experiments/make_table9.py

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_csv(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_method_map(rows, key="method"):
    out = {}
    for r in rows:
        out[r[key]] = r
    return out


def method_to_k(method: str) -> str:
    m = method.lower()
    if "k2" in m or "k_2" in m:
        return "2"
    if "k3" in m or "k_3" in m:
        return "3"
    if "k4" in m or "k_4" in m:
        return "4"
    raise ValueError(f"Cannot infer k from method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provenance_csv", required=True)
    ap.add_argument("--hh_csv", required=True)
    ap.add_argument("--truthfulqa_csv", required=True)
    ap.add_argument("--pubmedqa_csv", required=True)
    ap.add_argument("--refusal_csv", required=True)
    ap.add_argument("--ece_csv", required=True)
    ap.add_argument("--out_tex", required=True)
    args = ap.parse_args()

    provenance = read_csv(args.provenance_csv)
    hh_rows = read_csv(args.hh_csv)
    tqa_rows = read_csv(args.truthfulqa_csv)
    pub_rows = read_csv(args.pubmedqa_csv)
    refusal_rows = read_csv(args.refusal_csv)
    ece_rows = read_csv(args.ece_csv)

    hh_map = build_method_map(hh_rows)
    tqa_map = build_method_map(tqa_rows)
    pub_map = build_method_map(pub_rows)
    refusal_map = build_method_map(refusal_rows)
    ece_map = build_method_map(ece_rows)

    hh_prov = next(r for r in provenance if r["dataset"] == "hh")

    rows = []
    for method in sorted(hh_map.keys()):
        k = method_to_k(method)
        row = {
            "k": k,
            "retention": hh_prov[f"retention_pct_k{k}"],
            "hh": f'{100 * float(hh_map[method]["preference_accuracy"]):.2f}',
            "truthfulqa": f'{100 * float(tqa_map[method]["accuracy"]):.2f}',
            "pubmedqa": f'{100 * float(pub_map[method]["accuracy"]):.2f}',
            "safety": f'{100 * float(refusal_map[method]["safety"]):.2f}',
            "over_refusal": f'{100 * float(refusal_map[method]["over_refusal"]):.2f}',
            "ece": f'{float(ece_map[method]["ece"]):.3f}',
            "notes": {
                "2": "Balanced",
                "3": "More selective",
                "4": "Very strict",
            }[k],
        }
        rows.append(row)

    out_path = Path(args.out_tex)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Effect of consensus threshold $k$ on data retention and downstream performance. Lower thresholds are more permissive, whereas larger thresholds are more selective.}\n")
        f.write("\\label{tab:k_ablation}\n")
        f.write("\\begin{adjustbox}{width=\\textwidth}\n")
        f.write("\\begin{tabular}{c r r r r r r r l}\n")
        f.write("\\toprule\n")
        f.write("$k$ & Retention \\% & HH $\\uparrow$ & TruthfulQA $\\uparrow$ & PubMedQA $\\uparrow$ & Safety $\\uparrow$ & Over-Refusal $\\downarrow$ & ECE $\\downarrow$ & Notes \\\\\n")
        f.write("\\midrule\n")
        for r in sorted(rows, key=lambda x: int(x["k"])):
            f.write(
                f'{r["k"]} & {r["retention"]} & {r["hh"]} & {r["truthfulqa"]} & {r["pubmedqa"]} & {r["safety"]} & {r["over_refusal"]} & {r["ece"]} & {r["notes"]} \\\\\n'
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{adjustbox}\n")
        f.write("\\end{table*}\n")

    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()