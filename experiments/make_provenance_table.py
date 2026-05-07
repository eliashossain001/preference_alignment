# experiments/make_provenance_table.py

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Any, List


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_summaries(root_dir: Path) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for path in root_dir.rglob("summary.json"):
        try:
            summaries.append(read_json(path))
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
    return summaries


def build_table_rows(
    summaries: List[Dict[str, Any]],
    train_datasets: List[str],
    eval_datasets: List[str],
    ks: List[int],
) -> List[Dict[str, Any]]:
    by_dataset_k: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for s in summaries:
        dataset = s["dataset"]
        k = int(s["k"])
        by_dataset_k.setdefault(dataset, {})
        by_dataset_k[dataset][k] = s

    rows: List[Dict[str, Any]] = []

    for dataset in train_datasets:
        row: Dict[str, Any] = {
            "dataset": dataset,
            "role": "Train",
            "raw_pairs": "--",
        }

        raw_pairs_set = False
        for k in ks:
            s = by_dataset_k.get(dataset, {}).get(k)
            if s is not None and not raw_pairs_set:
                row["raw_pairs"] = s["raw_pairs"]
                raw_pairs_set = True

            row[f"retained_k{k}"] = s["kept_pairs"] if s is not None else "--"
            row[f"retention_pct_k{k}"] = (
                f'{s["retention_pct"]:.2f}' if s is not None else "--"
            )

        rows.append(row)

    for dataset in eval_datasets:
        row = {
            "dataset": dataset,
            "role": "Eval only",
            "raw_pairs": "--",
        }
        for k in ks:
            row[f"retained_k{k}"] = "--"
            row[f"retention_pct_k{k}"] = "--"
        rows.append(row)

    return rows


def write_csv(rows: List[Dict[str, Any]], ks: List[int], out_csv: Path):
    fieldnames = ["dataset", "role", "raw_pairs"]
    for k in ks:
        fieldnames.extend([f"retained_k{k}", f"retention_pct_k{k}"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_latex(rows: List[Dict[str, Any]], ks: List[int], out_tex: Path):
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    header_cols = ["Dataset", "Role", "Raw Pairs"]
    for k in ks:
        header_cols.extend([f"Retained @ $k={k}$", f"Retention \\% @ $k={k}$"])

    with open(out_tex, "w", encoding="utf-8") as f:
        f.write("% Auto-generated provenance table rows\n")
        f.write("\\begin{table*}[t]\n")
        f.write("\\centering\n")
        f.write("\\caption{Dataset provenance and verifier-based retention across consensus thresholds.}\n")
        f.write("\\label{tab:dataset_provenance_auto}\n")
        f.write("\\begin{adjustbox}{width=\\textwidth}\n")

        align = "l l r " + " ".join(["r r" for _ in ks])
        f.write(f"\\begin{{tabular}}{{{align}}}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header_cols) + " \\\\\n")
        f.write("\\midrule\n")

        for row in rows:
            vals = [
                str(row["dataset"]),
                str(row["role"]),
                str(row["raw_pairs"]),
            ]
            for k in ks:
                vals.append(str(row[f"retained_k{k}"]))
                vals.append(str(row[f"retention_pct_k{k}"]))
            f.write(" & ".join(vals) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{adjustbox}\n")
        f.write("\\end{table*}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", required=True, help="Root directory containing filter runs")
    ap.add_argument("--out_csv", required=True, help="Output CSV file")
    ap.add_argument("--out_tex", required=True, help="Output LaTeX file")
    ap.add_argument("--train_datasets", required=True, help="Comma-separated training datasets")
    ap.add_argument("--eval_datasets", default="", help="Comma-separated eval-only datasets")
    ap.add_argument("--ks", default="2,3,4", help="Comma-separated k values")
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    out_csv = Path(args.out_csv)
    out_tex = Path(args.out_tex)

    train_datasets = [x.strip() for x in args.train_datasets.split(",") if x.strip()]
    eval_datasets = [x.strip() for x in args.eval_datasets.split(",") if x.strip()]
    ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]

    summaries = collect_summaries(root_dir)
    rows = build_table_rows(summaries, train_datasets, eval_datasets, ks)

    write_csv(rows, ks, out_csv)
    write_latex(rows, ks, out_tex)

    print(f"[Saved CSV] {out_csv}")
    print(f"[Saved TEX] {out_tex}")


if __name__ == "__main__":
    main()