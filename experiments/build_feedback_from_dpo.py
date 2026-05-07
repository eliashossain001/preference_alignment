# experiments/build_feedback_from_dpo.py

from __future__ import annotations

import json
import argparse
from pathlib import Path


def build_feedback(dpo_path: str, out_path: str, dataset: str, split: str = "train"):
    dpo_path = Path(dpo_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(dpo_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            ex = json.loads(line)
            fb = {
                "id": f"{dataset}_{split}_{i:06d}",
                "dataset": dataset,
                "split": split,
                "source_row": i,
                "prompt": ex["prompt"],
                "response_a": ex["chosen"],
                "response_b": ex["rejected"],
                "user_choice": "A",
            }
            rows.append(fb)

    with open(out_path, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[Saved] {out_path} ({len(rows)} items)")
    print(f"[Dataset] {dataset}")
    print(f"[Split]   {split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_path", required=True, help="Input DPO jsonl file")
    parser.add_argument("--out_path", required=True, help="Output feedback jsonl file")
    parser.add_argument("--dataset", required=True, help="Dataset short name, e.g. hh, tqa, med")
    parser.add_argument("--split", default="train", help="Split name, default=train")
    args = parser.parse_args()

    build_feedback(
        dpo_path=args.dpo_path,
        out_path=args.out_path,
        dataset=args.dataset,
        split=args.split,
    )