# experiments/run_train.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from agents.base import Feedback
from fine_tuning.dpo_trainer import DPOTrainerHelper


def count_jsonl(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def write_status(output_dir: str, payload: dict):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "train_status.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_feedback(path: str) -> List[Feedback]:
    items: List[Feedback] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)

            if "chosen" in x and "rejected" in x:
                prompt = x["prompt"]
                response_a = x["chosen"]
                response_b = x["rejected"]
                user_choice = "A"
            else:
                prompt = x["prompt"]
                response_a = x["response_a"]
                response_b = x["response_b"]
                user_choice = x.get("user_choice", "A")

            fb = Feedback(
                prompt=prompt,
                response_a=response_a,
                response_b=response_b,
                user_choice=user_choice,
                id=x.get("id"),
                dataset=x.get("dataset"),
                split=x.get("split"),
                source_row=x.get("source_row"),
                is_clean=x.get("is_clean"),
            )
            items.append(fb)

    return items


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--beta", type=float, default=0.1)
    args = ap.parse_args()

    num_rows = count_jsonl(args.train_file)

    if num_rows == 0:
        status = {
            "status": "not_applicable",
            "reason": "zero_retention",
            "train_file": args.train_file,
            "num_rows": 0,
        }
        write_status(args.output_dir, status)
        print(json.dumps(status, indent=2))
        raise SystemExit(0)

    feedback = load_feedback(args.train_file)

    config = {
        "train_file": args.train_file,
        "output_dir": args.output_dir,
        "base_model": args.base_model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "beta": args.beta,
        "num_rows": num_rows,
    }

    write_status(args.output_dir, {"status": "starting", **config})

    trainer = DPOTrainerHelper(
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        beta=args.beta,
    )

    trainer.train(feedback)

    write_status(args.output_dir, {"status": "finished", **config})