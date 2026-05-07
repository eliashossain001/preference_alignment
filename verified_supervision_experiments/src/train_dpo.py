"""DPO (or noise-aware DPO) training with LoRA on a filtered preference jsonl.

Same hyperparameters across all methods so differences are attributable to the
filter, not the optimizer. The only knob the noise-aware baseline twists is
`label_smoothing` (cDPO-style), per Mitchell-style robust-DPO formulations.

Usage:
    python -m src.train_dpo \
        --train_file data/filtered/hh_train_struct_eta20/consensus_k3.train.jsonl \
        --output_dir results/checkpoints/struct_eta20/consensus_k3 \
        --base_model Qwen/Qwen2.5-1.5B-Instruct \
        --epochs 1 --batch_size 2 --grad_accum 8 --lr 5e-6 --beta 0.1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def load_dpo_jsonl(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x = json.loads(line)
            rows.append({"prompt": x["prompt"], "chosen": x["chosen"], "rejected": x["rejected"]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help=">0 enables cDPO-style noise-aware loss")
    ap.add_argument("--max_train_samples", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_dpo_jsonl(args.train_file)
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))

    if len(ds) == 0:
        status = {"status": "skipped", "reason": "empty_train_file", "train_file": args.train_file}
        with open(out / "train_status.json", "w") as f:
            json.dump(status, f, indent=2)
        print(json.dumps(status, indent=2))
        return

    print(f"[train] {len(ds)} pairs from {args.train_file}")
    print(f"[train] base_model={args.base_model} epochs={args.epochs} bs={args.batch_size} ga={args.grad_accum} lr={args.lr} beta={args.beta} ls={args.label_smoothing}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    cfg = DPOConfig(
        output_dir=str(out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        seed=args.seed,
        logging_steps=20,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        bf16=True,
        remove_unused_columns=False,
        label_smoothing=args.label_smoothing,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,            # use frozen base when peft_config is provided
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(str(out))

    with open(out / "train_status.json", "w") as f:
        json.dump(
            {
                "status": "finished",
                "train_file": args.train_file,
                "num_train": len(ds),
                "base_model": args.base_model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lr": args.lr,
                "beta": args.beta,
                "label_smoothing": args.label_smoothing,
                "lora_r": args.lora_r,
            },
            f,
            indent=2,
        )
    print(f"[done] checkpoint at {out}")


if __name__ == "__main__":
    main()
