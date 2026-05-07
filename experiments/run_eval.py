# experiments/run_eval.py

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model_and_tokenizer(model_dir: str, base_model: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    adapter_config = Path(model_dir) / "adapter_config.json"

    model_kwargs = dict(
        trust_remote_code=True,
        device_map="auto" if device.startswith("cuda") else None,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    )

    if adapter_config.exists():
        base = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, **model_kwargs)

    model.eval()
    return model, tokenizer


@torch.no_grad()
def average_continuation_logprob(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    device: str,
    max_length: int = 1024,
) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cont_ids = tokenizer(continuation, add_special_tokens=False)["input_ids"]

    if len(cont_ids) == 0:
        return float("-inf")

    full_ids = prompt_ids + cont_ids
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        full_ids = prompt_ids + cont_ids
        if len(full_ids) > max_length:
            full_ids = full_ids[-max_length:]

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    if not hasattr(model, "hf_device_map"):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    target_ids = input_ids[:, 1:]

    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)

    prompt_len = len(prompt_ids)
    cont_start = max(prompt_len - 1, 0)
    cont_end = cont_start + len(cont_ids)
    cont_token_log_probs = token_log_probs[0, cont_start:cont_end]

    if cont_token_log_probs.numel() == 0:
        return float("-inf")

    return float(cont_token_log_probs.mean().item())


def evaluate_preference_accuracy(model, tokenizer, rows: List[Dict], device: str, max_length: int = 1024):
    total = 0
    correct = 0
    ties = 0
    chosen_scores = []
    rejected_scores = []

    for ex in rows:
        prompt = ex["prompt"]
        chosen = ex["chosen"]
        rejected = ex["rejected"]

        chosen_lp = average_continuation_logprob(model, tokenizer, prompt, chosen, device, max_length)
        rejected_lp = average_continuation_logprob(model, tokenizer, prompt, rejected, device, max_length)

        chosen_scores.append(chosen_lp)
        rejected_scores.append(rejected_lp)

        total += 1
        if chosen_lp > rejected_lp:
            correct += 1
        elif chosen_lp == rejected_lp:
            ties += 1

    acc = correct / total if total else 0.0
    mean_chosen = sum(chosen_scores) / len(chosen_scores) if chosen_scores else 0.0
    mean_rejected = sum(rejected_scores) / len(rejected_scores) if rejected_scores else 0.0

    return {
        "num_examples": total,
        "preference_accuracy": acc,
        "num_correct": correct,
        "num_ties": ties,
        "mean_chosen_logprob": mean_chosen,
        "mean_rejected_logprob": mean_rejected,
        "mean_margin": mean_chosen - mean_rejected,
    }


def append_result_csv(out_csv: str, row: Dict):
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "dataset",
        "split",
        "method",
        "model_dir",
        "base_model",
        "num_examples",
        "preference_accuracy",
        "num_correct",
        "num_ties",
        "mean_chosen_logprob",
        "mean_rejected_logprob",
        "mean_margin",
    ]

    write_header = not out_path.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", default="experiments/results/final_results.csv")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model, device)
    metrics = evaluate_preference_accuracy(model, tokenizer, rows, device, args.max_length)

    result_row = {
        "dataset": args.dataset,
        "split": args.split,
        "method": args.method,
        "model_dir": args.model_dir,
        "base_model": args.base_model,
        **metrics,
    }
    append_result_csv(args.out, result_row)

    print("[EVAL DONE]")
    print(json.dumps(result_row, indent=2))


if __name__ == "__main__":
    main()