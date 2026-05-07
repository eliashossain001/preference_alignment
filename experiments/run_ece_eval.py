# experiments/run_ece_eval.py

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
def avg_logprob_for_answer(model, tokenizer, prompt: str, answer: str, device: str, max_length: int = 1024) -> float:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    ans_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]

    if len(ans_ids) == 0:
        return float("-inf")

    full_ids = prompt_ids + ans_ids
    if len(full_ids) > max_length:
        overflow = len(full_ids) - max_length
        prompt_ids = prompt_ids[overflow:]
        full_ids = prompt_ids + ans_ids
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
    start = max(prompt_len - 1, 0)
    end = start + len(ans_ids)
    ans_token_log_probs = token_log_probs[0, start:end]

    if ans_token_log_probs.numel() == 0:
        return float("-inf")

    return float(ans_token_log_probs.mean().item())


def compute_ece(confidences: List[float], corrects: List[int], n_bins: int = 10) -> float:
    if not confidences:
        return 0.0

    ece = 0.0
    total = len(confidences)

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == n_bins - 1 and lo <= c <= hi)]
        if not idx:
            continue
        bin_conf = sum(confidences[i] for i in idx) / len(idx)
        bin_acc = sum(corrects[i] for i in idx) / len(idx)
        ece += (len(idx) / total) * abs(bin_acc - bin_conf)

    return ece


def evaluate_ece(model, tokenizer, rows: List[Dict], device: str, max_length: int = 1024, n_bins: int = 10):
    confidences = []
    corrects = []

    for ex in rows:
        question = ex["question"]
        choices = ex["choices"]
        answer_idx = int(ex["answer_idx"])

        scores = [
            avg_logprob_for_answer(model, tokenizer, question, choice, device, max_length)
            for choice in choices
        ]
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(score_tensor, dim=0)

        pred = int(torch.argmax(probs).item())
        conf = float(torch.max(probs).item())

        confidences.append(conf)
        corrects.append(1 if pred == answer_idx else 0)

    ece = compute_ece(confidences, corrects, n_bins=n_bins)
    acc = sum(corrects) / len(corrects) if corrects else 0.0

    return {"num_examples": len(corrects), "accuracy": acc, "ece": ece}


def append_result_csv(out_csv: str, row: Dict):
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["dataset", "split", "method", "model_dir", "base_model", "num_examples", "accuracy", "ece"]
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
    ap.add_argument("--split", default="test")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_jsonl(args.eval_file)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model, device)
    metrics = evaluate_ece(model, tokenizer, rows, device)

    result_row = {
        "dataset": args.dataset,
        "split": args.split,
        "method": args.method,
        "model_dir": args.model_dir,
        "base_model": args.base_model,
        **metrics,
    }
    append_result_csv(args.out, result_row)

    print("[ECE EVAL DONE]")
    print(json.dumps(result_row, indent=2))


if __name__ == "__main__":
    main()