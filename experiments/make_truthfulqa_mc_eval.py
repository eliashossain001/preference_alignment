import json
from datasets import load_dataset

ds = load_dataset("EleutherAI/truthful_qa_mc", split="validation")

label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

with open("data/truthfulqa_mc_eval.jsonl", "w", encoding="utf-8") as f:
    for i, ex in enumerate(ds):
        label = ex["label"]
        if isinstance(label, str):
            answer_idx = label_map[label]
        else:
            answer_idx = int(label)

        row = {
            "id": f"truthfulqa_{i}",
            "question": ex["question"],
            "choices": list(ex["choices"]),
            "answer_idx": answer_idx,
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Saved data/truthfulqa_mc_eval.jsonl")