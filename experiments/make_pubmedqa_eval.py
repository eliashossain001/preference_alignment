import json
from datasets import load_dataset

# Try this first
ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

label_map = {
    "yes": 0,
    "no": 1,
    "maybe": 2,
}

with open("data/pubmedqa_eval.jsonl", "w", encoding="utf-8") as f:
    kept = 0
    for i, ex in enumerate(ds):
        question = ex.get("question", "").strip()
        answer = ex.get("final_decision", ex.get("answer", "")).strip().lower()

        if not question or answer not in label_map:
            continue

        row = {
            "id": f"pubmedqa_{kept}",
            "question": question,
            "choices": ["yes", "no", "maybe"],
            "answer_idx": label_map[answer],
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1

print(f"Saved data/pubmedqa_eval.jsonl with {kept} rows")