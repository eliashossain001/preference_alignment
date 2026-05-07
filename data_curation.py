# pip install -q datasets huggingface_hub pandas pyarrow

# fine_tuning/data_curation.py

from datasets import load_dataset, Dataset, DatasetDict, Features, Value
from huggingface_hub import login
from pathlib import Path
import pandas as pd
import json, os, re, random

# =========================
# CONFIG (EDIT THESE)
# =========================
HF_TOKEN     = os.getenv("HF_TOKEN")  # set HF_TOKEN env var (or in .env) before running
HF_USERNAME  = os.getenv("HF_USERNAME", "your-hf-username")
PRIVATE      = False           # set True if you want private repos
VAL_RATIO    = 0.05
SEED         = 42
INCLUDE_COT_IN_SFT = False     # medical: prepend Complex_CoT before Response
TQA_USE_ALL_NEG = True         # TruthfulQA DPO: all wrong answers as negatives

# Your dataset ids on the Hub
# Source dataset IDs on the HF Hub. Replace with your own forks/mirrors.
ID_HH   = os.getenv("HH_DATASET_ID",  "Anthropic/hh-rlhf")
ID_TQA  = os.getenv("TQA_DATASET_ID", "truthful_qa")
ID_MED  = os.getenv("MED_DATASET_ID", "FreedomIntelligence/medical-o1-reasoning-SFT")

# Output repo names on the Hub (distinct names as you asked)
REPOS = {
    "hh":  {"sft": f"{HF_USERNAME}/sft-hh-rlhf",
            "dpo": f"{HF_USERNAME}/dpo-hh-rlhf"},
    "tqa": {"sft": f"{HF_USERNAME}/sft-truthfulqa",
            "dpo": f"{HF_USERNAME}/dpo-truthfulqa"},
    "med": {"sft": f"{HF_USERNAME}/sft-medical-o1",
            "dpo": f"{HF_USERNAME}/dpo-medical-o1-synth"},  # synthetic negatives
}

random.seed(SEED)
OUT = Path("prepared_datasets"); OUT.mkdir(exist_ok=True)

# =========================
# HELPERS
# =========================
def split_train_val(df: pd.DataFrame, val_ratio=VAL_RATIO, seed=SEED):
    if df.empty: return df, df
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(len(df)*val_ratio)) if len(df) > 20 else 0
    return df.iloc[n_val:].reset_index(drop=True), df.iloc[:n_val].reset_index(drop=True)

def safe_split_listish(x):
    if x is None: return []
    if isinstance(x, list): return [str(t).strip() for t in x if str(t).strip()]
    s = str(x).strip()
    if not s: return []
    if s.startswith("[") and s.endswith("]"):
        try:
            return [str(t).strip() for t in json.loads(s) if str(t).strip()]
        except Exception:
            pass
    for sep in ["||",";","\n","|"]:
        if sep in s:
            return [t.strip() for t in s.split(sep) if t.strip()]
    return [s]

def parse_hh_text(txt):
    # expects "Human: ...\nAssistant: ..."
    if not isinstance(txt, str): return (None, None)
    parts = re.split(r"\bAssistant:\s*", txt, maxsplit=1)
    if len(parts) != 2: return (None, None)
    prompt = re.sub(r"^\s*Human:\s*", "", parts[0].strip())
    assistant = parts[1].strip()
    return (prompt or None, assistant or None)

def push_pair_to_hub(train_path, val_path, repo_id, features):
    train_ds = Dataset.from_json(str(train_path), features=features)
    val_ds   = Dataset.from_json(str(val_path),   features=features)
    DatasetDict({"train": train_ds, "validation": val_ds}).push_to_hub(repo_id, private=PRIVATE)
    print(f"[PUSHED] {repo_id}")

# =========================
# 1) ANTHROPIC HH
# =========================
print("=== Building Anthropic HH ===")
hh = load_dataset(ID_HH)["train"]

# --- SFT ---
sft_hh = []
for ex in hh:
    p, a = parse_hh_text(ex.get("chosen"))
    if p and a: sft_hh.append({"prompt": p, "response": a})

df = pd.DataFrame(sft_hh)
tr, va = split_train_val(df)
(tr).to_json(OUT/"sft_hh_train.jsonl", orient="records", lines=True, force_ascii=False)
(va).to_json(OUT/"sft_hh_val.jsonl",   orient="records", lines=True, force_ascii=False)
print(f"[HH SFT] {len(df)} rows")

# --- DPO ---
dpo_hh = []
for ex in hh:
    p_c, a_c = parse_hh_text(ex.get("chosen"))
    p_r, a_r = parse_hh_text(ex.get("rejected"))
    p = p_c or p_r
    if p and a_c and a_r:
        dpo_hh.append({"prompt": p, "chosen": a_c, "rejected": a_r})

df = pd.DataFrame(dpo_hh)
tr, va = split_train_val(df)
(tr).to_json(OUT/"dpo_hh_train.jsonl", orient="records", lines=True, force_ascii=False)
(va).to_json(OUT/"dpo_hh_val.jsonl",   orient="records", lines=True, force_ascii=False)
print(f"[HH DPO] {len(df)} rows")

# =========================
# 2) TRUTHFULQA
# =========================
print("=== Building TruthfulQA ===")
tqa = load_dataset(ID_TQA)["train"]

# --- SFT ---
sft_tqa = []
for ex in tqa:
    q = (ex.get("Question") or "").strip()
    a = (ex.get("Best Answer") or "").strip()
    if q and a: sft_tqa.append({"prompt": q, "response": a})

df = pd.DataFrame(sft_tqa)
tr, va = split_train_val(df)
(tr).to_json(OUT/"sft_tqa_train.jsonl", orient="records", lines=True, force_ascii=False)
(va).to_json(OUT/"sft_tqa_val.jsonl",   orient="records", lines=True, force_ascii=False)
print(f"[TQA SFT] {len(df)} rows")

# --- DPO ---
dpo_tqa = []
for ex in tqa:
    q = (ex.get("Question") or "").strip()
    best = (ex.get("Best Answer") or "").strip()
    wrongs = safe_split_listish(ex.get("Incorrect Answers"))
    if not (q and best and wrongs): continue
    negatives = wrongs if TQA_USE_ALL_NEG else random.sample(wrongs, k=1)
    for neg in negatives:
        neg = neg.strip()
        if neg and neg != best:
            dpo_tqa.append({"prompt": q, "chosen": best, "rejected": neg})

df = pd.DataFrame(dpo_tqa)
tr, va = split_train_val(df)
(tr).to_json(OUT/"dpo_tqa_train.jsonl", orient="records", lines=True, force_ascii=False)
(va).to_json(OUT/"dpo_tqa_val.jsonl",   orient="records", lines=True, force_ascii=False)
print(f"[TQA DPO] {len(df)} rows")

# =========================
# 3) MEDICAL-O1-REASONING
# =========================
print("=== Building medical-o1-reasoning ===")
med = load_dataset(ID_MED)["train"]

# --- SFT ---
sft_med = []
for ex in med:
    q   = (ex.get("Question") or "").strip()
    cot = (ex.get("Complex_CoT") or "").strip()
    ans = (ex.get("Response") or "").strip()
    if not (q and ans): continue
    resp = f"{cot}\n\n{ans}" if (INCLUDE_COT_IN_SFT and cot) else ans
    sft_med.append({"prompt": q, "response": resp})

df = pd.DataFrame(sft_med)
tr, va = split_train_val(df)
(tr).to_json(OUT/"sft_med_train.jsonl", orient="records", lines=True, force_ascii=False)
(va).to_json(OUT/"sft_med_val.jsonl",   orient="records", lines=True, force_ascii=False)
print(f"[MED SFT] {len(df)} rows")

# --- DPO (synthetic negatives sampled from other answers) ---
# Note: these are *synthetic* rejections; we suffix the repo with '-synth'.
dpo_med = []
all_answers = [r["response"] for r in sft_med]
for i, row in enumerate(sft_med):
    prompt  = row["prompt"]
    chosen  = row["response"]
    # sample a negative that is different from the chosen
    if len(all_answers) > 1:
        while True:
            neg = random.choice(all_answers)
            if neg != chosen: break
        dpo_med.append({"prompt": prompt, "chosen": chosen, "rejected": neg})

df = pd.DataFrame(dpo_med)
tr, va = split_train_val(df)
(tr).to_json(OUT/"dpo_med_train.jsonl", orient="records", lines=True, force_ascii=False)
(va).to_json(OUT/"dpo_med_val.jsonl",   orient="records", lines=True, force_ascii=False)
print(f"[MED DPO (synthetic)] {len(df)} rows")

# =========================
# 4) OPTIONAL: PUSH EACH TO HF
# =========================
if HF_TOKEN and HF_TOKEN != "YOUR_HF_TOKEN_HERE":
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"[WARN] HF login failed: {e}")

# Push SFT + DPO for HH
sft_features = Features({"prompt": Value("string"), "response": Value("string")})
dpo_features = Features({"prompt": Value("string"), "chosen": Value("string"), "rejected": Value("string")})

push_pair_to_hub(OUT/"sft_hh_train.jsonl",  OUT/"sft_hh_val.jsonl",  REPOS["hh"]["sft"],  sft_features)
push_pair_to_hub(OUT/"dpo_hh_train.jsonl",  OUT/"dpo_hh_val.jsonl",  REPOS["hh"]["dpo"],  dpo_features)

# Push SFT + DPO for TruthfulQA
push_pair_to_hub(OUT/"sft_tqa_train.jsonl", OUT/"sft_tqa_val.jsonl", REPOS["tqa"]["sft"], sft_features)
push_pair_to_hub(OUT/"dpo_tqa_train.jsonl", OUT/"dpo_tqa_val.jsonl", REPOS["tqa"]["dpo"], dpo_features)

# Push SFT + DPO (synthetic) for medical-o1
push_pair_to_hub(OUT/"sft_med_train.jsonl", OUT/"sft_med_val.jsonl", REPOS["med"]["sft"], sft_features)
push_pair_to_hub(OUT/"dpo_med_train.jsonl", OUT/"dpo_med_val.jsonl", REPOS["med"]["dpo"], dpo_features)

print("All six datasets built and (optionally) pushed.")
