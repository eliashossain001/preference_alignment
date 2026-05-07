# experiments/build_vpd.py

import argparse, json, random
from pathlib import Path

def load_dpo(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

def save_jsonl(data, path):
    with open(path, "w") as f:
        for x in data:
            f.write(json.dumps(x) + "\n")

def proxy_accept(p_correct):
    return random.random() < p_correct

def build_vpd(dpo_data, k, proxy_p):
    accepted = []
    for ex in dpo_data:
        passes = sum(proxy_accept(proxy_p) for _ in range(4))  # 4 agents
        if passes >= k:
            accepted.append(ex)
    return accepted

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--proxy_p", type=float, required=True)
    args = ap.parse_args()

    data = load_dpo(args.input)
    vpd = build_vpd(data, args.k, args.proxy_p)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(vpd, args.output)

    print(f"[VPD] retained {len(vpd)}/{len(data)} ({len(vpd)/len(data):.3f})")
