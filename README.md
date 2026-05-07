# Verified Supervision for Preference-Based Alignment

Research code for studying **pre-optimization preference verification** in RLHF / DPO pipelines. The repository contains:

1. A **multi-agent verifier framework** that filters preference pairs through a heterogeneous pool of role-prompted LLM judges (safety, helpfulness, factuality, policy) before they reach the DPO loss.
2. A **controlled preference-corruption stress test** that injects directional preference corruption at known rates and compares Raw DPO, noise-aware DPO, single-verifier filtering, *k*-of-*n* consensus, and oracle filtering on the resulting reliability and downstream metrics.

The central question this code is designed to investigate: when preference corruption is *directional* (correlated with a target unsafe behaviour) rather than symmetric stochastic noise, can objective-level robustness recover the missing provenance from the loss alone, or does reliability information have to enter the data layer before any gradient is computed?

---

## Repository structure

```
.
├── agents/                              # Four heterogeneous LLM-based verifiers
│   ├── base.py                          #  Feedback / AgentResult abstractions
│   ├── knowledge_verifier.py            #  Factuality
│   ├── behavior_auditor.py              #  Safety / policy compliance
│   ├── ethics_evaluator.py              #  Ethics
│   ├── trust_assessor.py                #  Trustworthiness
│   ├── proxy_verifiers.py               #  Stochastic proxy verifier (controlled-noise experiments)
│   └── local_llm_backend.py             #  Singleton-cached HF causal-LM backend
│
├── feedback_pipeline/
│   └── pipeline.py                      # k-of-n consensus filter (proxy or LLM mode)
│
├── fine_tuning/
│   └── dpo_trainer.py                   # DPOTrainer wrapper with LoRA + 8-bit quantization
│
├── data_curation.py                     # HH-RLHF / TruthfulQA / Medical-O1 -> SFT and DPO splits
├── run_dpo.py                           # Proxy-verifier sweep over (eta, k) with corruption injection
│
├── experiments/                         # End-to-end harness (real-LLM verifiers, multi-eta, MCQ/refusal/ECE eval)
│
└── verified_supervision_experiments/    # Controlled corruption stress test
    ├── configs/{dpo,corruption,verifier}_config.yaml
    ├── src/
    │   ├── load_datasets.py             #  Subsample HH-RLHF to 10K/1K/1K
    │   ├── create_corruption.py         #  Three corruption regimes: random_flip, structured_unsafe, coordinated_poisoning
    │   ├── verifiers.py                 #  Four role-prompted verifiers: Safety, Helpfulness, Factuality, Policy
    │   ├── batched_backend.py           #  Batched HF backend (4 verifier prompts per forward pass)
    │   ├── precompute_verifiers.py      #  Score both (prompt, response) directions once, sharded by GPU
    │   ├── consensus_filter_precomputed.py  #  Apply raw / single / consensus_k3 / oracle filters
    │   ├── train_dpo.py                 #  TRL DPOTrainer with LoRA
    │   ├── train_noise_aware_dpo.py     #  cDPO label-smoothing wrapper
    │   ├── evaluate_model.py            #  HH-test preference acc + safety/refusal split
    │   ├── analyze_rejections.py        #  Rejection-set composition
    │   ├── analyze_verifier_correlation.py  #  rho-hat + phi heatmap
    │   ├── aggregate_results.py         #  CSV + LaTeX table assembly
    │   └── plot_results.py              #  Harmful-survival and safety-coverage plots
    ├── scripts/
    │   ├── run_pilot_hh.sh              #  End-to-end orchestration (idempotent)
    │   └── pilot_status.sh              #  One-shot status check
    └── results/
        ├── tables/                      #  CSVs and LaTeX produced by aggregate/analyze (committed)
        ├── figures/                     #  Figures as PDF and PNG (committed)
        ├── checkpoints/                 #  Trained LoRA adapters (gitignored)
        ├── verifier_outputs/            #  Per-pair verifier scores (gitignored, large)
        └── eval/                        #  Per-checkpoint eval JSON (gitignored)
```

---

## Setup

```bash
# Python 3.11+ recommended
python -m venv .venv && source .venv/bin/activate

pip install "torch>=2.0" "transformers>=4.46" "datasets>=2.18" \
            "trl>=0.18" "peft>=0.13" "accelerate>=0.34" "bitsandbytes>=0.43" \
            "pandas" "matplotlib" "seaborn" "pydantic"

cp .env.example .env   # then edit and set HF_TOKEN
```

GPU is required for any LLM-verifier or DPO step. The pilot was developed on 2 x NVIDIA TITAN RTX (24 GB each) with bf16 and LoRA r=16.

---

## Reproducing the pilot

```bash
cd verified_supervision_experiments
bash scripts/run_pilot_hh.sh           # ~6-8 hours wall time on 2 GPUs
```

The script will, in order:

1. Subsample HH-RLHF to 10 000 train / 1 000 dev / 1 000 test (seeded).
2. Inject `structured_unsafe` corruption at eta in {0.0, 0.10, 0.20}.
3. Score every (prompt, response) direction once with four heterogeneous role-prompted verifiers (precompute, sharded across GPUs).
4. Apply Raw / Single-verifier / Consensus k=3 / Oracle filters per eta.
5. Train Qwen2.5-1.5B-Instruct with LoRA-DPO under each (eta, method) cell -- 5 methods x 3 eta = 15 checkpoints.
6. Evaluate every checkpoint on a 500-pair HH-test preference-accuracy split and a 500-prompt safety/refusal split.
7. Run rejection-set analysis and pairwise verifier-correlation analysis.
8. Aggregate result tables, figures, and the populated section text.

Status check at any time:

```bash
bash verified_supervision_experiments/scripts/pilot_status.sh
```

The pipeline is **idempotent**: each step skips if its primary output already exists. To re-run a stage, delete its output and rerun the script.

---

## Headline result

Filter-side ordering on HH-RLHF, structured-unsafe corruption (10 K pairs, Qwen2.5-1.5B verifiers and policy):

| eta | Method          | Retention | Harm. survival | Det. F1 |
|-----|-----------------|-----------|----------------|---------|
| 20% | Raw             | 100.0%    | 20.31%         | 0.00    |
| 20% | Single (Safety) | 59.1%     | 19.14%         | 0.29    |
| 20% | Consensus k=3   | 40.2%     | 18.08%         | 0.33    |
| 20% | Oracle          | 79.7%     | 0.00%          | 1.00    |

The strict ordering Raw &succ; Single &succ; Consensus &succ; Oracle holds at every eta > 0. The estimated effective verifier correlation in this pool is rho-hat = 3.07 (mean off-diagonal phi = 0.689).

---

## License

MIT.
