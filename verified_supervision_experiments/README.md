# Verified Supervision — Controlled Preference-Corruption Experiments

Empirical harness for testing whether pre-optimization preference verification
reduces the impact of directional preference corruption on DPO training. The
pilot compares Raw DPO, noise-aware DPO, single-verifier filtering, k-of-n
consensus filtering, and oracle filtering under controlled levels of
structured-unsafe preference corruption.

## Pilot scope (Phase 2)

- **Dataset:** Anthropic HH-RLHF, 10K train / 1K dev / 1K test (seeded subsample of `dpo_hh_*.jsonl`).
- **Corruption:** `structured_unsafe` only (priority); `random_flip` and `coordinated_poisoning` are wired up but kept for Phase 4.
- **η values:** 0.0, 0.10, 0.20.
- **Methods:** Raw DPO, Noise-aware DPO (cDPO label smoothing 0.1), Single-verifier (safety) + DPO, Consensus k=3-of-4 + DPO, Oracle filter + DPO.
- **Base model:** `Qwen/Qwen2.5-1.5B-Instruct`, LoRA r=16 α=32.
- **Verifiers:** four heterogeneous role-prompted Qwen2.5-1.5B judges (Safety / Helpfulness / Factuality / Policy), threshold 0.7. Same backend, different system prompts.

Total: 3 corrupted variants × 1 verifier pass each + 5 methods × 3 etas = 15 DPO runs + 15 evals.

## Layout

```
verified_supervision_experiments/
├── configs/{dpo,corruption,verifier}_config.yaml
├── data/
│   ├── raw/         symlinks to ../data/dpo_hh_{train,val}.jsonl + safety_refusal_eval.jsonl
│   ├── processed/   subsampled hh_train/dev/test.jsonl
│   ├── corrupted/   one jsonl per (regime, eta) with full _corruption metadata
│   └── filtered/    one subdir per (regime, eta) holding {raw,single,consensus_k3,oracle}.train.jsonl
├── src/
│   ├── load_datasets.py          # subsample + Feedback-format conversion
│   ├── create_corruption.py      # 3 regimes, full provenance
│   ├── verifiers.py              # SafetyVerifier / HelpfulnessVerifier / FactualityVerifier / PolicyVerifier
│   ├── run_verifiers.py          # batch all 4 verifiers over a corrupted file
│   ├── consensus_filter.py       # k-of-n + reliability metrics + rejection sets
│   ├── train_dpo.py              # TRL 0.29 DPOTrainer + LoRA
│   ├── train_noise_aware_dpo.py  # wrapper that injects label_smoothing=0.1
│   ├── evaluate_model.py         # preference acc + refusal split
│   ├── aggregate_results.py      # collect eval/*.json + filter summaries → CSV + LaTeX
│   ├── analyze_rejections.py     # Table 3 (rejection-set composition)
│   ├── analyze_verifier_correlation.py  # Table 4 + Figure 3 heatmap
│   └── plot_results.py           # Figures 1, 2
├── scripts/run_pilot_hh.sh       # end-to-end pilot (idempotent)
└── results/{tables,figures,verifier_outputs,checkpoints,eval}/
```

## Running the pilot

```bash
cd verified_supervision_experiments
bash scripts/run_pilot_hh.sh
```

Each step is **idempotent** — it re-uses earlier outputs and skips work already done. To rerun a specific stage, delete its output and rerun the script.

Environment variables override defaults:

| var | default | purpose |
|---|---|---|
| `TRAIN_SIZE` | 10000 | HH-train subsample |
| `BASE_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | base for both judge and policy |
| `ETAS` | `0.0 0.10 0.20` | corruption sweep |
| `METHODS` | `raw noise_aware single consensus_k3 oracle` | training pipelines |
| `EPOCHS` | 1.0 | DPO epochs |
| `EVAL_MAX_SAMPLES` | 500 | per-test cap |

## Expected wall time on 2× TITAN RTX

| stage | per (regime, eta) | total (3 etas) |
|---|---|---|
| corruption | seconds | ~10 s |
| verifier pass on 10K | ~75 min | ~3.75 hr |
| DPO train | ~40 min × 5 methods | ~10 hr |
| evaluation | ~12 min × 5 | ~3 hr |

≈ **17 hours sequential**, ≈ **9 hours** if methods within an eta are run on the two GPUs in parallel.

## Outputs

After the pilot run completes, the paper-ready artifacts land at:

- `results/tables/main_results.csv`, `table1_main.tex`
- `results/tables/verifier_correlation.csv`
- `results/tables/rejection_breakdown.csv`
- `results/figures/fig1_harmful_survival_vs_eta.pdf`
- `results/figures/fig2_safety_coverage.pdf`
- `results/figures/verifier_corr_heatmap_*.pdf`

A short paper subsection and the rebuttal paragraph are produced in Phase 3 from these tables.
