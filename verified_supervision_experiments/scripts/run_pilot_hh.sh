#!/usr/bin/env bash
# Pilot: HH-RLHF / structured_unsafe / eta in {0, 10, 20} / four methods.
# Uses *precomputed* verifier scores (one pass for both response directions),
# sharded across the available GPUs to halve verifier wall time. Idempotent:
# each step skips if its primary output already exists.

set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="$PWD"
TRAIN_SIZE="${TRAIN_SIZE:-10000}"
DEV_SIZE="${DEV_SIZE:-1000}"
TEST_SIZE="${TEST_SIZE:-1000}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
VERIFIER_MODEL="${VERIFIER_MODEL:-$BASE_MODEL}"
EPOCHS="${EPOCHS:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-6}"
BETA="${BETA:-0.1}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-500}"

ETAS=(${ETAS:-0.0 0.10 0.20})
CORR="${CORR:-structured_unsafe}"
METHODS=(${METHODS:-raw noise_aware single consensus_k3 oracle})
NUM_GPUS="${NUM_GPUS:-2}"

echo "[pilot] base=$BASE_MODEL verifier=$VERIFIER_MODEL train=$TRAIN_SIZE etas=${ETAS[*]} methods=${METHODS[*]} num_gpus=$NUM_GPUS"

# ---------- 1. subsample ----------
if [[ ! -f "$ROOT/data/processed/hh_train.jsonl" || \
      $(wc -l < "$ROOT/data/processed/hh_train.jsonl") -lt $TRAIN_SIZE ]]; then
    python3 -m src.load_datasets \
        --train_size "$TRAIN_SIZE" --dev_size "$DEV_SIZE" --test_size "$TEST_SIZE" --seed "$SEED"
fi

# ---------- 2. corruption sweep (cheap, runs sequentially) ----------
for ETA in "${ETAS[@]}"; do
    ETATAG="$(python3 -c "print(int(${ETA}*100))")"
    OUT="$ROOT/data/corrupted/hh_train_${CORR}_eta${ETATAG}.jsonl"
    if [[ ! -f "$OUT" ]]; then
        python3 -m src.create_corruption \
            --input "$ROOT/data/processed/hh_train.jsonl" \
            --output "$OUT" \
            --corruption "$CORR" --eta "$ETA" --seed "$SEED"
    fi
done

# ---------- 3. precompute verifier scores ONCE for both directions, sharded by GPU ----------
PRECOMP_DIR="$ROOT/results/verifier_outputs"
mkdir -p "$PRECOMP_DIR"
PRECOMP_FINAL="$PRECOMP_DIR/hh_train.precomputed.jsonl"
if [[ ! -f "$PRECOMP_FINAL" ]]; then
    SHARD_FILES=()
    for ((i=0; i<NUM_GPUS; i++)); do
        SHARD="$PRECOMP_DIR/hh_train.precomputed.shard${i}.jsonl"
        SHARD_FILES+=("$SHARD")
        if [[ ! -f "$SHARD" ]]; then
            echo "[precompute] launching shard $i/$NUM_GPUS on GPU$i"
            CUDA_VISIBLE_DEVICES=$i python3 -m src.precompute_verifiers \
                --input "$ROOT/data/processed/hh_train.jsonl" \
                --output "$SHARD" \
                --model_name "$VERIFIER_MODEL" --threshold 0.7 \
                --shard_index $i --shard_total $NUM_GPUS \
                > "$SHARD.log" 2>&1 &
        else
            echo "[precompute] shard $i already complete: $SHARD"
        fi
    done
    wait
    echo "[precompute] all shards complete; merging"
    cat "${SHARD_FILES[@]}" > "$PRECOMP_FINAL"
fi
echo "[precompute] $(wc -l < "$PRECOMP_FINAL") rows in $PRECOMP_FINAL"

# ---------- 4. apply consensus filter per eta ----------
for ETA in "${ETAS[@]}"; do
    ETATAG="$(python3 -c "print(int(${ETA}*100))")"
    FILT_DIR="$ROOT/data/filtered/hh_train_${CORR}_eta${ETATAG}"
    REJ_DIR="$PRECOMP_DIR/hh_train_${CORR}_eta${ETATAG}"
    if [[ ! -f "$FILT_DIR/filter_summary.json" ]]; then
        python3 -m src.consensus_filter_precomputed \
            --corrupted "$ROOT/data/corrupted/hh_train_${CORR}_eta${ETATAG}.jsonl" \
            --precomputed "$PRECOMP_FINAL" \
            --outdir "$FILT_DIR" --rejected_dir "$REJ_DIR" \
            --k 3 --single_verifier safety
    fi
done

# ---------- 5. train methods, parallelized across GPUs (round-robin) ----------
# Use +e so one OOM doesn't kill the whole pipeline; per-run failures show up as
# missing train_status.json and stage 6 will skip them.
set +e
GPU_IDX=0
for ETA in "${ETAS[@]}"; do
    ETATAG="$(python3 -c "print(int(${ETA}*100))")"
    FILT_DIR="$ROOT/data/filtered/hh_train_${CORR}_eta${ETATAG}"
    for METHOD in "${METHODS[@]}"; do
        case "$METHOD" in
            raw|noise_aware) TRAIN_FILE="$FILT_DIR/raw.train.jsonl" ;;
            single)          TRAIN_FILE="$FILT_DIR/single.train.jsonl" ;;
            consensus_k3)    TRAIN_FILE="$FILT_DIR/consensus_k3.train.jsonl" ;;
            oracle)          TRAIN_FILE="$FILT_DIR/oracle.train.jsonl" ;;
            *) echo "unknown method $METHOD"; exit 1 ;;
        esac
        CKPT="$ROOT/results/checkpoints/${CORR}_eta${ETATAG}/${METHOD}"
        if [[ -f "$CKPT/train_status.json" ]]; then
            echo "  skipping trained $CKPT"
            continue
        fi
        EXTRA=""
        if [[ "$METHOD" == "noise_aware" ]]; then
            EXTRA="--label_smoothing 0.1"
        fi
        echo "[train] GPU=$GPU_IDX  $CORR eta=$ETA  method=$METHOD"
        mkdir -p "$CKPT"          # ensure parent dir exists for the redirect
        CUDA_VISIBLE_DEVICES=$GPU_IDX python3 -m src.train_dpo \
            --train_file "$TRAIN_FILE" --output_dir "$CKPT" \
            --base_model "$BASE_MODEL" \
            --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" --grad_accum "$GRAD_ACCUM" \
            --lr "$LR" --beta "$BETA" --seed "$SEED" $EXTRA \
            > "$CKPT.log" 2>&1 &
        # round-robin GPU assignment with at most NUM_GPUS jobs in flight
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
        # block when we've launched NUM_GPUS jobs without waiting
        if (( $(jobs -r | wc -l) >= NUM_GPUS )); then
            wait -n
        fi
    done
done
wait
echo "[train] all training runs complete"

# ---------- 6. evaluate, also parallelized ----------
# Stay in +e mode: one bad checkpoint shouldn't block the others
GPU_IDX=0
for ETA in "${ETAS[@]}"; do
    ETATAG="$(python3 -c "print(int(${ETA}*100))")"
    for METHOD in "${METHODS[@]}"; do
        CKPT="$ROOT/results/checkpoints/${CORR}_eta${ETATAG}/${METHOD}"
        OUT="$ROOT/results/eval/${CORR}_eta${ETATAG}_${METHOD}.json"
        if [[ -f "$OUT" ]]; then continue; fi
        if [[ ! -f "$CKPT/train_status.json" ]]; then
            echo "  no train_status for $CKPT — skipping eval"; continue
        fi
        echo "[eval] GPU=$GPU_IDX  $CORR eta=$ETA  method=$METHOD"
        mkdir -p "$(dirname "$OUT")"  # ensure parent dir exists for redirect
        CUDA_VISIBLE_DEVICES=$GPU_IDX python3 -m src.evaluate_model \
            --model_dir "$CKPT" --base_model "$BASE_MODEL" \
            --hh_test "$ROOT/data/processed/hh_test.jsonl" \
            --refusal_eval "$ROOT/data/raw/safety_refusal_eval.jsonl" \
            --out_json "$OUT" \
            --max_samples "$EVAL_MAX_SAMPLES" > "$OUT.log" 2>&1 &
        GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))
        if (( $(jobs -r | wc -l) >= NUM_GPUS )); then
            wait -n
        fi
    done
done
wait
echo "[eval] all evaluations complete"
set -e

# ---------- 7. aggregate + analyses ----------
python3 -m src.aggregate_results || echo "[warn] aggregate_results failed"
python3 -m src.plot_results || echo "[warn] plot_results failed (likely no main_results.csv)"

# Verifier-correlation analysis runs on the per-eta SCORED file. Build it from
# the precomputed results by attaching the eta-specific user_choice direction.
for ETA in "${ETAS[@]}"; do
    ETATAG="$(python3 -c "print(int(${ETA}*100))")"
    SCORED="$PRECOMP_DIR/hh_train_${CORR}_eta${ETATAG}.scored.jsonl"
    if [[ ! -f "$SCORED" ]]; then
        python3 - <<PY
import json
from pathlib import Path
corrupt = [json.loads(l) for l in open("$ROOT/data/corrupted/hh_train_${CORR}_eta${ETATAG}.jsonl")]
pre = {json.loads(l)["id"]: json.loads(l) for l in open("$PRECOMP_FINAL")}
out = Path("$SCORED")
with open(out, "w") as f:
    for r in corrupt:
        prec = pre[r["id"]]
        r["verifier_results"] = prec["verifier_results_a"] if r["user_choice"] == "A" else prec["verifier_results_b"]
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {out}")
PY
    fi
    python3 -m src.analyze_verifier_correlation \
        --scored "$SCORED" \
        --out_csv "$ROOT/results/tables/verifier_correlation.csv" \
        --out_heatmap "$ROOT/results/figures/verifier_corr_heatmap_${CORR}_eta${ETATAG}.pdf" \
        --tag "${CORR}_eta${ETATAG}"
    REJ="$PRECOMP_DIR/hh_train_${CORR}_eta${ETATAG}/consensus_k3.rejected.jsonl"
    if [[ -f "$REJ" ]]; then
        python3 -m src.analyze_rejections \
            --rejected "$REJ" \
            --out_csv "$ROOT/results/tables/rejection_breakdown.csv" \
            --tag "${CORR}_eta${ETATAG}_consensus_k3"
    fi
done

echo "[pilot] done."
