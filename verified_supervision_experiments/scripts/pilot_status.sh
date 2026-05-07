#!/usr/bin/env bash
# One-shot status check for a running pilot. Print where the pipeline is and
# what's left.
cd "$(dirname "$0")/.."

PID_FILE="logs/pilot.pid"
if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if ps -p "$PID" > /dev/null 2>&1; then
        ETIME="$(ps -p "$PID" -o etime= | tr -d ' ')"
        echo "[pilot] running   pid=$PID   elapsed=$ETIME"
    else
        echo "[pilot] NOT running (pid=$PID exited)"
    fi
else
    echo "[pilot] no pid file"
fi

echo
echo "[stage 1: subsample]"
for f in data/processed/{hh_train,hh_dev,hh_test}.jsonl; do
    if [[ -f "$f" ]]; then echo "  $(wc -l < "$f") $f"; fi
done

echo
echo "[stage 2: corruption]"
for f in data/corrupted/*.summary.json; do
    if [[ -f "$f" ]]; then
        eta=$(python3 -c "import json; d=json.load(open('$f')); print(f\"eta={d['nominal_eta']:>4} effective={d['effective_eta']:.4f} corrupted={d['corrupted_rows']}/{d['input_rows']}\")")
        echo "  $eta"
    fi
done

echo
echo "[stage 3: precompute]"
TARGET=$(wc -l < data/processed/hh_train.jsonl 2>/dev/null || echo 0)
for s in 0 1; do
    f="results/verifier_outputs/hh_train.precomputed.shard${s}.jsonl"
    if [[ -f "$f" ]]; then
        n=$(wc -l < "$f")
        target_each=$((TARGET / 2))
        pct=$((100*n/target_each))
        echo "  shard $s: $n / $target_each rows ($pct%)"
    fi
done

echo
echo "[stage 4: filtered]"
for d in data/filtered/*/; do
    if [[ -f "$d/filter_summary.json" ]]; then
        echo "  $(basename "$d"): filter_summary.json present"
    fi
done

echo
echo "[stage 5: trained checkpoints]"
for d in results/checkpoints/*/*/; do
    if [[ -f "$d/train_status.json" ]]; then
        echo "  $(echo "$d" | sed 's,results/checkpoints/,,')"
    fi
done

echo
echo "[stage 6: evaluations]"
ls results/eval/*.json 2>/dev/null | sed 's,results/eval/,  ,'

echo
echo "[GPU]"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null

echo
echo "[recent log]"
tail -10 logs/pilot.log 2>/dev/null
