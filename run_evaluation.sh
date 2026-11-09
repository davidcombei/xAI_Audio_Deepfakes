#!/usr/bin/env bash
# Evaluate every explanation method on realistic, freq-swapped and freq-swapped-single, compute
# each setup's detector EER, then print the results.
#
#   ./run_evaluation.sh                 run all jobs, then print the results
#   ./run_evaluation.sh summary         only print the results of finished jobs
#
#   GPUS="2 3 4 5" ./run_evaluation.sh     choose GPUs
#
# All runs use real fill: masked-out bins are filled from the bona fide clip. Nothing uses zero fill.
#
# Jobs per setup:
#   optimized          the trained U-Net (printed as "learned_mask")
#   random             random masks, the baseline
#   saliency, input-x-gradient, gradient-shap
#   eer                detector EER on the setup's fakes + their bona fide clips (compute_eer.py)
# Metrics: Fidelity, Faithfulness, Average Drop/Increase/Gain for every setup; AUC and RMA only for
# the swap setups (realistic has no ground-truth region).
#
# Each GPU takes the next job from a shared queue until the queue is empty.
# Random masks are evaluated once per setup as their own job ("-e random"), so the other jobs
# run with --no-random-baseline instead of repeating the same random baseline four times.
# Logs: logs/evaluation/<setup>__<job>.out (results) and .err (progress bars, warnings).
# Example clips (wav + audio .npy + every method's mask .npy): images_npy/<setup>/, listed in files.txt.

set -u
cd "$(dirname "$0")"

# Listed slowest first (more clips, slower methods), so the long jobs start first and the GPUs
# finish at about the same time.
SETUPS=${SETUPS:-"freq-swapped realistic freq-swapped-single"}
METHODS=${METHODS:-"gradient-shap saliency input-x-gradient optimized random"}
GPUS=${GPUS:-"2 3 4 5 6 7"}
FILL=real
BATCH_SIZE=${BATCH_SIZE:-4}
# Gradient-SHAP peaks at 29.4 GiB on a 32 GiB GPU with batch 4.
BATCH_SIZE_GRADIENT_SHAP=${BATCH_SIZE_GRADIENT_SHAP:-2}
PYTHON=${PYTHON:-.venv/bin/python}
LOG_DIR=logs/evaluation
# Jobs that need extra flags, as "<setup>/<method>=<flags>".
EXTRA_FLAGS=${EXTRA_FLAGS:-""}
# If another evaluate.py run of the same setup/method is already filling its cache, use --no-cache
# for that job so the two processes never read each other's half-written files.
for setup in $SETUPS; do
    for method in $METHODS; do
        if pgrep -f "evaluate.py .*(-s|--setup)[ =]$setup( |$).*(-e|--explanation)[ =]$method( |$)" > /dev/null; then
            echo "another evaluate.py $setup/$method is running: that job will use --no-cache"
            EXTRA_FLAGS="$EXTRA_FLAGS $setup/$method=--no-cache"
        fi
    done
done

summary() {
    "$PYTHON" - "$LOG_DIR" <<'EOF'
import re, sys
from pathlib import Path

log_dir = Path(sys.argv[1])
rows, eer_rows, missing = [], [], []
for out in sorted(log_dir.glob("*.out")):
    setup, job = out.stem.split("__")
    # evaluate.py and compute_eer.py print "<label>:" and then "  <metric>: <value>" lines
    blocks, label = {}, None
    for line in out.read_text().splitlines():
        if re.fullmatch(r"\S+:", line):
            label = line[:-1]
            blocks[label] = {}
        elif label and (m := re.fullmatch(r"  (.+): (-?[\d.]+(?:e-?\d+)?|nan)", line)):
            blocks[label][m[1]] = m[2]
        else:
            label = None
    blocks = {k: v for k, v in blocks.items() if v}
    if not blocks:
        missing.append(f"{setup}/{job}")
    for label, metrics in blocks.items():
        (eer_rows if job == "eer" else rows).append((setup, label, metrics))


def print_table(rows, first_cols, short={}):
    names = []
    for _, _, m in rows:
        names += [k for k in m if k not in names]
    head = first_cols + [short.get(n, n) for n in names]
    table = [[s, l][:len(first_cols)] + [m.get(n, "-") for n in names] for s, l, m in rows]
    widths = [max(len(str(r[i])) for r in [head] + table) for i in range(len(head))]
    fmt = "  ".join(f"{{:{'<' if i < len(first_cols) else '>'}{w}}}" for i, w in enumerate(widths))
    print(fmt.format(*head))
    print("  ".join("-" * w for w in widths))
    prev = None
    for r in table:
        if prev is not None and r[0] != prev and len(first_cols) > 1:
            print()
        prev = r[0]
        print(fmt.format(*r))


if rows:
    print("EXPLANATIONS (real fill)\n")
    print_table(rows, ["setup", "method"], {
        "Average Drop": "AvgDrop", "Average Increase": "AvgInc", "Average Gain": "AvgGain",
        "Localization": "AUC", "Localization RMA": "RMA", "Localization RMA (chance)": "RMA chance"})
if eer_rows:
    print("\nDETECTOR EER (setup's fakes + their bona fide clips)\n")
    print_table(eer_rows, ["setup"])
if missing:
    print("\nno results (still running or failed, see the .err log):", ", ".join(missing))

for d in sorted(Path("images_npy").glob("*/files.txt")):
    print(f"\nexample clips in {d.parent}/:")
    for line in d.read_text().split():
        print("  " + line)
EOF
}

if [[ "${1:-}" == "summary" ]]; then
    summary
    exit 0
fi

mkdir -p "$LOG_DIR"
QUEUE="$LOG_DIR/queue.txt"
: > "$QUEUE"
for method in $METHODS; do
    for setup in $SETUPS; do
        echo "$setup $method" >> "$QUEUE"
    done
done
for setup in $SETUPS; do
    echo "$setup eer" >> "$QUEUE"
done

next_job() {
    # Pop the first line of the queue (flock: several GPU workers share it).
    flock "$QUEUE.lock" bash -c "head -n 1 '$QUEUE'; sed -i 1d '$QUEUE'"
}

extra_flags() {
    for entry in $EXTRA_FLAGS; do
        [[ "${entry%%=*}" == "$1" ]] && echo "${entry#*=}" | tr ',' ' '
    done
}

worker() {
    local gpu=$1
    while true; do
        local job setup method log flags batch status
        job=$(next_job)
        [[ -z "$job" ]] && break
        read -r setup method <<< "$job"
        log="$LOG_DIR/${setup}__${method}"
        if [[ "$method" == "eer" ]]; then
            echo "[$(date +%H:%M)] GPU $gpu: $setup / detector EER"
            OMP_NUM_THREADS=8 "$PYTHON" compute_eer.py -s "$setup" --device "cuda:$gpu" \
                --scores-csv "$LOG_DIR/${setup}__eer_scores.csv" > "$log.out" 2> "$log.err"
        else
            flags="$(extra_flags "$setup/$method")"
            [[ "$method" != "random" ]] && flags="$flags --no-random-baseline"
            batch=$BATCH_SIZE
            [[ "$method" == "gradient-shap" ]] && batch=$BATCH_SIZE_GRADIENT_SHAP
            echo "[$(date +%H:%M)] GPU $gpu: $setup / $method -b $batch --fill $FILL $flags"
            OMP_NUM_THREADS=8 "$PYTHON" evaluate.py -s "$setup" -e "$method" -b "$batch" \
                --device "cuda:$gpu" --fill "$FILL" $flags > "$log.out" 2> "$log.err"
        fi
        status=$?
        echo "[$(date +%H:%M)] GPU $gpu: $setup / $method finished (exit code $status)"
    done
}

echo "$(wc -l < "$QUEUE") jobs on GPUs $GPUS, fill=$FILL, logs in $LOG_DIR/"
for gpu in $GPUS; do
    worker "$gpu" &
done
wait
echo
summary
