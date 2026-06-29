#!/usr/bin/env bash
#
# Record mean generation time + constraint violation for each (example, method,
# denoising-steps) combination below:
#
#   examples:        star, mnist
#   methods:         pigdm (Pseudoinverse), pcfm (PCFM), ours (Dual Flows)
#   denoising steps: 10, 100
#
# Edit the PENALTY_* variables below to set the penalty weight per scenario. The
# other methods use their benchmark.py defaults.
#
# Usage:
#   ./run_benchmark_table.sh                 # default num samples
#   NUM_SAMPLES=10 ./run_benchmark_table.sh  # override sample count
#
set -euo pipefail

cd "$(dirname "$0")"

# Per-scenario penalty weights for our method.
PENALTY_STAR_10=${PENALTY_STAR_10:-3.0}
PENALTY_STAR_100=${PENALTY_STAR_100:-5.0}
PENALTY_MNIST_10=${PENALTY_MNIST_10:-10.0}
PENALTY_MNIST_100=${PENALTY_MNIST_100:-20.0}

# Number of samples to average over (more samples -> steadier timing).
NUM_SAMPLES=${NUM_SAMPLES:-20}

PYTHON=${PYTHON:-python3}

# Collected table rows, filled in as cases run.
RESULTS=()

# Look up the penalty for a scenario.
penalty_for() {
    local example=$1 steps=$2
    case "${example}_${steps}" in
        star_10)   echo "$PENALTY_STAR_10" ;;
        star_100)  echo "$PENALTY_STAR_100" ;;
        mnist_10)  echo "$PENALTY_MNIST_10" ;;
        mnist_100) echo "$PENALTY_MNIST_100" ;;
        *) echo "unknown scenario ${example}_${steps}" >&2; exit 1 ;;
    esac
}

# Pretty labels for the table.
example_label() { case "$1" in star) echo Star ;; mnist) echo MNIST ;; esac; }
method_label() {
    case "$1" in
        pigdm) echo "Pseudoinverse" ;;
        pcfm)  echo "PCFM" ;;
        ours)  echo "Dual Flows (ours)" ;;
    esac
}

# Run a single benchmark case and append a row to RESULTS.
run_case() {
    local example=$1 method=$2 steps=$3 penalty=$4
    # dt = 1/steps controls denoising resolution for the ODE-based methods.
    local dt
    dt=$($PYTHON -c "print(1.0/$steps)")

    echo ">>> $example / $method / ${steps} steps (penalty=$penalty)" >&2
    local extra=()
    case "$method" in
        ours)  extra=(--dt "$dt" --penalty-weight "$penalty") ;;
        pigdm) extra=(--dt "$dt") ;;
        pcfm)  extra=(--num-steps "$steps") ;;
    esac

    # Capture benchmark.py's stdout so we can scrape its summary lines.
    local log
    log=$($PYTHON benchmark.py \
        --example "$example" \
        --method "$method" \
        --num-samples "$NUM_SAMPLES" \
        "${extra[@]}")
    echo "$log" >&2

    # Summary lines look like:
    #   time/sample (ms): mean=   1.96  std= ...
    #   violation:        mean=2.345e-08  max= ...
    local time_ms viol
    time_ms=$(echo "$log" | grep 'time/sample' | sed -E 's/.*mean=[[:space:]]*([0-9.eE+-]+).*/\1/')
    viol=$(echo "$log" | grep 'violation:' | sed -E 's/.*mean=[[:space:]]*([0-9.eE+nan-]+).*/\1/')

    RESULTS+=("$(printf '%s\t%s\t%s\t%s\t%s' \
        "$(example_label "$example")" "$steps" "$(method_label "$method")" \
        "$time_ms" "$viol")")
}

EXAMPLES=(star mnist)
STEPS=(10 100)
METHODS=(pigdm pcfm ours)

for example in "${EXAMPLES[@]}"; do
    for steps in "${STEPS[@]}"; do
        penalty=$(penalty_for "$example" "$steps")
        for method in "${METHODS[@]}"; do
            run_case "$example" "$method" "$steps" "$penalty"
        done
    done
done

# ---------------------------------------------------------------------------
# Print the results table.
# ---------------------------------------------------------------------------
echo
fmt='%-8s %-15s %-18s %-20s %-20s\n'
printf "$fmt" "Example" "Denoising Steps" "Method" "Generation Time (ms)" "Constraint Violation"
printf "$fmt" "-------" "---------------" "------" "--------------------" "--------------------"
for row in "${RESULTS[@]}"; do
    IFS=$'\t' read -r example steps method time viol <<< "$row"
    printf "$fmt" "$example" "$steps" "$method" "$time" "$viol"
done
