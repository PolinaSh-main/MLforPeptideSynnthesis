#!/usr/bin/env bash
# run_all_experiments.sh
# Usage: bash run_all_experiments.sh --data_path /path/to/data --output_base /path/to/results
# Optional: --n_jobs 4   (parallel workers, default=4)
#           --n_repeats 20 (for whole_set_shuffled, default=20)

set -euo pipefail

# ── Args ────────────────────────────────────────────────────────────────────
DATA_PATH=""
OUTPUT_BASE="../results"

N_JOBS=4
N_REPEATS=20

WOF_START=2
WOF_END=2

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)   DATA_PATH="$2";   shift 2 ;;
        --output_base) OUTPUT_BASE="$2"; shift 2 ;;
        --n_jobs)      N_JOBS="$2";      shift 2 ;;
        --n_repeats)   N_REPEATS="$2";   shift 2 ;;
        --wof_start)   WOF_START="$2";   shift 2 ;;
        --wof_end)     WOF_END="$2";     shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$DATA_PATH" ]]; then
    echo "ERROR: --data_path is required"; exit 1
fi

mkdir -p "$OUTPUT_BASE"
LOGDIR="$OUTPUT_BASE/_logs"
mkdir -p "$LOGDIR"
WOF_FLAGS="--wof_start $WOF_START --wof_end $WOF_END"

# ── Job queue ────────────────────────────────────────────────────────────────
# Each entry: "tag|loader|preprocessor|model|extra_flags"
# extra_flags — любые доп. аргументы train_sklearn_models.py через пробел

JOBS=()

# ── OLD: статья 2026 — occurency + rff/xgb, wof_set и whole_set ─────────────
JOBS+=("old__wof__occurency__xgb|wof_set|occurency|xgb|$WOF_FLAGS")
JOBS+=("old__wof__occurency__rff|wof_set|occurency|rff|$WOF_FLAGS")
JOBS+=("old__whole__occurency__xgb|whole_set|occurency|xgb|")
JOBS+=("old__whole__occurency__rff|whole_set|occurency|rff|")
JOBS+=("old__whole__occurency__knn|whole_set|occurency|knn|")

# ── OLD: fingerprint нативный ────────────────────────────────────────────────
JOBS+=("old__wof__fingerprint__xgb|wof_set|fingerprint|xgb|$WOF_FLAGS")
JOBS+=("old__wof__fingerprint__rff|wof_set|fingerprint|rff|$WOF_FLAGS")
JOBS+=("old__whole__fingerprint__xgb|whole_set|fingerprint|xgb|")
JOBS+=("old__whole__fingerprint__rff|whole_set|fingerprint|rff|")

# ── OLD: shuffled baseline ───────────────────────────────────────────────────
JOBS+=("old__shuffled__occurency__xgb|whole_set_shuffled|occurency|xgb|--n_repeats $N_REPEATS")
JOBS+=("old__shuffled__fingerprint__xgb|whole_set_shuffled|fingerprint|xgb|--n_repeats $N_REPEATS")

# ── NEW A1: protected fingerprint ────────────────────────────────────────────
JOBS+=("new__wof__protected_fp__xgb|wof_set|protected_fingerprint|xgb|$WOF_FLAGS")
JOBS+=("new__wof__protected_fp__rff|wof_set|protected_fingerprint|rff|$WOF_FLAGS")
JOBS+=("new__whole__protected_fp__xgb|whole_set|protected_fingerprint|xgb|")
JOBS+=("new__whole__protected_fp__rff|whole_set|protected_fingerprint|rff|")
JOBS+=("new__shuffled__protected_fp__xgb|whole_set_shuffled|protected_fingerprint|xgb|--n_repeats $N_REPEATS")

# ── NEW A2: positional fingerprint ───────────────────────────────────────────
JOBS+=("new__wof__positional_fp__xgb|wof_set|positional_fingerprint|xgb|$WOF_FLAGS")
JOBS+=("new__wof__positional_fp__rff|wof_set|positional_fingerprint|rff|$WOF_FLAGS")
JOBS+=("new__whole__positional_fp__xgb|whole_set|positional_fingerprint|xgb|")
JOBS+=("new__whole__positional_fp__rff|whole_set|positional_fingerprint|rff|")

# ── NEW A3: whole peptide fingerprint (native) ───────────────────────────────
JOBS+=("new__wof__whole_fp__xgb|wof_set|whole_peptide_fingerprint|xgb|$WOF_FLAGS")
JOBS+=("new__wof__whole_fp__rff|wof_set|whole_peptide_fingerprint|rff|$WOF_FLAGS")
JOBS+=("new__whole__whole_fp__xgb|whole_set|whole_peptide_fingerprint|xgb|")
JOBS+=("new__whole__whole_fp__rff|whole_set|whole_peptide_fingerprint|rff|")

# ── NEW A3+A1: protected whole peptide fingerprint ───────────────────────────
JOBS+=("new__wof__prot_whole_fp__xgb|wof_set|protected_whole_peptide_fingerprint|xgb|$WOF_FLAGS")
JOBS+=("new__wof__prot_whole_fp__rff|wof_set|protected_whole_peptide_fingerprint|rff|$WOF_FLAGS")
JOBS+=("new__whole__prot_whole_fp__xgb|whole_set|protected_whole_peptide_fingerprint|xgb|")
JOBS+=("new__whole__prot_whole_fp__rff|whole_set|protected_whole_peptide_fingerprint|rff|")

# ── NEW B1: regression — первый раз на whole_set ─────────────────────────────
JOBS+=("new__whole__occurency__xgb__regression|whole_set|occurency|xgb|--task regression")
JOBS+=("new__whole__occurency__rff__regression|whole_set|occurency|rff|--task regression")
JOBS+=("new__whole__protected_fp__xgb__regression|whole_set|protected_fingerprint|xgb|--task regression")
JOBS+=("new__whole__protected_fp__rff__regression|whole_set|protected_fingerprint|rff|--task regression")
JOBS+=("new__whole__whole_fp__xgb__regression|whole_set|whole_peptide_fingerprint|xgb|--task regression")
JOBS+=("new__whole__prot_whole_fp__xgb__regression|whole_set|protected_whole_peptide_fingerprint|xgb|--task regression")

# ── NEW B2: step-level predictions ───────────────────────────────────────────
JOBS+=("new__wof__occurency__xgb__steps|wof_set|occurency|xgb|--save_step_predictions|$WOF_FLAGS")
JOBS+=("new__wof__protected_fp__xgb__steps|wof_set|protected_fingerprint|xgb|--save_step_predictions|$WOF_FLAGS")

# ── Runner ───────────────────────────────────────────────────────────────────
PIDS=()
TAGS=()
STATUSES=()

run_job() {
    local entry="$1"
    local tag loader preprocessor model extra_flags
    IFS='|' read -r tag loader preprocessor model extra_flags <<< "$entry"

    local out_dir="$OUTPUT_BASE/$tag"
    mkdir -p "$out_dir"

    local logfile="$LOGDIR/${tag}.log"

    # shellcheck disable=SC2086
    train_sklearn_model \
        --data_path "$DATA_PATH" \
        --output_path "$out_dir" \
        --loader "$loader" \
        --preprocessor "$preprocessor" \
        --model "$model" \
        $extra_flags \
        > "$logfile" 2>&1
}

export -f run_job
export DATA_PATH OUTPUT_BASE LOGDIR

echo "── Starting ${#JOBS[@]} experiments with $N_JOBS parallel workers ──"

# GNU parallel if available, else xargs
FAILED=0

if command -v parallel &>/dev/null; then
    printf '%s\n' "${JOBS[@]}" | \
        parallel --jobs "$N_JOBS" --bar run_job {} || FAILED=1
else
    printf '%s\n' "${JOBS[@]}" | \
        xargs -P "$N_JOBS" -I{} bash -c 'run_job "$@"' _ {} || FAILED=1
fi

if [[ $FAILED -eq 1 ]]; then
    echo ""
    echo "WARNING: Some experiments failed."
    echo "Continuing with aggregation."
    echo ""
fi

echo "── All jobs finished. Collecting results ──"

python3 -m ai4agg.cli.explainability \
    --aggregate_only \
    --output_path "$OUTPUT_BASE"
    
echo ""
echo "── Done. Results: $OUTPUT_BASE/all_results.csv ──"