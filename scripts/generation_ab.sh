#!/usr/bin/env bash
# Compare the generation designs implicated by the HHH game pilots.
#
# This script deliberately holds the generator and primary judge fixed.  A
# changed judge changes what "accepted" means, so a judge swap must be tested
# by agreement with an audit judge, not by comparing its raw kept rate.  To
# compare generator models, run this script once per GEN_MODEL with a distinct
# OUT directory, then compare the same arm across those otherwise identical
# runs.

set -euo pipefail

if [ -f scripts/paid_env.sh ]; then
    # Keys and provider routing stay outside version control.
    # shellcheck source=/dev/null
    source scripts/paid_env.sh
fi

VALUE_SET=${VALUE_SET:-HHH_new}
GEN_MODEL=${GEN_MODEL:-claude-sonnet-5}
JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.6-luna}
SCREEN_MODEL=${SCREEN_MODEL:-${JUDGE_MODEL}}
N=${N:-30}
OUT=${OUT:-analysis/generation_ab}
BATCH=${BATCH:-1}
MAX_TOKENS=${MAX_TOKENS:-16384}
PARALLEL=${PARALLEL:-0}

mkdir -p "${OUT}"

# Full checks expose every failure, which makes sole-cause diagnostics valid.
# Shadow screening means every outline still reaches the primary judge, so its
# precision (including shadow-only realism) can be measured in every arm.
COMMON=(
    -v "${VALUE_SET}" -m "${GEN_MODEL}" --judge-model "${JUDGE_MODEL}"
    --seed 0 --batch-size "${BATCH}" --max-tokens "${MAX_TOKENS}"
    --full-checks --prescreen shadow --screen-model "${SCREEN_MODEL}"
    -d -dt 0.8 -k 99999 --round-batch "${N}" --max-rounds 1 --max-stall-rounds 1
)

ARMS=(guided-bothAI none-bothAI paper-bothAI paper-oneAI)

arm_flags() {
    case "$1" in
        guided-bothAI)
            printf '%s\n' --prompt-style guided --generation-guidance checklist,recipes,stage2 --assistant-seat-prob 1.0
            ;;
        none-bothAI)
            printf '%s\n' --prompt-style guided --generation-guidance none --assistant-seat-prob 1.0
            ;;
        paper-bothAI)
            printf '%s\n' --prompt-style paper --assistant-seat-prob 1.0
            ;;
        paper-oneAI)
            printf '%s\n' --prompt-style paper --one-assistant-seat
            ;;
        *)
            echo "unknown arm: $1" >&2
            return 2
            ;;
    esac
}

run_arm() {
    local arm=$1
    local -a flags
    mapfile -t flags < <(arm_flags "${arm}")
    echo "=== ${arm} ==="
    python -m conflictxgame.games.generate_balanced "${COMMON[@]}" \
        -o "${OUT}/${arm}/games" --csv-dir "${OUT}/${arm}/data" \
        "${flags[@]}"
}

if [ "${PARALLEL}" = "1" ]; then
    pids=()
    for arm in "${ARMS[@]}"; do
        run_arm "${arm}" >"${OUT}/${arm}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}"
    done
else
    for arm in "${ARMS[@]}"; do
        run_arm "${arm}" | tee "${OUT}/${arm}.log"
    done
fi

# Kept rate plus a Wilson interval.  The run cap means these are pilot
# estimates, not a declaration of a winning configuration.
python - "${OUT}" "${ARMS[@]}" <<'PY'
import glob
import json
import os
import sys

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

root, *arms = sys.argv[1:]
for arm in arms:
    paths = glob.glob(os.path.join(root, arm, "data", "*_games.csv"))
    if not paths:
        print(f"{arm:15s}: no corpus produced")
        continue
    frame = pd.read_csv(paths[0])
    kept = frame["keep_scenario"].astype(str).str.lower().eq("true")
    n, k = len(frame), int(kept.sum())
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson") if n else (0, 0)
    line = f"{arm:15s}: {k}/{n} kept = {k / n:.1%} (95% CI {lo:.1%}-{hi:.1%})" if n else f"{arm:15s}: empty"
    coverage = os.path.join(root, arm, "games", "pair_coverage.json")
    if os.path.exists(coverage):
        with open(coverage, encoding="utf-8") as f:
            cost = json.load(f).get("cost", {})
        if cost.get("requests_per_kept") is not None:
            line += f" | {cost['requests_per_kept']:.1f} requests/kept"
    print(line)
PY

for arm in "${ARMS[@]}"; do
    echo "--- ${arm}: rejection causes ---"
    python -m conflictxgame.common.generation_quality \
        --csv "${OUT}/${arm}/data/"*_games.csv
    echo "--- ${arm}: pre-screen calibration ---"
    python -m conflictxgame.common.generation_quality --prescreen \
        --csv "${OUT}/${arm}/data/"*_games.csv
done

# Optional: re-judge each arm with a different model family.  Do not call a
# raw acceptance-rate change from a different judge an improvement; inspect
# agreement and the audited rejection reasons instead.
if [ -n "${AUDIT_JUDGE:-}" ]; then
    for arm in "${ARMS[@]}"; do
        echo "--- ${arm}: audit judge (${AUDIT_JUDGE}) ---"
        python -m conflictxgame.games.filter_game_scenarios \
            -i "${OUT}/${arm}/games/"*_games.json \
            -m "${AUDIT_JUDGE}" -v "${VALUE_SET}" \
            -o "${OUT}/${arm}/audit"
    done
fi
