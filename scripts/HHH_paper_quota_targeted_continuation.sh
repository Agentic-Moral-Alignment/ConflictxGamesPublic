#!/usr/bin/env bash
# Resume the HHH_new paper/both-assistant corpus while assigning only value
# pairs with a positive accepted-count deficit.  The unordered pair target is
# validated immediately after stage one, before screen, elaboration, or judge
# spend.  Smaller rounds limit overshoot when a remaining pair nears 50.
#
# Optional overrides:
#   RUN_ROOT=... MAX_ROUNDS=... ROUND_BATCH=... SEED=... \
#       bash scripts/HHH_paper_quota_targeted_continuation.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# shellcheck source=paid_env.sh
source "${SCRIPT_DIR}/paid_env.sh"

RUN_ROOT=${RUN_ROOT:-data/HHH_new_paper_bothAI_20260825_01}
MAX_ROUNDS=${MAX_ROUNDS:-100}
ROUND_BATCH=${ROUND_BATCH:-10}
SEED=${SEED:-0}

exec python -u -m conflictxgame.games.generate_balanced \
    -v HHH_new -m claude-sonnet-5 --judge-model gpt-5.6-luna \
    -k 50 -o "${RUN_ROOT}/games" --csv-dir "${RUN_ROOT}" \
    --prompt-style paper --paper-quota-targeting --generation-guidance none \
    --assistant-seat-prob 1.0 --seed "${SEED}" \
    --batch-size 1 --max-tokens 16384 -d -dt 0.8 \
    --prescreen shadow --screen-model gpt-5.6-luna \
    --round-batch "${ROUND_BATCH}" --max-rounds "${MAX_ROUNDS}" \
    --max-stall-rounds 5
