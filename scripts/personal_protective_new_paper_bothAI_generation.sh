#!/usr/bin/env bash
# Fill every alternate personalprotective_new value-pair quota with
# paper-style, all-assistant games in a new, isolated production run.
#
# Optional overrides:
#   RUN_ROOT=... MAX_ROUNDS=... ROUND_BATCH=... SEED=... \
#       bash scripts/personal_protective_new_paper_bothAI_generation.sh

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# Paid-provider credentials and source-tree import path. This also removes the
# institutional gateway base URL so the paid OpenAI key is routed to api.openai.com.
# shellcheck source=paid_env.sh
source "${SCRIPT_DIR}/paid_env.sh"

RUN_ROOT=${RUN_ROOT:-data/personal_protective_new_paper_bothAI_20260825_01}
MAX_ROUNDS=${MAX_ROUNDS:-100}
ROUND_BATCH=${ROUND_BATCH:-30}
SEED=${SEED:-0}

exec python -u -m conflictxgame.games.generate_balanced \
    -v value_set_alt/personalprotective_new \
    -m claude-sonnet-5 --judge-model gpt-5.6-luna \
    -k 50 -o "${RUN_ROOT}/games" --csv-dir "${RUN_ROOT}" \
    --prompt-style paper --generation-guidance none \
    --assistant-seat-prob 1.0 --seed "${SEED}" \
    --batch-size 1 --max-tokens 16384 -d -dt 0.8 \
    --prescreen shadow --screen-model gpt-5.6-luna \
    --round-batch "${ROUND_BATCH}" --max-rounds "${MAX_ROUNDS}" \
    --max-stall-rounds 5
