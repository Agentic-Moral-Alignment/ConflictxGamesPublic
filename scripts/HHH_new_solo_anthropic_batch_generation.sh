#!/usr/bin/env bash
# Launch the HHH-only solo Message Batches arm from
# configs/solo/HHH_new_solo_20260826_01_generate_generator-claude-sonnet-5.yaml.
#
# This is a separate Anthropic queue from the games arm
# data/HHH_new_paper_bothAI_anthropic_batch_20260826_01. Do not share
# message_batch_state_dir with that run.
#
# Only the three HHH pairs are generated (harmlessness/helpfulness/honesty)
# using the HHH_new wording. 50 skeletons per pair.
#
#   bash scripts/HHH_new_solo_anthropic_batch_generation.sh

set -euo pipefail

if (( $# != 0 )); then
    echo "This launcher accepts no positional options." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# shellcheck source=paid_env.sh
source "${SCRIPT_DIR}/paid_env.sh"

CONFIG_FILE=${CONFIG_FILE:-configs/solo/HHH_new_solo_20260826_01_generate_generator-claude-sonnet-5.yaml}
RUN_ROOT=${RUN_ROOT:-data/HHH_new_solo_20260826_01}
mkdir -p "${RUN_ROOT}"

exec python -u -m conflictxgame.solo.generate_scenarios \
    --config "${CONFIG_FILE}" \
    --all-hhh-pairs \
    >>"${RUN_ROOT}/generation.stdout.log" \
    2>>"${RUN_ROOT}/generation.stderr.log"
