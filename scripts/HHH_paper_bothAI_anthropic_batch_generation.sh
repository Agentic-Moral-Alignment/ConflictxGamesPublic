#!/usr/bin/env bash
# Launch the isolated HHH Anthropic Message Batches arm from one auditable YAML
# configuration. Generation settings belong in that file, never in this shell
# wrapper. The wrapper only loads paid credentials, validates the staged fork,
# and gives the run fixed root-local logs.
#
# Optional override:
#
#   CONFIG_FILE=configs/another_run.json \
#       bash scripts/HHH_paper_bothAI_anthropic_batch_generation.sh
#
# Read-only validation (runs the fork/hash checks and quota status, no clients):
#
#   CHECK_ONLY=1 bash scripts/HHH_paper_bothAI_anthropic_batch_generation.sh
#
# Safe periodic completion check for this shadow-to-off continuation:
#
#   python -m conflictxgame.games.quota_monitor \
#       --run-root data/HHH_new_paper_bothAI_anthropic_batch_20260826_01 \
#       --value-set HHH_new --model claude-sonnet-5 --target 50 \
#       --expected-prompt-style paper --expected-assistant-seats prob:1.0 \
#       --allow-shadow-to-off-prescreen-transition

set -euo pipefail

if (( $# != 0 )); then
    echo "This launcher accepts no positional options; set CONFIG_FILE instead." >&2
    exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

# shellcheck source=paid_env.sh
source "${SCRIPT_DIR}/paid_env.sh"

CONFIG_FILE=${CONFIG_FILE:-configs/games/HHH_new_paper_bothAI_anthropic_batch_20260826_01_generate_generator-claude-sonnet-5_judge-gpt-5.6-luna.yaml}
CHECK_ONLY=${CHECK_ONLY:-0}
if [[ ${CHECK_ONLY} != 0 && ${CHECK_ONLY} != 1 ]]; then
    echo "CHECK_ONLY must be 0 or 1." >&2
    exit 2
fi

# Print only the validated absolute run root to stdout; diagnostics go to
# stderr so command substitution remains unambiguous.
RUN_ROOT=$(python - "${CONFIG_FILE}" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

from conflictxgame.common.config import load_config
from conflictxgame.games.pair_quota import kept_pair_counts_from_csv


config_path = Path(sys.argv[1])
try:
    config = load_config(str(config_path))
except (OSError, ValueError) as exc:
    raise SystemExit(f"Cannot read CONFIG_FILE {config_path}: {exc}")
if not isinstance(config, dict):
    raise SystemExit("CONFIG_FILE must contain one mapping/object.")
if config.get("anthropic_message_batches") is not True:
    raise SystemExit(
        "Refusing to start this batch launcher: CONFIG_FILE must set "
        '"anthropic_message_batches": true.'
    )

raw_run_root = config.get("run_root")
if not isinstance(raw_run_root, str) or not raw_run_root.strip():
    raise SystemExit("CONFIG_FILE must set a non-empty string run_root.")
run_root = Path(raw_run_root).resolve()
state_dir = Path(
    config.get("message_batch_state_dir") or run_root / "message_batches"
).resolve()

try:
    state_dir.relative_to(run_root)
except ValueError:
    raise SystemExit(
        "Message Batch state must be inside run_root; refusing a shared journal."
    )

manifest_path = run_root / "fork_manifest.json"
csv_path = run_root / "claude-sonnet-5_games.csv"
required = (
    manifest_path,
    csv_path,
    run_root / "games" / "claude-sonnet-5_games.json",
    run_root / "games" / "claude-sonnet-5_game_prompts.json",
    run_root / "games" / "coverage_report.json",
    run_root / "games" / "pair_coverage.json",
)
missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise SystemExit(
        "Refusing to start: the staged fork is missing required non-empty files: "
        + ", ".join(missing)
    )

try:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"Cannot read staged fork manifest: {exc}")
if manifest.get("schema_version") != 1:
    raise SystemExit("Unsupported staged fork manifest schema.")

raw_source_root = manifest.get("source_run_root")
if not isinstance(raw_source_root, str) or not raw_source_root:
    raise SystemExit("Staged fork manifest has no source_run_root.")
source_root = Path(raw_source_root).resolve()
if run_root == source_root or run_root in source_root.parents or source_root in run_root.parents:
    raise SystemExit(
        "Refusing to start: run_root and the synchronous source root are equal or nested."
    )

# A non-empty provider journal means this fork has legitimately evolved beyond
# its seed hash. Before its first batch submission, require the exact verified
# CSV checkpoint recorded in fork_manifest.json.
state_has_entries = state_dir.is_dir() and next(state_dir.iterdir(), None) is not None
if not state_has_entries:
    checkpoint = manifest.get("source_checkpoint")
    if not isinstance(checkpoint, dict):
        raise SystemExit("Staged fork manifest has no source_checkpoint.")
    expected_bytes = checkpoint.get("csv_bytes")
    expected_digest = checkpoint.get("csv_sha256")
    digest = hashlib.sha256()
    with csv_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if (
        csv_path.stat().st_size != expected_bytes
        or digest.hexdigest().upper() != expected_digest
    ):
        raise SystemExit(
            "Staged CSV does not match fork_manifest.json, and no Message Batch "
            "journal exists to explain an evolved corpus."
        )

value_names = ("harmlessness", "helpfulness", "honesty")
kept, generated = kept_pair_counts_from_csv(str(csv_path), value_names)
completed_pair = ("helpfulness", "honesty")
completed_kept = kept.get(completed_pair, 0)
if completed_kept < 50:
    raise SystemExit(
        "Staged corpus is not the required continuation seed: "
        f"helpfulness|honesty has only {completed_kept}/50 accepted marginals."
    )
print(
    "Fork preflight passed: helpfulness|honesty has "
    f"{completed_kept} accepted and {generated.get(completed_pair, 0)} generated "
    "marginals; completed-only outlines can be pruned after stage one.",
    file=sys.stderr,
)
print(run_root)
PY
)

if [[ ${CHECK_ONLY} == 1 ]]; then
    exec python -m conflictxgame.games.generate_balanced \
        --config "${CONFIG_FILE}" --status
fi

STDOUT_LOG=${RUN_ROOT}/generation.stdout.log
STDERR_LOG=${RUN_ROOT}/generation.stderr.log

exec python -u -m conflictxgame.games.generate_balanced \
    --config "${CONFIG_FILE}" \
    >>"${STDOUT_LOG}" 2>>"${STDERR_LOG}"
