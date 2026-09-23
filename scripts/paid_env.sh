# Load the PAID API credentials from .env for a real (non-gateway) run.
#
# Source this, do not execute it:   source scripts/paid_env.sh
#
# .env keeps two kinds of credentials side by side: the institutional gateway
# (OPENAI_BASE_URL + GATEWAY_API_KEY_*) and the paid provider keys. Only the
# paid keys are exported here, and OPENAI_BASE_URL is explicitly unset: if it
# leaked into the environment (from .env, or from the conda env's config vars),
# the OpenAI SDK would send the paid key to the gateway and every judge call
# would be rejected. Nothing here prints a key value.
#
# The Anthropic SDK reads ANTHROPIC_API_KEY; .env spells it Anthropic_API_KEY.

set -a
eval "$(grep -E '^(OPENAI_API_KEY|Anthropic_API_KEY)=' .env)"
set +a
export ANTHROPIC_API_KEY="$Anthropic_API_KEY"
export PAID_OPENAI_API_KEY="$OPENAI_API_KEY"
unset OPENAI_BASE_URL Anthropic_API_KEY

# The sentence-transformer used for scenario deduplication is already cached
# locally. Prevent its loaders from making update-check HEAD requests, which
# add retries and can trigger a Miniconda console in restricted environments.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The package is importable from the source tree without an install.
export PYTHONPATH=src

# Overlaps stage-two generation calls (batch_generate); stage one and the
# judge issue one call at a time regardless. Paid endpoints tolerate this.
export CONFLICTXGAME_MAX_WORKERS=${CONFLICTXGAME_MAX_WORKERS:-4}
