# Route the focal model through OpenRouter without exposing other credentials.
# Source this file:  source scripts/openrouter_env.sh
#
# Mirrors gateway_env.sh. OpenRouter is OpenAI-compatible, so a model id
# prefixed `openai:` reaches it once OPENAI_BASE_URL points there. The paid
# OpenAI key is preserved under PAID_OPENAI_API_KEY first, so interactive runs
# can still use `paid-openai:` for the simulated user and the judge while the
# assistant seat runs on OpenRouter.
#
# Unlike the institutional gateway, OpenRouter bills per token. Nothing here prints
# a key value, and nothing here sets a spend cap -- that belongs on the
# OpenRouter account.

if [[ ! -f .env ]]; then
    echo "Missing .env in the repository root." >&2
    return 1
fi

set -a
eval "$(grep -E '^(OPENAI_API_KEY|OPENROUTER_API_KEY)=' .env)"
set +a

if [[ -z ${OPENROUTER_API_KEY:-} ]]; then
    echo "Missing OPENROUTER_API_KEY in .env." >&2
    return 1
fi

# Preserve the paid OpenAI key before OPENAI_API_KEY is replaced, exactly as the
# gateway script does, so `paid-openai:` keeps reaching the public endpoint.
export PAID_OPENAI_API_KEY="${OPENAI_API_KEY:-}"

export OPENAI_API_KEY="${OPENROUTER_API_KEY}"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
unset OPENROUTER_API_KEY

export PYTHONPATH=src

# The sentence-transformer used for deduplication is cached locally; keep its
# loaders from making update-check requests.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Three requests at a time. OpenRouter caps *in-flight* spend against
# available credits, and exceeding it returns HTTP 402
# (`in_flight_budget_exhausted`) rather than queueing. At 4 workers a
# ~$0.00025/call model failed all 150 scenarios that way while a
# ~$0.00003/call model on the same account succeeded, so the ceiling scales
# with per-call cost and cannot be assumed from one model's behaviour. The
# budget also rises with available credits, and the balance that failed at 4
# was a fraction of the present one, so 3 stays below the only value known to
# have broken while still tripling throughput. Raise further only after adding
# credits, which is what lifts the in-flight budget.
export CONFLICTXGAME_MAX_WORKERS=${CONFLICTXGAME_MAX_WORKERS:-3}

# A spent in-flight budget is the expected failure here, and no amount of
# retrying clears it -- backoff only burns wall-clock before the circuit
# breaker trips. One attempt surfaces a 402 while the queue can still react.
export CONFLICTXGAME_MAX_RETRIES=${CONFLICTXGAME_MAX_RETRIES:-1}
