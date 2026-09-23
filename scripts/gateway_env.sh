# Load one institutional gateway API key without exposing paid-provider credentials.
# Source this file; optionally select key 1, 2, or 3 with GATEWAY_KEY_INDEX.

if [[ ! -f .env ]]; then
    echo "Missing .env in the repository root." >&2
    return 1
fi

GATEWAY_KEY_INDEX=${GATEWAY_KEY_INDEX:-1}
case "${GATEWAY_KEY_INDEX}" in
    1|2|3) ;;
    *)
        echo "GATEWAY_KEY_INDEX must be 1, 2, or 3." >&2
        return 2
        ;;
esac

set -a
eval "$(grep -E '^(OPENAI_API_KEY|OPENAI_BASE_URL|GATEWAY_API_KEY_[123])=' .env)"
set +a

# Preserve the paid OpenAI key under a client-specific name before replacing
# OPENAI_API_KEY with the selected institutional gateway key. Model strings using
# paid-openai: can then use the public endpoint in the same process.
export PAID_OPENAI_API_KEY="${OPENAI_API_KEY:-}"

GATEWAY_KEY_VARIABLE="GATEWAY_API_KEY_${GATEWAY_KEY_INDEX}"
if [[ -z ${OPENAI_BASE_URL:-} || -z ${!GATEWAY_KEY_VARIABLE:-} ]]; then
    echo "Missing OPENAI_BASE_URL or ${GATEWAY_KEY_VARIABLE} in .env." >&2
    unset GATEWAY_KEY_VARIABLE
    return 1
fi

export OPENAI_API_KEY="${!GATEWAY_KEY_VARIABLE}"
unset GATEWAY_API_KEY_1 GATEWAY_API_KEY_2 GATEWAY_API_KEY_3
unset GATEWAY_KEY_VARIABLE
export PYTHONPATH=src
# Requests in flight per batch_generate call. Raised from 4 after measuring the
# gateway: Gemma's cost is almost entirely prefill (~17ms per input token against
# GPT-OSS's ~1ms), so a 550-token scenario takes it ~10.9s against GPT-OSS's
# ~1.3s and the runs are latency-bound, not compute-bound. A sweep at 4/8/16
# concurrent against a gateway already serving three live Gemma jobs returned
# 0.20/0.28/0.42 req/s with no errors at any level -- more workers keep helping,
# sub-linearly, and queue rather than fail. 8 takes the reliable part of that
# gain; 16 also measured clean if a lane needs pushing harder.
export CONFLICTXGAME_MAX_WORKERS=${CONFLICTXGAME_MAX_WORKERS:-8}
