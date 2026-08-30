#!/usr/bin/env bash
# Load local credentials without ever storing a key in this repository.
# Usage from the repository root: `source scripts/set_api_key.sh`

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set. Copy .env.example to .env and add it locally." >&2
    return 1 2>/dev/null || exit 1
fi

export OPENAI_API_KEY
