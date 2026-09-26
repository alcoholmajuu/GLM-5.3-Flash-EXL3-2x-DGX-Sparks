#!/bin/bash
# SPDX-License-Identifier: MIT
# Run Claude Code against the GLM-5.3-Flash vLLM server (Anthropic Messages API).
#
# Usage: scripts/run_claude_code.sh [--list-models] [--shared-config] [claude args...]
#
# Model handling — selectable, nothing global is overwritten:
#   - The /model picker keeps its built-in rows and additionally lists every
#     model the server serves (gateway discovery via GET /v1/models).
#   - The session default is the first served model, unless you pass
#     --model, set GLM53_MODEL, or already export ANTHROPIC_DEFAULT_MODEL /
#     ANTHROPIC_MODEL yourself (those win).
#   - Claude Code's config dir is redirected to ~/.claude-glm53 so a model
#     saved from the picker (Enter) never rewrites your normal ~/.claude
#     settings. Opt out with --shared-config or GLM53_SHARED_CONFIG=1.
#   - The opus/sonnet/haiku alias env vars are NOT set. Background tasks run
#     on the session model because ANTHROPIC_AUTH_TOKEN is set.
set -eu

PORT=${GLM53_PORT:-8000}
HOST=${GLM53_SSH_HOST:-}
if [ -z "$HOST" ] && [ -f .env ]; then
    # shellcheck disable=SC1091
    . ./.env
    HOST=${RANK0_HOST:-}
fi

list_only=0; shared=0
args=()
while [ $# -gt 0 ]; do
    case $1 in
        --list-models) list_only=1 ;;
        --shared-config) shared=1 ;;
        *) args+=("$1") ;;
    esac
    shift
done

if ! nc -z 127.0.0.1 "$PORT" 2>/dev/null; then
    if [ -z "$HOST" ]; then
        echo "error: no tunnel on port $PORT and no host known; set GLM53_SSH_HOST or RANK0_HOST in .env" >&2
        exit 1
    fi
    echo "opening tunnel: ssh -f -N -L $PORT:127.0.0.1:$PORT $HOST" >&2
    ssh -f -N -L "$PORT:127.0.0.1:$PORT" "$HOST"
    sleep 1
fi

# Served models, one "id max_model_len" pair per line.
models=$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    python3 -c 'import json,sys
try:
    for m in json.load(sys.stdin).get("data", []):
        print(m.get("id", ""), m.get("max_model_len", ""))
except Exception:
    pass' 2>/dev/null || true)

if [ "$list_only" = 1 ]; then
    echo "Served models (tunnel on port $PORT):"
    if [ -n "$models" ]; then
        printf '%s\n' "$models" | while read -r id ctx; do
            echo "  $id${ctx:+  (context $ctx)}"
        done
    else
        echo "  (could not list; server did not answer /v1/models)"
    fi
    exit 0
fi

default_id=$(printf '%s\n' "$models" | head -1 | awk '{print $1}')
[ -n "$default_id" ] || default_id=glm53-exl3

ctx_for() {
    printf '%s\n' "$models" | awk -v m="$1" '$1==m && $2!="" {print $2; exit}'
}

# Session default model: --model flag > GLM53_MODEL > pre-set env > first served.
pass_model=0
for a in "${args[@]+"${args[@]}"}"; do
    case $a in --model|--model=*) pass_model=1 ;; esac
done
if [ "$pass_model" = 0 ]; then
    if [ -n "${GLM53_MODEL:-}" ]; then
        export ANTHROPIC_DEFAULT_MODEL=$GLM53_MODEL
    elif [ -z "${ANTHROPIC_DEFAULT_MODEL:-}" ] && [ -z "${ANTHROPIC_MODEL:-}" ]; then
        export ANTHROPIC_DEFAULT_MODEL=$default_id
    fi
fi

# Declare the real context window of the selected model so auto-compact does
# not assume 200k.
if [ -z "${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-}" ]; then
    chosen=${ANTHROPIC_DEFAULT_MODEL:-${GLM53_MODEL:-$default_id}}
    detected=$(ctx_for "$chosen")
    [ -n "$detected" ] && export CLAUDE_CODE_MAX_CONTEXT_TOKENS=$detected
fi

# /model picker: list what the server serves, keeping the built-in rows.
# With discovery disabled, fall back to one explicit custom entry.
if [ "${GLM53_NO_DISCOVERY:-0}" != "1" ]; then
    export CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1
elif [ -z "${ANTHROPIC_CUSTOM_MODEL_OPTION:-}" ]; then
    export ANTHROPIC_CUSTOM_MODEL_OPTION=${GLM53_MODEL:-$default_id}
    export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME=${ANTHROPIC_CUSTOM_MODEL_OPTION_NAME:-GLM-5.3-Flash EXL3 (local vLLM)}
fi

# Keep picker saves (Enter) out of the normal ~/.claude settings.
if [ "$shared" = 0 ] && [ "${GLM53_SHARED_CONFIG:-0}" != "1" ]; then
    export CLAUDE_CONFIG_DIR=$HOME/.claude-glm53
    mkdir -p "$CLAUDE_CONFIG_DIR"
fi

export ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-http://localhost:$PORT}
export ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-dummy}
export ANTHROPIC_AUTH_TOKEN=${ANTHROPIC_AUTH_TOKEN:-dummy}

echo "glm53: base_url=$ANTHROPIC_BASE_URL default_model=${ANTHROPIC_DEFAULT_MODEL:-(--model flag / inherited env)} config_dir=${CLAUDE_CONFIG_DIR:-~/.claude (shared)}" >&2

exec claude "${args[@]+"${args[@]}"}"
