#!/usr/bin/env bash
# One-command entry point. Examples:
#   bash scripts/run_modelscope.sh
#   RUN_CHAPTERS=4-6 bash scripts/run_modelscope.sh
#   bash scripts/run_modelscope.sh chapter 02
#   bash scripts/run_modelscope.sh doctor
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/workspace/agentic-rl-lab}"
export RUN_PROFILE="${RUN_PROFILE:-smoke}"
export MAX_EXTERNAL_COST_CNY="${MAX_EXTERNAL_COST_CNY:-50}"
export ALLOW_PAID_OVERAGE="${ALLOW_PAID_OVERAGE:-0}"
export FREE_TOKEN_LIMIT="${FREE_TOKEN_LIMIT:-1000000}"
export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$WORKSPACE_ROOT/cache/modelscope}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

mkdir -p "$WORKSPACE_ROOT" "$HF_HOME" "$MODELSCOPE_CACHE"
exec python3 scripts/run_modelscope.py "$@"
