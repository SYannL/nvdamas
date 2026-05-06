#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

export MODEL="${MODEL:-claude-haiku-4-5}"
export MAS_MEMORY="${MAS_MEMORY:-g-memory}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.anthropic.com/v1/}"

bash scripts/fever/run_global_fever.sh
