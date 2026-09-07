#!/usr/bin/env bash
# Qwen3-32B ALFWorld Wilson experiment with:
#   1. paper equations (22), (24), (25), (26), and (27) set to one; and
#   2. Appendix A.3.2 section activation bounds set to one.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export NV_MEMCO_PAPER_A3_2_ACTIVATION_ALL_ONES=1

RUN_TIMEZONE="${RUN_TIMEZONE:-America/Vancouver}"
RUN_TS="${RUN_TS:-$(TZ="${RUN_TIMEZONE}" date +%Y%m%d_%H%M%S)}"
export RUN_TIMEZONE RUN_TS
export RUN_ID="${RUN_ID:-alfworld_memco_wilson_qwen32b_eq22_27_a3_2_activation_allones_${RUN_TS}}"

exec bash scripts/run_memco_wilson_qwen32b_alfworld_allones.sh
