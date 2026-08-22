#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-${SCRIPT_DIR}/combustion_stsr_mvnet.list}"
CONDA_ENV="${CONDA_ENV:-compression}"
RUN_TOKEN="${RUN_TOKEN:-combustion_stsr_mvnet_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
    CONFIG_LIST_FILE="$(cygpath -u "${CONFIG_LIST_FILE}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
BATCH_LABEL="STSR-INR + MVNet Combustion"
STAGE_LABELS=("STSR-INR Combustion" "MVNet Combustion")
STAGE_PATTERNS=(
    'configs/main/STSR-INR/combustion_40NH3_1.yaml'
    'configs/main/MVNet/combustion_40NH3_1.yaml'
)
STAGE_EXPECTED=(1 1)

source "${SCRIPT_DIR}/../lib/batch_runner.sh"
source "${SCRIPT_DIR}/../lib/combustion_batch_runner.sh"
combustion_batch_main
