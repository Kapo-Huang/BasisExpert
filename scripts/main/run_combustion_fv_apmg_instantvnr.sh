#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-${SCRIPT_DIR}/combustion_fv_apmg_instantvnr.list}"
RUN_TOKEN="${RUN_TOKEN:-combustion_fv_apmg_instantvnr_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
    CONFIG_LIST_FILE="$(cygpath -u "${CONFIG_LIST_FILE}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
BATCH_LABEL="fV-SRN + APMGSRN + InstantVNR Combustion"
STAGE_LABELS=("fV-SRN Combustion" "APMGSRN Combustion" "InstantVNR Combustion")
STAGE_PATTERNS=(
    'configs/main/fV-SRN/combustion_40NH3_1__*.yaml'
    'configs/main/APMGSRN/combustion_40NH3_1__*.yaml'
    'configs/main/InstantVNR/combustion_40NH3_1__*.yaml'
)
source "${SCRIPT_DIR}/../lib/batch_runner.sh"
source "${SCRIPT_DIR}/../lib/combustion_batch_runner.sh"
combustion_batch_main
