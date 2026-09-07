#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATASET="katrina"

RUN_TOKEN="${RUN_TOKEN:-stsr_inr_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi
CONFIG_LIST_FILE="${LOG_ROOT}/configs.list"

mkdir -p "${LOG_ROOT}"
{
    printf '# STSR-INR %s main experiment.\n' "${DATASET}"
    printf 'configs/main/STSR-INR/%s.yaml\n' "${DATASET}"
} > "${CONFIG_LIST_FILE}"

export RUN_TOKEN
export BATCH_LOG_ROOT="${LOG_ROOT}"
export CONFIG_LIST_FILE
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

exec bash "${SCRIPT_DIR}/run_all.sh" "$@"
