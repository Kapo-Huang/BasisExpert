#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATASET="combustion_40NH3_1"

RUN_TOKEN="${RUN_TOKEN:-neural_expert_${DATASET}_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi
CONFIG_LIST_FILE="${LOG_ROOT}/configs.list"

mkdir -p "${LOG_ROOT}"
{
    printf '# NeuralExpert %s main experiment: manager pretraining, then main training.\n' "${DATASET}"
    for config in "${REPO_ROOT}/configs/main/NeuralExpert/${DATASET}"__*__managerpretrain.yaml; do
        printf 'configs/main/NeuralExpert/%s\n' "${config##*/}"
    done
    for config in "${REPO_ROOT}/configs/main/NeuralExpert/${DATASET}"__*.yaml; do
        [[ "${config}" != *__managerpretrain.yaml ]] || continue
        printf 'configs/main/NeuralExpert/%s\n' "${config##*/}"
    done
} > "${CONFIG_LIST_FILE}"

export RUN_TOKEN
export BATCH_LOG_ROOT="${LOG_ROOT}"
export CONFIG_LIST_FILE
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

exec bash "${SCRIPT_DIR}/run_all.sh" "$@"
