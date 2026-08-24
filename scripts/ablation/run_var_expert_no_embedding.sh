#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?

CONFIG_PATH="${REPO_ROOT}/configs/ablation/variable_conditioning/VarExpertNoEmbedding/ionization.yaml"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/ablation/variable_conditioning/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"

source "${SCRIPT_DIR}/../lib/batch_runner.sh"
batch_init_status || exit $?

cd "${REPO_ROOT}"
failures=0
if ! batch_run_one_config "${CONFIG_PATH}" 1 1; then
    failures=1
fi
batch_rebuild_failures

printf 'Completed VarExpertNoEmbedding Ionization ablation; failures=%d; status=%s\n' \
    "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
