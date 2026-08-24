#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
export RUN_TOKEN
export CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-${SCRIPT_DIR}/var_expert_topk.list}"
export BATCH_LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/sensitivity/var_expert_topk/${RUN_TOKEN}}"

exec bash "${SCRIPT_DIR}/../main/run_all.sh" "$@"
