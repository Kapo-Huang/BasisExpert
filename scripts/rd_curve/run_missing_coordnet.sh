#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
RUN_TOKEN="${RUN_TOKEN:-missing_coordnet_rd_$(date +%Y%m%d_%H%M%S)}"
export RUN_TOKEN
export CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-${SCRIPT_DIR}/missing_coordnet.list}"
export BATCH_LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/missing/coordnet_rd/${RUN_TOKEN}}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"

exec bash "${SCRIPT_DIR}/../main/run_all.sh" "$@"
