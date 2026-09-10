#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
export CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-${SCRIPT_DIR}/miner_combustion_size041.list}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"

exec bash "${SCRIPT_DIR}/../main/run_all.sh" "$@"
