#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?

CONFIG_ROOT="${REPO_ROOT}/configs/exploration/rd_curve_smoke"
MINER_CONFIG_ROOT="${CONFIG_ROOT}/MINER"
RUN_ROOT="${RUNS_ROOT}/exploration_v3"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration_v3_miner/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
STRICT_VALIDATION="${STRICT_VALIDATION:-1}"

source "${SCRIPT_DIR}/../lib/batch_runner.sh"
batch_init_status

mapfile -t configs < <(
    find "${MINER_CONFIG_ROOT}" -mindepth 2 -maxdepth 2 -type f -name '*.yaml' -print |
        LC_ALL=C sort
)
total="${#configs[@]}"

wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then failures=$((failures + 1)); fi
    unset 'pids[index]'
    pids=("${pids[@]}")
}

cd "${REPO_ROOT}"
failures=0
completed=0
pids=()
for config in "${configs[@]}"; do
    completed=$((completed + 1))
    if [[ "${MAX_PARALLEL_JOBS}" -gt 0 ]]; then
        while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do
            wait_for_pid_at 0
        done
    fi
    batch_run_one_config "${config}" "${completed}" "${total}" &
    pids+=("$!")
done
while [[ "${#pids[@]}" -gt 0 ]]; do wait_for_pid_at 0; done

batch_rebuild_failures
if [[ "${DRY_RUN}" != "1" ]]; then
    summary_args=(
        --config-root "${CONFIG_ROOT}"
        --family MINER
        --validation-mode native-stages
        --status "${STATUS_FILE}"
        --output "${LOG_ROOT}/exploration_summary.tsv"
        --attention-output "${LOG_ROOT}/needs_attention.txt"
        --repo-root "${REPO_ROOT}"
        --run-root "${RUN_ROOT}"
    )
    if [[ "${STRICT_VALIDATION}" == "1" ]]; then summary_args+=(--fail-on-attention); fi
    if ! server_python "${SCRIPT_DIR}/summarize_rd_curve_smoke.py" "${summary_args[@]}"; then
        printf 'FAILED: MINER validation found incomplete or non-finite scale metrics.\n' >&2
        failures=$((failures + 1))
    fi
fi

printf 'Completed %d MINER exploration-v3 configs; failures=%d; status=%s\n' \
    "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
