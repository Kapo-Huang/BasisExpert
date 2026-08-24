#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
CONFIG_ROOT="${REPO_ROOT}/configs/ablation/depth_and_regularization"
RUN_ROOT="${RUNS_ROOT}/exploration_v4"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration_v4/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"; fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
STRICT_VALIDATION="${STRICT_VALIDATION:-1}"
COLLAPSE_THRESHOLD_DB="${COLLAPSE_THRESHOLD_DB:-1.0}"
MINIMUM_GAIN_DB="${MINIMUM_GAIN_DB:-0.1}"
source "${SCRIPT_DIR}/../lib/batch_runner.sh"
if [[ ! "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
    exit 2
fi
if ! batch_init_status; then exit 2; fi

mapfile -t COORDNET_DEPTH < <(
    find "${CONFIG_ROOT}/CoordNet" -type f -path '*_base_lr/*.yaml' -print | LC_ALL=C sort
)

total=${#COORDNET_DEPTH[@]}
wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then failures=$((failures + 1)); fi
    unset 'pids[index]'
    pids=("${pids[@]}")
}

run_stage() {
    local label="$1"
    shift
    local -a configs=("$@")
    printf '== %s (%d configs, max_parallel=%d) ==\n' "${label}" "${#configs[@]}" "${MAX_PARALLEL_JOBS}"
    pids=()
    local config
    for config in "${configs[@]}"; do
        completed=$((completed + 1))
        if [[ "${MAX_PARALLEL_JOBS}" -gt 0 ]]; then
            while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do wait_for_pid_at 0; done
        fi
        batch_run_one_config "${config}" "${completed}" "${total}" &
        pids+=("$!")
    done
    while [[ "${#pids[@]}" -gt 0 ]]; do wait_for_pid_at 0; done
}

cd "${REPO_ROOT}"
failures=0
completed=0
run_stage "CoordNet equal-budget depth" "${COORDNET_DEPTH[@]}"

batch_rebuild_failures
if [[ "${DRY_RUN}" != "1" ]]; then
    summary_args=(
        --config-root "${CONFIG_ROOT}"
        --status "${STATUS_FILE}"
        --output "${LOG_ROOT}/exploration_summary.tsv"
        --profile-output "${LOG_ROOT}/profile_summary.tsv"
        --attention-output "${LOG_ROOT}/needs_attention.txt"
        --repo-root "${REPO_ROOT}"
        --run-root "${RUN_ROOT}"
        --collapse-threshold-db "${COLLAPSE_THRESHOLD_DB}"
        --minimum-gain-db "${MINIMUM_GAIN_DB}"
    )
    if [[ "${STRICT_VALIDATION}" == "1" ]]; then summary_args+=(--fail-on-attention); fi
    if ! server_python "${SCRIPT_DIR}/summarize_depth_and_regularization.py" \
        "${summary_args[@]}"; then
        printf 'FAILED: exploration-v4 validation found runs needing attention.\n' >&2
        failures=$((failures + 1))
    fi
fi

printf 'Completed %d exploration-v4 configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
