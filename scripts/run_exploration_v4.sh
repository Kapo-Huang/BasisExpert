#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${REPO_ROOT}/configs_exploration_v4"
RUN_ROOT="${REPO_ROOT}/runs/exploration_v4"
CONDA_ENV="${CONDA_ENV:-compression}"
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
EXPECTED_TOTAL=81

source "${SCRIPT_DIR}/lib/batch_runner.sh"
if [[ ! "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
    exit 2
fi
if ! batch_init_status; then exit 2; fi

mapfile -t RMDSRN_MAIN < <(
    find "${CONFIG_ROOT}/RMDSRN" -type f -path '*/schedule900k_lambda10/*.yaml' -print | LC_ALL=C sort
)
mapfile -t RMDSRN_ABLATIONS < <(
    find "${CONFIG_ROOT}/RMDSRN" -type f \( -path '*/schedule900k_lambda1/*.yaml' -o -path '*/schedule900k_lambda0/*.yaml' \) -print | LC_ALL=C sort
)
mapfile -t COORDNET_DEPTH < <(
    find "${CONFIG_ROOT}/CoordNet" -type f -path '*_base_lr/*.yaml' -print | LC_ALL=C sort
)
mapfile -t COORDNET_CONTROLS < <(
    find "${CONFIG_ROOT}/CoordNet" -type f ! -path '*_base_lr/*.yaml' -print | LC_ALL=C sort
)

if [[ "${#RMDSRN_MAIN[@]}" -ne 15 || "${#RMDSRN_ABLATIONS[@]}" -ne 12 || \
      "${#COORDNET_DEPTH[@]}" -ne 45 || "${#COORDNET_CONTROLS[@]}" -ne 9 ]]; then
    printf 'Unexpected v4 stage counts: rmdsrn_main=%d rmdsrn_ablations=%d coordnet_depth=%d coordnet_controls=%d\n' \
        "${#RMDSRN_MAIN[@]}" "${#RMDSRN_ABLATIONS[@]}" \
        "${#COORDNET_DEPTH[@]}" "${#COORDNET_CONTROLS[@]}" >&2
    exit 2
fi

total=$(( ${#RMDSRN_MAIN[@]} + ${#RMDSRN_ABLATIONS[@]} + ${#COORDNET_DEPTH[@]} + ${#COORDNET_CONTROLS[@]} ))
if [[ "${total}" -ne "${EXPECTED_TOTAL}" ]]; then
    printf 'Expected %d exploration-v4 configs, found %d.\n' "${EXPECTED_TOTAL}" "${total}" >&2
    exit 2
fi

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
run_stage "RMDSRN corrected schedule" "${RMDSRN_MAIN[@]}"
run_stage "RMDSRN lambda ablations" "${RMDSRN_ABLATIONS[@]}"
run_stage "CoordNet equal-budget depth" "${COORDNET_DEPTH[@]}"
run_stage "CoordNet causal controls" "${COORDNET_CONTROLS[@]}"

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
    if ! conda run --no-capture-output -n "${CONDA_ENV}" python \
        "${SCRIPT_DIR}/summarize_exploration_v4.py" "${summary_args[@]}"; then
        printf 'FAILED: exploration-v4 validation found runs needing attention.\n' >&2
        failures=$((failures + 1))
    fi
fi

printf 'Completed %d exploration-v4 configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
