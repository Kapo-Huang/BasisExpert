#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
CONFIG_LIST_FILE="${SCRIPT_DIR}/selected_datasets.list"
RUN_TOKEN="${RUN_TOKEN:-coordnet_mvnet_stsr_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
    STATUS_FILE="${LOG_ROOT}/status.tsv"
    FAILURE_FILE="${LOG_ROOT}/failed.txt"
fi

source "${SCRIPT_DIR}/../lib/batch_runner.sh"

declare -a coordnet_configs=()
declare -a mvnet_configs=()
declare -a stsr_configs=()
declare -a additional_configs=()

while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line="${raw%$'\r'}"
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] || continue
    line="${line#./}"
    case "${line}" in
        configs/main/CoordNet/combustion_40NH3_1__*.yaml)
            coordnet_configs+=("${REPO_ROOT}/${line}")
            ;;
        configs/main/MVNet/katrina.yaml)
            mvnet_configs+=("${REPO_ROOT}/${line}")
            ;;
        configs/main/STSR-INR/redsea.yaml)
            stsr_configs+=("${REPO_ROOT}/${line}")
            ;;
        *)
            additional_configs+=("${REPO_ROOT}/${line}")
            ;;
    esac
done < "${CONFIG_LIST_FILE}"

if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
    exit 2
fi

batch_init_status || exit $?
cd "${REPO_ROOT}"

failures=0
completed=0
total=$((${#coordnet_configs[@]} + ${#mvnet_configs[@]} + ${#stsr_configs[@]} + ${#additional_configs[@]}))

wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then
        failures=$((failures + 1))
    fi
    unset "pids[${index}]"
    pids=("${pids[@]}")
}

run_stage() {
    local label="$1"
    shift
    local -a stage_configs=("$@")
    local config
    printf '== %s (%d config%s, max_parallel=%s) ==\n' \
        "${label}" "${#stage_configs[@]}" \
        "$([[ "${#stage_configs[@]}" -eq 1 ]] && printf '' || printf 's')" \
        "${MAX_PARALLEL_JOBS}"
    pids=()
    for config in "${stage_configs[@]}"; do
        completed=$((completed + 1))
        while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do
            wait_for_pid_at 0
        done
        batch_run_one_config "${config}" "${completed}" "${total}" &
        pids+=("$!")
    done
    while [[ "${#pids[@]}" -gt 0 ]]; do
        wait_for_pid_at 0
    done
}

printf 'CoordNet-Combustion + MVNet-Katrina + STSR-INR-RedSea: %d configs, max_parallel=%s\n' \
    "${total}" "${MAX_PARALLEL_JOBS}"
run_stage "CoordNet Combustion" "${coordnet_configs[@]}"
run_stage "MVNet Katrina" "${mvnet_configs[@]}"
run_stage "STSR-INR RedSea" "${stsr_configs[@]}"
if [[ "${#additional_configs[@]}" -gt 0 ]]; then
    run_stage "Additional configs" "${additional_configs[@]}"
fi

batch_rebuild_failures
printf 'Completed %d configs; failures=%d; status=%s\n' \
    "${completed}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
