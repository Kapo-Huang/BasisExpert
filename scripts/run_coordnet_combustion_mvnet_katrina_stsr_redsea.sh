#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_LIST_FILE="${SCRIPT_DIR}/run_coordnet_combustion_mvnet_katrina_stsr_redsea.list"
CONDA_ENV="${CONDA_ENV:-compression}"
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

source "${SCRIPT_DIR}/lib/batch_runner.sh"

declare -A seen=()
declare -a coordnet_configs=()
declare -a mvnet_configs=()
declare -a stsr_configs=()

while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line="${raw%$'\r'}"
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] || continue
    line="${line#./}"
    if [[ -n "${seen[${line}]+x}" ]]; then
        printf 'Duplicate config: %s\n' "${line}" >&2
        exit 2
    fi
    if [[ ! -f "${REPO_ROOT}/${line}" ]]; then
        printf 'Config not found: %s\n' "${line}" >&2
        exit 2
    fi
    seen["${line}"]=1
    case "${line}" in
        configs/CoordNet/combustion_40NH3_1__*.yaml)
            coordnet_configs+=("${REPO_ROOT}/${line}")
            ;;
        configs/MVNet/katrina.yaml)
            mvnet_configs+=("${REPO_ROOT}/${line}")
            ;;
        configs/STSR-INR/redsea.yaml)
            stsr_configs+=("${REPO_ROOT}/${line}")
            ;;
        *)
            printf 'Out-of-scope config: %s\n' "${line}" >&2
            exit 2
            ;;
    esac
done < "${CONFIG_LIST_FILE}"

if [[ "${#coordnet_configs[@]}" -ne 13 || "${#mvnet_configs[@]}" -ne 1 || "${#stsr_configs[@]}" -ne 1 ]]; then
    printf 'Expected CoordNet=13, MVNet=1, STSR-INR=1; found %d/%d/%d.\n' \
        "${#coordnet_configs[@]}" "${#mvnet_configs[@]}" "${#stsr_configs[@]}" >&2
    exit 2
fi

if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
    exit 2
fi

batch_init_status || exit $?
cd "${REPO_ROOT}"

failures=0
completed=0
total=15

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

printf 'CoordNet-Combustion + MVNet-Katrina + STSR-INR-RedSea: 15 configs, max_parallel=%s\n' \
    "${MAX_PARALLEL_JOBS}"
run_stage "CoordNet Combustion" "${coordnet_configs[@]}"
run_stage "MVNet Katrina" "${mvnet_configs[@]}"
run_stage "STSR-INR RedSea" "${stsr_configs[@]}"

batch_rebuild_failures
printf 'Completed %d configs; failures=%d; status=%s\n' \
    "${completed}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
