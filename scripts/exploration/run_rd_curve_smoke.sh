#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
CONFIG_ROOT="${REPO_ROOT}/configs/exploration/rd_curve_smoke"
RUN_ROOT="${REPO_ROOT}/runs/exploration_v3"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration_v3/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
STRICT_VALIDATION="${STRICT_VALIDATION:-1}"
COLLAPSE_THRESHOLD_DB="${COLLAPSE_THRESHOLD_DB:-1.0}"
MINIMUM_GAIN_DB="${MINIMUM_GAIN_DB:-0.1}"
GROUP_DELIM=$'\034'
source "${SCRIPT_DIR}/../lib/batch_runner.sh"
batch_init_status

declare -a GROUP_LABELS=()
declare -a GROUP_CONFIGS=()

add_group() {
    local label="$1"
    shift
    [[ "$#" -gt 0 ]] || return
    local IFS="${GROUP_DELIM}"
    GROUP_LABELS+=("${label}")
    GROUP_CONFIGS+=("$*")
}

collect_group() {
    local family="$1"
    local size="$2"
    local mode="${3:-all}"
    local -a configs=()
    local config
    while IFS= read -r config; do
        case "${mode}" in
            manager) [[ "${config}" == *"__managerpretrain.yaml" ]] || continue ;;
            main) [[ "${config}" != *"__managerpretrain.yaml" ]] || continue ;;
            all) ;;
            *) printf 'Unknown collection mode: %s\n' "${mode}" >&2; return 2 ;;
        esac
        configs+=("${config}")
    done < <(find "${CONFIG_ROOT}/${family}/${size}" -maxdepth 1 -type f -name '*.yaml' -print | LC_ALL=C sort)
    add_group "${family}:${size}:${mode}" "${configs[@]}"
}

declare -a SIZES=(Size082 Size163 Size326 Size652)
declare -a FAMILIES=(VarExpert CoordNet MoE-INR fV-SRN MINER STSR-INR)

for family in "${FAMILIES[@]}"; do
    for size in "${SIZES[@]}"; do
        collect_group "${family}" "${size}"
    done
done

total=0
for group in "${GROUP_CONFIGS[@]}"; do
    IFS="${GROUP_DELIM}" read -r -a configs <<< "${group}"
    total=$((total + ${#configs[@]}))
done
wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then failures=$((failures + 1)); fi
    unset 'pids[index]'
    pids=("${pids[@]}")
}

wait_for_all_pids() {
    while [[ "${#pids[@]}" -gt 0 ]]; do wait_for_pid_at 0; done
}

cd "${REPO_ROOT}"
failures=0
completed=0
for group_index in "${!GROUP_CONFIGS[@]}"; do
    label="${GROUP_LABELS[${group_index}]}"
    IFS="${GROUP_DELIM}" read -r -a configs <<< "${GROUP_CONFIGS[${group_index}]}"
    printf '== %s (%d configs) ==\n' "${label}" "${#configs[@]}"
    pids=()
    for config in "${configs[@]}"; do
        completed=$((completed + 1))
        if [[ "${MAX_PARALLEL_JOBS}" -gt 0 ]]; then
            while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do wait_for_pid_at 0; done
        fi
        batch_run_one_config "${config}" "${completed}" "${total}" &
        pids+=("$!")
    done
    wait_for_all_pids
done

batch_rebuild_failures
if [[ "${DRY_RUN}" != "1" ]]; then
    summary_args=(
        --config-root "${CONFIG_ROOT}"
        --status "${STATUS_FILE}"
        --output "${LOG_ROOT}/exploration_summary.tsv"
        --attention-output "${LOG_ROOT}/needs_attention.txt"
        --repo-root "${REPO_ROOT}"
        --run-root "${RUN_ROOT}"
        --collapse-threshold-db "${COLLAPSE_THRESHOLD_DB}"
        --minimum-gain-db "${MINIMUM_GAIN_DB}"
    )
    if [[ "${STRICT_VALIDATION}" == "1" ]]; then summary_args+=(--fail-on-attention); fi
    if ! server_python "${SCRIPT_DIR}/summarize_rd_curve_smoke.py" \
        "${summary_args[@]}"; then
        printf 'FAILED: exploration-v3 validation found incomplete, non-improving, or collapsed runs.\n' >&2
        failures=$((failures + 1))
    fi
fi

printf 'Completed %d exploration-v3 configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
