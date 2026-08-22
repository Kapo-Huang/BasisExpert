#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
CONFIG_ROOT="${REPO_ROOT}/configs/ablation/architecture"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration/${RUN_TOKEN}}"
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-0}"
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
    local profile="$2"
    local mode="${3:-all}"
    local -a configs=()
    local config
    while IFS= read -r config; do
        case "${mode}" in
            manager) [[ "${config}" == *"__managerpretrain.yaml" ]] || continue ;;
            main) [[ "${config}" != *"__managerpretrain.yaml" ]] || continue ;;
        esac
        configs+=("${config}")
    done < <(find "${CONFIG_ROOT}/${family}/Size163/${profile}" -maxdepth 1 -type f -name '*.yaml' -print | LC_ALL=C sort)
    add_group "${family}:${profile}:${mode}" "${configs[@]}"
}

for profile in depth1 depth2 depth3; do collect_group "NeuralExpert" "${profile}" manager; done
for family in SIREN CoordNet MoE-INR VarExpert MC-INR APMGSRN fV-SRN RMDSRN; do
    while IFS= read -r profile_dir; do
        collect_group "${family}" "$(basename "${profile_dir}")"
    done < <(find "${CONFIG_ROOT}/${family}/Size163" -mindepth 1 -maxdepth 1 -type d -print | LC_ALL=C sort)
done
for profile in depth1 depth2 depth3; do collect_group "NeuralExpert" "${profile}" main; done

total=0
for group in "${GROUP_CONFIGS[@]}"; do
    IFS="${GROUP_DELIM}" read -r -a configs <<< "${group}"
    total=$((total + ${#configs[@]}))
done
wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then failures=$((failures + 1)); fi
    unset "pids[${index}]"
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
    if ! server_python "${SCRIPT_DIR}/../ablation/summarize_architecture.py" \
        --config-root "${CONFIG_ROOT}" \
        --status "${STATUS_FILE}" \
        --output "${LOG_ROOT}/exploration_summary.tsv" \
        --repo-root "${REPO_ROOT}"; then
        printf 'FAILED: could not build exploration summary.\n' >&2
        failures=$((failures + 1))
    fi
fi
printf 'Completed %d exploration configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
