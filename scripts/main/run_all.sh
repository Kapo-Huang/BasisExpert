#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONDA_ENV="${CONDA_ENV:-compression}"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-${SCRIPT_DIR}/all_configs.list}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
    CONFIG_LIST_FILE="$(cygpath -u "${CONFIG_LIST_FILE}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-0}"
GROUP_DELIM=$'\034'
source "${SCRIPT_DIR}/../lib/batch_runner.sh"

declare -a GROUP_LABELS=()
declare -a GROUP_CONFIGS=()
declare -A SELECTED_CONFIGS=()

join_configs() {
    local IFS="${GROUP_DELIM}"
    printf '%s' "$*"
}

add_group() {
    local label="$1"
    shift
    if [[ "$#" -eq 0 ]]; then
        return
    fi
    GROUP_LABELS+=("${label}")
    GROUP_CONFIGS+=("$(join_configs "$@")")
}

collect_files() {
    local dir="$1"
    local pattern="$2"
    if [[ ! -d "${dir}" ]]; then
        return
    fi
    find "${dir}" -maxdepth 1 -type f -name "${pattern}" -print | LC_ALL=C sort
}

append_serial_main_family() {
    local family="$1"
    local config
    while IFS= read -r config; do
        add_group "main:${family}:${config#${REPO_ROOT}/configs/main/${family}/}" "${config}"
    done < <(collect_files "${REPO_ROOT}/configs/main/${family}" '*.yaml')
}

append_serial_size_family() {
    local family="$1"
    local size config
    for size in Size082 Size163 Size326 Size652; do
        while IFS= read -r config; do
            add_group "size:${family}:${size}:${config##*/}" "${config}"
        done < <(collect_files "${REPO_ROOT}/configs/rd_curve/${family}/${size}" '*.yaml')
    done
}

append_attribute_main_groups() {
    local family="$1"
    local mode="${2:-all}"
    local dataset config
    local -a configs=()
    for dataset in bathymetry combustion_40NH3_1 katrina ionization; do
        configs=()
        while IFS= read -r config; do
            case "${mode}" in
                manager) [[ "${config}" == *"__managerpretrain.yaml" ]] || continue ;;
                main) [[ "${config}" != *"__managerpretrain.yaml" ]] || continue ;;
            esac
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/main/${family}" "${dataset}__*.yaml")
        add_group "main:${family}:${dataset}:${mode}" "${configs[@]}"
    done
}

append_attribute_size_groups() {
    local family="$1"
    local mode="${2:-all}"
    local size config
    local -a configs=()
    for size in Size082 Size163 Size326 Size652; do
        configs=()
        while IFS= read -r config; do
            case "${mode}" in
                manager) [[ "${config}" == *"__managerpretrain.yaml" ]] || continue ;;
                main) [[ "${config}" != *"__managerpretrain.yaml" ]] || continue ;;
            esac
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/rd_curve/${family}/${size}" 'ionization__*.yaml')
        add_group "size:${family}:${size}:ionization:${mode}" "${configs[@]}"
    done
}

append_volume_main_group() {
    local family="$1"
    local config
    local -a configs=()
    local dataset
    for dataset in combustion_40NH3_1 ionization; do
        configs=()
        while IFS= read -r config; do
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/main/${family}" "${dataset}__*.yaml")
        add_group "main:${family}:${dataset}" "${configs[@]}"
    done
}

append_volume_size_groups() {
    local family="$1"
    local size config
    local -a configs=()
    for size in Size082 Size163 Size326 Size652; do
        configs=()
        while IFS= read -r config; do
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/rd_curve/${family}/${size}" 'ionization__*.yaml')
        add_group "size:${family}:${size}:ionization" "${configs[@]}"
    done
}

group_config_count() {
    local count=0 group
    local -a configs=()
    for group in "${GROUP_CONFIGS[@]}"; do
        IFS="${GROUP_DELIM}" read -r -a configs <<< "${group}"
        count=$((count + ${#configs[@]}))
    done
    printf '%d\n' "${count}"
}

load_config_selection() {
    local raw line line_number=0
    if [[ ! -f "${CONFIG_LIST_FILE}" ]]; then
        printf 'Config list not found: %s\n' "${CONFIG_LIST_FILE}" >&2
        return 2
    fi
    while IFS= read -r raw || [[ -n "${raw}" ]]; do
        line_number=$((line_number + 1))
        line="${raw%$'\r'}"
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -n "${line}" ]] || continue
        line="${line#./}"
        if [[ "${line}" != configs/*.yaml ]]; then
            printf 'Invalid config list entry at %s:%d: %s\n' "${CONFIG_LIST_FILE}" "${line_number}" "${line}" >&2
            return 2
        fi
        if [[ -n "${SELECTED_CONFIGS[${line}]+x}" ]]; then
            printf 'Duplicate config list entry at %s:%d: %s\n' "${CONFIG_LIST_FILE}" "${line_number}" "${line}" >&2
            return 2
        fi
        if [[ ! -f "${REPO_ROOT}/${line}" ]]; then
            printf 'Selected config does not exist at %s:%d: %s\n' "${CONFIG_LIST_FILE}" "${line_number}" "${line}" >&2
            return 2
        fi
        SELECTED_CONFIGS["${line}"]="${line_number}"
    done < "${CONFIG_LIST_FILE}"
    if [[ "${#SELECTED_CONFIGS[@]}" -eq 0 ]]; then
        printf 'Config list contains no active entries: %s\n' "${CONFIG_LIST_FILE}" >&2
        return 2
    fi
}

apply_config_selection() {
    local group_index config relative selected_path
    local -a configs=() selected=() filtered_labels=() filtered_groups=()
    local -A matched=()
    for group_index in "${!GROUP_CONFIGS[@]}"; do
        IFS="${GROUP_DELIM}" read -r -a configs <<< "${GROUP_CONFIGS[${group_index}]}"
        selected=()
        for config in "${configs[@]}"; do
            relative="${config#${REPO_ROOT}/}"
            if [[ -n "${SELECTED_CONFIGS[${relative}]+x}" ]]; then
                selected+=("${config}")
                matched["${relative}"]=1
            fi
        done
        if [[ "${#selected[@]}" -gt 0 ]]; then
            filtered_labels+=("${GROUP_LABELS[${group_index}]}")
            filtered_groups+=("$(join_configs "${selected[@]}")")
        fi
    done
    for selected_path in "${!SELECTED_CONFIGS[@]}"; do
        if [[ -z "${matched[${selected_path}]+x}" ]]; then
            printf 'Selected config is not part of the formal matrix: %s\n' "${selected_path}" >&2
            return 2
        fi
    done
    GROUP_LABELS=("${filtered_labels[@]}")
    GROUP_CONFIGS=("${filtered_groups[@]}")
}

# Phase 1: finish every model's main experiments on all datasets before Size runs.
append_serial_main_family "VarExpert"
append_serial_main_family "MVNet"
append_attribute_main_groups "SIREN" "all"
append_attribute_main_groups "CoordNet" "all"
append_serial_main_family "STSR-INR"
append_attribute_main_groups "MoE-INR" "all"
append_volume_main_group "InstantNGP"
append_volume_main_group "InstantVNR"
append_serial_main_family "MC-INR"
append_attribute_main_groups "NeuralExpert" "manager"
append_attribute_main_groups "NeuralExpert" "main"
append_volume_main_group "APMGSRN"
append_volume_main_group "fV-SRN"
append_volume_main_group "RMDSRN"
append_volume_main_group "ECNR"
append_volume_main_group "MINER"

# Phase 2: run the complete Size matrix only after all main experiments finish.
append_serial_size_family "VarExpert"
append_attribute_size_groups "CoordNet" "all"
append_attribute_size_groups "MoE-INR" "all"
append_volume_size_groups "fV-SRN"
append_volume_size_groups "MINER"
append_serial_size_family "STSR-INR"

wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then
        failures=$((failures + 1))
    fi
    unset "pids[${index}]"
    pids=("${pids[@]}")
}

wait_for_all_pids() {
    while [[ "${#pids[@]}" -gt 0 ]]; do
        wait_for_pid_at 0
    done
}

matrix_total="$(group_config_count)"
if [[ "${matrix_total}" -ne 354 ]]; then
    printf 'Expected 354 configs, found %d. Regenerate with scripts/main/generate_configs.py.\n' "${matrix_total}" >&2
    exit 2
fi
load_config_selection || exit $?
apply_config_selection || exit $?
total="$(group_config_count)"
batch_init_status || exit $?
printf 'Selected %d of %d configs from %s\n' "${total}" "${matrix_total}" "${CONFIG_LIST_FILE}"

cd "${REPO_ROOT}"

failures=0
completed=0
for group_index in "${!GROUP_CONFIGS[@]}"; do
    label="${GROUP_LABELS[${group_index}]}"
    IFS="${GROUP_DELIM}" read -r -a configs <<< "${GROUP_CONFIGS[${group_index}]}"
    printf '== %s (%d config%s) ==\n' "${label}" "${#configs[@]}" "$([[ "${#configs[@]}" -eq 1 ]] && printf '' || printf 's')"
    if [[ "${#configs[@]}" -eq 1 ]]; then
        completed=$((completed + 1))
        if ! batch_run_one_config "${configs[0]}" "${completed}" "${total}"; then
            failures=$((failures + 1))
        fi
        continue
    fi
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
    wait_for_all_pids
done

batch_rebuild_failures

printf 'Completed %d configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
