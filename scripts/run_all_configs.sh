#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-compression}"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-0}"
GROUP_DELIM=$'\034'
source "${SCRIPT_DIR}/lib/batch_runner.sh"
batch_init_status

declare -a GROUP_LABELS=()
declare -a GROUP_CONFIGS=()

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

append_serial_family() {
    local family="$1"
    local config
    while IFS= read -r config; do
        add_group "${family}:${config#${REPO_ROOT}/configs/${family}/}" "${config}"
    done < <(find "${REPO_ROOT}/configs/${family}" -type f -name '*.yaml' -print | LC_ALL=C sort)
}

append_attribute_groups() {
    local family="$1"
    local mode="${2:-all}"
    local dataset size config
    local -a configs=()
    for dataset in bathymetry katrina ionization; do
        configs=()
        while IFS= read -r config; do
            case "${mode}" in
                manager) [[ "${config}" == *"__managerpretrain.yaml" ]] || continue ;;
                main) [[ "${config}" != *"__managerpretrain.yaml" ]] || continue ;;
            esac
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/${family}" "${dataset}__*.yaml")
        add_group "${family}:${dataset}:${mode}" "${configs[@]}"
    done
    for size in Size082 Size163 Size326 Size652 Size1304; do
        configs=()
        while IFS= read -r config; do
            case "${mode}" in
                manager) [[ "${config}" == *"__managerpretrain.yaml" ]] || continue ;;
                main) [[ "${config}" != *"__managerpretrain.yaml" ]] || continue ;;
            esac
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/${family}/${size}" 'ionization__*.yaml')
        add_group "${family}:${size}:ionization:${mode}" "${configs[@]}"
    done
}

append_volume_attribute_groups() {
    local family="$1"
    local size config
    local -a configs=()
    configs=()
    while IFS= read -r config; do
        configs+=("${config}")
    done < <(collect_files "${REPO_ROOT}/configs/${family}" 'ionization__*.yaml')
    add_group "${family}:ionization" "${configs[@]}"
    for size in Size082 Size163 Size326 Size652 Size1304; do
        configs=()
        while IFS= read -r config; do
            configs+=("${config}")
        done < <(collect_files "${REPO_ROOT}/configs/${family}/${size}" 'ionization__*.yaml')
        add_group "${family}:${size}:ionization" "${configs[@]}"
    done
}

append_serial_family "VarExpert"
append_serial_family "MVNet"
append_attribute_groups "SIREN" "all"
append_attribute_groups "CoordNet" "all"
append_attribute_groups "MoE-INR" "all"
append_volume_attribute_groups "CompactNGP"
append_volume_attribute_groups "InstantNGP"
append_volume_attribute_groups "FA-TR-INR"
append_serial_family "MC-INR"
append_attribute_groups "NeuralExpert" "manager"
append_attribute_groups "NeuralExpert" "main"
append_volume_attribute_groups "APMGSRN"
append_volume_attribute_groups "DC-INR"
append_volume_attribute_groups "fV-SRN"
append_volume_attribute_groups "RMDSRN"
append_volume_attribute_groups "ECNR"

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

total=0
for group in "${GROUP_CONFIGS[@]}"; do
    IFS="${GROUP_DELIM}" read -r -a configs <<< "${group}"
    total=$((total + ${#configs[@]}))
done
if [[ "${total}" -ne 355 ]]; then
    printf 'Expected 355 configs, found %d. Regenerate with scripts/generate_config_matrix.py.\n' "${total}" >&2
    exit 2
fi

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
