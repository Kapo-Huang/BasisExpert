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

mkdir -p "${LOG_ROOT}/logs"
printf 'config\tstatus\texit_code\tlog\n' > "${STATUS_FILE}"
: > "${FAILURE_FILE}"

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
append_attribute_groups "SIREN" "all"
append_attribute_groups "CoordNet" "all"
append_attribute_groups "MoE-INR" "all"
append_volume_attribute_groups "CompactNGP"
append_serial_family "MC-INR"
append_attribute_groups "NeuralExpert" "manager"
append_attribute_groups "NeuralExpert" "main"
append_volume_attribute_groups "APMGSRN"
append_volume_attribute_groups "DC-INR"
append_volume_attribute_groups "fV-SRN"
append_volume_attribute_groups "RMDSRN"

module_for_config() {
    case "$1" in
        */MC-INR/*) echo "var_expert_inr.mc_inr.cli" ;;
        */NeuralExpert/*) echo "var_expert_inr.neural_expert.cli" ;;
        */APMGSRN/*) echo "var_expert_inr.apmgsrn.cli" ;;
        */DC-INR/*) echo "var_expert_inr.dc_inr.cli" ;;
        */fV-SRN/*) echo "var_expert_inr.fv_srn.cli" ;;
        */RMDSRN/*) echo "var_expert_inr.rmdsrn.cli" ;;
        *) echo "var_expert_inr.cli" ;;
    esac
}

run_one_config() {
    local config="$1"
    local index="$2"
    local total="$3"
    local relative safe_name log_path module exit_code
    relative="${config#${REPO_ROOT}/}"
    safe_name="${relative//\//__}"
    safe_name="${safe_name%.yaml}"
    log_path="${LOG_ROOT}/logs/${safe_name}.log"
    module="$(module_for_config "${config}")"
    local -a command=(conda run --no-capture-output -n "${CONDA_ENV}" python -m "${module}" train --config "${config}")

    printf '[%d/%d] %s\n' "${index}" "${total}" "${relative}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        printf '%s\tok\t0\t%s\n' "${relative}" "${log_path}" >> "${STATUS_FILE}"
        return 0
    fi

    "${command[@]}" 2>&1 | tee "${log_path}"
    exit_code="${PIPESTATUS[0]}"
    if [[ "${exit_code}" -eq 0 ]]; then
        printf '%s\tok\t0\t%s\n' "${relative}" "${log_path}" >> "${STATUS_FILE}"
    else
        printf '%s\tfailed\t%d\t%s\n' "${relative}" "${exit_code}" "${log_path}" >> "${STATUS_FILE}"
        printf '%s\n' "${relative}" >> "${FAILURE_FILE}"
        printf 'FAILED (%d): %s\n' "${exit_code}" "${relative}" >&2
    fi
    return "${exit_code}"
}

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
if [[ "${total}" -ne 342 ]]; then
    printf 'Expected 342 configs, found %d. Regenerate with scripts/generate_config_matrix.py.\n' "${total}" >&2
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
        if ! run_one_config "${configs[0]}" "${completed}" "${total}"; then
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
        run_one_config "${config}" "${completed}" "${total}" &
        pids+=("$!")
    done
    wait_for_all_pids
done

printf 'Completed %d configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
