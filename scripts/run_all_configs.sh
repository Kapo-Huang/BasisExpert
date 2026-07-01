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

mkdir -p "${LOG_ROOT}/logs"
printf 'config\tstatus\texit_code\tlog\n' > "${STATUS_FILE}"
: > "${FAILURE_FILE}"

declare -a CONFIGS=()

append_configs() {
    local family="$1"
    local pattern="${2:-*.yaml}"
    while IFS= read -r config; do
        CONFIGS+=("${config}")
    done < <(find "${REPO_ROOT}/configs/${family}" -type f -name "${pattern}" -print | LC_ALL=C sort)
}

append_configs "VarExpert"
append_configs "SIREN"
append_configs "CoordNet"
append_configs "MoE-INR"
append_configs "MC-INR"
append_configs "NeuralExpert" "*__managerpretrain.yaml"
while IFS= read -r config; do
    [[ "${config}" == *"__managerpretrain.yaml" ]] || CONFIGS+=("${config}")
done < <(find "${REPO_ROOT}/configs/NeuralExpert" -type f -name '*.yaml' -print | LC_ALL=C sort)
append_configs "APMGSRN"
append_configs "DC-INR"
append_configs "fV-SRN"
append_configs "RMDSRN"

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

failures=0
completed=0
total="${#CONFIGS[@]}"
if [[ "${total}" -ne 336 ]]; then
    printf 'Expected 336 configs, found %d. Regenerate with scripts/generate_config_matrix.py.\n' "${total}" >&2
    exit 2
fi
cd "${REPO_ROOT}"

for config in "${CONFIGS[@]}"; do
    completed=$((completed + 1))
    relative="${config#${REPO_ROOT}/}"
    safe_name="${relative//\//__}"
    safe_name="${safe_name%.yaml}"
    log_path="${LOG_ROOT}/logs/${safe_name}.log"
    module="$(module_for_config "${config}")"
    command=(conda run --no-capture-output -n "${CONDA_ENV}" python -m "${module}" train --config "${config}")

    printf '[%d/%d] %s\n' "${completed}" "${total}" "${relative}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        printf '%s\tok\t0\t%s\n' "${relative}" "${log_path}" >> "${STATUS_FILE}"
        continue
    fi

    "${command[@]}" 2>&1 | tee "${log_path}"
    exit_code="${PIPESTATUS[0]}"
    if [[ "${exit_code}" -eq 0 ]]; then
        printf '%s\tok\t0\t%s\n' "${relative}" "${log_path}" >> "${STATUS_FILE}"
    else
        failures=$((failures + 1))
        printf '%s\tfailed\t%d\t%s\n' "${relative}" "${exit_code}" "${log_path}" >> "${STATUS_FILE}"
        printf '%s\n' "${relative}" >> "${FAILURE_FILE}"
        printf 'FAILED (%d): %s\n' "${exit_code}" "${relative}" >&2
    fi
done

printf 'Completed %d configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then
    exit 1
fi
