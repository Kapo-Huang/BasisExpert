#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_LIST_FILE="${SCRIPT_DIR}/run_neural_expert_non_ionization_main.list"
RUN_TOKEN="${RUN_TOKEN:-siren_neural_expert_non_ionization_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
CONDA_ENV="${CONDA_ENV:-compression}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
EVALUATION_DEVICE="${EVALUATION_DEVICE:-}"

if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi

declare -A selected_configs=()
declare -a configs=()
siren_count=0
neural_manager_count=0
neural_main_count=0
neural_bathymetry_manager_count=0
neural_bathymetry_main_count=0
neural_combustion_manager_count=0
neural_combustion_main_count=0
while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line="${raw%$'\r'}"
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] || continue
    line="${line#./}"
    case "${line}" in
        configs/SIREN/combustion_40NH3_1__*.yaml)
            [[ "${line}" != *"managerpretrain"* ]] || { printf 'Out-of-scope SIREN config: %s\n' "${line}" >&2; exit 2; }
            siren_count=$((siren_count + 1))
            ;;
        configs/NeuralExpert/combustion_40NH3_1__Velocity.yaml|configs/NeuralExpert/combustion_40NH3_1__Velocity__managerpretrain.yaml)
            printf 'Out-of-scope NeuralExpert vector target: %s\n' "${line}" >&2
            exit 2
            ;;
        configs/NeuralExpert/bathymetry__*.yaml|configs/NeuralExpert/combustion_40NH3_1__*.yaml)
            if [[ "${line}" == *"__managerpretrain.yaml" ]]; then
                neural_manager_count=$((neural_manager_count + 1))
                if [[ "${line}" == configs/NeuralExpert/bathymetry__* ]]; then
                    neural_bathymetry_manager_count=$((neural_bathymetry_manager_count + 1))
                else
                    neural_combustion_manager_count=$((neural_combustion_manager_count + 1))
                fi
            else
                neural_main_count=$((neural_main_count + 1))
                if [[ "${line}" == configs/NeuralExpert/bathymetry__* ]]; then
                    neural_bathymetry_main_count=$((neural_bathymetry_main_count + 1))
                else
                    neural_combustion_main_count=$((neural_combustion_main_count + 1))
                fi
            fi
            ;;
        *)
            printf 'Out-of-scope combined experiment config: %s\n' "${line}" >&2
            exit 2
            ;;
    esac
    if [[ -n "${selected_configs[${line}]+x}" ]]; then
        printf 'Duplicate combined experiment config: %s\n' "${line}" >&2
        exit 2
    fi
    selected_configs["${line}"]=1
    configs+=("${line}")
done < "${CONFIG_LIST_FILE}"

if [[ "${#configs[@]}" -ne 45 || "${siren_count}" -ne 13 || "${neural_manager_count}" -ne 16 || "${neural_main_count}" -ne 16 \
    || "${neural_bathymetry_manager_count}" -ne 4 || "${neural_bathymetry_main_count}" -ne 4 \
    || "${neural_combustion_manager_count}" -ne 12 || "${neural_combustion_main_count}" -ne 12 ]]; then
    printf 'Expected SIREN=13, NeuralExpert manager=16, NeuralExpert main=16; found %d/%d/%d (%d total).\n' \
        "${siren_count}" "${neural_manager_count}" "${neural_main_count}" "${#configs[@]}" >&2
    exit 2
fi

export CONFIG_LIST_FILE
export RUN_TOKEN
export BATCH_LOG_ROOT="${LOG_ROOT}"
export CONDA_ENV
export DRY_RUN
export MAX_PARALLEL_JOBS

printf 'SIREN + NeuralExpert non-Ionization matrix: 45 configs, max_parallel=%s\n' "${MAX_PARALLEL_JOBS}"
runner_exit=0
bash "${SCRIPT_DIR}/run_all_configs.sh" || runner_exit=$?
if [[ "${DRY_RUN}" == "1" ]]; then
    exit "${runner_exit}"
fi

STATUS_FILE="${LOG_ROOT}/status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'Combined experiment status file not found: %s\n' "${STATUS_FILE}" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}/evaluations"
SUMMARY_FILE="${LOG_ROOT}/experiment_psnr.tsv"
SUMMARY_TEMP="${SUMMARY_FILE}.tmp"
printf 'family\tconfig\ttarget\tpsnr\tscope\tresult_path\n' > "${SUMMARY_TEMP}"

validation_failures=0
validated_siren=0
validated_managers=0
validated_neural=0

status_record() {
    local relative="$1"
    awk -F '\t' -v wanted="${relative}" '
        NR > 1 && $1 == wanted { status=$2; log_path=$4 }
        END { printf "%s\t%s\n", status, log_path }
    ' "${STATUS_FILE}"
}

is_finite_number() {
    local value="$1"
    [[ "${value}" =~ ^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$ ]]
}

for relative in "${configs[@]}"; do
    record="$(status_record "${relative}")"
    status="${record%%$'\t'*}"
    log_path="${record#*$'\t'}"
    if [[ "${status}" != "ok" ]]; then
        printf 'INVALID status=%s: %s\n' "${status:-missing}" "${relative}" >&2
        validation_failures=$((validation_failures + 1))
        continue
    fi
    if [[ -z "${log_path}" || ! -f "${log_path}" ]]; then
        printf 'INVALID missing log: %s (%s)\n' "${relative}" "${log_path:-no path}" >&2
        validation_failures=$((validation_failures + 1))
        continue
    fi

    filename="${relative##*/}"
    target="${filename#*__}"
    target="${target%__managerpretrain.yaml}"
    target="${target%.yaml}"

    if [[ "${relative}" == configs/SIREN/* ]]; then
        psnr_line="$(grep -F 'PSNR epoch 600/600:' "${log_path}" | tail -n 1 || true)"
        psnr="$(printf '%s\n' "${psnr_line}" | sed -n 's/.*aggregate=\([^[:space:]]*\).*/\1/p')"
        if [[ -z "${psnr_line}" ]] || ! is_finite_number "${psnr}"; then
            printf 'INVALID missing or non-finite final SIREN PSNR: %s (%s)\n' "${relative}" "${log_path}" >&2
            validation_failures=$((validation_failures + 1))
            continue
        fi
        printf 'SIREN\t%s\t%s\t%s\tsampled_10_percent\t%s\n' \
            "${relative}" "${target}" "${psnr}" "${log_path}" >> "${SUMMARY_TEMP}"
        validated_siren=$((validated_siren + 1))
        continue
    fi

    if [[ "${relative}" == *"__managerpretrain.yaml" ]]; then
        if ! grep -Fq 'Exported manager pretrain checkpoint to ' "${log_path}"; then
            printf 'INVALID missing manager checkpoint export: %s (%s)\n' "${relative}" "${log_path}" >&2
            validation_failures=$((validation_failures + 1))
            continue
        fi
        validated_managers=$((validated_managers + 1))
        continue
    fi

    if ! grep -Fq 'Saved final state dict to ' "${log_path}"; then
        printf 'INVALID missing NeuralExpert final state: %s (%s)\n' "${relative}" "${log_path}" >&2
        validation_failures=$((validation_failures + 1))
        continue
    fi

    safe_name="${relative//\//__}"
    safe_name="${safe_name%.yaml}"
    evaluation_log="${LOG_ROOT}/evaluations/${safe_name}.log"
    command=(conda run --no-capture-output -n "${CONDA_ENV}" python "${SCRIPT_DIR}/evaluate_neural_expert_config.py" --config "${REPO_ROOT}/${relative}")
    if [[ -n "${EVALUATION_DEVICE}" ]]; then
        command+=(--device "${EVALUATION_DEVICE}")
    fi
    "${command[@]}" 2>&1 | tee "${evaluation_log}"
    evaluation_exit="${PIPESTATUS[0]}"
    if [[ "${evaluation_exit}" -ne 0 ]]; then
        printf 'INVALID NeuralExpert PSNR evaluation failed (%d): %s\n' "${evaluation_exit}" "${relative}" >&2
        validation_failures=$((validation_failures + 1))
        continue
    fi
    marker_line="$(grep $'^NEURAL_EXPERT_PSNR\t' "${evaluation_log}" | tail -n 1 || true)"
    IFS=$'\t' read -r marker marker_target marker_psnr marker_run_dir marker_metrics_path <<< "${marker_line}"
    if [[ "${marker}" != "NEURAL_EXPERT_PSNR" || "${marker_target}" != "${target}" ]] || ! is_finite_number "${marker_psnr}"; then
        printf 'INVALID missing or malformed NeuralExpert PSNR result: %s (%s)\n' "${relative}" "${evaluation_log}" >&2
        validation_failures=$((validation_failures + 1))
        continue
    fi
    printf 'NeuralExpert\t%s\t%s\t%s\tfull\t%s\n' \
        "${relative}" "${target}" "${marker_psnr}" "${marker_metrics_path}" >> "${SUMMARY_TEMP}"
    validated_neural=$((validated_neural + 1))
done

mv "${SUMMARY_TEMP}" "${SUMMARY_FILE}"
printf 'Validated SIREN=%d/13, NeuralExpert managers=%d/16, NeuralExpert mains with full PSNR=%d/16.\n' \
    "${validated_siren}" "${validated_managers}" "${validated_neural}"
printf 'PSNR summary: %s\n' "${SUMMARY_FILE}"
if [[ "${runner_exit}" -ne 0 || "${validation_failures}" -ne 0 ]]; then
    exit 1
fi
