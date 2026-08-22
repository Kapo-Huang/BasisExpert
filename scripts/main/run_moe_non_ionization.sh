#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?
CONFIG_LIST_FILE="${SCRIPT_DIR}/moe_non_ionization.list"
RUN_TOKEN="${RUN_TOKEN:-moe_non_ionization_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/${RUN_TOKEN}}"
DRY_RUN="${DRY_RUN:-0}"

if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi

declare -a configs=()
moe_count=0
while IFS= read -r raw || [[ -n "${raw}" ]]; do
    line="${raw%$'\r'}"
    line="${line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -n "${line}" ]] || continue
    line="${line#./}"
    case "${line}" in
        configs/main/MoE-INR/redsea__*.yaml|configs/main/MoE-INR/combustion_40NH3_1__*.yaml|configs/main/MoE-INR/katrina__*.yaml)
            moe_count=$((moe_count + 1))
            ;;
    esac
    configs+=("${line}")
done < "${CONFIG_LIST_FILE}"

export CONFIG_LIST_FILE
export RUN_TOKEN
export BATCH_LOG_ROOT="${LOG_ROOT}"
export DRY_RUN

runner_exit=0
bash "${SCRIPT_DIR}/run_all.sh" "$@" || runner_exit=$?
if [[ "${DRY_RUN}" == "1" ]]; then
    exit "${runner_exit}"
fi

STATUS_FILE="${LOG_ROOT}/status.tsv"
if [[ ! -f "${STATUS_FILE}" ]]; then
    printf 'MoE rerun status file not found: %s\n' "${STATUS_FILE}" >&2
    exit 1
fi

validation_failures=0
validated=0
for relative in "${configs[@]}"; do
    case "${relative}" in
        configs/main/MoE-INR/redsea__*.yaml|configs/main/MoE-INR/combustion_40NH3_1__*.yaml|configs/main/MoE-INR/katrina__*.yaml) ;;
        *) continue ;;
    esac
    record="$(awk -F '\t' -v wanted="${relative}" '
        NR > 1 && $1 == wanted { status=$2; log_path=$4 }
        END { printf "%s\t%s\n", status, log_path }
    ' "${STATUS_FILE}")"
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
    if ! grep -Fq 'PSNR epoch 600/600:' "${log_path}"; then
        printf 'INVALID missing final PSNR: %s (%s)\n' "${relative}" "${log_path}" >&2
        validation_failures=$((validation_failures + 1))
        continue
    fi
    validated=$((validated + 1))
done

printf 'Validated %d/%d MoE configs with status=ok and final PSNR.\n' "${validated}" "${moe_count}"
if [[ "${runner_exit}" -ne 0 || "${validation_failures}" -ne 0 ]]; then
    exit 1
fi
