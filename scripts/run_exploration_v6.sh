#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${REPO_ROOT}/configs_exploration_v6"
RUN_ROOT="${REPO_ROOT}/runs/exploration_v6"
CONDA_ENV="${CONDA_ENV:-compression}"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration_v6/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"; fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
DEVICE="${DEVICE:-cuda:0}"
STRICT_VALIDATION="${STRICT_VALIDATION:-1}"
PSNR_REGRESSION_TOLERANCE_DB="${PSNR_REGRESSION_TOLERANCE_DB:-1.0}"
EXPECTED_TOTAL=18

source "${SCRIPT_DIR}/lib/batch_runner.sh"
if [[ ! "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
    exit 2
fi
if [[ ! "${DEVICE}" =~ ^cuda:[0-9]+$ ]]; then
    printf 'DEVICE must use cuda:N form, for example cuda:0 or cuda:1; got %s.\n' "${DEVICE}" >&2
    exit 2
fi
if [[ "${STRICT_VALIDATION}" != "0" && "${STRICT_VALIDATION}" != "1" ]]; then
    printf 'STRICT_VALIDATION must be 0 or 1, got %s.\n' "${STRICT_VALIDATION}" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${DEVICE#cuda:}"
printf 'Selected device: %s (CUDA_VISIBLE_DEVICES=%s)\n' "${DEVICE}" "${CUDA_VISIBLE_DEVICES}"
if ! batch_init_status; then exit 2; fi

mapfile -t CONFIGS < <(
    find "${CONFIG_ROOT}/ECNR" -type f -name '*.yaml' -print | LC_ALL=C sort
)
if [[ "${#CONFIGS[@]}" -ne "${EXPECTED_TOTAL}" ]]; then
    printf 'Expected %d exploration-v6 configs, found %d. Regenerate with scripts/generate_exploration_v6_configs.py.\n' \
        "${EXPECTED_TOTAL}" "${#CONFIGS[@]}" >&2
    exit 2
fi

wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then failures=$((failures + 1)); fi
    unset 'pids[index]'
    pids=("${pids[@]}")
}

cd "${REPO_ROOT}"
printf '== ECNR main-training sweep (%d configs, max_parallel=%d) ==\n' "${#CONFIGS[@]}" "${MAX_PARALLEL_JOBS}"
failures=0
pids=()
for index in "${!CONFIGS[@]}"; do
    while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do wait_for_pid_at 0; done
    batch_run_one_config "${CONFIGS[${index}]}" "$((index + 1))" "${#CONFIGS[@]}" &
    pids+=("$!")
done
while [[ "${#pids[@]}" -gt 0 ]]; do wait_for_pid_at 0; done

batch_rebuild_failures
if [[ "${DRY_RUN}" != "1" ]]; then
    summary_args=(
        --config-root "${CONFIG_ROOT}"
        --status "${STATUS_FILE}"
        --output "${LOG_ROOT}/exploration_summary.tsv"
        --profile-output "${LOG_ROOT}/profile_summary.tsv"
        --attention-output "${LOG_ROOT}/needs_attention.txt"
        --repo-root "${REPO_ROOT}"
        --run-root "${RUN_ROOT}"
        --regression-tolerance-db "${PSNR_REGRESSION_TOLERANCE_DB}"
    )
    if [[ "${STRICT_VALIDATION}" == "1" ]]; then
        summary_args+=(--fail-if-no-eligible-profile)
    fi
    if ! conda run --no-capture-output -n "${CONDA_ENV}" python \
        "${SCRIPT_DIR}/summarize_exploration_v6.py" "${summary_args[@]}"; then
        printf 'FAILED: exploration-v6 has no eligible three-target ECNR profile.\n' >&2
        failures=$((failures + 1))
    fi
fi

printf 'Completed %d exploration-v6 configs; failures=%d; status=%s\n' \
    "${#CONFIGS[@]}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
