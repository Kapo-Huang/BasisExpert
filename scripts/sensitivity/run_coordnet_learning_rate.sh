#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_ROOT="${REPO_ROOT}/configs/sensitivity/coordnet_learning_rate"
RUN_ROOT="${REPO_ROOT}/runs/exploration_CoordNet"
CONDA_ENV="${CONDA_ENV:-compression}"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration_CoordNet/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"; fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
STRICT_VALIDATION="${STRICT_VALIDATION:-1}"
COLLAPSE_THRESHOLD_DB="${COLLAPSE_THRESHOLD_DB:-1.0}"
MINIMUM_GAIN_DB="${MINIMUM_GAIN_DB:-0.1}"
EXPECTED_TOTAL=30

source "${SCRIPT_DIR}/../lib/batch_runner.sh"
if [[ ! "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
    exit 2
fi
if ! batch_init_status; then exit 2; fi

declare -a SIZES=(Size082 Size163 Size326 Size652 Size1304)
declare -a PROFILES=(lr1e-5 lr5e-6)
declare -a CONFIGS=()
for profile in "${PROFILES[@]}"; do
    for size in "${SIZES[@]}"; do
        while IFS= read -r config; do CONFIGS+=("${config}"); done < <(
            find "${CONFIG_ROOT}/CoordNet/${size}/${profile}" -maxdepth 1 -type f -name '*.yaml' -print | LC_ALL=C sort
        )
    done
done

if [[ "${#CONFIGS[@]}" -ne "${EXPECTED_TOTAL}" ]]; then
    printf 'Expected %d CoordNet learning-rate configs, found %d. Regenerate with scripts/sensitivity/generate_coordnet_learning_rate.py.\n' \
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
failures=0
completed=0
pids=()
for config in "${CONFIGS[@]}"; do
    completed=$((completed + 1))
    while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do wait_for_pid_at 0; done
    batch_run_one_config "${config}" "${completed}" "${EXPECTED_TOTAL}" &
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
        --collapse-threshold-db "${COLLAPSE_THRESHOLD_DB}"
        --minimum-gain-db "${MINIMUM_GAIN_DB}"
    )
    if [[ "${STRICT_VALIDATION}" == "1" ]]; then summary_args+=(--fail-on-attention); fi
    if ! conda run --no-capture-output -n "${CONDA_ENV}" python \
        "${SCRIPT_DIR}/summarize_coordnet_learning_rate.py" "${summary_args[@]}"; then
        printf 'FAILED: exploration_CoordNet validation found runs needing attention.\n' >&2
        failures=$((failures + 1))
    fi
fi

printf 'Completed %d exploration_CoordNet configs; failures=%d; status=%s\n' \
    "${EXPECTED_TOTAL}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
