#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_ROOT="${REPO_ROOT}/configs_exploration_v2"
RUN_ROOT="${REPO_ROOT}/runs/exploration_v2"
CONDA_ENV="${CONDA_ENV:-compression}"
RUN_TOKEN="${RUN_TOKEN:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${BATCH_LOG_ROOT:-${REPO_ROOT}/batch_logs/exploration_v2/${RUN_TOKEN}}"
if command -v cygpath >/dev/null 2>&1; then
    LOG_ROOT="$(cygpath -u "${LOG_ROOT}")"
fi
STATUS_FILE="${LOG_ROOT}/status.tsv"
FAILURE_FILE="${LOG_ROOT}/failed.txt"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-0}"
GROUP_DELIM=$'\034'
EXPECTED_TOTAL=53

source "${SCRIPT_DIR}/lib/batch_runner.sh"
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

collect_sorted() {
    local label="$1"
    shift
    local -a configs=()
    local pattern path
    for pattern in "$@"; do
        while IFS= read -r path; do configs+=("${path}"); done < <(compgen -G "${pattern}" | LC_ALL=C sort)
    done
    add_group "${label}" "${configs[@]}"
}

collect_neural() {
    local label="$1"
    local mode="$2"
    local pattern="$3"
    local -a configs=()
    local path
    while IFS= read -r path; do
        case "${mode}" in
            manager) [[ "${path}" == *"__managerpretrain.yaml" ]] || continue ;;
            main) [[ "${path}" != *"__managerpretrain.yaml" ]] || continue ;;
            *) printf 'Unknown NeuralExpert collection mode: %s\n' "${mode}" >&2; return 2 ;;
        esac
        configs+=("${path}")
    done < <(compgen -G "${pattern}" | LC_ALL=C sort)
    add_group "${label}" "${configs[@]}"
}

# Managers must exist before their matching NeuralExpert main runs.
for profile in depth1 depth2 depth3; do
    collect_neural "NeuralExpert:Size326:${profile}:manager" manager \
        "${CONFIG_ROOT}/NeuralExpert/Size326/${profile}/*.yaml"
done

# Re-run the previously blocked method after the code fix.
collect_sorted "MC-INR:Size163:fixed" "${CONFIG_ROOT}/MC-INR/Size163/*/*.yaml"

# Stage A compares the previous winner with experts9/10 under the same top-3 routing.
collect_sorted "VarExpert:Size163:expert-count-control" \
    "${CONFIG_ROOT}/VarExpert/Size163/experts8_top3/*.yaml" \
    "${CONFIG_ROOT}/VarExpert/Size163/experts9_top3/*.yaml" \
    "${CONFIG_ROOT}/VarExpert/Size163/experts10_top3/*.yaml"

# Stage B scans top-1 through top-max for experts9 and experts10.  top-3 rows
# are skipped here because Stage A already ran them.
declare -a var_top_sweep=()
for experts in 9 10; do
    for top_k in $(seq 1 "${experts}"); do
        [[ "${top_k}" -eq 3 ]] && continue
        var_top_sweep+=("${CONFIG_ROOT}/VarExpert/Size163/experts${experts}_top${top_k}/ionization.yaml")
    done
done
add_group "VarExpert:Size163:top-k-sweep" "${var_top_sweep[@]}"

for profile in depth1 depth2 depth3; do
    collect_neural "NeuralExpert:Size326:${profile}:main" main \
        "${CONFIG_ROOT}/NeuralExpert/Size326/${profile}/*.yaml"
done

total=0
for group in "${GROUP_CONFIGS[@]}"; do
    IFS="${GROUP_DELIM}" read -r -a configs <<< "${group}"
    total=$((total + ${#configs[@]}))
done
if [[ "${total}" -ne "${EXPECTED_TOTAL}" ]]; then
    printf 'Expected %d exploration-v2 configs, found %d. Regenerate with scripts/generate_exploration_v2_configs.py.\n' \
        "${EXPECTED_TOTAL}" "${total}" >&2
    exit 2
fi

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
    if ! conda run --no-capture-output -n "${CONDA_ENV}" python "${SCRIPT_DIR}/summarize_size_exploration.py" \
        --config-root "${CONFIG_ROOT}" \
        --status "${STATUS_FILE}" \
        --output "${LOG_ROOT}/exploration_summary.tsv" \
        --repo-root "${REPO_ROOT}" \
        --run-root "${RUN_ROOT}"; then
        printf 'FAILED: could not build exploration-v2 summary.\n' >&2
        failures=$((failures + 1))
    fi
fi
printf 'Completed %d exploration-v2 configs; failures=%d; status=%s\n' "${total}" "${failures}" "${STATUS_FILE}"
if [[ "${failures}" -ne 0 ]]; then exit 1; fi
