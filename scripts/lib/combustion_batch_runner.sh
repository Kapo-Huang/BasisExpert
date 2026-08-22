#!/usr/bin/env bash

# Shared staged runner for the fixed Combustion experiment entrypoints.
# The caller must define SCRIPT_DIR, REPO_ROOT, CONFIG_LIST_FILE, LOG_ROOT,
# STATUS_FILE, FAILURE_FILE, SERVER_ENV, CONDA_ENV, PYTHON_BIN, DRY_RUN,
# MAX_PARALLEL_JOBS,
# STAGE_LABELS and STAGE_PATTERNS.

COMBUSTION_GROUP_DELIM=$'\034'

combustion_join_configs() {
    local IFS="${COMBUSTION_GROUP_DELIM}"
    printf '%s' "$*"
}

combustion_wait_for_pid_at() {
    local index="$1"
    local pid="${pids[${index}]}"
    if ! wait "${pid}"; then
        failures=$((failures + 1))
    fi
    unset "pids[${index}]"
    pids=("${pids[@]}")
}

combustion_run_stage() {
    local stage_index="$1"
    local group="${stage_groups[${stage_index}]}"
    local -a configs=()
    local config
    IFS="${COMBUSTION_GROUP_DELIM}" read -r -a configs <<< "${group}"
    printf '== %s (%d config%s, max_parallel=%s) ==\n' \
        "${STAGE_LABELS[${stage_index}]}" "${#configs[@]}" \
        "$([[ "${#configs[@]}" -eq 1 ]] && printf '' || printf 's')" \
        "${MAX_PARALLEL_JOBS}"
    pids=()
    for config in "${configs[@]}"; do
        completed=$((completed + 1))
        while [[ "${#pids[@]}" -ge "${MAX_PARALLEL_JOBS}" ]]; do
            combustion_wait_for_pid_at 0
        done
        batch_run_one_config "${config}" "${completed}" "${total}" &
        pids+=("$!")
    done
    while [[ "${#pids[@]}" -gt 0 ]]; do
        combustion_wait_for_pid_at 0
    done
}

combustion_batch_main() {
    if ! [[ "${MAX_PARALLEL_JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        printf 'MAX_PARALLEL_JOBS must be a positive integer, got %s.\n' "${MAX_PARALLEL_JOBS}" >&2
        return 2
    fi

    local -a stage_groups=()
    local raw line relative matched_stage index
    local total=0
    local additional_group=""
    for index in "${!STAGE_LABELS[@]}"; do
        stage_groups+=("")
    done

    while IFS= read -r raw || [[ -n "${raw}" ]]; do
        line="${raw%$'\r'}"
        line="${line%%#*}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [[ -n "${line}" ]] || continue
        relative="${line#./}"
        matched_stage=-1
        for index in "${!STAGE_PATTERNS[@]}"; do
            if [[ "${relative}" == ${STAGE_PATTERNS[${index}]} ]]; then
                matched_stage="${index}"
                break
            fi
        done
        if [[ "${matched_stage}" -eq -1 ]]; then
            if [[ -n "${additional_group}" ]]; then
                additional_group="${additional_group}${COMBUSTION_GROUP_DELIM}${REPO_ROOT}/${relative}"
            else
                additional_group="${REPO_ROOT}/${relative}"
            fi
        elif [[ -n "${stage_groups[${matched_stage}]}" ]]; then
            stage_groups[${matched_stage}]="${stage_groups[${matched_stage}]}${COMBUSTION_GROUP_DELIM}${REPO_ROOT}/${relative}"
        else
            stage_groups[${matched_stage}]="${REPO_ROOT}/${relative}"
        fi
        total=$((total + 1))
    done < "${CONFIG_LIST_FILE}"

    if [[ -n "${additional_group}" ]]; then
        STAGE_LABELS+=("Additional configs")
        stage_groups+=("${additional_group}")
    fi

    batch_init_status || return $?
    cd "${REPO_ROOT}" || return $?
    local failures=0
    local completed=0
    local -a pids=()
    printf '%s: %d configs, max_parallel=%s\n' "${BATCH_LABEL}" "${total}" "${MAX_PARALLEL_JOBS}"
    for index in "${!STAGE_LABELS[@]}"; do
        [[ -n "${stage_groups[${index}]}" ]] || continue
        combustion_run_stage "${index}"
    done
    batch_rebuild_failures
    printf 'Completed %d configs; failures=%d; status=%s\n' \
        "${completed}" "${failures}" "${STATUS_FILE}"
    if [[ "${failures}" -ne 0 ]]; then
        return 1
    fi
}
