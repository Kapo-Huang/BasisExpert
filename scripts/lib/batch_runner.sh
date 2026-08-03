#!/usr/bin/env bash

# Shared status/retry primitives for the formal and exploration batch entrypoints.
# The caller must define REPO_ROOT, LOG_ROOT, STATUS_FILE, FAILURE_FILE,
# CONDA_ENV, and DRY_RUN before invoking these functions.

batch_init_status() {
    mkdir -p "${LOG_ROOT}/logs"
    if [[ ! -f "${STATUS_FILE}" ]]; then
        printf 'config\tstatus\texit_code\tlog\n' > "${STATUS_FILE}"
    else
        local header
        IFS= read -r header < "${STATUS_FILE}" || true
        if [[ "${header}" != $'config\tstatus\texit_code\tlog' ]]; then
            printf 'Unsupported Status.tsv header in %s: %s\n' "${STATUS_FILE}" "${header}" >&2
            return 2
        fi
    fi
    touch "${FAILURE_FILE}"
}

batch_latest_status() {
    local relative="$1"
    awk -F '\t' -v wanted="${relative}" 'NR > 1 && $1 == wanted { value=$2 } END { print value }' "${STATUS_FILE}"
}

batch_attempt_number() {
    local relative="$1"
    local previous
    previous="$(awk -F '\t' -v wanted="${relative}" 'NR > 1 && $1 == wanted && $2 == "running" { count += 1 } END { print count + 0 }' "${STATUS_FILE}")"
    printf '%d\n' "$((previous + 1))"
}

batch_append_status() {
    local relative="$1"
    local status="$2"
    local exit_code="$3"
    local log_path="$4"
    if command -v flock >/dev/null 2>&1; then
        (
            flock 9
            printf '%s\t%s\t%s\t%s\n' "${relative}" "${status}" "${exit_code}" "${log_path}" >> "${STATUS_FILE}"
        ) 9>"${STATUS_FILE}.lock"
    else
        local lock_dir="${STATUS_FILE}.lockdir"
        local owner_file="${lock_dir}/owner"
        local owner_pid=""
        local wait_count=0
        while ! mkdir "${lock_dir}" 2>/dev/null; do
            wait_count=$((wait_count + 1))
            if [[ -f "${owner_file}" ]]; then
                IFS= read -r owner_pid < "${owner_file}" || true
                if [[ "${owner_pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${owner_pid}" 2>/dev/null; then
                    rm -f "${owner_file}"
                    rmdir "${lock_dir}" 2>/dev/null || true
                    continue
                fi
            elif [[ "${wait_count}" -ge 100 ]]; then
                rmdir "${lock_dir}" 2>/dev/null || true
                wait_count=0
                continue
            fi
            sleep 0.05
        done
        printf '%s\n' "${BASHPID:-$$}" > "${owner_file}"
        (
            trap 'rm -f "${owner_file}"; rmdir "${lock_dir}" 2>/dev/null || true' EXIT INT TERM
            printf '%s\t%s\t%s\t%s\n' "${relative}" "${status}" "${exit_code}" "${log_path}" >> "${STATUS_FILE}"
        )
    fi
}

batch_module_for_config() {
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

batch_run_one_config() {
    local config="$1"
    local index="$2"
    local total="$3"
    local relative status attempt safe_name log_path module exit_code
    relative="${config#${REPO_ROOT}/}"
    status="$(batch_latest_status "${relative}")"
    if [[ "${status}" == "ok" ]]; then
        printf '[%d/%d] SKIP ok: %s\n' "${index}" "${total}" "${relative}"
        return 0
    fi

    attempt="$(batch_attempt_number "${relative}")"
    safe_name="${relative//\//__}"
    safe_name="${safe_name%.yaml}"
    log_path="${LOG_ROOT}/logs/${safe_name}.attempt-${attempt}.log"
    module="$(batch_module_for_config "${config}")"
    local -a command=(conda run --no-capture-output -n "${CONDA_ENV}" python -m "${module}" train --config "${config}")

    printf '[%d/%d] RUN attempt=%d previous=%s: %s\n' "${index}" "${total}" "${attempt}" "${status:-missing}" "${relative}"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf 'DRY_RUN:'
        printf ' %q' "${command[@]}"
        printf '\n'
        return 0
    fi

    batch_append_status "${relative}" "running" "" "${log_path}"
    "${command[@]}" 2>&1 | tee "${log_path}"
    exit_code="${PIPESTATUS[0]}"
    if [[ "${exit_code}" -eq 0 ]]; then
        batch_append_status "${relative}" "ok" "0" "${log_path}"
    else
        batch_append_status "${relative}" "failed" "${exit_code}" "${log_path}"
        printf 'FAILED (%d): %s\n' "${exit_code}" "${relative}" >&2
    fi
    return "${exit_code}"
}

batch_rebuild_failures() {
    local temporary="${FAILURE_FILE}.tmp"
    awk -F '\t' '
        NR > 1 { latest[$1]=$2 }
        END { for (config in latest) if (latest[config] == "failed") print config }
    ' "${STATUS_FILE}" | LC_ALL=C sort > "${temporary}"
    mv "${temporary}" "${FAILURE_FILE}"
}
