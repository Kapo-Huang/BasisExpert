#!/usr/bin/env bash

# Shared server-profile selection and Python command construction.
# User-facing entrypoints call server_env_init "$@" before launching Python.

server_env_is_positive_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

server_env_configure_threads() {
    local default_threads="${VAR_EXPERT_INR_NUM_THREADS:-64}"
    if ! server_env_is_positive_int "${default_threads}"; then
        default_threads=64
    fi

    local key value
    local -a thread_env_keys=(
        OMP_NUM_THREADS
        MKL_NUM_THREADS
        OPENBLAS_NUM_THREADS
        NUMEXPR_NUM_THREADS
        VECLIB_MAXIMUM_THREADS
        BLIS_NUM_THREADS
        LOKY_MAX_CPU_COUNT
    )
    for key in "${thread_env_keys[@]}"; do
        value="${!key-}"
        if ! server_env_is_positive_int "${value}"; then
            printf -v "${key}" '%s' "${default_threads}"
        fi
        export "${key}"
    done
}

server_env_init() {
    local selected="${SERVER_ENV:-original}"
    local argument next_argument
    local -a arguments=("$@")
    local index

    for index in "${!arguments[@]}"; do
        argument="${arguments[${index}]}"
        case "${argument}" in
            --env)
                next_argument="${arguments[$((index + 1))]:-}"
                if [[ -z "${next_argument}" ]]; then
                    printf '%s\n' '--env requires a value: original or autodl.' >&2
                    return 2
                fi
                selected="${next_argument}"
                ;;
            --env=*) selected="${argument#--env=}" ;;
            env=*) selected="${argument#env=}" ;;
        esac
    done

    case "${selected}" in
        original|autodl) ;;
        *)
            printf 'Unsupported server environment %q; expected original or autodl.\n' "${selected}" >&2
            return 2
            ;;
    esac

    SERVER_ENV="${selected}"
    CONDA_ENV="${CONDA_ENV:-compression}"
    PYTHON_BIN="${PYTHON_BIN:-python}"
    if [[ -z "${RUNS_ROOT:-}" ]]; then
        if [[ "${SERVER_ENV}" == "autodl" ]]; then
            RUNS_ROOT="/root/autodl-tmp/runs"
        elif [[ -n "${REPO_ROOT:-}" ]]; then
            RUNS_ROOT="${REPO_ROOT}/runs"
        else
            printf '%s\n' 'REPO_ROOT must be defined before server_env_init for the original environment.' >&2
            return 2
        fi
    fi
    export SERVER_ENV CONDA_ENV PYTHON_BIN RUNS_ROOT
    server_env_configure_threads
}

server_python_command() {
    local result_name="$1"
    shift
    local -n result_ref="${result_name}"

    if [[ "${SERVER_ENV:-original}" == "autodl" ]]; then
        result_ref=("${PYTHON_BIN:-python}" "$@")
    else
        result_ref=(
            conda run --no-capture-output
            -n "${CONDA_ENV:-compression}"
            "${PYTHON_BIN:-python}"
            "$@"
        )
    fi
}

server_python() {
    local -a command=()
    server_python_command command "$@"
    "${command[@]}"
}
