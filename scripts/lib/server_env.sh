#!/usr/bin/env bash

# Shared server-profile selection and Python command construction.
# User-facing entrypoints call server_env_init "$@" before launching Python.

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
    export SERVER_ENV CONDA_ENV PYTHON_BIN
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
