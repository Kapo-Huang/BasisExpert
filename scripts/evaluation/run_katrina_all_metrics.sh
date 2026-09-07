#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/../lib/server_env.sh"
server_env_init "$@" || exit $?

cd "${REPO_ROOT}"
command=()
server_python_command command scripts/evaluation/run_batch.py \
    --config configs/evaluation/evaluation_katrina_all_metrics.yaml
printf 'Recomputing all Katrina quality metrics.\n'
"${command[@]}"
