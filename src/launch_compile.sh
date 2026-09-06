#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${WM_PROJECT_DIR:-}" ]]; then
    printf 'OpenFOAM is not loaded; source $WM_PROJECT_DIR/etc/bashrc first.\n' >&2
    exit 1
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${script_dir}/modularWKPressure"
wmake
