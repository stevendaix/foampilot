#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CASE_NAME=${1:?usage: $0 CASE_NAME}
: "${WM_PROJECT_DIR:?Source OpenFOAM Foundation 13 before running a case}"
command -v urbanMicroclimateFoam >/dev/null 2>&1 || {
    echo "urbanMicroclimateFoam is not built; run ./Allwmake first" >&2
    exit 3
}
export PYTHONPATH="$ROOT/../../foampilot/src${PYTHONPATH:+:$PYTHONPATH}"
exec python3 "$ROOT/run.py" --case "$CASE_NAME"
