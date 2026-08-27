#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
: "${WM_PROJECT_DIR:?Source OpenFOAM Foundation 13 before validation}"
export PYTHONPATH="$ROOT/../../foampilot/src${PYTHONPATH:+:$PYTHONPATH}"
python3 "$ROOT/run.py" --all --generate --overwrite
for name in streetCanyon_CFD streetCanyon_CFDHAM streetCanyon_CFDHAM_grass streetCanyon_CFDHAM_veg windAroundBuildings_CFDHAM windAroundBuildings_CFDHAM_veg; do
    case_dir="$ROOT/cases/$name"
    [ -f "$case_dir/system/controlDict" ] || { echo "$name: missing generated system/controlDict" >&2; exit 1; }
    grep -q '^application[[:space:]]\+urbanMicroclimateFoam;' "$case_dir/system/controlDict" || {
        echo "$name: wrong application" >&2; exit 1;
    }
    [ -f "$case_dir/system/decomposeParDict" ] || { echo "$name: missing generated decomposeParDict" >&2; exit 1; }
    [ -f "$case_dir/Allrun" ] || { echo "$name: missing generated Allrun" >&2; exit 1; }
    grep -Rqs 'nu[[:space:]]' "$case_dir/constant" || {
        echo "$name: missing explicit nu in constant dictionaries" >&2; exit 1;
    }
    echo "$name: generated Foundation 13 case checks passed"
done
