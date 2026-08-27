#!/usr/bin/env bash
set -euo pipefail

: "${WM_PROJECT_DIR:?Chargez OpenFOAM Foundation 13 avant d’exécuter ce script}"

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for dir in \
    "$ROOT/ported/boundaryConditions/ZYturbulentInlet" \
    "$ROOT/ported/boundaryConditions/turbulentInletTable" \
    "$ROOT/ported/MachineLearningTurbulenceModels/calculateNut" \
    "$ROOT/ported/MachineLearningTurbulenceModels/calculateGamma" \
    "$ROOT/ported/MachineLearningTurbulenceModels/calculateRFV" \
    "$ROOT/ported/MachineLearningTurbulenceModels/calculateRFVperp" \
    "$ROOT/ported/MachineLearningTurbulenceModels/calculateRperp"
do
    echo "==> Building ${dir#$ROOT/}"
    (cd "$dir" && wclean libso >/dev/null 2>&1 || true)
    if grep -q '^EXE =' "$dir/Make/files"; then
        (cd "$dir" && wmake)
    else
        (cd "$dir" && wmake libso)
    fi
done

echo "OpenFOAM 13 ports built successfully."
