#!/usr/bin/env bash
set -euo pipefail

: "${WM_PROJECT_DIR:?Source OpenFOAM 13 avant d’exécuter ce script}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
module_root="$repo_root/third_party/openHFDIB-DEM"

if [[ "${WM_PROJECT_VERSION:-}" != "13" ]]; then
    echo "Erreur: ce script exige OpenFOAM Foundation 13 (WM_PROJECT_VERSION=13)" >&2
    echo "Version détectée: ${WM_PROJECT_VERSION:-non définie}" >&2
    exit 2
fi

cd "$module_root"
rm -rf src/HFDIBDEM/lnInclude
wmakeLnInclude src/HFDIBDEM
wmake -j1 libso src/HFDIBDEM

lib="$FOAM_USER_LIBBIN/libHFDIBDEM.so"
test -s "$lib"
echo "OK: $lib"
