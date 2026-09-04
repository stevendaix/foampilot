#!/usr/bin/env bash
set -eo pipefail

# Build the Foundation 13 ports shipped with Foampilot.
# Usage: ./tools/build_ported_physics_of13.sh

foam_bashrc="${FOAM_BASHRC:-}"
if [[ -z "$foam_bashrc" && -n "${WM_PROJECT_DIR:-}" ]]; then
    foam_bashrc="$WM_PROJECT_DIR/etc/bashrc"
fi
if [[ -z "$foam_bashrc" || ! -f "$foam_bashrc" ]]; then
    echo "FOAM_BASHRC or WM_PROJECT_DIR must identify an OpenFOAM bashrc" >&2
    exit 2
fi
# The official bashrc may contain optional shell hooks that are not errexit-safe.
set +e
source "$foam_bashrc"
foamrc=$?
set -e
if [[ "$foamrc" -ne 0 ]]; then
    echo "Failed to source OpenFOAM bashrc: $foam_bashrc (status ${foamrc})" >&2
    exit "$foamrc"
fi

if [[ "${WM_PROJECT_VERSION:-}" != "13" ]]; then
    echo "OpenFOAM Foundation 13 must be sourced (WM_PROJECT_VERSION=${WM_PROJECT_VERSION:-unset})" >&2
    exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

build_one() {
    local label="$1"
    local dir="$2"
    echo "==> ${label}: ${dir}"
    (cd "$dir" && wclean && wmake libso)
}

build_one "sediFoam drag models" \
    "$ROOT/third_party/sediFoam/lammpsFoam/dragModels"

build_one "libAcoustics core" \
    "$ROOT/third_party/libAcoustics/Sources/lib"

echo "OpenFOAM 13 physics libraries built successfully."
ls -l "$FOAM_USER_LIBBIN/libLagrangianInterfacialModels.so" \
      "$FOAM_USER_LIBBIN/libAcoustics.so"
