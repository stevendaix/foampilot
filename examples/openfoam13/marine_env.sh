#!/usr/bin/env bash
# Common FoamPilot marine environment resolver for OpenFOAM Foundation 13.
# Source this file from a runner; it does not assume a fixed installation path.

_marine_env_source="${BASH_SOURCE[0]}"
MARINE_OPENFOAM_ROOT="$(cd "$(dirname "$_marine_env_source")" && pwd)"
MARINE_REPO_ROOT="$(cd "$MARINE_OPENFOAM_ROOT/.." && pwd)"

if [ -z "${WM_PROJECT_DIR:-}" ]; then
    for _marine_bashrc in \
        "${FOAM_INST_DIR:-}/OpenFOAM-${WM_PROJECT_VERSION:-13}/etc/bashrc" \
        "${HOME}/OpenFOAM/OpenFOAM-${WM_PROJECT_VERSION:-13}/etc/bashrc"; do
        if [ -n "$_marine_bashrc" ] && [ -f "$_marine_bashrc" ]; then
            # shellcheck disable=SC1090
            source "$_marine_bashrc"
            break
        fi
    done
fi

if [ -z "${WM_PROJECT_DIR:-}" ]; then
    echo "OpenFOAM Foundation 13 is not loaded; source its etc/bashrc or set WM_PROJECT_DIR" >&2
    return 1 2>/dev/null || exit 1
fi

case "${WM_PROJECT_VERSION:-}" in
    13|13.*) ;;
    *)
        echo "OpenFOAM Foundation 13 required; found ${WM_PROJECT_VERSION:-unknown}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

export MARINE_OPENFOAM_ROOT MARINE_REPO_ROOT
marine_require_command()
{
    command -v "$1" >/dev/null 2>&1 || {
        echo "Required Foundation 13 command not found: $1" >&2
        return 1
    }
}
