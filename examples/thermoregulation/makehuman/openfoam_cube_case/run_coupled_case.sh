#!/usr/bin/env bash
set +e
CASE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$CASE/../../../.." && pwd)"
DRIVER="$ROOT/examples/thermoregulation/openfoam_jos3_coupling/openfoam13_jos3_driver.py"
cd "$CASE"
. /opt/openfoam13/etc/bashrc || true
set -eo pipefail
rm -f comms/data.in comms/data.out comms/OpenFOAM.lock comms/Python.lock
python3 "$DRIVER" "$CASE" > coupled_jos3.log 2>&1 &
DRIVER_PID=$!
cleanup() { kill "$DRIVER_PID" 2>/dev/null || true; }
trap cleanup EXIT
for i in $(seq 1 60); do
    if [ -d "comms" ]; then
        break
    fi
    sleep 0.5
done
foamRun > coupled_openfoam.log 2>&1
wait "$DRIVER_PID" || true
printf '%s\n' 'Couplage terminé.'
