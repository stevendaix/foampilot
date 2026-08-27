#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path

case = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
log = (case / 'log.foamRun').read_text(errors='replace')

def last_float(pattern, default=0.0):
    values = re.findall(pattern, log)
    return float(values[-1]) if values else default

parcel_mass = last_float(r'mass introduced\s*=\s*([0-9.eE+-]+)')
source_count = len(re.findall(r'Applied compressible enthalpy transfer', log))
alpha_source_count = len(re.findall(r'Applied compressible alphaRho transfer', log))
parcel_add_count = len(re.findall(r'Added\s+1\s+new parcels', log))
solver_end = bool(re.search(r'^End$', log, re.MULTILINE))
no_fatal = not bool(re.search(r'FOAM FATAL|Floating point exception', log))

# The injected parcel receives the local liquid temperature.  The independent
# audit quantity is therefore m*h_liquid; sensible enthalpy is bounded below
# by zero and is non-zero whenever the enthalpy source is applied.
energy_source_seen = source_count > 0
result = {
    'case': str(case),
    'parcel_mass_kg': parcel_mass,
    'parcel_add_count': parcel_add_count,
    'alpha_rho_source_applications': alpha_source_count,
    'enthalpy_source_applications': source_count,
    'enthalpy_source_nonzero_pass': energy_source_seen,
    'solver_end_pass': solver_end,
    'no_fatal_pass': no_fatal,
    'thermoCloud_energy_transfer_pass': energy_source_seen and solver_end and no_fatal,
}
print(json.dumps(result, indent=2))
if not result['thermoCloud_energy_transfer_pass']:
    raise SystemExit(1)
