"""Third medical_build example: Windkessel pressure waveform at the inlet.

The Windkessel model is used as a 0-D upstream cardiovascular model. Its
computed aortic pressure p1(t) is exported as an OpenFOAM uniformFixedValue
table at ``aorta_surface_inlet``. The validated snappy case remains untouched.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from foampilot.model_addon.windkessel import Windkessel

ROOT = Path(__file__).resolve().parents[2]
COA = ROOT / 'examples' / 'coa'
OUT = ROOT / 'examples' / 'medical_build' / 'outputs' / 'windkessel_inlet_example'
OUT.mkdir(parents=True, exist_ok=True)
CASE = ROOT / 'examples' / 'medical_build' / 'openfoam_case'


def load_flow():
    raw = np.loadtxt(COA / 'data_typec_q.csv', delimiter=',', skiprows=1)
    return raw[:, 0], raw[:, 1] * 1e-6  # ml/s -> m3/s


def write_pressure_field(path: Path, t: np.ndarray, p: np.ndarray):
    rows = '\n'.join(f'        ({ti:.8g} {pi:.8g})' for ti, pi in zip(t, p))
    path.write_text(f'''FoamFile
{{
    version 2.0;
    format ascii;
    class volScalarField;
    object p;
}}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    outer {{ type zeroGradient; }}
    aorta_surface_inlet
    {{
        type uniformFixedValue;
        uniformValue table
        (
{rows}
        );
    }}
    aorta_surface_outlet_0 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_1 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_2 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_3 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_5 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_6 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_7 {{ type fixedValue; value uniform 0; }}
    aorta_surface_outlet_8 {{ type fixedValue; value uniform 0; }}
    aorta_surface_wall {{ type zeroGradient; }}
}}
''')


def main():
    t, q = load_flow()
    wk = Windkessel(t_flow=t, q_flow=q, Rc=1.0e6, Rp=2.0e9, C=2.0e-7, L=5.0e3, Cprox=1.0e-8, periodic=True)
    sol = wk.solve(t_start=0.0, t_end=5.0 * (t[-1] - t[0]), n_steps=5000)
    # Keep the final cycle after transient convergence.
    mask = sol.t >= sol.t[-1] - (t[-1] - t[0])
    tc = sol.t[mask] - sol.t[mask][0]
    pc = np.asarray(sol.p1[mask])
    qc = np.asarray(wk.Q(sol.t[mask]))
    np.savetxt(OUT / 'windkessel_inlet_waveform.csv', np.c_[tc, qc, pc], delimiter=',', header='time_s,flow_m3_s,pressure_Pa', comments='')
    write_pressure_field(OUT / 'p.windkessel', tc, pc)
    report = {'model': '5-element Windkessel', 'boundary_patch': 'aorta_surface_inlet', 'period_s': float(t[-1]-t[0]), 'parameters': {'Rc':wk.Rc,'Rp':wk.Rp,'C':wk.C,'L':wk.L,'Cprox':wk.Cprox}, 'pressure_min_Pa':float(pc.min()), 'pressure_max_Pa':float(pc.max()), 'flow_min_m3_s':float(qc.min()), 'flow_max_m3_s':float(qc.max()), 'solver_success':bool(sol.success), 'reference_case':str(CASE)}
    (OUT / 'windkessel_inlet_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
