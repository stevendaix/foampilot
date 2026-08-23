from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JOS3_SRC = Path('/home/ubuntu/JOS-3/src')
if str(JOS3_SRC) not in sys.path:
    sys.path.insert(0, str(JOS3_SRC))
FOAMPILOT_SRC = ROOT / 'foampilot' / 'src'
if str(FOAMPILOT_SRC) not in sys.path:
    sys.path.insert(0, str(FOAMPILOT_SRC))

import jos3
import importlib.util
import types

# Load only the physiology package, avoiding FoamPilot's heavy CAD imports.
_pkg = types.ModuleType('foampilot')
_pkg.__path__ = [str(FOAMPILOT_SRC / 'foampilot')]
sys.modules.setdefault('foampilot', _pkg)
_phys = types.ModuleType('foampilot.physiology')
_phys.__path__ = [str(FOAMPILOT_SRC / 'foampilot' / 'physiology')]
sys.modules.setdefault('foampilot.physiology', _phys)
_jpkg = types.ModuleType('foampilot.physiology.jos3')
_jpkg.__path__ = [str(FOAMPILOT_SRC / 'foampilot' / 'physiology' / 'jos3')]
sys.modules.setdefault('foampilot.physiology.jos3', _jpkg)
_spec = importlib.util.spec_from_file_location(
    'foampilot.physiology.jos3.jos3',
    FOAMPILOT_SRC / 'foampilot' / 'physiology' / 'jos3' / 'jos3.py',
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _mod
_spec.loader.exec_module(_mod)
_jpkg.JOS3 = _mod.JOS3
_jpkg.BODY_NAMES = _mod.BODY_NAMES
_spec2 = importlib.util.spec_from_file_location(
    'foampilot.physiology.coupling',
    FOAMPILOT_SRC / 'foampilot' / 'physiology' / 'coupling.py',
)
_coupling = importlib.util.module_from_spec(_spec2)
sys.modules[_spec2.name] = _coupling
_spec2.loader.exec_module(_coupling)
EmbeddedJOS3 = _mod.JOS3
SurfaceMapping = _coupling.SurfaceMapping
DistributedSurfaceNetwork = _coupling.DistributedSurfaceNetwork


def check_original():
    model = jos3.JOS3(ex_output='all')
    model.Ta = 28.0
    model.Tr = 28.0
    model.RH = 50.0
    model.Va = 0.1
    model.simulate(times=1, dtime=1.0)
    assert model._cap.shape == (85,)
    assert model._cdt.shape == (85, 85)
    assert np.all(model._cap > 0)
    # Internal conduction/perfusion matrices must conserve pairwise exchange.
    # This is checked on the coefficient matrices before division by capacity.
    from jos3 import matrix, thermoregulation as threg
    bf = np.zeros(17)
    local = matrix.localarr(bf, bf, bf, bf, 0.0, 0.0)
    whole = matrix.wholebody(*matrix.vessel_bloodflow(bf, bf, bf, bf, 0.0, 0.0), 0.0, 0.0)
    internal = local + whole + model._cdt
    asym = np.max(np.abs(internal - internal.T))
    row_sum_max = np.max(np.abs(internal.sum(axis=1)))
    operator = -internal + np.diag(internal.sum(axis=1))
    operator_row_sum_max = np.max(np.abs(operator.sum(axis=1)))
    return {
        'num_nodes': int(model._cap.size),
        'matrix_shape': tuple(model._cdt.shape),
        'internal_exchange_asymmetry_W_per_K': float(asym),
        'internal_row_sum_max_W_per_K': float(row_sum_max),
        'assembled_operator_row_sum_max_W_per_K': float(operator_row_sum_max),
        'capacity_min_J_per_K': float(model._cap.min()),
        'capacity_max_J_per_K': float(model._cap.max()),
        't_skin_mean_C': float(model.TskMean),
    }


def check_distributed():
    model = EmbeddedJOS3(ex_output='all')
    zone_ids = np.repeat(np.arange(17), 2)
    areas = np.tile(np.array([0.04, 0.06]), 17)
    mapping = SurfaceMapping(zone_ids=zone_ids, areas=areas)
    initial = np.linspace(33.0, 35.0, 34)
    network = DistributedSurfaceNetwork(model, mapping, surface_temperature=initial)
    h = np.full(34, 10.0)
    ta = np.full(34, 20.0)
    before = network.surface_temperature.copy()
    exchange = network.step(h, ta, dtime=0.5)
    # Physical signs: environment receives heat for Ts > Ta; body receives the
    # opposite of outward dry heat only in the separate nodal_flux convention.
    env_positive = bool(np.all(exchange.environment_power > 0.0))
    # Surface state must change independently, and total capacities must match JOS3 skin capacities.
    independent = bool(not np.isclose(network.surface_temperature[0], network.surface_temperature[1]))
    capacity_error = np.max(np.abs(
        np.bincount(zone_ids, weights=network.capacity, minlength=17)
        - model._cap[model.skin_node_indices]
    ))
    return {
        'environment_power_positive_for_hot_surface': env_positive,
        'independent_surface_states': independent,
        'max_zone_capacity_error_J_per_K': float(capacity_error),
        'total_environment_power_W': float(exchange.environment_power.sum()),
        'total_body_power_W': float(exchange.body_power.sum()),
        'surface_delta_min_C': float((network.surface_temperature - before).min()),
        'surface_delta_max_C': float((network.surface_temperature - before).max()),
    }


if __name__ == '__main__':
    print('ORIGINAL', check_original())
    print('DISTRIBUTED', check_distributed())
