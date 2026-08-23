#!/usr/bin/env python3
"""Validation de l’extension de surface distribuée JOS-3."""
from pathlib import Path
import importlib.util
import sys
import types
import numpy as np

HERE = Path(__file__).resolve()
SRC = HERE.parents[3] / "foampilot" / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE.parents[4] / "JOS-3" / "src"))

sys.modules.setdefault("foampilot", types.ModuleType("foampilot"))
spec = importlib.util.spec_from_file_location(
    "foampilot.physiology",
    SRC / "foampilot" / "physiology" / "__init__.py",
    submodule_search_locations=[str(SRC / "foampilot" / "physiology")],
)
physiology = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = physiology
spec.loader.exec_module(physiology)
JOS3 = physiology.JOS3
SurfaceMapping = physiology.SurfaceMapping
DistributedSurfaceNetwork = physiology.DistributedSurfaceNetwork


def main():
    n = 34
    zone_ids = np.repeat(np.arange(17), 2)
    areas = np.full(n, 0.02)
    mapping = SurfaceMapping(zone_ids=zone_ids, areas=areas)
    model = JOS3(ex_output="all")
    model.Ta = 25.0
    model.Tr = 25.0
    network = DistributedSurfaceNetwork(
        model, mapping, surface_temperature=np.full(n, 34.0)
    )

    np.testing.assert_allclose(
        np.bincount(zone_ids, weights=network.capacity, minlength=17),
        network.skin_capacity,
    )
    np.testing.assert_allclose(
        np.bincount(zone_ids, weights=network.anchor_conductance, minlength=17),
        np.sum(model._cdt[model.skin_node_indices, :], axis=1),
    )

    h = np.full(n, 6.0)
    ta = np.full(n, 25.0)
    ta[0] = 15.0
    ta[1] = 35.0
    old = network.surface_temperature.copy()
    exchange = network.step(h, ta, dtime=0.5)
    assert network.surface_temperature[0] != network.surface_temperature[1]
    assert not np.allclose(network.surface_temperature, old)
    np.testing.assert_allclose(
        exchange.zone_body_power,
        np.bincount(zone_ids, weights=exchange.body_power, minlength=17),
    )
    assert model.environment_mode == "external_surface"
    assert np.isfinite(model.Tsk).all()
    print("Réseau de surface distribué validé.")
    print("Températures locales zone Head :", network.surface_temperature[:2])
    print("Puissance environnementale totale [W] :", exchange.environment_power.sum())


if __name__ == "__main__":
    main()
