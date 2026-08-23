#!/usr/bin/env python3
"""Test du couplage JOS-3 en mémoire, sans OpenFOAM ni fichiers d’échange."""

from pathlib import Path
import importlib.util
import sys
import types

import numpy as np

HERE = Path(__file__).resolve()
SRC = HERE.parents[3] / "foampilot" / "src"
JOS3_SRC = HERE.parents[4] / "JOS-3" / "src"
sys.path[:0] = [str(SRC), str(JOS3_SRC)]

import jos3 as reference_jos3

# Chargement ciblé du sous-paquet pour tester le couplage sans les extensions CAD.
sys.modules.setdefault("foampilot", types.ModuleType("foampilot"))
physiology_pkg = types.ModuleType("foampilot.physiology")
physiology_pkg.__path__ = [str(SRC / "foampilot" / "physiology")]
sys.modules["foampilot.physiology"] = physiology_pkg
_spec = importlib.util.spec_from_file_location(
    "foampilot.physiology",
    SRC / "foampilot" / "physiology" / "__init__.py",
    submodule_search_locations=[str(SRC / "foampilot" / "physiology")],
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
JOS3 = _module.JOS3
JOS3NodeCoupler = _module.JOS3NodeCoupler
SurfaceMapping = _module.SurfaceMapping


def main():
    n = 170
    mapping = SurfaceMapping(
        zone_ids=np.arange(n) % 17,
        areas=np.full(n, 0.01),
    )
    reference = reference_jos3.JOS3(ex_output="all")
    reference.To = 22.0
    reference.simulate(times=2, dtime=60.0)

    model = JOS3(ex_output="all")
    model.To = 22.0
    model.simulate(times=2, dtime=60.0)
    np.testing.assert_allclose(model.Tsk, reference.Tsk, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(model.Tcr, reference.Tcr, atol=1e-12, rtol=0.0)
    coupler = JOS3NodeCoupler(model, mapping)

    h = np.full(n, 8.0)
    ta = np.full(n, 22.0)
    ts = ta + 2.0
    exchange = coupler.step_steady(h, ts, ta, dtime=60.0)
    np.testing.assert_allclose(exchange.body_flux, -16.0)
    np.testing.assert_allclose(exchange.zone_power, -1.6)
    assert model.external_heat_flux[model.skin_node_indices].shape == (17,)

    class FakeOpenFOAM:
        def __init__(self):
            self.k = 0
            self.received = []

        def read_nodal_fields(self):
            self.k += 1
            air = np.full(n, 20.0 + self.k)
            return {"h": h, "surface_temperature": air + 1.0, "air_temperature": air}

        def write_nodal_flux(self, flux):
            self.received.append(flux)

    provider = FakeOpenFOAM()
    transient = coupler.run_transient(provider, dtime=60.0, steps=3)
    assert len(transient) == 3
    assert len(provider.received) == 3
    np.testing.assert_allclose(provider.received[-1], -8.0)
    print("Couplage mémoire steady/transitoire validé.")


if __name__ == "__main__":
    main()
