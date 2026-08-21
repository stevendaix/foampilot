from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "foampilot" / "src"
JOS_SRC = Path("/home/ubuntu/JOS-3/src")
sys.path.insert(0, str(JOS_SRC))

import jos3

# Charge uniquement les modules physiologiques et le provider, sans les dépendances CAD.
_pkg = types.ModuleType("foampilot")
_pkg.__path__ = [str(SRC / "foampilot")]
sys.modules.setdefault("foampilot", _pkg)
_phys = types.ModuleType("foampilot.physiology")
_phys.__path__ = [str(SRC / "foampilot" / "physiology")]
sys.modules.setdefault("foampilot.physiology", _phys)
_jpkg = types.ModuleType("foampilot.physiology.jos3")
_jpkg.__path__ = [str(SRC / "foampilot" / "physiology" / "jos3")]
sys.modules.setdefault("foampilot.physiology.jos3", _jpkg)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


jos3_embedded = load(
    "foampilot.physiology.jos3.jos3",
    SRC / "foampilot" / "physiology" / "jos3" / "jos3.py",
)
_jpkg.JOS3 = jos3_embedded.JOS3
_jpkg.BODY_NAMES = jos3_embedded.BODY_NAMES
coupling = load(
    "foampilot.physiology.coupling",
    SRC / "foampilot" / "physiology" / "coupling.py",
)
provider_module = load(
    "foampilot.postprocess.openfoam_external_coupled",
    SRC / "foampilot" / "postprocess" / "openfoam_external_coupled.py",
)

JOS3 = jos3_embedded.JOS3
SurfaceMapping = coupling.SurfaceMapping
DistributedSurfaceNetwork = coupling.DistributedSurfaceNetwork
OpenFOAMExternalCoupledProvider = provider_module.OpenFOAMExternalCoupledProvider


class TransientFileBridge:
    """Producteur de champs OpenFOAM et collecteur du flux retourné."""

    def __init__(self, directory, n):
        self.directory = Path(directory)
        self.n = n
        self.step_id = 0
        self.returned_fluxes = []

    def write_outgoing_fields(self):
        # Champs exprimés comme OpenFOAM : h en W/m²/K, températures en K.
        h = np.linspace(5.0, 15.0, self.n) + 0.25 * self.step_id
        ta_c = np.linspace(18.0, 24.0, self.n) + 0.5 * self.step_id
        tr_c = ta_c + 4.0 * np.sin(np.linspace(0.0, np.pi, self.n))
        np.savetxt(self.directory / "h.out", h)
        np.savetxt(self.directory / "air_temperature.out", ta_c + 273.15)
        np.savetxt(self.directory / "radiative_temperature.out", tr_c + 273.15)

    def collect_returned_flux(self):
        flux = np.loadtxt(self.directory / "qJOS3.in").reshape(-1)
        self.returned_fluxes.append(flux.copy())
        assert np.all(np.isfinite(flux))
        assert flux.size == self.n
        assert (self.directory / "OpenFOAM.lock").exists()
        (self.directory / "OpenFOAM.lock").unlink()
        self.step_id += 1


def main():
    n = 34
    zone_ids = np.repeat(np.arange(17), 2)
    areas = np.tile(np.array([0.04, 0.06]), 17)
    mapping = SurfaceMapping(zone_ids=zone_ids, areas=areas)
    model = JOS3(ex_output="all")
    initial_temperature = np.linspace(33.0, 35.0, n)
    network = DistributedSurfaceNetwork(
        model, mapping, surface_temperature=initial_temperature
    )

    with tempfile.TemporaryDirectory() as directory:
        bridge = TransientFileBridge(directory, n)
        provider = OpenFOAMExternalCoupledProvider(
            directory,
            timeout=1.0,
            fields=("h", "air_temperature", "radiative_temperature"),
            output_field="qJOS3",
            temperature_unit="K",
        )

        history = []
        for _ in range(4):
            bridge.write_outgoing_fields()
            fields = provider.read_nodal_fields()
            exchange = network.step(
                fields["h"],
                fields["air_temperature"],
                radiative_temperature=fields["radiative_temperature"],
                dtime=0.5,
                hr=4.5,
            )
            # OpenFOAM reçoit un flux surfacique, non une puissance nodale.
            flux = exchange.environment_power / areas
            provider.write_nodal_flux(flux)
            bridge.collect_returned_flux()
            history.append(exchange.surface_temperature.copy())

        assert len(history) == 4
        assert not np.allclose(history[0], history[-1])
        assert not np.allclose(bridge.returned_fluxes[0], bridge.returned_fluxes[-1])
        np.testing.assert_allclose(
            bridge.returned_fluxes[-1] * areas,
            network.last_exchange.environment_power,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isfinite(model.TskMean)

    print("Échange transitoire bidirectionnel validé sur 4 pas.")
    print(f"Flux final min/max [W/m²] : {bridge.returned_fluxes[-1].min():.6g} / {bridge.returned_fluxes[-1].max():.6g}")
    print(f"Température de surface finale min/max [°C] : {history[-1].min():.6g} / {history[-1].max():.6g}")


if __name__ == "__main__":
    main()
