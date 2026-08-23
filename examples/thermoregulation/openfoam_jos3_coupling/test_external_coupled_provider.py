#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import tempfile
import numpy as np

MODULE = Path(__file__).resolve().parents[3] / "foampilot" / "src" / "foampilot" / "postprocess" / "openfoam_external_coupled.py"
spec = importlib.util.spec_from_file_location("openfoam_external_coupled", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OpenFOAMExternalCoupledProvider = module.OpenFOAMExternalCoupledProvider


def main():
    with tempfile.TemporaryDirectory() as directory:
        comms = Path(directory)
        (comms / "h.out").write_text("5\n6\n", encoding="utf-8")
        (comms / "air_temperature.out").write_text("293.15\n294.15\n", encoding="utf-8")
        provider = OpenFOAMExternalCoupledProvider(
            comms, timeout=1.0, fields=("h", "air_temperature"), output_field="qJOS3"
        )
        fields = provider.read_nodal_fields()
        np.testing.assert_allclose(fields["h"], [5.0, 6.0])
        np.testing.assert_allclose(fields["air_temperature"], [20.0, 21.0])
        provider.write_nodal_flux(np.array([10.0, -4.0]))
        np.testing.assert_allclose(np.loadtxt(comms / "qJOS3.in"), [10.0, -4.0])
        assert (comms / "OpenFOAM.lock").exists()
    print("Provider externalCoupled validé.")


if __name__ == "__main__":
    main()
