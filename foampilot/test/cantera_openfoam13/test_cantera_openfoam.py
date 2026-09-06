from pathlib import Path
import importlib.util
import sys

MODULE = Path(__file__).parents[2] / "src" / "foampilot" / "coupling" / "cantera_openfoam.py"
spec = importlib.util.spec_from_file_location("cantera_openfoam", MODULE)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
assert spec.loader is not None
spec.loader.exec_module(module)
CanteraOpenFOAMCoupler = module.CanteraOpenFOAMCoupler


def test_equilibrium_is_physical():
    coupler = CanteraOpenFOAMCoupler()
    state = coupler.equilibrate(1000.0, 101325.0, "H2:2,O2:1,N2:3.76")
    assert state.temperature > 1000.0
    assert abs(sum(state.mass_fractions) - 1.0) < 1e-12


def test_csv_exchange(tmp_path: Path):
    source = tmp_path / "openfoam_cells.csv"
    source.write_text(
        "cell,T,p,composition\n0,1000,101325,H2:2,O2:1,N2:3.76".replace(
            "cell,T,p,composition\n0,1000,101325,H2:2,O2:1,N2:3.76",
            "cell,T,p,composition\n0,1000,101325,\"H2:2,O2:1,N2:3.76\"",
        ),
        encoding="utf-8",
    )
    target = tmp_path / "cantera_cells.csv"
    CanteraOpenFOAMCoupler().equilibrate_csv(source, target)
    text = target.read_text(encoding="utf-8")
    assert "T_eq" in text
    assert "thermal_conductivity" in text
