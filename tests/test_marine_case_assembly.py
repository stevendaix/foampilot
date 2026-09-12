import importlib.util
import sys
from pathlib import Path


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).parents[2]
controls = load("assembly_controls", ROOT / "foampilot/src/foampilot/solver/marine_controls.py")
forces = load("assembly_forces", ROOT / "foampilot/src/foampilot/solver/marine_forces.py")
actuation = load("assembly_actuation", ROOT / "foampilot/src/foampilot/solver/marine_actuation_disk.py")
case_module = load("assembly_case", ROOT / "foampilot/src/foampilot/solver/marine_case.py")


def test_propeller_case_assembly_is_structurally_complete(tmp_path):
    (tmp_path / "system").mkdir()
    (tmp_path / "constant").mkdir()
    (tmp_path / "system" / "controlDict").write_text(
        "application marineFoam;\nsolver incompressibleFluid;\n", encoding="utf-8"
    )
    for name in ("fvSchemes", "fvSolution"):
        (tmp_path / "system" / name).touch()
    (tmp_path / "constant" / "g").touch()
    (tmp_path / "constant" / "marineProperties").write_text(
        "mode propeller_mrf;\n", encoding="utf-8"
    )
    (tmp_path / "constant" / "MRFProperties").write_text("cellZone rotor;\n", encoding="utf-8")
    actuation.write_actuation_disk(
        tmp_path,
        actuation.actuation_disk_from_propeller(
            cell_zone="rotor",
            diameter=0.25,
            disk_dir=(1, 0, 0),
            cp=0.1,
            ct=0.5,
            upstream_point=(0, 0, 0),
        ),
    )
    config = case_module.MarineCaseConfig.from_case(tmp_path)
    config.validate_files()
    assert (tmp_path / "constant" / "fvModels").is_file()
    assert "actuationDisk" in (tmp_path / "constant" / "fvModels").read_text()


def test_force_reference_file_contains_finite_loads(tmp_path):
    output = forces.write_force_model(
        tmp_path,
        propeller=forces.PropellerForceModel(1000, 0.2, 600, 0.1, 0.02),
        rudder=forces.RudderForceModel(1000, 0.04, 0.8, 2.0, 15, 1.5),
    )
    text = output.read_text()
    assert "thrust " in text and "torque " in text
    assert "sideForce " in text and "yawMoment " in text
    assert "nan" not in text.lower()
