from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest


_MODULE_PATH = Path(__file__).parents[2] / "src" / "foampilot" / "wind" / "floating_turbine.py"
_SPEC = spec_from_file_location("foampilot_wind_floating_turbine", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
FloatingTurbine = _MODULE.FloatingTurbine
MooringLine = _MODULE.MooringLine


def test_floating_turbine_rejects_non_unit_axes(tmp_path: Path):
    turbine = FloatingTurbine(rotor_axis=(2.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="unit vector"):
        turbine.write(tmp_path)


def test_fv_models_contains_openfoam13_actuator_line_configuration():
    turbine = FloatingTurbine(coupling=True)
    text = turbine.render_fv_models(cell_zone="rotorZone")
    assert "object      fvModels;" in text
    legacy = turbine.render_legacy_fv_options(cell_zone="rotorZone")
    assert "object      fvOptions;" in legacy
    assert "type                axialFlowTurbineALSource;" in text
    assert "cellZone            rotorZone;" in text
    assert "coupleLoads         true;" in text
    assert "rotorAxis           (1 0 0);" in text


def test_dynamic_mesh_contains_mooring_and_constraints():
    line = MooringLine(
        name="line1",
        anchor=(-100.0, 0.0, -200.0),
        attachment_point=(-20.0, 0.0, -14.0),
        mass_per_length=108.63,
        line_length=865.5,
    )
    turbine = FloatingTurbine(mooring_lines=(line,))
    text = turbine.render_dynamic_mesh(
        mass=1000.0,
        moment_of_inertia=(10.0, 20.0, 30.0),
        constraints={"surgeOnly": {"type": "line", "direction": (1.0, 0.0, 0.0)}},
    )
    assert "solver sixDoFRigidBodyMotion;" in text
    assert "sixDoFRigidBodyMotionRestraint mooringLine;" in text
    assert "lineLength          865.5;" in text
    assert "sixDoFRigidBodyMotionConstraint line;" in text


def test_configure_solver_composes_runtime_libraries():
    class ControlDict:
        def __init__(self):
            self.libs = []

        def add_library(self, library):
            if library not in self.libs:
                self.libs.append(library)

    class Solver:
        def __init__(self):
            self.system = type("System", (), {"controlDict": ControlDict()})()
            self.transient = False

    solver = Solver()
    FloatingTurbine(mooring_lines=(MooringLine("line1", (0, 0, -1), (0, 0, 0), 1, 2),)).configure_solver(solver)
    assert solver.transient is True
    assert solver.system.controlDict.libs == [
        "libturbinesFoam.so",
        "libfloatingSixDoFRigidBodyMotion.so",
    ]


def test_write_emits_physics_files(tmp_path: Path):
    line = MooringLine(
        name="line1",
        anchor=(-100.0, 0.0, -200.0),
        attachment_point=(-20.0, 0.0, -14.0),
        mass_per_length=100.0,
        line_length=500.0,
    )
    paths = FloatingTurbine(mooring_lines=(line,)).write(
        tmp_path, mass=1000.0, moment_of_inertia=(1.0, 2.0, 3.0)
    )
    assert set(paths) == {"fvModels", "dynamicMeshDict"}
    assert all(path.exists() for path in paths.values())
    assert "nu" not in paths["fvModels"].read_text(encoding="utf-8")


def test_mooring_line_requires_positive_physics():
    with pytest.raises(ValueError, match="positive"):
        MooringLine(
            name="line1",
            anchor=(0, 0, 0),
            attachment_point=(0, 0, 0),
            mass_per_length=0,
            line_length=1,
        )
