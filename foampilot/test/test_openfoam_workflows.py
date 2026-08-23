from pathlib import Path

from foampilot.constant.constantDirectory import ConstantDirectory
from foampilot.system.SystemDirectory import SystemDirectory
from foampilot.workflows.marine import (
    dtc_openfoam13_workflow,
    dtc_overset_workflow,
    maneuvering_turning_workflow,
    propeller_mrf_workflow,
    write_mrf_properties,
    write_openfoam13_rigid_body_mover,
    write_overset_dynamic_mesh,
)
from foampilot.workflows.openfoam import OpenFOAMWorkflow


class _FieldsStub:
    def __init__(self):
        self.fields = {}

    def get_field_names(self):
        return list(self.fields)


class _CaseStub:
    def __init__(self, case_path: Path):
        self.case_path = case_path
        self.solver_name = "simpleFoam"
        self.energy_activated = False
        self.is_vof = False
        self.compressible = False
        self.with_gravity = False
        self.fields_manager = _FieldsStub()

    def get_turbulence_configuration(self):
        return "laminar", None


def test_additional_system_and_constant_dictionaries_are_written(tmp_path):
    case = _CaseStub(tmp_path)
    system = SystemDirectory(case)
    system.add_dict_file("fvOptions", {"limitT": {"type": "limitTemperature", "min": 101}})
    system.write()

    constant = ConstantDirectory(case)
    constant.add_dict_file("MRFProperties", {"MRF1": {"cellZone": "rotor", "omega": 314.16}})
    constant.write()

    assert "limitT" in (tmp_path / "system" / "fvOptions").read_text()
    assert "cellZone rotor;" in (tmp_path / "constant" / "MRFProperties").read_text()


def test_workflow_executes_copy_cleanup_and_command(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "source.txt").write_text("input", encoding="utf-8")
    (tmp_path / "obsolete").mkdir()
    (tmp_path / "obsolete" / "result.txt").write_text("old", encoding="utf-8")
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "U.orig").write_text("initial velocity", encoding="utf-8")

    workflow = OpenFOAMWorkflow(tmp_path, "smoke")
    workflow.add_copy("copy-input", "inputs/source.txt", "work/source.txt")
    workflow.add_remove("clear-obsolete", "obsolete")
    workflow.add_restore_initial_fields("restore-fields")
    workflow.add_command("emit", "/bin/echo", "workflow-ok")

    results = workflow.run()

    assert [result.status for result in results] == ["completed", "completed", "completed", "completed"]
    assert (tmp_path / "work" / "source.txt").read_text(encoding="utf-8") == "input"
    assert not (tmp_path / "obsolete").exists()
    assert (tmp_path / "0" / "U").read_text(encoding="utf-8") == "initial velocity"
    assert "workflow-ok" in (tmp_path / "logs" / "04_emit.log").read_text(encoding="utf-8")


def test_marine_writers_and_workflows(tmp_path):
    mrf_path = write_mrf_properties(tmp_path / "propeller", omega=100.0)
    overset_path = write_overset_dynamic_mesh(tmp_path / "dtc", joints=("Pz", "Ry"))
    mover_path = write_openfoam13_rigid_body_mover(tmp_path / "dtc13", joints=("Pz", "Ry"))

    assert "omega 100;" in mrf_path.read_text(encoding="utf-8")
    overset_text = overset_path.read_text(encoding="utf-8")
    assert "dynamicOversetFvMesh" in overset_text
    assert "rigidBodyMotion" in overset_text
    assert "joints (" in overset_text
    assert ");" in overset_text
    mover_text = mover_path.read_text(encoding="utf-8")
    assert "mover" in mover_text
    assert "librigidBodyMeshMotion.so" in mover_text
    assert "dynamicOversetFvMesh" not in mover_text

    assert "rhoSimpleFoam -parallel" in propeller_mrf_workflow(tmp_path).preview()
    assert "overInterDyMFoam -parallel" in dtc_overset_workflow(tmp_path).preview()
    openfoam13_preview = dtc_openfoam13_workflow(tmp_path).preview()
    assert "foamRun" in openfoam13_preview
    assert "copy-compatible-mesh" in openfoam13_preview
    assert "run-turning" in maneuvering_turning_workflow(tmp_path).preview()
