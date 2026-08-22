from pathlib import Path

from foampilot.tutorials import WolfDynamicsTutorial, validate_generated_case


def _write_minimal_tutorial_source(root: Path, *, include_zero: bool = True) -> None:
    (root / "system").mkdir(parents=True)
    (root / "constant").mkdir()
    if include_zero:
        (root / "0").mkdir()
        (root / "0" / "U").write_text("internalField uniform (0 0 0);\n", encoding="utf-8")
    (root / "system" / "controlDict").write_text(
        "application foamRun;\nsolver multicomponentFluid;\nendTime 1000;\nwriteInterval 100;\n",
        encoding="utf-8",
    )
    (root / "system" / "fvSchemes").write_text("ddtSchemes { default Euler; }\n", encoding="utf-8")
    (root / "system" / "fvSolution").write_text("solvers {}\n", encoding="utf-8")
    (root / "constant" / "physicalProperties").write_text("thermoType {}\n", encoding="utf-8")


def test_wolfdynamics_tutorial_setup_copies_files(tmp_path):
    source = tmp_path / "source"
    _write_minimal_tutorial_source(source)

    target = tmp_path / "target"
    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=target,
        foamrun_module="multicomponentFluid",
        compressible=True,
    )

    tutorial.setup_case()
    assert target.exists()
    assert (target / "system" / "controlDict").exists()


def test_wolfdynamics_tutorial_fails_validation_if_initial_fields_missing(tmp_path):
    source = tmp_path / "source"
    _write_minimal_tutorial_source(source, include_zero=False)

    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=tmp_path / "target",
        foamrun_module="incompressibleFluid",
        compressible=False,
    )

    tutorial.setup_case()
    tutorial.write_case()

    validation = validate_generated_case(tutorial.case_path, compressible=tutorial.compressible)
    assert not validation.valid
    assert "0" in validation.missing_files


def test_wolfdynamics_tutorial_applies_smoke_test_controls(tmp_path):
    source = tmp_path / "source"
    _write_minimal_tutorial_source(source)

    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=tmp_path / "target",
        foamrun_module="multicomponentFluid",
        compressible=True,
        end_time=20,
        write_interval=10,
    )

    tutorial.setup_case()
    tutorial.write_case()
    control = (tutorial.case_path / "system" / "controlDict").read_text(
        encoding="utf-8"
    )
    assert "application     foamRun;" in control or "application foamRun;" in control
    assert "solver          multicomponentFluid;" in control or "solver multicomponentFluid;" in control
    assert "endTime 20;" in control
    assert "writeInterval 10;" in control
    assert (tutorial.case_path / "foampilot-input-manifest.json").exists()


def test_wolfdynamics_tutorial_setup_is_idempotent(tmp_path):
    source = tmp_path / "source"
    _write_minimal_tutorial_source(source)
    (source / "sentinel.txt").write_text("source", encoding="utf-8")

    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=tmp_path / "target",
        foamrun_module="multicomponentFluid",
        compressible=True,
    )
    tutorial.setup_case()
    sentinel = tutorial.case_path / "sentinel.txt"
    sentinel.write_text("generated", encoding="utf-8")

    tutorial.setup_case()
    assert sentinel.read_text(encoding="utf-8") == "generated"


def test_wolfdynamics_tutorial_manifest_records_all_input_roles(tmp_path):
    source = tmp_path / "source"
    _write_minimal_tutorial_source(source)

    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=tmp_path / "target",
        foamrun_module="multicomponentFluid",
        compressible=True,
    )
    tutorial.setup_case()
    tutorial.write_case()

    manifest = (tutorial.case_path / "foampilot-input-manifest.json").read_text(
        encoding="utf-8"
    )
    assert '"generator": "FoamPilot"' in manifest
    assert "initial_or_boundary_field" in manifest
    assert "numerical_or_run_control" in manifest
    assert "physical_or_chemistry_model" in manifest
