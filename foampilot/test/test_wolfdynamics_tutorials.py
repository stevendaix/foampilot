import pytest
from pathlib import Path

from foampilot.tutorials import WolfDynamicsTutorial, validate_generated_case


def test_wolfdynamics_tutorial_setup_copies_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "system").mkdir()
    (source / "system" / "controlDict").write_text("application foamRun;\n", encoding="utf-8")
    
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


def test_wolfdynamics_tutorial_fails_validation_if_incomplete(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "system").mkdir()
    (source / "system" / "controlDict").write_text("application foamRun;\n", encoding="utf-8")
    
    target = tmp_path / "target"
    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=target,
        foamrun_module="incompressibleFluid",
        compressible=False,
    )
    
    tutorial.setup_case()
    # Write case does nothing by default, so the missing files (0/, constant/, nu) remain missing.
    tutorial.write_case()
    
    validation = validate_generated_case(tutorial.case_path, compressible=tutorial.compressible)
    assert not validation.valid
    assert "constant" in validation.missing_files
    assert "0" in validation.missing_files
    assert any("nu" in w for w in validation.warnings)


def test_wolfdynamics_tutorial_applies_smoke_test_controls(tmp_path):
    source = tmp_path / "source"
    (source / "system").mkdir(parents=True)
    (source / "system" / "controlDict").write_text(
        "application foamRun;\nendTime 1000;\nwriteInterval 100;\n",
        encoding="utf-8",
    )

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
    assert "endTime 20;" in control
    assert "writeInterval 10;" in control


def test_wolfdynamics_tutorial_rejects_ambiguous_control_entry(tmp_path):
    source = tmp_path / "source"
    (source / "system").mkdir(parents=True)
    (source / "system" / "controlDict").write_text(
        "endTime 1;\nendTime 2;\n", encoding="utf-8"
    )

    tutorial = WolfDynamicsTutorial(
        source_case_path=source,
        target_case_path=tmp_path / "target",
        foamrun_module="multicomponentFluid",
        compressible=True,
        end_time=20,
    )

    tutorial.setup_case()
    with pytest.raises(ValueError, match="exactly one 'endTime'"):
        tutorial.write_case()


def test_wolfdynamics_tutorial_setup_is_idempotent(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
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
