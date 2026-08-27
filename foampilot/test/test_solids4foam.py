from pathlib import Path

import pytest

from foampilot.solids4foam import (
    SolidMaterial,
    Solids4FoamCase,
    Solids4FoamConfigurationError,
)


def test_material_validation():
    with pytest.raises(Solids4FoamConfigurationError):
        SolidMaterial(young_modulus=0)


def test_fsi_dictionaries_match_solids4foam_structure():
    case = Solids4FoamCase(
        "/tmp/beam",
        fluid_patch="interface",
        solid_patch="interface",
        material=SolidMaterial(
            name="rubber", law="neoHookeanElastic", density=1000, young_modulus=1e4
        ),
    )
    assert "type fluidSolidInteraction;" in case.physics_properties()
    assert "fluidSolidInterface    IQNILS;" in case.fsi_properties()
    assert "solidModel" in case.solid_properties()
    assert "neoHookeanElastic" in case.mechanical_properties()
    assert "region              fluid;" in case.functions()


def test_write_and_run_plan(tmp_path: Path):
    case = Solids4FoamCase(tmp_path)
    files = case.write()
    assert (tmp_path / "constant/physicsProperties").is_file()
    assert (tmp_path / "constant/fsiProperties").is_file()
    assert (tmp_path / "constant/solid/solidProperties").is_file()
    assert (tmp_path / "constant/solid/mechanicalProperties").is_file()
    assert "constant/fsiProperties" in files
    assert case.run_plan(parallel=False)[-1] == ["solids4Foam"]
    assert ["solids4Foam", "-parallel"] in case.run_plan(parallel=True)


def test_custom_fluid_properties_are_written(tmp_path: Path):
    case = Solids4FoamCase(tmp_path, fluid_properties="nu  nu [0 2 -1 0 0 0 0] 1e-3;")
    case.write()
    content = (tmp_path / "constant/fluid/fluidProperties").read_text()
    assert "1e-3" in content
