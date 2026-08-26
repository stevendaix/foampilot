from pathlib import Path
import json

from foampilot.openfoam13 import PROFILES, UrbanClimateCase


ROOT = Path(__file__).parents[3]
TEMPLATES = ROOT / "examples" / "urbanclimate" / "templates"


def test_six_profiles_are_public_and_have_templates():
    assert len(PROFILES) == 6
    for name in PROFILES:
        assert (TEMPLATES / name).is_dir()
        assert (TEMPLATES / name / "0").is_dir()
        assert (TEMPLATES / name / "constant").is_dir()
        assert (TEMPLATES / name / "system").is_dir()


def test_profile_generation_uses_foampilot_api(tmp_path: Path):
    for name in PROFILES:
        destination = tmp_path / name
        case = UrbanClimateCase.from_name(name, TEMPLATES)
        result = case.write_case(destination)
        assert result == destination
        assert not case.validate(destination)
        assert (destination / "foampilotPhysics.json").exists()
        assert (destination / "foampilotUrbanClimate.json").exists()
        manifest = json.loads((destination / "foampilotUrbanClimate.json").read_text())
        assert manifest["profile"] == name
        assert manifest["openfoam"]["version"] == 13


def test_generation_is_non_destructive_by_default(tmp_path: Path):
    case = UrbanClimateCase.from_name("streetCanyon_CFD", TEMPLATES)
    destination = case.write_case(tmp_path / "case")
    marker = destination / "user.marker"
    marker.write_text("keep")
    try:
        case.write_case(destination)
    except FileExistsError:
        pass
    else:
        raise AssertionError("write_case must refuse accidental overwrite")
    assert marker.read_text() == "keep"
