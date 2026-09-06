from pathlib import Path
import json

from foampilot.openfoam13 import (
    PROFILES,
    RegionSpec,
    UrbanClimateCase,
    UrbanClimateNativeCaseBuilder,
)


def _regions(profile):
    regions = [RegionSpec("air", "fluid", temperature=300.0, velocity=(1.0, 0.0, 0.0))]
    if profile.ham:
        regions.extend((RegionSpec("ground", "solid"), RegionSpec("buildings", "solid")))
    if profile.vegetation:
        regions.append(RegionSpec("vegetation", "vegetation"))
    return tuple(regions)


def test_six_profiles_are_public_and_native():
    assert len(PROFILES) == 6
    assert not (Path(__file__).parents[3] / "examples" / "urbanclimate" / "templates").exists()


def test_profile_generation_uses_native_foampilot_api(tmp_path: Path):
    for name, profile in PROFILES.items():
        destination = tmp_path / name
        case = UrbanClimateCase.from_name(name)
        result = case.write_case(destination)
        assert result == destination
        assert not case.validate(destination)
        assert (destination / "system/blockMeshDict").exists()
        assert (destination / "0/air/U").exists()
        manifest = json.loads((destination / "foampilotUrbanClimate.json").read_text())
        assert manifest["profile"] == name
        assert manifest["openfoam"]["version"] == 13


def test_native_builder_writes_all_profile_roots(tmp_path: Path):
    for name, profile in PROFILES.items():
        destination = tmp_path / name
        builder = UrbanClimateNativeCaseBuilder(
            destination,
            _regions(profile),
            profile=name,
            ham=profile.ham,
            vegetation=profile.vegetation,
            radiation=profile.radiation,
        )
        builder.write_case()
        assert all((destination / path).is_dir() for path in ("0", "constant", "system"))
        assert (destination / "Allrun").stat().st_mode & 0o111


def test_generation_is_non_destructive_by_default(tmp_path: Path):
    case = UrbanClimateCase.from_name("streetCanyon_CFD")
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
