from pathlib import Path

import pytest

from foampilot.fsi import FSIConfigurationError, NativeRigidFSI, RigidBody


def test_rigid_body_rejects_non_positive_mass():
    with pytest.raises(FSIConfigurationError):
        RigidBody(mass=0)


def test_foundation13_dictionary_contains_native_motion_and_body():
    config = NativeRigidFSI(
        "/tmp/fsi",
        body=RigidBody(name="plate", patch="plate", mass=12.5),
        variant="foundation13",
    )
    content = config.dynamic_mesh_dict()
    assert "mover" in content
    assert "rigidBodyMotion" in content
    assert "patches         (plate)" in content
    assert "mass            12.5" in content
    assert "librigidBodyMeshMotion.so" in content


def test_legacy_dictionary_contains_six_dof_coeffs():
    config = NativeRigidFSI(
        "/tmp/fsi",
        body=RigidBody(name="flap", patch="flap", mass=2.0),
        variant="legacy",
    )
    content = config.dynamic_mesh_dict()
    assert "dynamicMotionSolverFvMesh" in content
    assert "sixDoFRigidBodyMotionCoeffs" in content
    assert "sixDoFRigidBodyMotion" in content


def test_write_creates_dictionaries(tmp_path: Path):
    config = NativeRigidFSI(
        tmp_path,
        body=RigidBody(name="body", patch="fsiWall"),
    )
    paths = config.write()
    assert paths["dynamicMeshDict"].is_file()
    assert paths["forcesFunctionObject"].is_file()
    assert "patches         (fsiWall)" in paths["forcesFunctionObject"].read_text()


def test_restraint_is_rendered():
    config = NativeRigidFSI(
        "/tmp/fsi",
        restraints={
            "spring": {
                "type": "linearSpring",
                "body": "body",
                "anchor": (0.0, 0.0, 0.0),
                "stiffness": 100.0,
            }
        },
    )
    content = config.dynamic_mesh_dict()
    assert "restraints" in content
    assert "linearSpring" in content
    assert "stiffness 100.0;" in content
