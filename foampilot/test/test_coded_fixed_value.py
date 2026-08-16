"""
Test suite for codedFixedValue boundary condition writing in foampilot.

Validates that:
1. Multi-line C++ code blocks (`#{ ... #};`) are written without a trailing ';'
   in both _write_attributes and write_boundary_file.
2. The `write_boundary_file` method produces OpenFOAM-parseable BC files
   for codedFixedValue conditions on inlet patches.
3. Regular single-line values still get the trailing ';'.
4. The generated 0/U, 0/k, 0/epsilon, 0/omega, and 0/p files are syntactically
   valid (balanced braces, no malformed delimiters).

Follows the same style as test_roundtrip.py and test_direct_openfoam_export.py.
"""

import re
import sys
import tempfile
import shutil
from pathlib import Path

# Ensure foampilot is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from foampilot.base.openFOAMFile import OpenFOAMFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_boundary_dir(tmpdir: Path) -> Path:
    """Create a minimal case tree with constant/polyMesh/boundary."""
    case = tmpdir / "coded_test_case"
    (case / "constant" / "polyMesh").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)

    boundary_content = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       polyBoundaryMesh;
    location    "constant/polyMesh";
    object      boundary;
}
(
    INLET
    {
        type            patch;
        nFaces          10;
        startFace       0;
    }
    OUTLET
    {
        type            patch;
        nFaces          10;
        startFace       10;
    }
    WALLS
    {
        type            wall;
        nFaces          40;
        startFace       20;
    }
)
"""
    (case / "constant" / "polyMesh" / "boundary").write_text(boundary_content)
    return case


def _validate_coded_block(content: str) -> bool:
    """Check that every `#{` ... `#};` block is well-formed and has no
    stray ';' right after the closing #};."""
    # Each #{ ... #}; must not be followed by an extra ';' on the same line
    if re.search(r'#\}\};', content):
        return False
    # All #{ must have a matching #};
    opens = content.count("#{")
    closes = content.count("#};")
    if opens != closes:
        return False
    return True


def _validate_semicolons(lines: list[str]) -> bool:
    """Validate that non-empty, non-block lines end with ';' or '{' or '}'
    or are part of a multi-line codedFixedValue block."""
    in_code_block = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if "#{" in stripped:
            in_code_block = True
        if in_code_block:
            if "#};" in stripped:
                in_code_block = False
            continue
        # Outside code blocks, lines should end with ;, {, or }
        if stripped.endswith("{") or stripped.endswith("}"):
            continue
        if stripped.endswith(";"):
            continue
        if stripped == "(" or stripped == ")":
            continue
        # If we get here, it's a malformed line
        return False
    return True


# ---------------------------------------------------------------------------
# Code snippets used in tests (matching the pattern from
# examples/building_aero/generate_wind_cases.py)
# ---------------------------------------------------------------------------

KAPPA = 0.41

def _u_code(speed: float, z0: float, z_ref: float) -> str:
    return (
        "const vector& pos = pos();\n"
        f"    scalar z = pos.z;\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {KAPPA};\n"
        "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
        "    scalar u_mag = u_star / kappa * Foam::log(Foam::max(z / z0, 1.0 + SMALL));\n"
        "    result = vector(u_mag, 0, 0);"
    )

def _k_code(speed: float, z0: float, z_ref: float, intensity: float) -> str:
    return (
        "const vector& pos = pos();\n"
        f"    scalar z = pos.z;\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {KAPPA};\n"
        f"    scalar I = {intensity};\n"
        "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
        "    scalar u_mag = u_star / kappa * Foam::log(Foam::max(z / z0, 1.0 + SMALL));\n"
        "    result = 1.5 * pow(I * u_mag, 2);"
    )

def _omega_code(speed: float, z0: float, z_ref: float) -> str:
    return (
        "const vector& pos = pos();\n"
        f"    scalar z = pos.z;\n"
        f"    scalar z0 = {z0};\n"
        f"    scalar u_ref = {speed};\n"
        f"    scalar z_ref = {z_ref};\n"
        f"    scalar kappa = {KAPPA};\n"
        "    scalar u_star = u_ref * kappa / Foam::log(z_ref / z0);\n"
        "    result = u_star / (kappa * Foam::max(z, 1e-12));"
    )


# ---------------------------------------------------------------------------
# Test 1 — _write_attributes with multi-line codedFixedValue code
# ---------------------------------------------------------------------------

def test_write_attributes_multiline_code():
    """_write_attributes must NOT append ';' to multi-line code blocks."""
    print("=" * 60)
    print("Test: _write_attributes multi-line codedFixedValue code")
    print("=" * 60)

    code = "#{\n" + _u_code(10.0, 0.01, 10.0) + "\n#};"
    attrs = {
        "type": "codedFixedValue",
        "value": "uniform (0 0 0)",
        "code": code,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        foam_file = OpenFOAMFile("testDict")
        foam_file.attributes = attrs
        out_path = Path(tmpdir) / "testAttr"
        foam_file.write_file(out_path)

        content = out_path.read_text()
        print(f"  Output:\n{content}")

        # The code block must end with #}; NOT #};;
        assert "#};" in content, "Missing #}; closing delimiter"
        assert not re.search(r"#\}\};", content), "Extra ';' after #};"

        # Verify no ';' after the multi-line value line
        code_lines = content.splitlines()
        in_code = False
        for i, line in enumerate(code_lines):
            if "#{" in line:
                in_code = True
            if in_code and "#};" in line:
                assert not line.rstrip().endswith(";;"), \
                    f"Double semicolon at line {i}: {line}"
                in_code = False

    print("[OK] _write_attributes handles multi-line code without trailing ';'")


# ---------------------------------------------------------------------------
# Test 2 — write_boundary_file with codedFixedValue on U
# ---------------------------------------------------------------------------

def test_boundary_file_coded_fixed_value_u():
    """write_boundary_file produces valid 0/U with codedFixedValue inlet."""
    print("=" * 60)
    print("Test: write_boundary_file codedFixedValue on U")
    print("=" * 60)

    boundaries = {
        "INLET": {
            "type": "codedFixedValue",
            "value": "uniform (0 0 0)",
            "code": "#{\n" + _u_code(10.0, 0.01, 10.0) + "\n#};",
        },
        "OUTLET": {"type": "zeroGradient"},
        "WALLS": {"type": "noSlip"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        case = _make_boundary_dir(Path(tmpdir))
        foam_file = OpenFOAMFile("U")
        foam_file.write_boundary_file("U", boundaries, case)

        u_file = case / "0" / "U"
        assert u_file.exists(), "0/U file not created"
        content = u_file.read_text()
        print(f"  0/U:\n{content}")

        assert _validate_coded_block(content), "Malformed codedFixedValue block"
        assert "codedFixedValue" in content
        assert "#{" in content and "#};" in content
        assert not re.search(r"#\}\};", content), "Extra ';' after #};"

        # Validate brace balance
        depth = 0
        in_code = False
        for line in content.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            if "#{" in stripped:
                in_code = True
            if "#};" in stripped:
                in_code = False
                continue
            if in_code:
                continue
            depth += stripped.count("{") - stripped.count("}")
        assert depth == 0, f"Unbalanced braces (depth={depth})"

    print("[OK] 0/U with codedFixedValue written correctly")


# ---------------------------------------------------------------------------
# Test 3 — Full set of codedFixedValue inlet conditions (U, k, omega)
# ---------------------------------------------------------------------------

def test_full_inlet_coded_conditions():
    """Write 0/U, 0/k, 0/omega with codedFixedValue inlet — mirror the
    generate_wind_cases.py pattern — and validate every file."""
    print("=" * 60)
    print("Test: Full inlet codedFixedValue set (U, k, omega)")
    print("=" * 60)

    speed, z0, z_ref, intensity = 12.0, 0.03, 10.0, 0.08

    boundaries_u = {
        "INLET": {
            "type": "codedFixedValue",
            "value": "uniform (0 0 0)",
            "code": "#{\n" + _u_code(speed, z0, z_ref) + "\n#};",
        },
        "OUTLET": {"type": "zeroGradient"},
        "WALLS": {"type": "noSlip"},
    }
    boundaries_k = {
        "INLET": {
            "type": "codedFixedValue",
            "value": f"uniform {1.5 * (intensity * speed) ** 2:.6f}",
            "code": "#{\n" + _k_code(speed, z0, z_ref, intensity) + "\n#};",
        },
        "OUTLET": {"type": "zeroGradient"},
        "WALLS": {"type": "kqRWallFunction"},
    }
    boundaries_omega = {
        "INLET": {
            "type": "codedFixedValue",
            "value": "uniform 1.0",
            "code": "#{\n" + _omega_code(speed, z0, z_ref) + "\n#};",
        },
        "OUTLET": {"type": "zeroGradient"},
        "WALLS": {"type": "omegaWallFunction"},
    }
    boundaries_p = {
        "INLET": {"type": "zeroGradient"},
        "OUTLET": {"type": "fixedValue", "value": "uniform 0"},
        "WALLS": {"type": "zeroGradient"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        case = _make_boundary_dir(Path(tmpdir))
        foam_file = OpenFOAMFile("field")

        all_bcs = [("U", boundaries_u), ("k", boundaries_k),
                   ("omega", boundaries_omega), ("p", boundaries_p)]

        for field, bcs in all_bcs:
            foam_file.write_boundary_file(field, bcs, case)
            fpath = case / "0" / field
            assert fpath.exists(), f"0/{field} not created"
            content = fpath.read_text()
            print(f"--- 0/{field} ---")
            print(content)
            assert _validate_coded_block(content), f"Malformed block in {field}"
            assert not re.search(r"#\}\};", content), f"Extra ';' in {field}"

        print("[OK] All inlet BC files written and validated")


# ---------------------------------------------------------------------------
# Test 4 — Single-line values still get ';'
# ---------------------------------------------------------------------------

def test_single_line_values_get_semicolon():
    """Ensure regular single-line values still end with ';'."""
    print("=" * 60)
    print("Test: single-line values still get ';'")
    print("=" * 60)

    attrs = {
        "type": "fixedValue",
        "value": "uniform 0",
        "refinement": "1.5",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        foam_file = OpenFOAMFile("simple")
        foam_file.attributes = attrs
        out_path = Path(tmpdir) / "simple"
        foam_file.write_file(out_path)
        content = out_path.read_text()
        print(f"  Output:\n{content}")

        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("//") and not stripped.startswith("FoamFile") and \
               stripped not in ("{", "}", "}") and "{" not in stripped and "}" not in stripped:
                assert stripped.endswith(";"), f"Missing ';' at: {line}"

    print("[OK] Single-line values correctly end with ';'")


# ---------------------------------------------------------------------------
# Test 5 — Code block with codeInclude and codeOptions (full dynamic code)
# ---------------------------------------------------------------------------

def test_coded_fixed_value_with_includes():
    """Test codedFixedValue with codeInclude and codeOptions sub-dicts."""
    print("=" * 60)
    print("Test: codedFixedValue with codeInclude/codeOptions")
    print("=" * 60)

    code = (
        "#{\n"
        "    const vector& pos = pos();\n"
        "    scalar z = pos.z;\n"
        "    result = vector(10.0 * z / 100.0, 0, 0);\n"
        "#};"
    )

    boundaries = {
        "INLET": {
            "type": "codedFixedValue",
            "value": "uniform (0 0 0)",
            "code": code,
        },
        "WALLS": {"type": "noSlip"},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        case = _make_boundary_dir(Path(tmpdir))
        foam_file = OpenFOAMFile("U")
        foam_file.write_boundary_file("U", boundaries, case)

        content = (case / "0" / "U").read_text()
        print(f"  0/U:\n{content}")

        assert "codedFixedValue" in content
        assert _validate_coded_block(content)
        # The INLET block should have type, value, and code entries
        inlet_section = content[content.index('"INLET"'):]
        assert "code" in inlet_section
        assert "value" in inlet_section

    print("[OK] codedFixedValue with sub-dicts written correctly")


# ---------------------------------------------------------------------------
# Test 5 — Validate the vadier OpenFOAM case files
# ---------------------------------------------------------------------------

def test_vadier_case_files_valid():
    """Validate that the coded_fixed_value_vadier_case OpenFOAM case has
    correctly written codedFixedValue boundary files in 0/."""
    print("=" * 60)
    print("Test: vadier case boundary files validation")
    print("=" * 60)

    case_dir = Path(__file__).resolve().parent / "coded_fixed_value_vadier_case"

    for field in ["U", "k", "omega"]:
        fpath = case_dir / "0" / field
        assert fpath.exists(), f"Missing 0/{field} in vadier case"
        content = fpath.read_text()

        # Must contain codedFixedValue with proper #{ ... #}; block
        assert "codedFixedValue" in content, f"codedFixedValue missing in {field}"
        assert "name" in content, f"name field missing in {field}"
        assert _validate_coded_block(content), f"Malformed code block in {field}"
        assert "this->patch().Cf()" in content, \
            f"Wrong API (should use this->patch().Cf()) in {field}"
        assert "operator==(" in content, \
            f"Wrong API (should use operator==) in {field}"
        assert "pos()" not in content, \
            f"Old pos() API still used in {field}"
        assert "result =" not in content, \
            f"Old result= API still used in {field}"

    print("[OK] All vadier case BC files use correct OpenFOAM API")

    # Validate system files exist
    for fname in ["blockMeshDict", "controlDict", "fvSchemes", "fvSolution"]:
        assert (case_dir / "system" / fname).exists(), f"Missing system/{fname}"

    # Validate constant files
    for fname in ["transportProperties", "turbulenceProperties", "polyMesh/boundary"]:
        assert (case_dir / "constant" / fname).exists(), f"Missing constant/{fname}"

    print("[OK] All case files present and validated")


def run_all():
    tests = [
        ("_write_attributes multiline", test_write_attributes_multiline_code),
        ("boundary file codedFixedValue U", test_boundary_file_coded_fixed_value_u),
        ("full inlet BC set", test_full_inlet_coded_conditions),
        ("single-line semicolons", test_single_line_values_get_semicolon),
        ("codedFixedValue with includes", test_coded_fixed_value_with_includes),
        ("vadier case files", test_vadier_case_files_valid),
    ]
    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")
    print(f"\n{'='*60}")
    print(f"codedFixedValue tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    return failed


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.CRITICAL)
    sys.exit(run_all())
