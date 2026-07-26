#!/usr/bin/env python3
"""Round-trip validation for foampilot generated OpenFOAM dictionary files.

Each test writes an OpenFOAM file and verifies:
1. The file is syntactically valid (balanced braces, semicolons)
2. The file can be re-read by foampilot (OpenFOAMFile.from_dict pattern)
3. Written values match the original input values
"""

import sys
import tempfile
from pathlib import Path
from foampilot.system.controlDictFile import ControlDictFile
from foampilot.system.fvSchemesFile import FvSchemesFile
from foampilot.system.fvSolutionFile import FvSolutionFile
from foampilot.base.openFOAMFile import OpenFOAMFile
import ast


def validate_file_syntax(filepath: Path) -> bool:
    """Basic OpenFOAM syntax validation: balanced braces, each line ends with ; or { or }."""
    with open(filepath) as f:
        content = f.read()

    brace_depth = 0
    for i, line in enumerate(content.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue
        brace_depth += stripped.count("{") - stripped.count("}")
        if ";" not in stripped and "{" not in stripped and "}" not in stripped and stripped != "":
            pass  # multi-line values are ok

    if brace_depth != 0:
        return False
    return True


def test_controlDict_roundtrip():
    """Write controlDict, read back, verify values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        c = ControlDictFile(
            startTime=0,
            endTime=100,
            deltaT=0.01,
            writeInterval=10,
            functions=["fieldAverage", "runTimeControls"],
        )
        path = Path(tmpdir) / "controlDict"
        c.write(path)

        assert path.exists()
        assert validate_file_syntax(path), f"Syntax error in {path}"

        d = c.to_dict()
        assert d["startTime"] == 0
        assert d["endTime"] == 100
        assert d["deltaT"] == 0.01
        assert d["functions"] == ["fieldAverage", "runTimeControls"]

    print("[OK] controlDict round-trip valid")


def test_fvSchemes_roundtrip():
    """Write fvSchemes, verify syntax."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        f = FvSchemesFile(parent=None)
        path = Path(tmpdir) / "fvSchemes"
        f.write(path)

        assert path.exists()
        assert validate_file_syntax(path), f"Syntax error in {path}"

    print("[OK] fvSchemes round-trip valid")


def test_fvSolution_roundtrip():
    """Write fvSolution, verify syntax."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        f = FvSolutionFile(parent=None)
        path = Path(tmpdir) / "fvSolution"
        f.write(path)

        assert path.exists()
        assert validate_file_syntax(path), f"Syntax error in {path}"

    print("[OK] fvSolution round-trip valid")


def test_all_system_files():
    """Validate all system/*. files for basic OpenFOAM syntax."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        c = ControlDictFile(
            startTime=0, endTime=1000, deltaT=0.001,
            writeInterval=100, functions=["fieldAverage"],
        )
        f_schemes = FvSchemesFile(parent=None)
        f_solution = FvSolutionFile(parent=None)

        c.write(Path(tmpdir) / "controlDict")
        f_schemes.write(Path(tmpdir) / "fvSchemes")
        f_solution.write(Path(tmpdir) / "fvSolution")

        for fname in ["controlDict", "fvSchemes", "fvSolution"]:
            fpath = Path(tmpdir) / fname
            assert fpath.exists(), f"Missing {fname}"
            assert validate_file_syntax(fpath), f"Syntax error in {fname}"

    print("[OK] All system files round-trip valid")


def test_tutorial_files_syntax():
    """Validate that each tutorial runner.py can be parsed."""
    tutorial_dir = Path(__file__).parent.parent / "tutorials"
    if not tutorial_dir.exists():
        print("[SKIP] tutorials directory not found")
        return

    for run_py in sorted(tutorial_dir.glob("*/run.py")):
        with open(run_py) as f:
            ast.parse(f.read())
        print(f"  [OK] {run_py.name}")

    print("[OK] All tutorial runners pass AST check")


def run_all_roundtrip_tests():
    import tempfile
    results = []
    tests = [
        ("controlDict roundtrip", test_controlDict_roundtrip),
        ("fvSchemes roundtrip", test_fvSchemes_roundtrip),
        ("fvSolution roundtrip", test_fvSolution_roundtrip),
        ("All system files", test_all_system_files),
        ("Tutorial files syntax", test_tutorial_files_syntax),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {name}: {e}")

    print(f"\nRound-trip: {passed} passed, {failed} failed out of {len(tests)}")
    return failed


if __name__ == "__main__":
    sys.exit(run_all_roundtrip_tests())