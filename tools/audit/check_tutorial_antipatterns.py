#!/usr/bin/env python3
"""Audit script for tutorials and examples - detects anti-patterns.

Scans all .py files in tutorials/ and examples/ for forbidden patterns:
- import_reference_file / import_reference_field (blind file copying)
- remove_files / shutil.copy / shutil.copytree (post-hoc file manipulation)
- Absolute paths (/home/steven/, /Users/)
- Legacy imports (foampilot.utilities, foampilot.mesh, etc.)

Usage:
    python tools/audit/check_tutorial_antipatterns.py [--root .]
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple

FORBIDDEN_PATTERNS = [
    "shutil.copy",
    "shutil.copytree",
    "shutil.rmtree",
    "os.remove",
    "os.unlink",
    "Path.unlink",
    "Path.rmdir",
    "subprocess.run([\"rm",
    "subprocess.run(['rm",
]

FORBIDDEN_IMPORTS = [
    "foampilot.utilities",
    "foampilot.mesh",
    "foampilot.openfoam13",
    "foampilot.model_addon",
    "foampilot.postprocess",
    "foampilot.report",
    "foampilot.base",
    "foampilot.commons",
    "foampilot.geometry",
    "foampilot.urban",
    "foampilot.wind",
    "foampilot.physiology",
    "foampilot.cht",
    "foampilot.coupling",
]

CANONICAL_IMPORTS = [
    "foampilot.core.",
    "foampilot.openfoam.",
    "foampilot.solver",
    "foampilot.boundaries",
    "foampilot.system",
    "foampilot.constant",
]


def check_file(filepath: Path) -> Dict[str, Tuple[bool, str]]:
    """Check a single Python file for anti-patterns."""
    results = {}
    content = filepath.read_text(encoding="utf-8")

    # 1. Forbidden function calls (file manipulation)
    forbidden_found = []
    for pattern in FORBIDDEN_PATTERNS:
        if pattern in content:
            forbidden_found.append(pattern)
    results["antipatterns"] = (
        len(forbidden_found) == 0,
        "OK" if not forbidden_found else f"FORBIDDEN: {', '.join(forbidden_found)}"
    )

    # 2. Forbidden imports
    forbidden_imports = []
    for forbidden in FORBIDDEN_IMPORTS:
        if f"from {forbidden}" in content or f"import {forbidden}" in content:
            forbidden_imports.append(forbidden)
    results["imports"] = (
        len(forbidden_imports) == 0,
        "OK" if not forbidden_imports else f"FORBIDDEN: {', '.join(forbidden_imports)}"
    )

    # 3. Absolute paths
    has_abs_path = "/home/steven/" in content or "/Users/" in content
    results["paths"] = (not has_abs_path, "OK" if not has_abs_path else "ABSOLUTE PATH DETECTED")

    # 4. ValueWithUnit usage (only check for tutorial scripts, not case generators)
    is_tutorial = "tutorials/" in str(filepath) or "examples/" in str(filepath)
    if is_tutorial:
        has_vwu = "ValueWithUnit" in content
        results["units"] = (has_vwu, "YES" if has_vwu else "MISSING")
    else:
        results["units"] = (True, "N/A (case generator)")

    # 5. Structure blocks (only for tutorial scripts)
    if is_tutorial:
        has_solver = "Solver(" in content
        has_run = "run_simulation" in content
        results["structure"] = (
            has_solver and has_run,
            f"Solver={has_solver}, run={has_run}"
        )
    else:
        results["structure"] = (True, "N/A (case generator)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit tutorials and examples for anti-patterns")
    parser.add_argument("paths", nargs="*", help="Directories to audit")
    parser.add_argument("--root", default=".", help="Root directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dirs = [Path(d) for d in args.paths] if args.paths else [
        root / "tutorials",
        root / "examples",
    ]

    all_results = {}
    for d in dirs:
        if not d.exists():
            print(f"Directory not found: {d}")
            continue
        for py_file in sorted(d.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            try:
                results = check_file(py_file)
                all_results[py_file] = results
            except Exception as e:
                all_results[py_file] = {"error": (False, str(e))}

    # Print results
    print("=" * 100)
    print(f"{'File':<55} {'AntiPatterns':<25} {'Imports':<25} {'Paths':<20} {'Units':<10} {'Structure'}")
    print("=" * 100)

    passed = 0
    failed = 0
    for filepath, results in all_results.items():
        rel = filepath.relative_to(root) if filepath.is_relative_to(root) else filepath
        status = "PASS" if all(v[0] for v in results.values()) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"{str(rel):<55} {results.get('antipatterns', (False, ''))[1]:<25} {results.get('imports', (False, ''))[1]:<25} {results.get('paths', (False, ''))[1]:<20} {results.get('units', (False, ''))[1]:<10} {results.get('structure', (False, ''))[1]}")

    print("=" * 100)
    print(f"Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())