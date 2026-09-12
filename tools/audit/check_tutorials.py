#!/usr/bin/env python3
"""Audit script for tutorials and examples.

Checks:
1. Canonical imports (no foampilot.utilities, foampilot.mesh, etc.)
2. ValueWithUnit usage for physical quantities
3. Relative paths (no absolute /home/steven/... paths)
4. Structure blocks (Solver, run_simulation, FoamPostProcessing)
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Tuple

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
    results = {}
    content = filepath.read_text(encoding="utf-8")
    
    # 1. Forbidden imports
    forbidden_found = []
    for forbidden in FORBIDDEN_IMPORTS:
        if f"from {forbidden}" in content or f"import {forbidden}" in content:
            forbidden_found.append(forbidden)
    results["imports"] = (
        len(forbidden_found) == 0,
        f"OK" if not forbidden_found else f"FORBIDDEN: {', '.join(forbidden_found)}"
    )
    
    # 2. ValueWithUnit usage
    has_vwu = "ValueWithUnit" in content
    results["units"] = (has_vwu, "YES" if has_vwu else "MISSING")
    
    # 3. Relative paths
    has_abs_path = "/home/steven/" in content or "/Users/" in content
    results["paths"] = (not has_abs_path, "OK" if not has_abs_path else "ABSOLUTE PATH DETECTED")
    
    # 4. Structure blocks
    has_solver = "Solver(" in content
    has_run = "run_simulation" in content
    has_post = "FoamPostProcessing" in content
    results["structure"] = (
        has_solver and has_run,
        f"Solver={has_solver}, run={has_run}, post={has_post}"
    )
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit tutorials and examples")
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
    print("=" * 80)
    print(f"{'File':<60} {'Imports':<15} {'Units':<10} {'Paths':<15} {'Structure'}")
    print("=" * 80)
    
    passed = 0
    failed = 0
    for filepath, results in all_results.items():
        rel = filepath.relative_to(root) if filepath.is_relative_to(root) else filepath
        status = "PASS" if all(v[0] for v in results.values()) else "FAIL"
        if status == "PASS":
            passed += 1
        else:
            failed += 1
        print(f"{str(rel):<60} {results.get('imports', (False, ''))[1]:<15} {results.get('units', (False, ''))[1]:<10} {results.get('paths', (False, ''))[1]:<15} {results.get('structure', (False, ''))[1]}")
    
    print("=" * 80)
    print(f"Passed: {passed}, Failed: {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
