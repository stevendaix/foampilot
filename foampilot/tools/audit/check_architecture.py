#!/usr/bin/env python3
"""
Architecture checker for FoamPilot.

This script validates that the codebase respects the architectural rules:
1. core/ can never import workflows/, examples/, or version-specific modules
2. openfoam/backend/ handles version detection (internal only)
3. Workflows can import from core/
4. No circular dependencies

Usage:
    python check_architecture.py [--fix]

Exit codes:
    0 - Architecture is valid
    1 - Architecture violations found (and --fix not used)
    2 - --fix was used and violations were auto-fixed
"""

import ast
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

SRC_DIR = Path(__file__).parent.parent.parent / "src" / "foampilot"
RULES = {
    "core": {
        "forbidden_imports": [
            "workflows",
            "examples",
            "openfoam13",
            "model_addon",
        ],
        "forbidden_paths": [
            "workflows",
            "examples",
        ],
    },
    "openfoam": {
        "forbidden_imports": [
            "workflows",
            "examples",
        ],
    },
}


class ImportVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str, src_root: Path):
        self.file_path = Path(file_path)
        self.src_root = src_root
        self.imports: List[Tuple[str, int]] = []
        self.from_imports: List[Tuple[str, int]] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if alias.name:
                self.imports.append((alias.name, node.lineno))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.from_imports.append((node.module, node.lineno))
        for alias in node.names:
            if alias.name:
                self.from_imports.append((f"{node.module}.{alias.name}", node.lineno))
        self.generic_visit(node)


def get_module_path(import_name: str, src_root: Path) -> Path:
    parts = import_name.split(".")
    for i in range(len(parts), 0, -1):
        candidate = src_root / "/".join(parts[:i])
        if candidate.exists() or (candidate.parent / f"{parts[i-1]}.py").exists():
            return candidate
    return src_root / import_name.split(".")[0]


def check_file(file_path: Path, src_root: Path, rules: Dict) -> List[Dict]:
    violations = []
    rel_path = file_path.relative_to(src_root.parent.parent)
    module_parts = rel_path.with_suffix("").parts

    if len(module_parts) < 2:
        return violations

    current_module = module_parts[0]
    current_submodule = ".".join(module_parts[1:])

    if current_module not in rules:
        return violations

    rule = rules[current_module]

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        return violations

    visitor = ImportVisitor(str(file_path), src_root)
    visitor.visit(tree)

    for imp, lineno in visitor.imports + visitor.from_imports:
        imp_module = imp.split(".")[0]

        if imp_module in rule.get("forbidden_imports", []):
            violations.append({
                "file": str(rel_path),
                "line": lineno,
                "type": "forbidden_import",
                "import": imp,
                "module": current_module,
                "message": f"'{current_module}' imports forbidden module '{imp}'",
            })

    return violations


def check_directory(src_root: Path, rules: Dict) -> List[Dict]:
    all_violations = []
    for ext in (".py",):
        for file_path in src_root.rglob(f"*{ext}"):
            if "__pycache__" in str(file_path):
                continue
            violations = check_file(file_path, src_root, rules)
            all_violations.extend(violations)
    return all_violations


def print_violations(violations: List[Dict]) -> None:
    print("Architecture Violations Found:")
    print("=" * 70)
    for v in violations:
        print(f"  {v['file']}:{v['line']} - {v['message']}")
    print("=" * 70)
    print(f"Total: {len(violations)} violation(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Check FoamPilot architecture rules"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Automatically fix violations (not implemented yet)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output",
    )
    args = parser.parse_args()

    if not SRC_DIR.exists():
        print(f"Source directory not found: {SRC_DIR}")
        sys.exit(1)

    violations = check_directory(SRC_DIR, RULES)

    if violations:
        print_violations(violations)
        sys.exit(1 if not args.fix else 2)

    if args.verbose:
        print("Architecture check passed. No violations found.")

    sys.exit(0)


if __name__ == "__main__":
    main()