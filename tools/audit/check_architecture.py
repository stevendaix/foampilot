#!/usr/bin/env python3
"""Static dependency guard for the FoamPilot v3 migration.

The checker is intentionally conservative: it reports only path-level rules
that can be evaluated without importing optional scientific dependencies.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path


def layer(path: Path) -> str:
    text = "/".join(path.parts)
    if "workflows" in path.parts:
        return "workflow"
    if "examples" in path.parts or "tutorial" in text.lower():
        return "example"
    if "validation" in path.parts:
        return "validation"
    if "extensions" in path.parts or "third_party" in path.parts:
        return "extension"
    if "core" in path.parts or "src" in path.parts:
        return "core"
    return "other"


def imported_text(node: ast.AST) -> str:
    if isinstance(node, ast.Import):
        return " ".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module or ""
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    violations = []
    for path in root.rglob("*.py"):
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        source_layer = layer(path.relative_to(root))
        for node in ast.walk(tree):
            target = imported_text(node)
            if not target:
                continue
            forbidden = (
                source_layer == "core" and any(x in target for x in ("examples", "tutorial", "validation", "workflows"))
            )
            if forbidden:
                violations.append({"file": str(path.relative_to(root)), "import": target})
    for item in violations:
        print(f"ARCHITECTURE VIOLATION: {item['file']} imports {item['import']}")
    print(f"violations={len(violations)}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
