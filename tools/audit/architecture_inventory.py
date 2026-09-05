#!/usr/bin/env python3
"""Create a path-based architecture inventory for the FoamPilot v3 migration."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def classify(path: Path) -> str:
    parts = path.parts
    if "third_party" in parts:
        return "third_party"
    if "validation" in parts:
        return "validation"
    if "tutorial" in "/".join(parts).lower():
        return "tutorial"
    if "examples" in parts:
        return "example_or_workflow"
    if "openfoam13" in parts:
        return "openfoam_versioned"
    if "test" in parts or "tests" in parts:
        return "test"
    if "src" in parts:
        return "python_or_cpp_source"
    if "docs" in parts:
        return "documentation"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("architecture-inventory.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root)
        records.append({"path": rel.as_posix(), "category": classify(rel), "suffix": path.suffix})
    payload = {
        "root": str(root),
        "file_count": len(records),
        "categories": dict(Counter(item["category"] for item in records)),
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"file_count": payload["file_count"], "categories": payload["categories"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
