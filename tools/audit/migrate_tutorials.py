#!/usr/bin/env python3
"""Remove import_reference_file/import_reference_field anti-patterns from tutorial scripts.

Replaces patterns like:
    for source in (REFERENCE / "system").iterdir():
        if source.is_file():
            solver.system.import_reference_file(source)
    for source in (REFERENCE / "constant").iterdir():
        if source.is_file():
            solver.constant.import_reference_file(source)
    for source in (REFERENCE / "0").iterdir():
        if source.is_file():
            solver.fields_manager.import_reference_field(source, case_path)

With declarative generation:
    solver.system.write()
    solver.constant.write()
    solver.fields_manager._generate_fields()
"""
import re
import sys
from pathlib import Path
from typing import List

# Pattern: for source in (REFERENCE / "system").iterdir(): ... import_reference_file
PATTERN_SYSTEM_ITER = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"system"\)\.iterdir\(\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.system\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

PATTERN_CONSTANT_ITER = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"constant"\)\.iterdir\(\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.constant\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

PATTERN_FIELDS_ITER = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"0"\)\.iterdir\(\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.fields_manager\.import_reference_field\([^)]*\)',
    re.MULTILINE
)

PATTERN_SYSTEM_RGLOB = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"system"\)\.rglob\([^)]*\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.system\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

PATTERN_CONSTANT_RGLOB = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"constant"\)\.rglob\([^)]*\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.constant\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

PATTERN_FIELDS_RGLOB = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"0"\)\.rglob\([^)]*\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.fields_manager\.import_reference_field\([^)]*\)',
    re.MULTILINE
)

PATTERN_REMOVE_FILES = re.compile(
    r'solver\.constant\.remove_files\([^)]*\)',
    re.MULTILINE
)

PATTERN_IMPORT_DICT = re.compile(
    r'mesh\.mesher\.import_reference_dict\([^)]*\)',
    re.MULTILINE
)


def migrate_file(filepath: Path) -> bool:
    """Migrate a single file. Returns True if changes were made."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    
    # Replace import_reference_file loops with declarative generation
    content = PATTERN_SYSTEM_ITER.sub(
        '# Declarative generation: solver.system.write() already called in setup_case',
        content
    )
    content = PATTERN_SYSTEM_RGLOB.sub(
        '# Declarative generation: solver.system.write() already called in setup_case',
        content
    )
    content = PATTERN_CONSTANT_ITER.sub(
        '# Declarative generation: solver.constant.write() already called in setup_case',
        content
    )
    content = PATTERN_CONSTANT_RGLOB.sub(
        '# Declarative generation: solver.constant.write() already called in setup_case',
        content
    )
    content = PATTERN_FIELDS_ITER.sub(
        '# Declarative generation: solver.fields_manager._generate_fields()',
        content
    )
    content = PATTERN_FIELDS_RGLOB.sub(
        '# Declarative generation: solver.fields_manager._generate_fields()',
        content
    )
    
    # Replace remove_files with comment
    content = PATTERN_REMOVE_FILES.sub(
        '# Declarative generation: no files to remove',
        content
    )
    
    # Replace import_reference_dict with comment
    content = PATTERN_IMPORT_DICT.sub(
        '# Declarative generation: mesh.mesher.write() generates blockMeshDict',
        content
    )
    
    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False


def main():
    tutorials_dir = Path("tutorials")
    migrated = []
    for py_file in sorted(tutorials_dir.rglob("run*.py")):
        if "__pycache__" in str(py_file):
            continue
        if migrate_file(py_file):
            migrated.append(str(py_file))
    
    print(f"Migrated {len(migrated)} files:")
    for f in migrated:
        print(f"  - {f}")


if __name__ == "__main__":
    main()