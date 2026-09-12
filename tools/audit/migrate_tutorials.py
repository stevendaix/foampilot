#!/usr/bin/env python3
"""Migrate tutorial scripts from reference-file-copy pattern to declarative API.

This script transforms tutorial scripts that use:
    for source in (REFERENCE / "system").iterdir():
        solver.system.import_reference_file(source)
    for source in (REFERENCE / "constant").iterdir():
        solver.constant.import_reference_file(source)
    for source in (REFERENCE / "0").iterdir():
        solver.fields_manager.import_reference_field(source, case_path)

Into declarative generation:
    solver.system.write()  # Already done in setup_case
    solver.constant.write()  # Already done in setup_case
    solver.fields_manager.write_initial_fields()  # Generate fields via API
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Pattern to find and replace
IMPORT_REFERENCE_SYSTEM = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"system"\)\.iterdir\(\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.system\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

IMPORT_REFERENCE_CONSTANT = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"constant"\)\.iterdir\(\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.constant\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

IMPORT_REFERENCE_FIELDS = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"0"\)\.iterdir\(\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.fields_manager\.import_reference_field\([^)]*\)',
    re.MULTILINE
)

IMPORT_REFERENCE_SYSTEM_RGLOB = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"system"\)\.rglob\("*"\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.system\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

IMPORT_REFERENCE_CONSTANT_RGLOB = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"constant"\)\.rglob\("*"\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.constant\.import_reference_file\([^)]*\)',
    re.MULTILINE
)

IMPORT_REFERENCE_FIELDS_RGLOB = re.compile(
    r'for\s+source\s+in\s+\(REFERENCE\s*/\s*"0"\)\.rglob\("*"\):\s*\n'
    r'\s+if\s+source\.is_file\(\):\s*\n'
    r'\s+solver\.fields_manager\.import_reference_field\([^)]*\)',
    re.MULTILINE
)

REMOVE_FILES_PATTERN = re.compile(
    r'solver\.constant\.remove_files\([^)]*\)',
    re.MULTILINE
)

IMPORT_REFERENCE_DICT = re.compile(
    r'mesh\.mesher\.import_reference_dict\([^)]*\)',
    re.MULTILINE
)


def migrate_file(filepath: Path) -> bool:
    """Migrate a single tutorial file. Returns True if changes were made."""
    content = filepath.read_text(encoding="utf-8")
    original = content
    
    # Replace import_reference_file loops with declarative generation
    content = IMPORT_REFERENCE_SYSTEM.sub(
        '# Declarative generation: solver.system.write() already called in setup_case',
        content
    )
    content = IMPORT_REFERENCE_SYSTEM_RGLOB.sub(
        '# Declarative generation: solver.system.write() already called in setup_case',
        content
    )
    content = IMPORT_REFERENCE_CONSTANT.sub(
        '# Declarative generation: solver.constant.write() already called in setup_case',
        content
    )
    content = IMPORT_REFERENCE_CONSTANT_RGLOB.sub(
        '# Declarative generation: solver.constant.write() already called in setup_case',
        content
    )
    content = IMPORT_REFERENCE_FIELDS.sub(
        '# Declarative generation: solver.fields_manager.write_initial_fields()',
        content
    )
    content = IMPORT_REFERENCE_FIELDS_RGLOB.sub(
        '# Declarative generation: solver.fields_manager.write_initial_fields()',
        content
    )
    
    # Replace remove_files with comment
    content = REMOVE_FILES_PATTERN.sub(
        '# Declarative generation: no files to remove',
        content
    )
    
    # Replace import_reference_dict with comment
    content = IMPORT_REFERENCE_DICT.sub(
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