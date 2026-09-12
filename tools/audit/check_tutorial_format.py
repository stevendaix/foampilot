#!/usr/bin/env python3
"""Audit des tutoriels - Vérifie les anti-patterns de copie de fichiers."""
import sys
from pathlib import Path

# ⛔ INTERDITS ABSOLUS (y compris les méthodes officielles de copie)
FORBIDDEN_PATTERNS = [
    "import_reference_file",
    "import_reference_field",
    "import shutil",
    "shutil.copy",
    "shutil.copytree",
    "shutil.rmtree",
    "os.system",
    "subprocess.run(['cp'",
    "subprocess.run(['cp ",
    "subprocess.run([\"cp\"",
]


def audit_file(filepath: Path):
    content = filepath.read_text(encoding='utf-8')
    issues = []

    for pattern in FORBIDDEN_PATTERNS:
        if pattern in content:
            issues.append(f"  ⛔ INTERDIT : '{pattern}'")

    return issues


def main():
    tutorials_dir = Path("tutorials")
    if not tutorials_dir.exists():
        print("❌ Dossier 'tutorials/' non trouvé.")
        sys.exit(1)

    print(f"🔍 Audit des tutoriels dans {tutorials_dir.resolve()}...\n")
    all_clean = True
    files_audited = 0

    for py_file in sorted(tutorials_dir.rglob("*.py")):
        if py_file.name.startswith("_"):  # Skip _template.py
            continue
        issues = audit_file(py_file)
        files_audited += 1
        if issues:
            all_clean = False
            print(f"📁 {py_file}")
            for issue in issues:
                print(issue)
            print("-" * 50)

    print(f"\n📊 {files_audited} fichiers audités.")
    if all_clean:
        print("✅ SUCCÈS : Aucun anti-pattern de copie détecté.")
        sys.exit(0)
    else:
        print("⚠️ ÉCHEC : Corrections nécessaires (voir ci-dessus).")
        sys.exit(1)


if __name__ == "__main__":
    main()