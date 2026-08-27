from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "foampilot" / "tutorials"
INTEGRATION = ROOT / "docs" / "openfoam13_foampilot_integration.md"
OUTPUT = ROOT / "docs" / "openfoam13_foampilot_rewrite_matrix_full.md"


def count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text))


def status_for(text: str) -> tuple[str, str]:
    imports = sum(count(text, p) for p in (
        r"import_reference_file", r"import_reference_field",
        r"import_reference_asset", r"import_reference_dict",
        r"copy_case_tree", r"rglob\(", r"glob\(",
    ))
    direct = sum(count(text, p) for p in (
        r"subprocess", r"os\.system", r"shutil", r"open\(",
    ))
    generated = sum(count(text, p) for p in (
        r"register_field", r"set_condition", r"set_raw_condition",
        r"\.write\(", r"Meshing\(", r"BlockMesher\(",
        r"Boundary\(", r"configure_", r"set_default_value",
    ))
    if direct:
        return "À corriger — opération directe", "Opération directe détectée"
    if imports == 0 and generated > 0:
        return "Réécrit FoamPilot à vérifier", "Pas d’import de référence détecté"
    if imports > 0 and generated > 0:
        return "Partiellement réécrit", "Imports et génération FoamPilot mélangés"
    if imports > 0:
        return "Fonctionnel — import de référence", "Mise en données de référence importée"
    return "À auditer", "Aucun motif de génération/import classifiable"


def api_names(text: str) -> str:
    names = []
    for name in (
        "import_reference_file", "import_reference_field", "import_reference_asset",
        "import_reference_dict", "copy_case_tree", "run_command", "run_simulation",
        "run_parallel", "register_field", "set_condition", "set_raw_condition",
        "set_patch_type", "update_dictionary_entries", "merge_reference_dictionary",
        "remove_files", "write", "Meshing", "BlockMesher", "Boundary",
    ):
        if re.search(re.escape(name), text):
            names.append(name)
    return ", ".join(names) if names else "—"


def validation_status(rel: str, integration: str) -> str:
    token = rel.split("/", 1)[-1].split("/", 1)[-1]
    matches = [line for line in integration.splitlines() if token and token in line]
    if not matches:
        return "Non retrouvée dans le suivi"
    line = matches[-1]
    if "Validé" in line or "VALIDÉ" in line:
        return "Validé fonctionnellement"
    if "réserve" in line.lower() or "Réserve" in line:
        return "Accepté avec réserve"
    if "Partiel" in line or "En cours" in line:
        return "Partiel / en cours"
    return "Mentionné dans le suivi"


def main() -> None:
    integration = INTEGRATION.read_text(encoding="utf-8", errors="replace") if INTEGRATION.exists() else ""
    runners = sorted(
        TUTORIALS.glob("*/run.py"),
        key=lambda p: (
            int(re.match(r"(\d+)", p.parent.name).group(1))
            if re.match(r"(\d+)", p.parent.name)
            else 10**9,
            p.parent.name,
        ),
    )
    rows = []
    for index, runner in enumerate(runners, 1):
        rel = runner.parent.relative_to(TUTORIALS).as_posix()
        text = runner.read_text(encoding="utf-8", errors="replace")
        status, reason = status_for(text)
        rows.append((index, rel, status, validation_status(rel, integration), api_names(text), reason))

    lines = [
        "# Matrice complète de réécriture des runners OpenFOAM 13",
        "",
        "> Cette matrice est générée à partir des 261 fichiers `run.py` présents dans `foampilot/tutorials`. Elle sépare le statut de validation fonctionnelle du statut de réécriture réelle. Les statuts sont fondés sur les motifs présents dans le code; une validation OF13 n’est jamais déduite de la seule présence d’un runner.",
        "",
        "| # | Runner | Réécriture | Validation OF13 suivie | API / opérations détectées | Justification automatique |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| {} | `{}` | {} | {} | `{}` | {} |".format(*row))
    lines += [
        "",
        "## Règle d’interprétation",
        "",
        "Les statuts `Fonctionnel — import de référence` et `Partiellement réécrit` ne sont pas des réécritures complètes. Un runner ne pourra passer à `Réécrit FoamPilot` qu’après remplacement des champs et dictionnaires de mise en données par des constructeurs FoamPilot et validation des fichiers générés sous OpenFOAM 13.",
        "",
        f"Nombre de runners analysés automatiquement: **{len(rows)}**.",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"generated {OUTPUT} with {len(rows)} runners")


if __name__ == "__main__":
    main()
