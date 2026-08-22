"""Create a FoamPilot tutorial manifest from a cloned OpenFOAMTutorials repo."""

from pathlib import Path
import csv
import sys

from foampilot.tutorials import OpenFOAMTutorialManifest


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: audit_repository.py REPOSITORY OUTPUT_CSV")
    repository = Path(sys.argv[1])
    output = Path(sys.argv[2])
    specs = OpenFOAMTutorialManifest(repository).discover()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family", "name", "has_run_script",
                "requires_external_geometry", "geometry_files", "source_path",
            ],
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow({
                "family": spec.family,
                "name": spec.name,
                "has_run_script": spec.has_run_script,
                "requires_external_geometry": spec.requires_external_geometry,
                "geometry_files": ";".join(spec.geometry_files),
                "source_path": str(spec.source_path),
            })
    print(f"Wrote {len(specs)} tutorial records to {output}")


if __name__ == "__main__":
    main()
