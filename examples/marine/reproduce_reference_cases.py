#!/usr/bin/env python3
"""Preview or execute the three marine OpenFOAM reference workflows.

Examples
--------
Preview a workflow from an existing checkout::

    python reproduce_reference_cases.py maneuvering /path/to/maneuveringLib/tutorial/Turning35

Execute it after sourcing the matching OpenFOAM release::

    python reproduce_reference_cases.py dtc /path/to/DTCMoving_Overset --execute --processors 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from foampilot.workflows.marine import (
    dtc_openfoam13_workflow,
    dtc_overset_workflow,
    maneuvering_turning_workflow,
    propeller_mrf_workflow,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "case",
        choices=("maneuvering", "propeller", "dtc", "dtc13"),
        help="Reference-case family to prepare.",
    )
    parser.add_argument("root", type=Path, help="Path to the reference case root.")
    parser.add_argument(
        "--processors",
        type=int,
        default=4,
        help="MPI ranks used by meshing and solver stages (default: 4).",
    )
    parser.add_argument(
        "--mesh-source",
        type=Path,
        default=Path("../DTCHull"),
        help="Chemin relatif au maillage DTC compatible OpenFOAM 13 (défaut : ../DTCHull).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the commands. Without this flag, only validate and preview the workflow.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    builders = {
        "maneuvering": maneuvering_turning_workflow,
        "propeller": propeller_mrf_workflow,
        "dtc": dtc_overset_workflow,
    }
    if arguments.case == "dtc13":
        workflow = dtc_openfoam13_workflow(
            arguments.root,
            mesh_source=arguments.mesh_source,
            processors=arguments.processors,
        )
    else:
        workflow = builders[arguments.case](arguments.root, processors=arguments.processors)

    if not arguments.execute:
        print(workflow.preview())
        print("\nPreview only. Add --execute after sourcing the compatible OpenFOAM environment.")
        return

    results = workflow.run()
    for result in results:
        print(f"{result.status:>10}  {result.name}: {result.detail}")


if __name__ == "__main__":
    main()
