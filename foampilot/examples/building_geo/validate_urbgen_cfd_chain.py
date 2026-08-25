from __future__ import annotations

import json
import shutil
from pathlib import Path

import gmsh
from foampilot.mesh.direct_openfoam_exporter import DirectOpenFOAMExporter
from foampilot.urban.generation import UrbGENConfig, generate_urbgen
from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "urbgen_cfd_validation"
CASE = OUT / "case"


def write_openfoam_case(case: Path, patches: list[str]) -> None:
    (case / "0").mkdir(parents=True, exist_ok=True)
    (case / "constant").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(parents=True, exist_ok=True)
    (case / "system" / "controlDict").write_text(
        "FoamFile\n{ version 2.0; format ascii; class dictionary; object controlDict; }\n"
        "application simpleFoam; startFrom startTime; startTime 0; stopAt endTime; endTime 1;\n"
        "deltaT 1; writeControl timeStep; writeInterval 1;\n"
    )
    (case / "system" / "fvSchemes").write_text(
        "FoamFile\n{ version 2.0; format ascii; class dictionary; object fvSchemes; }\n"
        "ddtSchemes { default steadyState; }\n"
        "gradSchemes { default cellLimited Gauss linear 1; }\n"
        "divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); }\n"
        "laplacianSchemes { default Gauss linear corrected; }\n"
        "interpolationSchemes { default linear; }\n"
        "snGradSchemes { default corrected; }\n"
    )
    (case / "system" / "fvSolution").write_text(
        "FoamFile\n{ version 2.0; format ascii; class dictionary; object fvSolution; }\n"
        "solvers { p { solver GAMG; tolerance 1e-6; relTol 0; } U { solver smoothSolver; tolerance 1e-6; relTol 0; } }\n"
        "SIMPLE { nNonOrthogonalCorrectors 0; }\n"
    )
    (case / "constant" / "transportProperties").write_text(
        "FoamFile\n{ version 2.0; format ascii; class dictionary; object transportProperties; }\n"
        "transportModel Newtonian;\nnu nu [0 2 -1 0 0 0 0] 1.5e-05;\n"
    )
    boundary = "\n".join(f"    {p} {{ type wall; }}" for p in patches)
    (case / "0" / "U").write_text(
        "FoamFile\n{ version 2.0; format ascii; class volVectorField; object U; }\n"
        "dimensions [0 1 -1 0 0 0 0]; internalField uniform (10 0 0);\n"
        f"boundaryField {{\n{boundary}\n}}\n"
    )
    (case / "0" / "p").write_text(
        "FoamFile\n{ version 2.0; format ascii; class volScalarField; object p; }\n"
        "dimensions [0 2 -2 0 0 0 0]; internalField uniform 0;\n"
        f"boundaryField {{\n{boundary}\n}}\n"
    )


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    site = Polygon([(10, 10), (190, 10), (190, 140), (10, 140)])
    result = generate_urbgen(site, UrbGENConfig(bcr=0.08, far=1.5, setback=5.0, seed=42, podium_floors=0))
    domain = (0.0, 0.0, 0.0, 240.0, 180.0, 90.0)

    gmsh.initialize()
    gmsh.model.add("urbgen_cfd_validation")
    try:
        fluid = gmsh.model.occ.addBox(domain[0], domain[1], domain[2], domain[3], domain[4], domain[5])
        building_tags = []
        for building in result.model.buildings():
            minx, miny, maxx, maxy = building.footprint.bounds
            building_tags.append(gmsh.model.occ.addBox(minx, miny, building.ground_z, maxx - minx, maxy - miny, building.height))
        gmsh.model.occ.synchronize()
        if building_tags:
            cut, _ = gmsh.model.occ.cut([(3, fluid)], [(3, tag) for tag in building_tags], removeObject=True, removeTool=False)
            fluid = cut[0][1]
        gmsh.model.occ.synchronize()
        fluid_entities = [(3, fluid)]
        fluid_group = gmsh.model.addPhysicalGroup(3, [fluid], name="FLUID")
        gmsh.model.setPhysicalName(3, fluid_group, "FLUID")
        boundary_entities = gmsh.model.getBoundary(fluid_entities, oriented=False, recursive=False)
        patches: dict[str, list[int]] = {"inlet": [], "outlet": [], "side_walls": [], "ground": [], "top": [], "building_walls": []}
        xmin, ymin, zmin, xmax, ymax, zmax = domain
        tol = 1e-6
        for dim, tag in boundary_entities:
            if dim != 2:
                continue
            cx, cy, cz = gmsh.model.occ.getCenterOfMass(2, tag)
            if abs(cx - xmin) < tol:
                name = "inlet"
            elif abs(cx - xmax) < tol:
                name = "outlet"
            elif abs(cy - ymin) < tol or abs(cy - ymax) < tol:
                name = "side_walls"
            elif abs(cz - zmin) < tol:
                name = "ground"
            elif abs(cz - zmax) < tol:
                name = "top"
            else:
                name = "building_walls"
            patches[name].append(tag)
        existing_surface_tags = {tag for _, tag in gmsh.model.getEntities(dim=2)}
        for name, tags in patches.items():
            tags = [tag for tag in tags if tag in existing_surface_tags]
            patches[name] = tags
            if tags:
                gid = gmsh.model.addPhysicalGroup(2, tags, name=name)
                gmsh.model.setPhysicalName(2, gid, name)
        gmsh.model.mesh.generate(3)
        mesh_file = OUT / "urbgen.msh"
        gmsh.write(str(mesh_file))
        stats = {
            "building_count": result.model.building_count(),
            "volume_count": len(gmsh.model.getEntities(3)),
            "surface_count": len(gmsh.model.getEntities(2)),
            "node_count": len(gmsh.model.mesh.getNodes()[0]),
            "physical_volume_names": [gmsh.model.getPhysicalName(3, t) for _, t in gmsh.model.getPhysicalGroups(3)],
            "physical_surface_names": [gmsh.model.getPhysicalName(2, t) for _, t in gmsh.model.getPhysicalGroups(2)],
            "actual_bcr": result.actual_bcr,
            "actual_far": result.actual_far,
        }
        (OUT / "gmsh_metrics.json").write_text(json.dumps(stats, indent=2))
        CASE.mkdir(parents=True, exist_ok=True)
        DirectOpenFOAMExporter(CASE).export_single_region("fluid")
        write_openfoam_case(CASE, [name for name, tags in patches.items() if tags])
        required = [CASE / "constant" / "polyMesh" / name for name in ("points", "faces", "owner", "neighbour", "boundary")]
        required += [CASE / "constant" / "transportProperties", CASE / "system" / "controlDict", CASE / "system" / "fvSchemes", CASE / "system" / "fvSolution", CASE / "0" / "U", CASE / "0" / "p"]
        missing = [str(p) for p in required if not p.exists()]
        stats["openfoam_missing_files"] = missing
        stats["openfoam_case_valid"] = not missing and "nu " in (CASE / "constant" / "transportProperties").read_text()
        (OUT / "validation_report.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    main()
