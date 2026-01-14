import os
import shutil
from pathlib import Path
from foampilot import Meshing, Quantity, FluidMechanics,Solver

def setup_coa_case():
    # 1. DEFINE CASE PATH
    current_path = Path.cwd()
    case_path = Path.cwd() / "CoA_test_foampilot"
    csv_inlet_data = current_path/ "data" / "2FBPM120.csv"
    stl_path = current_path/ "data"

    case_path.mkdir(parents=True, exist_ok=True)

    # 2. FLUID PROPERTIES
    # Blood: rho=1060, nu=3.77e-6
    # Since Blood is not in default FluidMechanics, we set it manually
    rho_blood = Quantity(1060, "kg/m^3")
    nu_blood = Quantity(3.7735849056603773e-06, "m^2/s")

    # 3. INITIALIZE SOLVER
    solver = Solver(case_path)
    solver.compressible = False
    solver.with_gravity = False
    solver.transient = True
    solver.turbulence_model = "laminar"
    
    # Set viscosity and density
    solver.constant.transportProperties.nu = nu_blood
    solver.constant.transportProperties.rho = rho_blood

    # 4. CONFIGURE CONTROLDICT & SYSTEM
    solver.system.controlDict.application = "foamRun"
    solver.system.controlDict.solver = "incompressibleFluid"
    solver.system.controlDict.startTime = 0.0
    solver.system.controlDict.endTime = 0.5
    solver.system.controlDict.deltaT = 1e-05
    solver.system.controlDict.writeInterval = 0.01
    solver.system.controlDict.adjustTimeStep = True
    solver.system.controlDict.maxCo = 0.8
    solver.system.controlDict.libs = ("libmodularWKPressure.so",)
    
    # Configure PIMPLE for CoA_test
    solver.system.fvSolution.PIMPLE.update({
        "nOuterCorrectors": 40,
        "nCorrectors": 2,
        "nNonOrthogonalCorrectors": 1,
        "residualControl": {
            "p": 0.01,
            "U": 0.001
        }
    })
    
    # Write system and constant folders
    solver.system.write()
    solver.constant.write()

    # 5. MESH GENERATION (snappyHexMesh)
    # Copy STL files from the original tutorial
    
    stl_dest = case_path / "constant" / "triSurface"
    stl_dest.mkdir(parents=True, exist_ok=True)
    
    for stl_file in stl_path.glob("*.stl"):
        shutil.copy(stl_file, stl_dest)
    

    # Initialize Meshing with snappy
    mesh = Meshing(case_path, mesher="snappy")
    snappy = mesh.mesher
    snappy.stl_file = Path("wall_aorta.stl")
    
    # Configure snappyHexMeshDict
    snappy.locationInMesh = (-16.3177, -21.6838, -12.3357)
    snappy.geometry = {
        "wall_aorta": {"type": "triSurfaceMesh", "file": "wall_aorta.stl", "name": "wall_aorta"},
        "inlet": {"type": "triSurfaceMesh", "file": "inlet.stl", "name": "inlet"},
        "outlet1": {"type": "triSurfaceMesh", "file": "outlet1.stl", "name": "outlet1"},
        "outlet2": {"type": "triSurfaceMesh", "file": "outlet2.stl", "name": "outlet2"},
        "outlet3": {"type": "triSurfaceMesh", "file": "outlet3.stl", "name": "outlet3"},
        "outlet4": {"type": "triSurfaceMesh", "file": "outlet4.stl", "name": "outlet4"},
    }
    snappy.castellatedMeshControls["refinementSurfaces"] = {
        "wall_aorta": {"level": (0, 1)},
        "inlet": {"level": (0, 1)},
        "outlet1": {"level": (0, 1)},
        "outlet2": {"level": (0, 1)},
        "outlet3": {"level": (0, 1)},
        "outlet4": {"level": (0, 1)},
    }
    for part in ["wall_aorta", "inlet", "outlet1", "outlet2", "outlet3", "outlet4"]:
        snappy.add_feature(f"{part}.eMesh", 1)
    stl_files = ["wall_aorta.stl", "inlet.stl", "outlet1.stl", "outlet2.stl", "outlet3.stl", "outlet4.stl"]

    snappy.addLayers = True
    snappy.add_layer("wall_aorta", 3)
    snappy.addLayersControls["finalLayerThickness"] = 0.3
    
    # Write snappyHexMeshDict
    mesh.write()

    # 1. Générer surfaceFeaturesDict
    snappy.write_surface_features_dict(case_path, stl_files, included_angle=30)

    snappy.write_snappyHexMesh_dict(case_path)

    snappy.run()

    # Initialize boundaries
    solver.boundary.initialize_boundary()
    
    # Define custom BCs for Windkessel
    inlet_bc_u = {
        "type": "timeVaryingMappedFixedValue",
        "offset": "(0 0 0)",
        "setAverage": "false"
    }
    inlet_bc_p = {"type": "zeroGradient"}
    
    wk_p_bc = {
        "type": "modularWKPressure",
        "phi": "phi",
        "order": "2",
        "R": "1000",
        "C": "1e-06",
        "Z": "100",
        "p0": "10666",
        "value": "uniform 10666"
    }
    wk_u_bc = {
        "type": "stabilizedWindkesselVelocity",
        "beta": "1.0",
        "enableStabilization": "true",
        "value": "uniform (0 0 0)"
    }
    
    # Apply to patches
    solver.boundary.fields["U"]["inlet"] = inlet_bc_u
    solver.boundary.fields["p"]["inlet"] = inlet_bc_p
    
    for i in range(1, 5):
        patch = f"outlet{i}"
        solver.boundary.fields["U"][patch] = wk_u_bc
        solver.boundary.fields["p"][patch] = wk_p_bc
        
    solver.boundary.fields["U"]["wall_aorta"] = {"type": "noSlip"}
    solver.boundary.fields["p"]["wall_aorta"] = {"type": "zeroGradient"}

    # Write all boundary files (this creates the 0/ directory files)
    solver.boundary.write_boundary_conditions()
    
    # Copy boundaryData for inlet

    # Charger ton CSV
    df_csv = pd.read_csv(csv_inlet_data)  # colonnes: Time, Flowrate

    # Charger le maillage
    integrator = CSVFoamIntegrator("CoA_test_foampilot")
    df_patch = integrator.get_patch_dataframe("inlet")  # patch d'entrée

    # Répliquer le CSV pour chaque point du patch
    df_full = []
    for _, row in df_csv.iterrows():
        for _, p in df_patch.iterrows():
            df_full.append({
                "time": row["Time"],
                "x": p["x"],
                "y": p["y"],
                "z": p["z"],
                "Flowrate": row["Flowrate"]
            })
    df_full = pd.DataFrame(df_full)

    # Exporter en boundaryData
    integrator.export_to_boundary_data("inlet", df_full, "Flowrate")


    print(f"Case {case_path} has been set up successfully using the modern foampilot API.")

if __name__ == "__main__":
    setup_coa_case()
