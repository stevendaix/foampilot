import gmsh
from pathlib import Path
from OpenFOAM.python_code.code_manage.tuto.cfd_mesher1 import GeometryCFD

def run_full_stl_test():
    model_name = "chess_pieces_full_test"
    case = GeometryCFD(model_name)

    try:
        # 1. Load the STL geometry
        stl_filepath = Path("/home/ubuntu/chess-pieces.stl")
        if not stl_filepath.exists():
            print(f"Error: STL file not found at {stl_filepath}")
            return
        
        case._log(f"Loading STL geometry from {stl_filepath}")
        case.load_geometry(stl_filepath)
        
        # 2. Wrap surfaces (crucial for STL to create a proper topology)
        case._log("Wrapping surfaces...")
        case.wrap_surfaces()

        # 3. Create external domain
        case._log("Creating external domain...")
        fluid_volume_tag = case.create_external_domain(padding=1.0)
        case._log(f"External fluid domain created with tag: {fluid_volume_tag}")

        # 4. Define physical groups for the external domain boundaries
        gmsh.model.occ.synchronize()
        
        # Get the boundary surfaces of the fluid domain (the external box)
        fluid_domain_boundary_entities = gmsh.model.getBoundary([(3, fluid_volume_tag)], oriented=False)
        fluid_domain_boundary_tags = [ent[1] for ent in fluid_domain_boundary_entities if ent[0] == 2]

        case._log("Detecting inlets/outlets on fluid domain boundary surfaces...")
        boundary_groups = case.detect_inlets_outlets(surface_tags_to_process=fluid_domain_boundary_tags)
        print(f"Detected boundary groups (external domain surfaces): {boundary_groups}")

        # Assign STL surfaces to a specific physical group
        stl_surfaces = gmsh.model.getEntities(dim=2)
        stl_surface_tags = [s[1] for s in stl_surfaces]
        
        # Filter out surfaces that are part of the external domain boundary (to avoid double assignment)
        # This is important if the STL surfaces are coincident with the external domain boundary
        # For a clean setup, STL surfaces should be internal to the fluid domain.
        internal_stl_surface_tags = [tag for tag in stl_surface_tags if tag not in fluid_domain_boundary_tags]
        
        if internal_stl_surface_tags:
            case._log("Defining physical group for internal STL walls...")
            case.define_physical_group(2, internal_stl_surface_tags, "stl_internal_walls")
        else:
            case._log("No internal STL surfaces found to define as physical group (they might be part of external boundary). ")

        # 5. Assign material to the fluid volume
        case._log(f"Assigning material \'fluid\' to volume {fluid_volume_tag}")
        case.set_material("fluid", [fluid_volume_tag])

        # 6. Generate volume mesh
        case._log("Generating volume mesh...")
        case.mesh_volume(lc=0.5) # Adjust lc as needed for mesh density

        # 7. Get and print basic mesh statistics
        mesh_stats = case.get_basic_mesh_stats()
        print(f"Basic mesh statistics: {mesh_stats}")

        # 8. Analyze and print mesh quality
        quality_metrics = case.analyze_mesh_quality()
        print(f"Mesh quality metrics: {quality_metrics}")

        # 9. Export to OpenFOAM
        output_folder = Path("openfoam_chess_full_test")
        case._log(f"Exporting mesh to OpenFOAM in {output_folder}")
        case.export_to_openfoam(output_folder, run_gmshtofoam=False) # Set to True if gmshToFoam is available

        # 10. Visualize the mesh (optional, requires Gmsh GUI)
        case._log("Visualizing mesh in Gmsh GUI...")
        case.visualize()

        print("Full STL test completed successfully!")

    except Exception as e:
        print(f"An error occurred during the full STL test: {e}")
    finally:
        case.finalize()

if __name__ == "__main__":
    run_full_stl_test()
