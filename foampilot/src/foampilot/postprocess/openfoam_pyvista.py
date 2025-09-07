import subprocess
from pathlib import Path
import pyvista as pv

# class FoamPostProcessing:
#     def __init__(self, case_path: str, vtk_dir: str = "VTK"):
#         self.case_path = Path(case_path)
#         self.vtk_dir = self.case_path / vtk_dir

#     def check_case(self):
#         if not self.case_path.exists() or not self.case_path.is_dir():
#             raise FileNotFoundError(f"OpenFOAM case path '{self.case_path}' does not exist or is not a directory.")

#     def foamToVTK(self, 
#                   all_regions=False,
#                   ascii=False,
#                   constant=False,
#                   latest_time=False,
#                   fields=None,
#                   no_boundary=False,
#                   no_internal=False):
#         self.check_case()
#         cmd = ["foamToVTK"]

#         if all_regions:
#             cmd.append("-allRegions")
#         if ascii:
#             cmd.append("-ascii")
#         if constant:
#             cmd.append("-constant")
#         if latest_time:
#             cmd.append("-latestTime")
#         if fields:
#             if isinstance(fields, list):
#                 fields_str = "(" + " ".join(fields) + ")"
#             else:
#                 fields_str = fields
#             cmd += ["-fields", fields_str]

#         if no_boundary:
#             cmd.append("-no-boundary")
#         if no_internal:
#             cmd.append("-no-internal")

#         cmd += ["-case", str(self.case_path)]

#         print(f"Running foamToVTK with command: {' '.join(cmd)}")
#         result = subprocess.run(cmd, text=True, capture_output=True)
#         if result.returncode != 0:
#             raise RuntimeError(f"foamToVTK failed:\n{result.stderr}")
#         print(result.stdout)
#         print(f"VTK files written to {self.vtk_dir}")

#     def list_time_steps(self):
#         """
#         Returns a sorted list of available time steps based on VTK files in the main directory.
#         """
#         vtk_files = list(self.vtk_dir.glob(f"{self.case_path.name}_*.vtk"))
#         time_steps = sorted([int(f.stem.split("_")[-1]) for f in vtk_files])
#         return time_steps

#     def get_structure(self, time_step=None, boundaries=["inlet","outlet","walls"]):
#         """
#         Returns a dictionary structure with the main cell mesh and boundary meshes.
#         """
#         if time_step is None:
#             # choisir le dernier pas disponible
#             steps = self.list_time_steps()
#             if not steps:
#                 raise FileNotFoundError("No VTK files found in directory.")
#             time_step = steps[-1]

#         structure = {}
#         # Cell mesh
#         cell_file = self.vtk_dir / f"{self.case_path.name}_{time_step}.vtk"
#         if not cell_file.exists():
#             raise FileNotFoundError(f"Cell file not found: {cell_file}")
#         structure['cell'] = pv.read(cell_file)

#         # Boundaries
#         structure['boundaries'] = {}
#         for b in boundaries:
#             b_file = self.vtk_dir / b / f"{b}_{time_step}.vtk"
#             if b_file.exists():
#                 structure['boundaries'][b] = pv.read(b_file)
#             else:
#                 print(f"Boundary file not found: {b_file}")

#         return structure

#     def plot_slice(self, structure=None, plane='z', scalars='U', opacity=0.25):
#         """
#         Generate a slice plot from the given structure dictionary.
#         """
#         if structure is None:
#             raise RuntimeError("No structure provided. Run get_structure() first.")

#         y_slice = structure["cell"].slice(plane)
#         pl = pv.Plotter()
#         pl.add_mesh(y_slice, scalars=scalars, lighting=False, scalar_bar_args={'title': scalars})
#         pl.add_mesh(structure["cell"], color='w', opacity=opacity)
#         for name, mesh in structure.get("boundaries", {}).items():
#             pl.add_mesh(mesh, opacity=0.5)
#         pl.enable_anti_aliasing()
#         pl.show()


import subprocess
from pathlib import Path
import pyvista as pv
import numpy as np
import pandas as pd
import json


class FoamPostProcessing:
    def __init__(self, case_path: str, vtk_dir: str = "VTK"):
        """
        Post-processing class for OpenFOAM case.

        Args:
            case_path (str): Path to OpenFOAM case directory
            vtk_dir (str): Directory name for VTK output (default: 'VTK')
        """
        self.case_path = Path(case_path)
        self.vtk_dir = self.case_path / vtk_dir

    def check_case(self):
        if not self.case_path.exists() or not self.case_path.is_dir():
            raise FileNotFoundError(
                f"OpenFOAM case path '{self.case_path}' does not exist or is not a directory."
            )

    def foamToVTK(
        self,
        all_regions=False,
        ascii=False,
        constant=False,
        latest_time=False,
        fields=None,
        no_boundary=False,
        no_internal=False,
    ):
        """
        Converts the OpenFOAM case to VTK files using foamToVTK.
        """
        self.check_case()
        cmd = ["foamToVTK"]

        if all_regions:
            cmd.append("-allRegions")
        if ascii:
            cmd.append("-ascii")
        if constant:
            cmd.append("-constant")
        if latest_time:
            cmd.append("-latestTime")
        if fields:
            if isinstance(fields, list):
                fields_str = "(" + " ".join(fields) + ")"
            else:
                fields_str = fields
            cmd += ["-fields", fields_str]

        cmd += ["-case", str(self.case_path)]

        print(f"Running foamToVTK with command: {' '.join(cmd)}")
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"foamToVTK failed:\n{result.stderr}")
        print(result.stdout)
        print(f"VTK files written to {self.vtk_dir}")

    def list_time_steps(self):
        """
        Returns a sorted list of available time steps based on VTK files in the main directory.
        """
        vtk_files = list(self.vtk_dir.glob(f"{self.case_path.name}_*.vtk"))
        time_steps = sorted([int(f.stem.split("_")[-1]) for f in vtk_files])
        return time_steps

    def get_structure(self, time_step=None):
        """
        Construit un dictionnaire avec la maille principale (cell)
        et toutes les boundaries trouvées automatiquement dans le dossier VTK.
        """
        if time_step is None:
            steps = self.list_time_steps()
            if not steps:
                raise FileNotFoundError("No VTK files found in directory.")
            time_step = steps[-1]

        structure = {}

        # Charger la maille principale (cell)
        cell_file = self.vtk_dir / f"{self.case_path.name}_{time_step}.vtk"
        if not cell_file.exists():
            raise FileNotFoundError(f"Cell file not found: {cell_file}")
        structure["cell"] = pv.read(cell_file)

        # Charger automatiquement toutes les boundaries
        structure["boundaries"] = {}
        for subdir in self.vtk_dir.iterdir():
            if subdir.is_dir():
                b_file = subdir / f"{subdir.name}_{time_step}.vtk"
                if b_file.exists():
                    structure["boundaries"][subdir.name] = pv.read(b_file)
                else:
                    print(f"Boundary file not found: {b_file}")

        return structure

    def load_time_step(self, time_step: int):
        """
        Loads the VTK data for a specific time step.
        """
        return self.get_structure(time_step=time_step)

    def get_all_time_steps(self):
        """
        Returns all available time steps.
        """
        return self.list_time_steps()

    def plot_slice(self, structure=None, plane="z", scalars="U", opacity=0.25):
        """
        Generate a slice plot from the given structure dictionary.
        """
        if structure is None:
            raise RuntimeError("No structure provided. Run get_structure() first.")

        y_slice = structure["cell"].slice(plane)
        pl = pv.Plotter()
        pl.add_mesh(
            y_slice,
            scalars=scalars,
            lighting=False,
            scalar_bar_args={"title": scalars},
        )
        pl.add_mesh(structure["cell"], color="w", opacity=opacity)
        for name, mesh in structure.get("boundaries", {}).items():
            pl.add_mesh(mesh, opacity=0.5)
        pl.enable_anti_aliasing()
        pl.show()

    def plot_contour(self, mesh, scalars: str, is_filled: bool = True, opacity: float = 1.0):
        """
        Generate a contour plot.
        """
        pl = pv.Plotter()
        if is_filled:
            pl.add_mesh(mesh.contour(), scalars=scalars, show_scalar_bar=True, opacity=opacity)
        else:
            pl.add_mesh(mesh.contour(isosurfaces=10), scalars=scalars, show_scalar_bar=True, opacity=opacity, style='wireframe')
        pl.show()

    def plot_vectors(self, mesh, vectors: str, scale: float = 1.0, color: str = 'blue'):
        """
        Generate a vector plot.
        """
        pl = pv.Plotter()
        if vectors not in mesh.point_data:
            raise ValueError(f"Vector field '{vectors}' not found in mesh point data.")
        arrows = mesh.glyph(orient=vectors, scale=vectors, factor=scale)
        pl.add_mesh(arrows, color=color)
        pl.show()

    def plot_streamlines(self, mesh, vectors: str, n_points: int = 100, max_time: float = 10.0):
        """
        Generate streamlines.
        """
        pl = pv.Plotter()
        streamlines = mesh.streamlines(vectors=vectors, n_points=n_points, max_time=max_time)
        pl.add_mesh(streamlines, color='red')
        pl.add_mesh(mesh, opacity=0.25)
        pl.show()

    def plot_mesh_style(self, mesh, style: str = 'surface', show_edges: bool = False, color: str = 'white', opacity: float = 1.0):
        """
        Visualize the mesh with different styles.
        """
        pl = pv.Plotter()
        pl.add_mesh(mesh, style=style, show_edges=show_edges, color=color, opacity=opacity)
        pl.show()

    def calculate_q_criterion(self, mesh, velocity_field: str = 'U'):
        """
        Calculate the Q-criterion.
        """
        if velocity_field not in mesh.point_data:
            raise ValueError(f"Velocity field '{velocity_field}' not found in mesh point data.")

        gradient = mesh.compute_derivative(scalars=velocity_field).point_data['gradient']

        grad_u = gradient.reshape(-1, 3, 3)

        S = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))

        Omega = 0.5 * (grad_u - np.transpose(grad_u, (0, 2, 1)))

        q_criterion = 0.5 * (np.linalg.norm(Omega, axis=(1, 2))**2 - np.linalg.norm(S, axis=(1, 2))**2)
        mesh.point_data['q_criterion'] = q_criterion
        return mesh

    def calculate_vorticity(self, mesh, velocity_field: str = 'U'):
        """
        Calculate the vorticity.
        """
        if velocity_field not in mesh.point_data:
            raise ValueError(f"Velocity field '{velocity_field}' not found in mesh point data.")

        vorticity = mesh.compute_derivative(scalars=velocity_field, vorticity=True).point_data['vorticity']
        mesh.point_data['vorticity'] = vorticity
        return mesh
    
    def export_plot(self, plotter, filename: Path, image_format: str = "png"):
        """
        Export the current plot to an image file.

        Args:
            plotter: The plotting object (ex: pyvista.Plotter).
            filename (Path): Nom du fichier (avec ou sans extension).
            image_format (str): Format de l'image (par défaut 'png').
        """
        filename = Path(filename)
        if filename.suffix != f".{image_format}":
            filename = filename.with_suffix(f".{image_format}")

        plotter.screenshot(str(filename))  # PyVista attend une string
        print(f"Plot exported to {filename}")

    def create_animation(self, scalars: str, filename: str, image_format: str = 'gif', fps: int = 10):
        """
        Create an animation across time steps.
        """
        if not filename.endswith(f'.{image_format}'):
            filename = f"{filename}.{image_format}"

        time_steps = self.list_time_steps()
        if not time_steps:
            raise FileNotFoundError("No VTK files found for animation.")

        pl = pv.Plotter(off_screen=True)
        pl.open_gif(filename, fps=fps)

        for step in time_steps:
            structure = self.load_time_step(step)
            mesh = structure["cell"]
            
            pl.clear()
            pl.add_mesh(
                mesh,
                scalars=scalars,
                lighting=False,
                scalar_bar_args={"title": scalars},
                clim=[mesh.get_data_range(scalars)[0], mesh.get_data_range(scalars)[1]]
            )
            pl.write_frame()
        
        pl.close()
        print(f"Animation saved to {filename}")

    def get_scalar_statistics(self, mesh, scalar_field: str):
        """
        Calculates statistics (mean, std, min, max) for a scalar field.
        """
        if scalar_field not in mesh.point_data:
            raise ValueError(f"Scalar field '{scalar_field}' not found in mesh point data.")
        
        data = mesh.point_data[scalar_field]
        stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        }
        return stats

    def get_time_series_data(self, scalar_field: str, point_coordinates: list):
        """
        Extracts time series data for a scalar field at a specific point.
        """
        time_steps = self.list_time_steps()
        if not time_steps:
            raise FileNotFoundError("No VTK files found for time series analysis.")

        time_series = []
        for step in time_steps:
            structure = self.load_time_step(step)
            mesh = structure["cell"]
            
            closest_point_id = mesh.find_closest_point(point_coordinates)
            
            if scalar_field not in mesh.point_data:
                raise ValueError(f"Scalar field '{scalar_field}' not found in mesh point data for time step {step}.")
            
            time_series.append(mesh.point_data[scalar_field][closest_point_id])
            
        return {"time_steps": time_steps, "data": time_series}

    def get_mesh_statistics(self, mesh):
        """
        Returns statistics about the mesh itself (e.g., number of points, cells).
        """
        stats = {
            "num_points": mesh.n_points,
            "num_cells": mesh.n_cells,
            "bounds": list(mesh.bounds),
            "volume": mesh.volume,
            "area": mesh.area if mesh.n_cells > 0 and mesh.get_cell(0).type == pv.CellType.TRIANGLE else None, # Check if it's a surface mesh for area
        }
        return stats

    def get_region_statistics(self, structure, region_name: str, scalar_field: str):
        """
        Calculates statistics for a scalar field within a specific region (cell or boundary).
        """
        mesh = None
        if region_name == "cell":
            mesh = structure["cell"]
        elif region_name in structure["boundaries"]:
            mesh = structure["boundaries"][region_name]
        else:
            raise ValueError(f"Region '{region_name}' not found.")

        if scalar_field not in mesh.point_data and scalar_field not in mesh.cell_data:
            raise ValueError(f"Scalar field '{scalar_field}' not found in region '{region_name}'.")

        # Get data, prioritizing point data, then cell data
        if scalar_field in mesh.point_data:
            data = mesh.point_data[scalar_field]
        elif scalar_field in mesh.cell_data:
            data = mesh.cell_data[scalar_field]
        else:
            raise ValueError(f"Scalar field '{scalar_field}' not found in region '{region_name}'.")

        stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data)
        }
        
        # Calculate volume-weighted average if it's a volume mesh
        if mesh.volume > 0:
            # Compute cell volumes
            mesh_with_volumes = mesh.compute_cell_sizes(length=False, area=False, volume=True)
            
            # If scalar_field is point data, transfer it to cell data for weighted average
            if scalar_field in mesh.point_data:
                cell_data_from_points = mesh_with_volumes.point_data_to_cell_data([scalar_field])
                data_for_weighted_avg = cell_data_from_points.cell_data[scalar_field]
            elif scalar_field in mesh.cell_data:
                data_for_weighted_avg = mesh.cell_data[scalar_field]
            else:
                data_for_weighted_avg = None # Should not happen due to prior checks

            if data_for_weighted_avg is not None and 'Volume' in mesh_with_volumes.cell_data:
                # Ensure data_for_weighted_avg is 1D for np.average
                if data_for_weighted_avg.ndim > 1:
                    # If it's a vector field, average each component separately
                    # Check if weights sum to zero before attempting to normalize
                    if np.sum(mesh_with_volumes.cell_data['Volume']) == 0:
                        weighted_means = [0.0] * data_for_weighted_avg.shape[1] # Assign 0 if weights sum to 0
                    else:
                        weighted_means = [np.average(data_for_weighted_avg[:, i], weights=mesh_with_volumes.cell_data['Volume']) for i in range(data_for_weighted_avg.shape[1])]
                    stats["volume_weighted_mean"] = weighted_means
                else:
                    if np.sum(mesh_with_volumes.cell_data['Volume']) == 0:
                        stats["volume_weighted_mean"] = 0.0 # Assign 0 if weights sum to 0
                    else:
                        stats["volume_weighted_mean"] = np.average(data_for_weighted_avg, weights=mesh_with_volumes.cell_data['Volume'])
            else:
                stats["volume_weighted_mean"] = None
        else:
            stats["volume_weighted_mean"] = None

        return stats

    def export_region_data_to_csv(self, structure, region_name: str, scalar_fields: list, output_filename: Path):
        """
        Exports XYZ coordinates and specified scalar field values for a given region to a CSV file.

        Args:
            structure: Dictionnaire contenant le maillage et les régions.
            region_name (str): Nom de la région (par ex. "cell" ou "boundary").
            scalar_fields (list): Champs scalaires à exporter.
            output_filename (Path): Chemin du fichier de sortie (csv).
        """
        output_filename = Path(output_filename)
        output_filename.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier si nécessaire
        if output_filename.suffix != ".csv":
            output_filename = output_filename.with_suffix(".csv")

        if region_name == "cell":
            mesh = structure["cell"]
        elif region_name in structure["boundaries"]:
            mesh = structure["boundaries"][region_name]
        else:
            raise ValueError(f"Region '{region_name}' not found.")

        data_to_export = {
            'X': mesh.points[:, 0],
            'Y': mesh.points[:, 1],
            'Z': mesh.points[:, 2]
        }

        for field in scalar_fields:
            if field not in mesh.point_data:
                raise ValueError(f"Scalar field '{field}' not found in region '{region_name}'.")
            
            field_data = mesh.point_data[field]
            if field_data.ndim > 1:  # Handle vector fields
                for i in range(field_data.shape[1]):
                    data_to_export[f'{field}_{i}'] = field_data[:, i]
            else:
                data_to_export[field] = field_data
        
        df = pd.DataFrame(data_to_export)
        df.to_csv(output_filename, index=False)
        print(f"Data for region '{region_name}' exported to {output_filename}")


    def export_statistics_to_json(self, stats_data: dict, output_filename: Path):
        """
        Exports statistical data to a JSON file.

        Args:
            stats_data (dict): Dictionnaire avec les statistiques.
            output_filename (Path): Chemin du fichier de sortie (json).
        """
        output_filename = Path(output_filename)
        output_filename.parent.mkdir(parents=True, exist_ok=True)  # crée le dossier si nécessaire
        if output_filename.suffix != ".json":
            output_filename = output_filename.with_suffix(".json")

        with open(output_filename, "w") as f:
            json.dump(stats_data, f, indent=4, cls=NumpyEncoder)  # Custom encoder for numpy types
        print(f"Statistics exported to {output_filename}")


# Custom JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)






