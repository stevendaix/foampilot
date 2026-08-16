import subprocess
import logging
from pathlib import Path
from typing import Optional, List
import pyvista as pv
import numpy as np
import pandas as pd
import json
from pyvirtualdisplay import Display

logger = logging.getLogger(__name__)


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

        logger.info("Running foamToVTK with command: %s", " ".join(cmd))
        try:
            subprocess.run(cmd, text=True, capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"foamToVTK failed:\n{e.stderr or e.stdout}")
    def read_direct(
        self,
        field_names: Optional[List[str]] = None,
        cache_directory: str = "cache",
    ) -> pv.MultiBlock:
        """Read OpenFOAM case data directly using POpenFOAMReader,
        avoiding the intermediate foamToVTK step.

        Args:
            field_names: List of field names to read (e.g., ['U', 'p', 'k']).
                If None, all fields are loaded.
            cache_directory: Sub-directory for PyVista cache.

        Returns:
            A PyVista MultiBlock containing the case data.
        """
        self.check_case()
        reader = pv.POpenFOAMReader(str(self.case_path))
        if field_names:
            for field in field_names:
                reader.set_active_scalars(field)
        return reader.read()

    def calc_y_plus(self, mesh: pv.DataSet, wall_patch_name: str = "walls", velocity_field: str = "U", viscosity: float = 1e-5) -> pv.DataSet:
        """Calculate y+ wall distance for a mesh.

        Args:
            mesh: The PyVista mesh object.
            wall_patch_name: Name of the wall patch.
            velocity_field: Velocity field name.
            viscosity: Kinematic viscosity (nu).

        Returns:
            Mesh with y_plus point data added.
        """
        if velocity_field not in mesh.point_data:
            raise ValueError(f"Velocity field '{velocity_field}' not found in mesh point data.")

        # Compute the magnitude of velocity
        vel = mesh.point_data[velocity_field]
        if vel.ndim == 1:
            u_mag = np.abs(vel)
        else:
            u_mag = np.linalg.norm(vel, axis=1)

        # Simplified y+ estimation using first cell center distance
        # In practice, this requires wall-normal distance computation
        y_plus = np.zeros(mesh.n_points)
        points = mesh.points
        if len(points) > 0:
            distances = np.linalg.norm(points - mesh.bounds[:3], axis=1)
            u_tau = np.sqrt(0.5 * viscosity * u_mag / (distances + 1e-10))
            y_plus = u_mag * distances / (u_tau + 1e-10)

        mesh.point_data["y_plus"] = y_plus
        return mesh

    def calc_strain_rate(self, mesh: pv.DataSet, velocity_field: str = "U") -> pv.DataSet:
        """Calculate the strain rate tensor magnitude from velocity field.

        Args:
            mesh: The PyVista mesh object.
            velocity_field: Velocity field name.

        Returns:
            Mesh with strain_rate point data added.
        """
        if velocity_field not in mesh.point_data:
            raise ValueError(f"Velocity field '{velocity_field}' not found in mesh point data.")

        gradient = mesh.compute_derivative(scalars=velocity_field).point_data["gradient"]
        grad_u = gradient.reshape(-1, 3, 3)
        S = 0.5 * (grad_u + np.transpose(grad_u, (0, 2, 1)))
        strain_rate = np.linalg.norm(S, axis=(1, 2))
        mesh.point_data["strain_rate"] = strain_rate
        return mesh

    def calc_wall_shear_stress(
        self,
        mesh: pv.DataSet,
        velocity_field: str = "U",
        viscosity: float = 1e-5,
        wall_normal: list = None,
    ) -> pv.DataSet:
        """Calculate wall shear stress magnitude.

        Args:
            mesh: The PyVista mesh object.
            velocity_field: Velocity field name.
            viscosity: Dynamic viscosity (mu).
            wall_normal: Wall normal vector (default: [0, 0, 1]).

        Returns:
            Mesh with wall_shear_stress point data added.
        """
        if velocity_field not in mesh.point_data:
            raise ValueError(f"Velocity field '{velocity_field}' not found in mesh point data.")

        gradient = mesh.compute_derivative(scalars=velocity_field).point_data["gradient"]
        grad_u = gradient.reshape(-1, 3, 3)
        if wall_normal is None:
            n = np.array([[0, 0, 1]], dtype=float)
        else:
            n = np.array([wall_normal], dtype=float)
        tau_w = viscosity * (grad_u @ n[..., np.newaxis]).squeeze()
        mesh.point_data["wall_shear_stress"] = np.linalg.norm(tau_w, axis=1)
        return mesh

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
                    logger.warning(f"Boundary file not found: {b_file}")

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

    def plot_slice(self, structure=None, plane="z", scalars="U", opacity=0.25, path_filename=None):
        """
        Generate a slice plot from the given structure dictionary.
        """
        if structure is None:
            raise RuntimeError("No structure provided. Run get_structure() first.")

        # Détermine si on est en mode off_screen
        off_screen = path_filename is not None

        y_slice = structure["cell"].slice(plane)
        pl = pv.Plotter(off_screen=off_screen)
        pl.set_background("white")

        pl.add_mesh(
            y_slice,
            scalars=scalars,
            lighting=False,
            scalar_bar_args={"title": scalars},
        )

        # Add full cell mesh as wireframe context (transparency breaks on some VTK versions)
        pl.add_mesh(structure["cell"], style="wireframe", color="lightgray", line_width=0.5)
        for name, mesh in structure.get("boundaries", {}).items():
            pl.add_mesh(mesh, color="black", style="wireframe", line_width=1)

        # Use a standard orthogonal view so the full domain is visible
        # (reset_camera alone can leave a cut-away appearance for thin slices)
        pl.reset_camera()
        view_map = {"x": "yz", "y": "xz", "z": "xy"}
        pl.camera_position = view_map.get(plane, "xy")

        if path_filename is not None:
            pl.render()
            pl.screenshot(path_filename)
            logger.info(f"Image sauvegardée : {path_filename}")
        else:
            # Affiche le rendu à l'écran
            pl.show()
        return pl


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

        Args:
            mesh: The PyVista mesh object.
            velocity_field: Velocity field name.

        Returns:
            Mesh with vorticity point data added.
        """
        if velocity_field not in mesh.point_data:
            raise ValueError(f"Velocity field '{velocity_field}' not found in mesh point data.")

        vorticity = mesh.compute_derivative(scalars=velocity_field, vorticity=True).point_data['vorticity']
        mesh.point_data['vorticity'] = vorticity
        return mesh

    def calc_temperature_gradient(
        self,
        mesh: pv.DataSet,
        temperature_field: str = 'T',
    ) -> pv.DataSet:
        """
        Calculate the temperature gradient vector field.

        Args:
            mesh: The PyVista mesh object.
            temperature_field: Temperature field name (default: 'T').

        Returns:
            Mesh with temperature_gradient point data added (shape: n_points, 3).
        """
        if temperature_field not in mesh.point_data:
            raise ValueError(f"Temperature field '{temperature_field}' not found in mesh point data.")

        grad_data = mesh.compute_derivative(scalars=temperature_field, gradient=True)
        gradient = grad_data.point_data['gradient']
        mesh.point_data['temperature_gradient'] = gradient
        return mesh

    def calc_heat_flux(
        self,
        mesh: pv.DataSet,
        temperature_field: str = 'T',
        thermal_conductivity: float = 0.026,
    ) -> pv.DataSet:
        """
        Calculate the heat flux vector field from Fourier's law: q = -k * grad(T).

        Args:
            mesh: The PyVista mesh object.
            temperature_field: Temperature field name (default: 'T').
            thermal_conductivity: Thermal conductivity k in W/(m·K) (default: 0.026 for air).

        Returns:
            Mesh with heat_flux point data added (shape: n_points, 3).
        """
        self.calc_temperature_gradient(mesh, temperature_field=temperature_field)
        gradient = mesh.point_data['temperature_gradient']
        heat_flux = -thermal_conductivity * gradient
        mesh.point_data['heat_flux'] = heat_flux
        mesh.point_data['heat_flux_magnitude'] = np.linalg.norm(heat_flux, axis=1)
        return mesh

    def calc_wall_heat_flux(
        self,
        mesh: pv.DataSet,
        temperature_field: str = 'T',
        thermal_conductivity: float = 0.026,
        wall_patch_name: str = 'walls',
    ) -> pv.DataSet:
        """
        Calculate the wall heat flux magnitude from the temperature gradient normal to the wall.

        Uses Fourier's law: q_wall = -k * (dT/dn) where dT/dn is the gradient normal to the wall.

        Args:
            mesh: The PyVista mesh object with wall patch data.
            temperature_field: Temperature field name (default: 'T').
            thermal_conductivity: Thermal conductivity k in W/(m·K).
            wall_patch_name: Name of the wall patch.

        Returns:
            Mesh with wall_heat_flux point data added (scalar field).
        """
        if wall_patch_name not in mesh.cell_data and wall_patch_name not in mesh.point_data:
            raise ValueError(f"Wall patch '{wall_patch_name}' not found in mesh data.")

        self.calc_temperature_gradient(mesh, temperature_field=temperature_field)
        gradient = mesh.point_data['temperature_gradient']

        if wall_patch_name in mesh.cell_data:
            wall_mask = mesh.cell_data[wall_patch_name].astype(bool)
            normals = mesh.cell_normals
        else:
            wall_mask = mesh.point_data[wall_patch_name].astype(bool)
            normals = mesh.point_normals

        if normals is None:
            normals = mesh.compute_normals().point_data['Normals']

        dTdn = np.einsum('ij,ij->i', gradient, normals)
        wall_heat_flux = -thermal_conductivity * dTdn

        if wall_patch_name in mesh.cell_data:
            mesh.cell_data['wall_heat_flux'] = wall_heat_flux
        else:
            mesh.point_data['wall_heat_flux'] = wall_heat_flux

        return mesh

    def calc_nusselt_number(
        self,
        mesh: pv.DataSet,
        temperature_field: str = 'T',
        thermal_conductivity: float = 0.026,
        reference_temperature: float = 300.0,
        characteristic_length: float = 1.0,
        wall_patch_name: str = 'walls',
    ) -> pv.DataSet:
        """
        Calculate the local Nusselt number at a wall.

        Nu = (q_wall * L) / (k * (T_wall - T_ref))

        Args:
            mesh: The PyVista mesh object.
            temperature_field: Temperature field name (default: 'T').
            thermal_conductivity: Thermal conductivity k in W/(m·K).
            reference_temperature: Reference (free-stream) temperature T_inf in K.
            characteristic_length: Characteristic length L in m.
            wall_patch_name: Name of the wall patch.

        Returns:
            Mesh with nusselt_number data added (scalar field).
        """
        self.calc_wall_heat_flux(
            mesh,
            temperature_field=temperature_field,
            thermal_conductivity=thermal_conductivity,
            wall_patch_name=wall_patch_name,
        )

        heat_flux_key = (
            'wall_heat_flux' if wall_patch_name in mesh.cell_data else 'wall_heat_flux'
        )
        q_wall = mesh.cell_data.get(heat_flux_key) or mesh.point_data.get(heat_flux_key)
        if q_wall is None:
            raise ValueError("Wall heat flux data not found. Run calc_wall_heat_flux() first.")

        T_wall = mesh.point_data.get(temperature_field)
        if T_wall is None:
            raise ValueError(f"Temperature field '{temperature_field}' not found in mesh point data.")

        delta_T = T_wall - reference_temperature
        delta_T_safe = np.where(np.abs(delta_T) < 1e-10, 1e-10, delta_T)

        nusselt = np.abs(q_wall) * characteristic_length / (thermal_conductivity * np.abs(delta_T_safe))

        if wall_patch_name in mesh.cell_data:
            mesh.cell_data['nusselt_number'] = nusselt
        else:
            mesh.point_data['nusselt_number'] = nusselt

        return mesh

    def calc_thermal_boundary_layer_thickness(
        self,
        mesh: pv.DataSet,
        temperature_field: str = 'T',
        reference_temperature: float = 300.0,
        threshold: float = 0.99,
        direction: str = 'x',
    ) -> float:
        """
        Estimate the thermal boundary layer thickness.

        The thickness is defined as the distance from the wall where the
        temperature reaches threshold * (T_wall - T_ref) + T_ref.

        Args:
            mesh: The PyVista mesh object.
            temperature_field: Temperature field name (default: 'T').
            reference_temperature: Free-stream temperature T_inf.
            threshold: Fraction of (T_wall - T_ref) to define the boundary layer edge (default: 0.99).
            direction: Direction normal to the wall ('x', 'y', or 'z').

        Returns:
            Estimated thermal boundary layer thickness in meters.
        """
        if temperature_field not in mesh.point_data:
            raise ValueError(f"Temperature field '{temperature_field}' not found in mesh point data.")

        T = mesh.point_data[temperature_field]
        T_ref = reference_temperature
        T_wall = np.max(T)
        T_edge = T_wall - threshold * (T_wall - T_ref)

        if direction == 'x':
            coords = mesh.points[:, 0]
        elif direction == 'y':
            coords = mesh.points[:, 1]
        elif direction == 'z':
            coords = mesh.points[:, 2]
        else:
            raise ValueError(f"Invalid direction '{direction}'. Use 'x', 'y', or 'z'.")

        wall_points = coords[T > T_edge]
        edge_points = coords[T <= T_edge]

        if len(wall_points) == 0 or len(edge_points) == 0:
            return 0.0

        delta = np.min(np.abs(wall_points[:, np.newaxis] - edge_points[np.newaxis, :]))
        return float(delta)
    
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

        plotter.render()
        plotter.screenshot(str(filename))  # PyVista attend une string
        logger.info(f"Plot exported to {filename}")

    def create_animation(self, scalars: str, filename: Path, image_format: str = 'gif', fps: int = 10):
        """
        Create an animation across time steps.
        """
        if filename.suffix != f'.{image_format}':
            filename = filename.with_suffix(f'.{image_format}')

        time_steps = self.list_time_steps()
        if not time_steps:
            raise FileNotFoundError("No VTK files found for animation.")

        pl = pv.Plotter(off_screen=True)
        pl.open_gif(str(filename), fps=fps)

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
        logger.info(f"Animation saved to {filename}")

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
                cell_data_from_points = mesh_with_volumes.point_data_to_cell_data(pass_point_data=False)
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
        logger.info(f"Data for region '{region_name}' exported to {output_filename}")


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
        logger.info(f"Statistics exported to {output_filename}")

    # ------------------------------------------------------------------
    # CHT post-processing
    # ------------------------------------------------------------------

    def calc_region_heat_flux(
        self,
        mesh: pv.DataSet,
        temperature_field: str = "T",
        thermal_conductivity: float = 0.026,
    ) -> pv.DataSet:
        """Calculate heat flux magnitude from temperature gradient
        using Fourier's law: q = -k * grad(T).

        Args:
            mesh: The PyVista mesh object with point data temperature.
            temperature_field: Temperature field name (default: "T").
            thermal_conductivity: Thermal conductivity k in W/(m·K).

        Returns:
            Mesh with heat_flux_magnitude cell data added.
        """
        if temperature_field not in mesh.point_data:
            raise ValueError(
                f"Temperature field '{temperature_field}' not found."
            )

        grad_data = mesh.compute_derivative(
            scalars=temperature_field, gradient=True
        )
        gradient = grad_data.point_data["gradient"]
        heat_flux = -thermal_conductivity * gradient
        q_mag = np.linalg.norm(heat_flux, axis=1)
        mesh.cell_data["heat_flux_magnitude"] = q_mag
        return mesh

    def calc_interface_heat_flux(
        self,
        fluid_mesh: pv.DataSet,
        solid_mesh: pv.DataSet,
        T_fluid_field: str = "T",
        T_solid_field: str = "T",
        h: float = 10.0,
        area: float = 1.0,
    ) -> dict:
        """Calculate heat flux across a fluid-solid interface.

        Args:
            fluid_mesh: PyVista mesh on the fluid side.
            solid_mesh: PyVista mesh on the solid side.
            T_fluid_field: Temperature field on fluid side.
            T_solid_field: Temperature field on solid side.
            h: Heat transfer coefficient in W/(m²·K).
            area: Interface area in m².

        Returns:
            Dictionary with total, convective, and conductive heat
            fluxes and the interface temperature.
        """
        T_f = np.mean(fluid_mesh.point_data[T_fluid_field])
        T_s = np.mean(solid_mesh.point_data[T_solid_field])
        T_interface = (T_f + T_s) / 2.0

        q_conv = h * (T_s - T_f) * area

        grad_f = fluid_mesh.compute_derivative(
            scalars=T_fluid_field
        ).point_data["gradient"]
        q_cond_f = np.mean(
            -0.026 * np.linalg.norm(grad_f, axis=1)
        ) * area

        grad_s = solid_mesh.compute_derivative(
            scalars=T_solid_field
        ).point_data["gradient"]
        q_cond_s = np.mean(
            -50.0 * np.linalg.norm(grad_s, axis=1)
        ) * area

        return {
            "q_total": float(q_conv + q_cond_f + q_cond_s),
            "q_conv": float(q_conv),
            "q_cond_fluid": float(q_cond_f),
            "q_cond_solid": float(q_cond_s),
            "T_interface": float(T_interface),
        }

    def calc_nusselt_number(
        self,
        q_wall: float,
        L: float,
        k_fluid: float,
        T_wall: float,
        T_bulk: float,
    ) -> float:
        """Calculate Nusselt number from wall heat flux.

        Nu = q_wall * L / (k_fluid * |T_wall - T_bulk|)

        Args:
            q_wall: Wall heat flux in W/m².
            L: Characteristic length in m.
            k_fluid: Fluid thermal conductivity in W/(m·K).
            T_wall: Wall temperature in K.
            T_bulk: Bulk fluid temperature in K.

        Returns:
            Nusselt number (dimensionless).
        """
        delta_T = T_wall - T_bulk
        if abs(delta_T) < 1e-10:
            return 0.0
        return abs(q_wall) * L / (k_fluid * abs(delta_T))

    def calc_thermal_boundary_layer_thickness(
        self,
        T_wall: float,
        T_bulk: float,
        T_field: np.ndarray,
        x_positions: np.ndarray,
        threshold: float = 0.99,
    ) -> float:
        """Estimate thermal boundary layer thickness.

        Args:
            T_wall: Wall temperature in K.
            T_bulk: Bulk fluid temperature in K.
            T_field: Temperature field along wall-normal direction.
            x_positions: Position array corresponding to T_field.
            threshold: Fraction defining the boundary layer edge.

        Returns:
            Thermal boundary layer thickness in meters.
        """
        T_edge = T_wall - threshold * (T_wall - T_bulk)
        wall_pts = x_positions[T_field > T_edge]
        edge_pts = x_positions[T_field <= T_edge]
        if len(wall_pts) == 0 or len(edge_pts) == 0:
            return 0.0
        return float(
            np.min(
                np.abs(
                    wall_pts[:, np.newaxis] - edge_pts[np.newaxis, :]
                )
            )
        )


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






