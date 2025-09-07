import subprocess
from pathlib import Path
import pyvista as pv

class FoamPostProcessing:
    def __init__(self, case_path: str, vtk_dir: str = "VTK"):
        self.case_path = Path(case_path)
        self.vtk_dir = self.case_path / vtk_dir

    def check_case(self):
        if not self.case_path.exists() or not self.case_path.is_dir():
            raise FileNotFoundError(f"OpenFOAM case path '{self.case_path}' does not exist or is not a directory.")

    def foamToVTK(self, 
                  all_regions=False,
                  ascii=False,
                  constant=False,
                  latest_time=False,
                  fields=None,
                  no_boundary=False,
                  no_internal=False):
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

        if no_boundary:
            cmd.append("-no-boundary")
        if no_internal:
            cmd.append("-no-internal")

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

    def get_structure(self, time_step=None, boundaries=["inlet","outlet","walls"]):
        """
        Returns a dictionary structure with the main cell mesh and boundary meshes.
        """
        if time_step is None:
            # choisir le dernier pas disponible
            steps = self.list_time_steps()
            if not steps:
                raise FileNotFoundError("No VTK files found in directory.")
            time_step = steps[-1]

        structure = {}
        # Cell mesh
        cell_file = self.vtk_dir / f"{self.case_path.name}_{time_step}.vtk"
        if not cell_file.exists():
            raise FileNotFoundError(f"Cell file not found: {cell_file}")
        structure['cell'] = pv.read(cell_file)

        # Boundaries
        structure['boundaries'] = {}
        for b in boundaries:
            b_file = self.vtk_dir / b / f"{b}_{time_step}.vtk"
            if b_file.exists():
                structure['boundaries'][b] = pv.read(b_file)
            else:
                print(f"Boundary file not found: {b_file}")

        return structure

    def plot_slice(self, structure=None, plane='z', scalars='U', opacity=0.25):
        """
        Generate a slice plot from the given structure dictionary.
        """
        if structure is None:
            raise RuntimeError("No structure provided. Run get_structure() first.")

        y_slice = structure["cell"].slice(plane)
        pl = pv.Plotter()
        pl.add_mesh(y_slice, scalars=scalars, lighting=False, scalar_bar_args={'title': scalars})
        pl.add_mesh(structure["cell"], color='w', opacity=opacity)
        for name, mesh in structure.get("boundaries", {}).items():
            pl.add_mesh(mesh, opacity=0.5)
        pl.enable_anti_aliasing()
        pl.show()
