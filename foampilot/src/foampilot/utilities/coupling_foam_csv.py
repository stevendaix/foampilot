import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree


class CSVFoamIntegrator:
    def __init__(self, case_path: str):
        self.case_path = Path(case_path)
        self.poly_mesh = self.case_path / "constant" / "polyMesh"
        self._points = None
        self._faces = None
        self._boundary = None

    # ---------- Mesh Reading (Direct Method) ----------

    def _read_points(self):
        path = self.poly_mesh / "points"
        with open(path, "r") as f:
            lines = f.readlines()
        start = lines.index("(\n") + 1
        end = lines.index(")\n")
        points = []
        for line in lines[start:end]:
            x, y, z = line.strip("()\n ").split()
            points.append([float(x), float(y), float(z)])
        return np.asarray(points)

    def _read_faces(self):
        path = self.poly_mesh / "faces"
        with open(path, "r") as f:
            lines = f.readlines()
        start = lines.index("(\n") + 1
        end = lines.index(")\n")
        faces = []
        for line in lines[start:end]:
            tokens = line.strip("()\n ").split()
            n = int(tokens[0])
            faces.append([int(i) for i in tokens[1:1 + n]])
        return faces

    def _read_boundary(self):
        path = self.poly_mesh / "boundary"
        with open(path, "r") as f:
            lines = f.readlines()
        boundary = {}
        i = lines.index("(\n") + 1
        while not lines[i].startswith(")"):
            patch = lines[i].strip()
            i += 2  # skip "{"
            entry = {}
            while not lines[i].startswith("}"):
                parts = lines[i].strip(";\n").split()
                if len(parts) == 2:
                    key, value = parts
                    entry[key] = int(value) if value.isdigit() else value
                i += 1
            boundary[patch] = entry
            i += 1
        return boundary

    def load_mesh(self):
        """Load mesh data into memory."""
        self._points = self._read_points()
        self._faces = self._read_faces()
        self._boundary = self._read_boundary()
        print(f"Mesh loaded: {len(self._points)} points, {len(self._faces)} faces.")

    def get_patch_dataframe(self, patch_name):
        """Get a DataFrame of face centers for a specific patch."""
        if self._boundary is None: self.load_mesh()
        if patch_name not in self._boundary:
            raise ValueError(f"Patch {patch_name} not found.")
        
        info = self._boundary[patch_name]
        start, nfaces = info["startFace"], info["nFaces"]
        rows = []
        for local_id, face_id in enumerate(range(start, start + nfaces)):
            center = np.mean(self._points[self._faces[face_id]], axis=0)
            rows.append({"localId": local_id, "x": center[0], "y": center[1], "z": center[2]})
        return pd.DataFrame(rows)

    # ---------- Data Mapping ----------

    def map_csv_to_patch(self, df_patch, df_csv):
        """Map CSV data to patch faces using nearest neighbor."""
        tree = cKDTree(df_csv[['x', 'y', 'z']].values)
        _, indices = tree.query(df_patch[['x', 'y', 'z']].values)
        return df_csv.iloc[indices].reset_index(drop=True)

    # ---------- BC Generation Methods ----------

    def write_nonuniform_bc(self, patch_name, df_mapped, fields_config):
        """
        Generate a 'nonuniform List' BC block.
        fields_config: dict like {'Ta': 'scalar', 'h': 'scalar'}
        """
        bc_str = f"    {patch_name}\n    {{\n"
        bc_str += "        type            externalWallHeatFluxTemperature;\n"
        bc_str += "        mode            coefficient;\n"
        
        for field, ftype in fields_config.items():
            values = df_mapped[field].values
            bc_str += f"        {field}            nonuniform List<{ftype}> {len(values)}\n        (\n"
            bc_str += "\n".join([f"            {v:g}" for v in values]) + "\n        );\n"
            
        bc_str += "        kappaMethod     fluidThermo;\n"
        bc_str += "        value           uniform 300;\n"
        bc_str += "    }\n"
        return bc_str

    def export_to_boundary_data(self, patch_name, df_mapped, field_name, time="0"):
        """Export to constant/boundaryData format (Spatio-Temporal)."""
        # Prepare df for the existing script
        df_mapped['time'] = time
        df_mapped['x'] = self.get_patch_dataframe(patch_name)['x'].values
        df_mapped['y'] = self.get_patch_dataframe(patch_name)['y'].values
        df_mapped['z'] = self.get_patch_dataframe(patch_name)['z'].values
        
        df_to_boundary_data_timeseries(df_mapped, patch_name, field_name, output_dir=str(self.case_path / "constant/boundaryData"))

# --- Example Usage ---
if __name__ == "__main__":
    # integrator = ManusFoamIntegrator(".")
    # df_patch = integrator.get_patch_dataframe("wall1")
    # df_csv = pd.read_csv("data.csv")
    # df_mapped = integrator.map_csv_to_patch(df_patch, df_csv)
    # print(integrator.write_nonuniform_bc("wall1", df_mapped, {'Ta': 'scalar', 'h': 'scalar'}))
    pass



def write_openfoam_header(f, class_name, object_name, location):
    header = f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    location    "{location}";
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

"""
    f.write(header)

def df_to_boundary_data_timeseries(df, patch_name, field_name, time_col='time', output_dir="constant/boundaryData"):
    """
    Convert a DataFrame with multiple time steps to OpenFOAM boundaryData format.
    The DataFrame should have columns: [time, x, y, z, value] or [time, x, y, z, vx, vy, vz]
    """
    patch_dir = os.path.join(output_dir, patch_name)
    os.makedirs(patch_dir, exist_ok=True)

    # Get unique time steps
    times = sorted(df[time_col].unique())
    
    # Check if it's a vector field
    is_vector = all(c in df.columns for c in ['vx', 'vy', 'vz'])
    class_name = "vectorField" if is_vector else "scalarField"

    for t in times:
        # Format time string (OpenFOAM prefers no trailing zeros if integer, but handles float)
        t_str = f"{t:g}"
        time_dir = os.path.join(patch_dir, t_str)
        os.makedirs(time_dir, exist_ok=True)
        
        df_t = df[df[time_col] == t]
        
        # 1. Write points file for this time step
        # Note: points can be the same for all times, but OpenFOAM allows them to be in each time dir
        points_path = os.path.join(time_dir, "points")
        with open(points_path, "w") as f:
            write_openfoam_header(f, "vectorField", "points", location=f"{output_dir}/{patch_name}/{t_str}")
            f.write(f"{len(df_t)}\n(\n")
            for _, row in df_t.iterrows():
                f.write(f"({row['x']} {row['y']} {row['z']})\n")
            f.write(")\n\n// ************************************************************************* //\n")

        # 2. Write field file
        field_path = os.path.join(time_dir, field_name)
        with open(field_path, "w") as f:
            write_openfoam_header(f, class_name, field_name, location=f"{output_dir}/{patch_name}/{t_str}")
            f.write(f"{len(df_t)}\n(\n")
            for _, row in df_t.iterrows():
                if is_vector:
                    f.write(f"({row['vx']} {row['vy']} {row['vz']})\n")
                else:
                    f.write(f"{row[field_name]}\n")
            f.write(")\n\n// ************************************************************************* //\n")
        
        print(f"Wrote data for time {t_str} to {time_dir}")
