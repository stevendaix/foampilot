
from pathlib import Path
import numpy as np
import pyvista as pv
import pandas as pd

LAWSON_THRESHOLDS = {
    "sitting": 4.0,
    "standing": 6.0,
    "walking": 8.0,
    "unsafe": 15.0,
}

LAWSON_MAX_PROBABILITY = 0.05
PEDESTRIAN_HEIGHT = 1.75

class WindRose:
    """
    Rose des vents météorologique.
    """
    def __init__(self, data: dict):
        """
        data = {
            direction_deg: [
                {"speed": 5.0, "frequency": 0.12},
                {"speed": 8.0, "frequency": 0.04}
            ]
        }
        """
        self.data = data


def angular_distance(a: float, b: float) -> float:
    """
    Distance angulaire minimale entre deux directions (en degrés).
    """
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)

class WindCaseResult:
    def __init__(
        self,
        post: "FoamPostProcessing",  # noqa: F821 — forward reference string
        direction_deg: float,
        u_ref: float = 10.0,
        field_name: str = "U",
        pedestrian_height: float = PEDESTRIAN_HEIGHT,
        resolution: int = 300,
    ):
        self.post = post
        self.direction_deg = direction_deg
        self.u_ref = u_ref
        self.field_name = field_name
        self.pedestrian_height = pedestrian_height
        self.resolution = resolution

        self.mesh = None
        self.plane = None
        self.sensitivity = None
        self.structure = None
        self.cell_mesh = None

    def extract_pedestrian_plane(self):
        structure = self.post.get_structure()
        self.structure = structure
        mesh = structure["cell"]
        self.cell_mesh = mesh

        bounds = mesh.bounds
        plane = pv.Plane(
            center=((bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    self.pedestrian_height),
            direction=(0, 0, 1),
            i_size=bounds[1] - bounds[0],
            j_size=bounds[3] - bounds[2],
            i_resolution=self.resolution,
            j_resolution=self.resolution,
        )

        sampled = plane.sample(mesh)
        self.mesh = sampled
        self.plane = plane

    def compute_sensitivity(self):
        U = self.mesh.point_data.get(self.field_name)
        if U is None:
            U = self.mesh.cell_data.get(self.field_name)
            if U is not None:
                self.mesh = self.mesh.cell_data_to_point_data()
                U = self.mesh.point_data[self.field_name]
        U_mag = np.linalg.norm(U, axis=1)
        self.sensitivity = U_mag / self.u_ref
        self.mesh["S"] = self.sensitivity

    def get_boundary(self, name: str):
        """Return the boundary mesh for patch *name*, or None if unavailable."""
        boundaries = self.structure.get("boundaries", {}) if self.structure else {}
        return boundaries.get(name)

    def get_cell_mesh(self):
        """Return the original (un-sampled) cell mesh from the latest time step."""
        if self.cell_mesh is None:
            self.extract_pedestrian_plane()
        return self.cell_mesh

    def run(self):
        self.extract_pedestrian_plane()
        self.compute_sensitivity()
        return self.mesh


class WindEnsemble:
    def __init__(self):
        self.cases = {}

    def add_case(self, direction_deg: float, case_result: WindCaseResult):
        self.cases[direction_deg] = case_result

    def run_all(self):
        results = {}
        for direction, case in self.cases.items():
            results[direction] = case.run()
        return results

    def compute_case_metrics(self, rho: float = 1.225):
        """Compute per-case engineering metrics for wind-rose aggregation.

        Returns a pandas DataFrame indexed by direction with columns:
          - mean_cp_buildings: mean pressure coefficient on building walls
          - max_velocity_street: max \|U\| at pedestrian height between buildings
          - mean_velocity_street: mean \|U\| at pedestrian height between buildings
          - comfort_index: 1 - mean Lawson walking probability (higher = better)
        """

        records = []
        for direction, case in sorted(self.cases.items()):
            cell_mesh = case.get_cell_mesh()
            boundaries = case.structure.get("boundaries", {}) if case.structure else {}

            # --- Cp moyen sur les bâtiments ---
            mean_cp = np.nan
            build_mesh = boundaries.get("buildings")
            if build_mesh is not None and "p" in build_mesh.point_data:
                p = build_mesh.point_data["p"]
                u_ref = case.u_ref
                p_ref = 0.5 * rho * u_ref ** 2
                cp = p / p_ref
                build_mesh["Cp"] = cp
                mean_cp = float(np.mean(cp))

            # --- Vitesse entre les bâtiments (plan piéton) ---
            max_vel = 0.0
            mean_vel = 0.0
            ped_slice = cell_mesh.slice(normal="z", origin=(0, 0, PEDESTRIAN_HEIGHT))
            if ped_slice.n_points > 0 and "U" in ped_slice.point_data:
                U = ped_slice.point_data["U"]
                U_mag = np.linalg.norm(U, axis=1)
                max_vel = float(np.max(U_mag))
                mean_vel = float(np.mean(U_mag))

            records.append({
                "direction": direction,
                "mean_cp_buildings": mean_cp,
                "max_velocity_street": max_vel,
                "mean_velocity_street": mean_vel,
            })

        df = pd.DataFrame(records).set_index("direction")
        df.index.name = "direction_deg"
        return df




            

class LawsonProcessor:
    def __init__(
        self,
        ensemble: WindEnsemble,
        wind_rose: WindRose,
        sector_half_width: float = 0.0,
    ):
        self.ensemble = ensemble
        self.wind_rose = wind_rose
        self.sector_half_width = sector_half_width

    def _wind_conditions_for_direction(self, direction_deg: float):
        """
        Regroupe les conditions météo dans le secteur angulaire
        centré sur direction_deg.
        """
        conditions = []

        for wd, wc_list in self.wind_rose.data.items():
            if angular_distance(wd, direction_deg) <= self.sector_half_width:
                conditions.extend(wc_list)

        return conditions

    def compute_probability_map(self, threshold: float):
        prob = None

        for direction, case in self.ensemble.cases.items():
            S = case.sensitivity

            wind_conditions = self._wind_conditions_for_direction(direction)

            for wc in wind_conditions:
                u_eq = S * wc["speed"]
                exceed = (u_eq > threshold).astype(float) * wc["frequency"]

                if prob is None:
                    prob = exceed.copy()
                else:
                    prob += exceed

        if prob is not None:
            prob = np.clip(prob, 0.0, LAWSON_MAX_PROBABILITY)

        return prob

    def compute_lawson_maps(self):
        maps = {}

        for label, threshold in LAWSON_THRESHOLDS.items():
            maps[label] = self.compute_probability_map(threshold)

        return maps

class LawsonVisualizer:
    def __init__(self, reference_mesh: pv.PolyData):
        self.mesh = reference_mesh.copy()
        self._grid_shape = None

    def add_probability_field(self, name: str, data: np.ndarray):
        self.mesh[name] = data
        if self._grid_shape is None and hasattr(self.mesh, 'dimensions'):
            self._grid_shape = self.mesh.dimensions

    def plot(self, field: str, filename: Path = None):
        data = self.mesh.point_data[field]
        dmin, dmax = data.min(), data.max()
        
        if self._grid_shape and len(data) == self._grid_shape[0] * self._grid_shape[1]:
            nx, ny = self._grid_shape[0], self._grid_shape[1]
            img_data = data.reshape(nx, ny).T
        else:
            # Fallback: try to reshape as square
            side = int(np.ceil(np.sqrt(len(data))))
            if side * side == len(data):
                img_data = data.reshape(side, side).T
            else:
                img_data = data.reshape(-1, 1).T
        
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
        
        norm = mcolors.Normalize(vmin=dmin, vmax=max(dmax, 1e-10))
        im = ax.imshow(img_data, origin="lower", cmap="viridis", norm=norm,
                       extent=[0, 100, 0, 100], aspect="auto")
        
        ax.set_title(f"{field}\nMax probability: {dmax:.6f}", fontsize=14)
        ax.set_xlabel("X (domain %)")
        ax.set_ylabel("Y (domain %)")
        plt.colorbar(im, ax=ax, label="Probability", shrink=0.8)
        
        if filename:
            fig.savefig(str(filename), dpi=100, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()