
from pathlib import Path
import numpy as np
import pyvista as pv
import pandas as pd
import json

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

class WindCaseResult:
    def __init__(
        self,
        post: "FoamPostProcessing",
        direction_deg: float,
        u_ref: float = 10.0,
        field_name: str = "U",
    ):
        self.post = post
        self.direction_deg = direction_deg
        self.u_ref = u_ref
        self.field_name = field_name

        self.mesh = None
        self.plane = None
        self.sensitivity = None

    def extract_pedestrian_plane(self):
        structure = self.post.get_structure()
        mesh = structure["cell"]

        bounds = mesh.bounds
        plane = pv.Plane(
            center=((bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    PEDESTRIAN_HEIGHT),
            direction=(0, 0, 1),
            i_size=bounds[1] - bounds[0],
            j_size=bounds[3] - bounds[2],
            i_resolution=300,
            j_resolution=300,
        )

        sampled = mesh.sample(plane)
        self.mesh = sampled
        self.plane = plane

    def compute_sensitivity(self):
        U = self.mesh[self.field_name]
        U_mag = np.linalg.norm(U, axis=1)
        self.sensitivity = U_mag / self.u_ref
        self.mesh["S"] = self.sensitivity

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


class LawsonProcessor:
    def __init__(self, ensemble: WindEnsemble, wind_rose: WindRose):
        self.ensemble = ensemble
        self.wind_rose = wind_rose

    def compute_probability_map(self, threshold: float):
        prob = None

        for direction, case in self.ensemble.cases.items():
            S = case.sensitivity
            mesh = case.mesh

            for wc in self.wind_rose.data.get(direction, []):
                u_eq = S * wc["speed"]
                exceed = (u_eq > threshold).astype(float) * wc["frequency"]

                if prob is None:
                    prob = exceed.copy()
                else:
                    prob += exceed

        return prob

    def compute_lawson_maps(self):
        maps = {}

        for label, threshold in LAWSON_THRESHOLDS.items():
            maps[label] = self.compute_probability_map(threshold)

        return maps

class LawsonVisualizer:
    def __init__(self, reference_mesh: pv.PolyData):
        self.mesh = reference_mesh.copy()

    def add_probability_field(self, name: str, data: np.ndarray):
        self.mesh[name] = data

    def plot(self, field: str, filename: Path = None):
        off_screen = filename is not None
        pl = pv.Plotter(off_screen=off_screen)

        pl.add_mesh(
            self.mesh,
            scalars=field,
            clim=[0, LAWSON_MAX_PROBABILITY],
            scalar_bar_args={"title": field},
        )

        if filename:
            pl.screenshot(str(filename))
        else:
            pl.show()