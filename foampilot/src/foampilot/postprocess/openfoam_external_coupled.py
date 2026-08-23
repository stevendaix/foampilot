"""Provider FoamPilot pour le functionObject OpenFOAM externalCoupled."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class OpenFOAMExternalCoupledProvider:
    """Échange de champs par le protocole natif ``externalCoupled``.

    OpenFOAM écrit les fichiers ``<field>.out`` dans ``comms_dir`` puis retire
    ``OpenFOAM.lock``. Le pilote Python lit les sorties, écrit les fichiers
    ``<field>.in`` et recrée le verrou pour rendre la main à OpenFOAM.
    """

    def __init__(self, comms_dir, *, timeout=300.0, poll_interval=0.05,
                 fields=("h", "air_temperature"), output_field="qJOS3",
                 temperature_unit="K"):
        self.comms_dir = Path(comms_dir)
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.fields = tuple(fields)
        self.output_field = output_field
        if temperature_unit not in ("K", "C"):
            raise ValueError("temperature_unit doit être 'K' ou 'C'")
        self.temperature_unit = temperature_unit

    @property
    def lock_path(self) -> Path:
        return self.comms_dir / "OpenFOAM.lock"

    def _wait_for_outputs(self) -> None:
        deadline = time.monotonic() + self.timeout
        while True:
            if all((self.comms_dir / f"{field}.out").exists() for field in self.fields):
                return
            if time.monotonic() >= deadline:
                missing = [
                    str(self.comms_dir / f"{field}.out")
                    for field in self.fields
                    if not (self.comms_dir / f"{field}.out").exists()
                ]
                raise TimeoutError(f"OpenFOAM externalCoupled: sorties absentes: {missing}")
            time.sleep(self.poll_interval)

    @staticmethod
    def _read_scalar_file(path: Path) -> np.ndarray:
        values = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # externalCoupled peut préfixer les lignes par des parenthèses.
            tokens = line.replace("(", " ").replace(")", " ").split()
            if tokens:
                values.append(float(tokens[0]))
        if not values:
            raise ValueError(f"Aucune valeur scalaire dans {path}")
        return np.asarray(values, dtype=float)

    def read_nodal_fields(self) -> Mapping[str, np.ndarray]:
        self._wait_for_outputs()
        result = {}
        for field in self.fields:
            result[field] = self._read_scalar_file(self.comms_dir / f"{field}.out")
        # Noms normalisés pour DistributedSurfaceNetwork.
        if "Ta" in result and "air_temperature" not in result:
            result["air_temperature"] = result.pop("Ta")
        if "T" in result and "surface_temperature" not in result:
            result["surface_temperature"] = result.pop("T")
        if self.temperature_unit == "K":
            for key in ("air_temperature", "surface_temperature"):
                if key in result:
                    result[key] = result[key] - 273.15
        return result

    @staticmethod
    def _write_scalar_file(path: Path, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float).reshape(-1)
        path.write_text(
            "\n".join(f"{value:.16g}" for value in values) + "\n",
            encoding="utf-8",
        )

    def write_nodal_flux(self, flux: np.ndarray) -> None:
        self._write_scalar_file(self.comms_dir / f"{self.output_field}.in", flux)
        # Le verrou est le signal de fin d’écriture pour externalCoupled.
        self.lock_path.touch()

    def read_nodal_fields_and_wait(self) -> Mapping[str, np.ndarray]:
        """Alias explicite pour une boucle transitoire pilotée par OpenFOAM."""
        return self.read_nodal_fields()


class OpenFOAM13TemperatureProvider:
    """Provider du format collaté OpenFOAM 13 ``externalCoupledTemperature``.

    OpenFOAM écrit une ligne par face dans ``data.out`` :
    ``area T qDot htc`` avec l’aire en m², la température en K, le flux en
    W/m² et le coefficient convectif effectif en W/m²/K. FoamPilot renvoie
    ``T snGrad valueFraction`` dans ``data.in`` ; cette condition impose ainsi
    la température de surface calculée par le réseau distribué.
    """

    def __init__(self, comms_dir, *, file="data", timeout=300.0,
                 poll_interval=0.05, air_temperature=20.0,
                 radiative_temperature=None, temperature_unit="K"):
        self.comms_dir = Path(comms_dir)
        self.file = file
        self.timeout = float(timeout)
        self.poll_interval = float(poll_interval)
        self.air_temperature = air_temperature
        self.radiative_temperature = radiative_temperature
        if temperature_unit not in ("K", "C"):
            raise ValueError("temperature_unit doit être 'K' ou 'C'")
        self.temperature_unit = temperature_unit
        self._last_data_mtime_ns = 0
        self._expected_n_faces = None

    @property
    def lock_path(self) -> Path:
        return self.comms_dir / "OpenFOAM.lock"

    @property
    def data_out_path(self) -> Path:
        return self.comms_dir / f"{self.file}.out"

    @property
    def data_in_path(self) -> Path:
        return self.comms_dir / f"{self.file}.in"

    def _wait_for_data(self) -> None:
        deadline = time.monotonic() + self.timeout
        while True:
            if self.data_out_path.exists():
                mtime_ns = self.data_out_path.stat().st_mtime_ns
                if mtime_ns > self._last_data_mtime_ns:
                    # Attendre que l’écriture collatée d’OpenFOAM soit stable.
                    size0 = self.data_out_path.stat().st_size
                    time.sleep(self.poll_interval)
                    if not self.data_out_path.exists() or self.data_out_path.stat().st_size != size0:
                        continue
                    self._last_data_mtime_ns = self.data_out_path.stat().st_mtime_ns
                    return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"OpenFOAM13: nouvelle sortie absente: {self.data_out_path}")
            time.sleep(self.poll_interval)

    @staticmethod
    def _read_data(path: Path) -> np.ndarray:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            tokens = line.replace("(", " ").replace(")", " ").split()
            if len(tokens) >= 4:
                rows.append([float(token) for token in tokens[:4]])
        if not rows:
            raise ValueError(f"Aucune ligne area/T/qDot/htc dans {path}")
        return np.asarray(rows, dtype=float)

    def read_nodal_fields(self) -> Mapping[str, np.ndarray]:
        self._wait_for_data()
        while True:
            table = self._read_data(self.data_out_path)
            if self._expected_n_faces is None:
                self._expected_n_faces = table.shape[0]
                break
            if table.shape[0] == self._expected_n_faces:
                break
            self._wait_for_data()
        area = table[:, 0]
        temperature = table[:, 1].copy()
        if self.temperature_unit == "K":
            temperature_c = temperature - 273.15
        else:
            temperature_c = temperature
        n = area.size
        air = np.asarray(self.air_temperature, dtype=float)
        if air.ndim == 0:
            air = np.full(n, float(air))
        air = air.reshape(-1)
        if air.size != n:
            raise ValueError("air_temperature doit être scalaire ou de longueur N")
        if self.radiative_temperature is None:
            rad = air.copy()
        else:
            rad = np.asarray(self.radiative_temperature, dtype=float)
            if rad.ndim == 0:
                rad = np.full(n, float(rad))
            rad = rad.reshape(-1)
            if rad.size != n:
                raise ValueError("radiative_temperature doit être scalaire ou de longueur N")
        return {
            "areas": area,
            "surface_temperature": temperature_c,
            "air_temperature": air,
            "radiative_temperature": rad,
            "q_dot": table[:, 2].copy(),
            "h": table[:, 3].copy(),
        }

    def write_surface_temperature(self, temperature_c: np.ndarray) -> None:
        temperature_c = np.asarray(temperature_c, dtype=float).reshape(-1)
        temperature = temperature_c + 273.15 if self.temperature_unit == "K" else temperature_c
        values = np.column_stack((temperature, np.zeros_like(temperature), np.ones_like(temperature)))
        self.data_in_path.write_text(
            "\n".join(" ".join(f"{value:.16g}" for value in row) for row in values) + "\n",
            encoding="utf-8",
        )
        self.lock_path.touch()
