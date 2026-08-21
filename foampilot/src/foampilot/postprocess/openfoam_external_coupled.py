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
