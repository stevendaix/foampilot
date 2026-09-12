"""Lightweight file-based Cantera/OpenFOAM thermochemistry coupling.

The adapter deliberately keeps the CFD solver in OpenFOAM and delegates
thermodynamic/kinetic state evaluation to Cantera.  The exchange format is a
small CSV table, which makes the coupling inspectable and usable from batch
jobs without an ABI-dependent OpenFOAM plug-in.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class ThermoState:
    temperature: float
    pressure: float
    mass_fractions: tuple[float, ...]


class CanteraOpenFOAMCoupler:
    """Evaluate Cantera states and exchange cell data with OpenFOAM CSV files."""

    def __init__(self, mechanism: str = "gri30.yaml", phase: str = "gri30") -> None:
        try:
            import cantera as ct
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Cantera est requis. Installez-le avec `pip install cantera`."
            ) from exc
        self._ct = ct
        self.gas = ct.Solution(mechanism, phaseid=phase)

    @property
    def species_names(self) -> tuple[str, ...]:
        return tuple(self.gas.species_names)

    def equilibrate(self, temperature: float, pressure: float,
                    composition: str | Sequence[float]) -> ThermoState:
        """Return the HP-equilibrium state for an OpenFOAM cell state."""
        self.gas.TPX = temperature, pressure, composition
        self.gas.equilibrate("HP")
        return ThermoState(self.gas.T, self.gas.P, tuple(self.gas.Y))

    def write_species_field(self, path: str | Path, values: Iterable[float]) -> None:
        """Write an OpenFOAM nonuniform scalar list for one species."""
        values = [float(value) for value in values]
        if not values:
            raise ValueError("Le champ OpenFOAM doit contenir au moins une cellule.")
        with Path(path).open("w", encoding="utf-8") as stream:
            stream.write(f"{len(values)}\n(\n")
            stream.writelines(f"{value:.16g}\n" for value in values)
            stream.write(")\n")

    @staticmethod
    def read_cell_states(path: str | Path) -> list[tuple[float, float, str]]:
        """Read ``cell,T,p,composition`` rows produced by an OpenFOAM function."""
        with Path(path).open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            required = {"cell", "T", "p", "composition"}
            if not required <= set(rows.fieldnames or ()):
                raise ValueError(f"Colonnes attendues: {sorted(required)}")
            result = []
            for row in rows:
                result.append((float(row["T"]), float(row["p"]), row["composition"]))
        return result

    def equilibrate_csv(self, input_path: str | Path, output_path: str | Path) -> None:
        """Evaluate every OpenFOAM row and write ``cell,T,p,rho,cp,kappa``."""
        with Path(input_path).open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        required = {"cell", "T", "p", "composition"}
        if not required <= set(rows[0] if rows else {}):
            raise ValueError(f"Colonnes attendues: {sorted(required)}")
        with Path(output_path).open("w", newline="", encoding="utf-8") as stream:
            fields = ["cell", "T_eq", "p_eq", "rho", "cp_mass", "thermal_conductivity"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                state = self.equilibrate(float(row["T"]), float(row["p"]), row["composition"])
                writer.writerow({
                    "cell": row["cell"], "T_eq": f"{state.temperature:.16g}",
                    "p_eq": f"{state.pressure:.16g}", "rho": f"{self.gas.density:.16g}",
                    "cp_mass": f"{self.gas.cp_mass:.16g}",
                    "thermal_conductivity": f"{self.gas.thermal_conductivity:.16g}",
                })

    def ignition_delay(self, temperature: float, pressure: float,
                       composition: str, end_time: float = 2e-3,
                       points: int = 200) -> tuple[float, float]:
        """Compute the first peak in OH for a homogeneous constant-volume reactor."""
        import numpy as np
        self.gas.TPX = temperature, pressure, composition
        reactor = self._ct.IdealGasReactor(self.gas, energy="on")
        net = self._ct.ReactorNet([reactor])
        times = np.linspace(0.0, end_time, points)
        oh = np.empty(points)
        for index, time in enumerate(times):
            net.advance(float(time))
            oh[index] = reactor.thermo["OH"].Y[0]
        peak = int(np.argmax(oh))
        return float(times[peak]), float(oh[peak])
