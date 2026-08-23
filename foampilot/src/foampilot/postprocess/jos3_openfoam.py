"""Couplage nodal entre résultats OpenFOAM et modèle physiologique JOS-3.

Le couplage utilise les champs scalaires OpenFOAM ``h``, ``Ta`` et ``T`` :

    q_out = h * (T_surface - Ta)       [W/m2]
    q_body = -q_out                    [W/m2]

``q_body`` est positif lorsque le fluide chauffe le corps. Il est intégré sur
les aires nodales puis regroupé sur les 17 segments JOS-3 en ``W``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from foampilot.postprocess.openfoam_direct import OpenFOAMDirectReader


JOS3_SEGMENT_NAMES = (
    "Head", "Neck", "Chest", "Back", "Pelvis", "LShoulder", "LArm",
    "LHand", "RShoulder", "RArm", "RHand", "LThigh", "LLeg", "LFoot",
    "RThigh", "RLeg", "RFoot",
)


@dataclass(frozen=True)
class NodalThermalExchange:
    """Résultats d’un échange OpenFOAM–JOS-3 à un instant donné."""

    time_step: str
    h: np.ndarray
    air_temperature: np.ndarray
    surface_temperature: np.ndarray
    outward_flux: np.ndarray
    body_flux: np.ndarray
    segment_heat: np.ndarray


class OpenFOAMJOS3Coupler:
    """Couple un champ thermique OpenFOAM nodal à une instance JOS-3.

    ``segment_ids`` contient, pour chaque nœud OpenFOAM, l’indice du segment
    JOS-3 compris entre 0 et 16. ``node_areas`` est l’aire duale associée à
    chaque nœud, en m². Le couplage ne suppose pas que le maillage possède
    exactement 17 nœuds : plusieurs milliers de nœuds peuvent être agrégés.

    Le mode ``raw_extra_heat`` injecte le flux intégré comme ``model.ex_q``.
    Le mode ``sensible_correction`` injecte la différence entre le flux CFD
    et la perte sensible déjà calculée par JOS-3, évitant un double comptage.
    """

    def __init__(
        self,
        model,
        segment_ids: Sequence[int],
        node_areas: Sequence[float],
        *,
        mode: str = "raw_extra_heat",
        outward_positive: bool = True,
    ) -> None:
        if len(segment_ids) != len(node_areas):
            raise ValueError("segment_ids et node_areas doivent avoir la même longueur")
        if mode not in {"raw_extra_heat", "sensible_correction"}:
            raise ValueError("mode doit être 'raw_extra_heat' ou 'sensible_correction'")
        self.model = model
        self.segment_ids = np.asarray(segment_ids, dtype=int)
        self.node_areas = np.asarray(node_areas, dtype=float)
        if np.any(self.segment_ids < 0) or np.any(self.segment_ids >= 17):
            raise ValueError("Chaque segment_id doit être compris entre 0 et 16")
        if np.any(self.node_areas <= 0):
            raise ValueError("Les aires nodales doivent être strictement positives")
        self.mode = mode
        self.outward_positive = outward_positive

    @classmethod
    def from_openfoam(
        cls,
        model,
        case_path: str | Path,
        segment_ids: Sequence[int],
        node_areas: Sequence[float],
        *,
        region: Optional[str] = None,
        mode: str = "raw_extra_heat",
    ) -> "OpenFOAMJOS3Coupler":
        """Construit le coupler avec un lecteur direct de cas OpenFOAM."""
        return cls(
            model,
            segment_ids,
            node_areas,
            mode=mode,
            outward_positive=True,
        ).with_reader(OpenFOAMDirectReader(case_path, region=region))

    def with_reader(self, reader: OpenFOAMDirectReader) -> "OpenFOAMJOS3Coupler":
        self.reader = reader
        return self

    @staticmethod
    def compute_flux(
        h: Sequence[float],
        surface_temperature: Sequence[float],
        air_temperature: Sequence[float],
        *,
        outward_positive: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calcule les flux sortant et entrant dans le corps, en W/m²."""
        h_arr = np.asarray(h, dtype=float)
        ts = np.asarray(surface_temperature, dtype=float)
        ta = np.asarray(air_temperature, dtype=float)
        if not (h_arr.shape == ts.shape == ta.shape):
            raise ValueError("h, surface_temperature et air_temperature doivent être alignés")
        outward = h_arr * (ts - ta)
        body = -outward if outward_positive else outward
        return outward, body

    def aggregate_to_segments(self, nodal_body_flux: Sequence[float]) -> np.ndarray:
        """Intègre le flux nodal en puissance reçue par chaque segment JOS-3."""
        q = np.asarray(nodal_body_flux, dtype=float)
        if q.shape != self.segment_ids.shape:
            raise ValueError("Le flux nodal ne correspond pas au mapping des segments")
        result = np.zeros(17, dtype=float)
        np.add.at(result, self.segment_ids, q * self.node_areas)
        return result

    def exchange_arrays(
        self,
        h: Sequence[float],
        air_temperature: Sequence[float],
        surface_temperature: Sequence[float],
        *,
        time_step: str = "0",
        apply: bool = True,
    ) -> NodalThermalExchange:
        """Calcule l’échange et l’applique à JOS-3 si ``apply=True``."""
        outward, body = self.compute_flux(
            h, surface_temperature, air_temperature,
            outward_positive=self.outward_positive,
        )
        segment_heat = self.aggregate_to_segments(body)
        result = NodalThermalExchange(
            str(time_step), np.asarray(h, float), np.asarray(air_temperature, float),
            np.asarray(surface_temperature, float), outward, body, segment_heat,
        )
        if apply:
            self.apply_to_model(result)
        return result

    def exchange_from_openfoam(
        self,
        *,
        time_step: str = "latest",
        h_field: str = "h",
        air_temperature_field: str = "Ta",
        surface_temperature_field: str = "T",
        apply: bool = True,
    ) -> NodalThermalExchange:
        """Lit trois champs scalaires OpenFOAM et réalise un échange nodal."""
        if not hasattr(self, "reader"):
            raise RuntimeError("Aucun lecteur OpenFOAM n’est attaché au coupler")
        ts = self.reader.get_latest_time() if time_step == "latest" else str(time_step)
        h = self.reader.read_field(h_field, ts)
        ta = self.reader.read_field(air_temperature_field, ts)
        t = self.reader.read_field(surface_temperature_field, ts)
        return self.exchange_arrays(h, ta, t, time_step=ts, apply=apply)

    def apply_to_model(self, exchange: NodalThermalExchange) -> None:
        """Injecte les puissances segmentées dans ``model.ex_q``."""
        q = exchange.segment_heat.copy()
        if self.mode == "sensible_correction":
            # JOS-3 calcule sa perte sensible à partir de Tsk, To et Rt.
            q_jos = (self.model.Tsk - self.model.To) / self.model.Rt * self.model.BSA
            q = q - q_jos
        self.model.ex_q = np.zeros_like(self.model.ex_q)
        self.model._set_ex_q("skin", q)

    def step(self, *, dtime: float = 60.0, **kwargs) -> NodalThermalExchange:
        """Effectue un échange OpenFOAM puis un pas JOS-3."""
        exchange = self.exchange_from_openfoam(apply=True, **kwargs)
        self.model.simulate(times=1, dtime=dtime)
        return exchange

    def write_point_scalar_field(
        self,
        values: Sequence[float],
        path: str | Path,
        *,
        field_name: str = "qJOS3",
        time_name: str = "0",
        object_name: Optional[str] = None,
    ) -> Path:
        """Écrit un champ ``pointScalarField`` OpenFOAM pour le retour CFD."""
        vals = np.asarray(values, dtype=float)
        if vals.shape != self.segment_ids.shape:
            raise ValueError("Le champ à écrire doit contenir une valeur par nœud")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = object_name or field_name
        body = "\n".join(f"{v:.12g}" for v in vals)
        text = f'''FoamFile\n{{\n    version     2.0;\n    format      ascii;\n    class       pointScalarField;\n    location    "{time_name}";\n    object      {obj};\n}}\n\ndimensions      [1 -2 -3 0 0 0 0];\ninternalField   nonuniform List<scalar>\n{len(vals)}\n(\n{body}\n)\n;\nboundaryField\n{{\n    walls\n    {{\n        type calculated;\n        value $internalField;\n    }}\n}}\n'''
        path.write_text(text, encoding="utf-8")
        return path

    @staticmethod
    def segment_names() -> tuple[str, ...]:
        return JOS3_SEGMENT_NAMES
