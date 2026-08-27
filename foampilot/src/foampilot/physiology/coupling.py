"""Couplage en mémoire entre une surface CFD et le modèle JOS-3 embarqué."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence
import csv

import numpy as np

from .jos3 import BODY_NAMES
from .units import as_magnitude, scalar

N_ZONES = len(BODY_NAMES)


class CallbackFieldProvider:
    """Adaptateur mémoire basé sur deux callbacks OpenFOAM/Python."""

    def __init__(self, reader: Callable[[], Mapping[str, np.ndarray]], writer: Callable[[np.ndarray], None]):
        self._reader = reader
        self._writer = writer

    def read_nodal_fields(self) -> Mapping[str, np.ndarray]:
        return self._reader()

    def write_nodal_flux(self, flux: np.ndarray) -> None:
        self._writer(flux)


class NodalFieldProvider(Protocol):
    """Interface minimale qu’un solveur OpenFOAM Python peut implémenter."""

    def read_nodal_fields(self) -> Mapping[str, np.ndarray]: ...

    def write_nodal_flux(self, flux: np.ndarray) -> None: ...


@dataclass(frozen=True)
class SurfaceMapping:
    """Géométrie et rattachement des points CFD aux 17 zones JOS-3."""

    zone_ids: np.ndarray
    areas: np.ndarray
    points: np.ndarray | None = None

    def __post_init__(self) -> None:
        zone_ids = np.asarray(self.zone_ids, dtype=int).reshape(-1)
        areas = as_magnitude(self.areas, "m^2", name="areas").reshape(-1)
        if zone_ids.size == 0:
            raise ValueError("Le mapping doit contenir au moins un nœud")
        if zone_ids.size != areas.size:
            raise ValueError("zone_ids et areas doivent avoir la même longueur")
        if np.any((zone_ids < 0) | (zone_ids >= N_ZONES)):
            raise ValueError(f"zone_ids doit être compris entre 0 et {N_ZONES - 1}")
        if not np.all(np.isfinite(areas)) or np.any(areas <= 0):
            raise ValueError("Les aires nodales doivent être finies et positives")
        object.__setattr__(self, "zone_ids", zone_ids)
        object.__setattr__(self, "areas", areas)
        if self.points is not None:
            points = as_magnitude(self.points, "m", name="points")
            if points.shape != (zone_ids.size, 3):
                raise ValueError("points doit être de forme (nombre_de_noeuds, 3)")
            if not np.all(np.isfinite(points)):
                raise ValueError("points contient une valeur non finie")
            object.__setattr__(self, "points", points)

    @classmethod
    def from_csv(cls, path, *, points=None):
        """Construit un mapping à partir de ``zone_mapping.csv``.

        Le CSV doit contenir ``zone_id`` et ``area_m2``. Les colonnes de nom
        et d’unités sont contrôlées lorsqu’elles sont présentes.
        """
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
        if not rows:
            raise ValueError("Le fichier de mapping est vide")
        required = {"zone_id", "area_m2"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Le mapping doit contenir zone_id et area_m2")
        for index, row in enumerate(rows, start=2):
            if any(row.get(column, "").strip() == "" for column in required):
                raise ValueError(f"Ligne CSV {index} incomplète")
        if "temperature_unit" in (reader.fieldnames or []) and any(r["temperature_unit"] != "K" for r in rows):
            raise ValueError("Le mapping CFD doit déclarer temperature_unit=K")
        if "h_unit" in (reader.fieldnames or []) and any(r["h_unit"] != "W/m2/K" for r in rows):
            raise ValueError("Unité h inattendue dans le mapping")
        try:
            zone_ids = np.array([int(r["zone_id"]) for r in rows])
            areas = np.array([float(r["area_m2"]) for r in rows])
        except (TypeError, ValueError) as error:
            raise ValueError("zone_id et area_m2 doivent être numériques") from error
        return cls(zone_ids=zone_ids, areas=areas, points=points)

    def validate_areas(self, areas, *, rtol=1e-8, atol=1e-12) -> None:
        values = as_magnitude(areas, "m^2", name="areas").reshape(-1)
        if values.size != self.areas.size:
            raise ValueError("Le champ areas ne correspond pas au mapping")
        if not np.allclose(values, self.areas, rtol=rtol, atol=atol):
            raise ValueError("Les aires du provider diffèrent du mapping")

    @property
    def zone_areas(self) -> np.ndarray:
        result = np.zeros(N_ZONES)
        np.add.at(result, self.zone_ids, self.areas)
        return result


@dataclass(frozen=True)
class ThermalExchange:
    """Échange conservant les valeurs point par point et les puissances par zone."""

    h: np.ndarray
    air_temperature: np.ndarray
    surface_temperature: np.ndarray
    outward_flux: np.ndarray
    body_flux: np.ndarray
    zone_flux_mean: np.ndarray
    zone_h_mean: np.ndarray
    zone_air_temperature: np.ndarray
    zone_power: np.ndarray


class JOS3NodeCoupler:
    """Compatibilité : couplage agrégé vers les 17 états cutanés JOS-3.

    Cette classe ne crée pas de température indépendante par point CFD.
    Pour le véritable échange nodal transitoire, utiliser
    :class:`DistributedSurfaceNetwork`.

    Le flux est calculé sur tous les points CFD. Comme JOS-3 possède 17 zones
    cutanées locales, la puissance de chaque zone est l’intégrale du flux sur
    ses aires nodales. Le mode steady applique un échange ponctuel après une
    moyenne spatiale par zone ; le mode transient refait cette opération à
    chaque pas de temps.
    """

    def __init__(self, model, mapping: SurfaceMapping, *, outward_positive=True):
        self.model = model
        self.mapping = mapping
        self.outward_positive = outward_positive
        self.last_exchange: ThermalExchange | None = None

    @staticmethod
    def nodal_flux(h, surface_temperature, air_temperature, *, outward_positive=True):
        h = as_magnitude(h, "W/m^2/K", name="h").reshape(-1)
        ts = as_magnitude(surface_temperature, "degC", name="surface_temperature").reshape(-1)
        ta = as_magnitude(air_temperature, "degC", name="air_temperature").reshape(-1)
        if not (h.size == ts.size == ta.size):
            raise ValueError("Les trois champs nodaux doivent avoir la même longueur")
        outward = h * (ts - ta)
        body = -outward if outward_positive else outward
        return outward, body

    def exchange(self, h, surface_temperature, air_temperature) -> ThermalExchange:
        h = as_magnitude(h, "W/m^2/K", name="h").reshape(-1)
        surface_temperature = as_magnitude(surface_temperature, "degC", name="surface_temperature").reshape(-1)
        air_temperature = as_magnitude(air_temperature, "degC", name="air_temperature").reshape(-1)
        outward, body = self.nodal_flux(
            h, surface_temperature, air_temperature,
            outward_positive=self.outward_positive,
        )
        if body.size != self.mapping.zone_ids.size:
            raise ValueError("Les champs nodaux ne correspondent pas au mapping")
        zone_area = self.mapping.zone_areas
        zone_power = np.zeros(N_ZONES)
        zone_h = np.zeros(N_ZONES)
        zone_ta = np.zeros(N_ZONES)
        np.add.at(zone_power, self.mapping.zone_ids, body * self.mapping.areas)
        np.add.at(zone_h, self.mapping.zone_ids, np.asarray(h) * self.mapping.areas)
        np.add.at(zone_ta, self.mapping.zone_ids, np.asarray(air_temperature) * self.mapping.areas)
        zone_flux_mean = np.divide(
            zone_power, zone_area, out=np.zeros(N_ZONES), where=zone_area > 0
        )
        zone_h_mean = np.divide(zone_h, zone_area, out=np.zeros(N_ZONES), where=zone_area > 0)
        zone_air_temperature = np.divide(
            zone_ta, zone_area, out=np.zeros(N_ZONES), where=zone_area > 0
        )
        result = ThermalExchange(
            np.asarray(h, float).copy(),
            np.asarray(air_temperature, float).copy(),
            np.asarray(surface_temperature, float).copy(),
            outward.copy(), body.copy(), zone_flux_mean, zone_h_mean,
            zone_air_temperature, zone_power,
        )
        self.last_exchange = result
        return result

    def apply(self, exchange: ThermalExchange) -> None:
        # Les températures CFD moyennées par zone alimentent aussi les termes
        # respiratoires/évaporatifs et la température opérative de JOS-3.
        self.model.Ta = exchange.zone_air_temperature
        self.model.set_external_heat_flux(exchange.zone_power, tissue="skin")

    def step_steady(self, h, surface_temperature, air_temperature, *, dtime=60.0):
        """Échange une fois, avec flux moyen par zone, puis avance JOS-3."""
        exchange = self.exchange(h, surface_temperature, air_temperature)
        self.apply(exchange)
        self.model.simulate(times=1, dtime=dtime)
        return exchange

    def step_transient(self, h, surface_temperature, air_temperature, *, dtime=60.0):
        """Échange tous les points à chaque pas puis avance JOS-3 d’un pas."""
        return self.step_steady(
            h, surface_temperature, air_temperature, dtime=dtime
        )

    def run_steady(self, fields, *, dtime=60.0, steps=1):
        """Applique un champ constant et réalise plusieurs pas JOS-3."""
        exchange = self.exchange(**fields)
        self.apply(exchange)
        self.model.simulate(times=steps, dtime=dtime)
        return exchange

    def run_transient(self, provider: NodalFieldProvider, *, dtime=60.0, steps=1):
        """Échange Pythonique avec un provider OpenFOAM à chaque pas."""
        exchanges = []
        for _ in range(steps):
            fields = provider.read_nodal_fields()
            exchange = self.step_transient(
                fields["h"], fields["surface_temperature"],
                fields["air_temperature"], dtime=dtime,
            )
            provider.write_nodal_flux(exchange.body_flux.copy())
            exchanges.append(exchange)
        return exchanges

    @staticmethod
    def zone_names() -> tuple[str, ...]:
        return tuple(BODY_NAMES)


@dataclass(frozen=True)
class DistributedSurfaceExchange:
    """État d’un échange avec une température indépendante par point CFD."""

    surface_temperature: np.ndarray
    air_temperature: np.ndarray
    h: np.ndarray
    environment_power: np.ndarray
    body_power: np.ndarray
    zone_body_power: np.ndarray


class DistributedSurfaceNetwork:
    """Extension distribuée de la peau JOS-3 sur les points du maillage CFD.

    Les 17 températures cutanées JOS-3 restent les états physiologiques de
    référence. Chaque point CFD possède en plus son propre état de surface,
    sa capacité thermique et une conductance vers la peau physiologique de sa
    zone. Les capacités sont réparties proportionnellement aux aires nodales,
    de sorte que leur somme par zone vaut la capacité cutanée JOS-3.
    """

    def __init__(self, model, mapping: SurfaceMapping, *, anchor_conductance=None,
                 surface_temperature=None):
        self.model = model
        self.mapping = mapping
        skin_indices = np.asarray(model.skin_node_indices, dtype=int)
        self.skin_capacity = np.asarray(model._cap[skin_indices], dtype=float)
        self.zone_area = mapping.zone_areas
        area_fraction = mapping.areas / self.zone_area[mapping.zone_ids]
        self.capacity = self.skin_capacity[mapping.zone_ids] * area_fraction
        if not np.all(np.isfinite(self.capacity)) or np.any(self.capacity <= 0):
            raise ValueError("Les capacités surfaciques doivent être finies et positives")
        if anchor_conductance is None:
            # Conductance from each JOS-3 skin node to its inner tissues.
            conductance = np.sum(model._cdt[skin_indices, :], axis=1)
        else:
            conductance = np.asarray(anchor_conductance, dtype=float).reshape(N_ZONES)
        if np.any(conductance <= 0):
            raise ValueError("Les conductances d’ancrage doivent être positives")
        self.anchor_conductance = conductance[mapping.zone_ids] * area_fraction
        if not np.all(np.isfinite(self.anchor_conductance)):
            raise ValueError("Les conductances d’ancrage doivent être finies")
        if surface_temperature is None:
            self.surface_temperature = model.Tsk[mapping.zone_ids].copy()
        else:
            surface_temperature = np.asarray(surface_temperature, dtype=float).reshape(-1)
            if surface_temperature.size != mapping.zone_ids.size:
                raise ValueError("surface_temperature ne correspond pas au mapping")
            self.surface_temperature = surface_temperature.copy()
        self.last_exchange: DistributedSurfaceExchange | None = None
        model.set_environment_mode("external_surface")

    def step(self, h, air_temperature, *, radiative_temperature=None,
             dtime=1.0, hr=0.0):
        """Avance le réseau d’un pas avec convection et rayonnement locaux.

        ``h`` et ``hr`` sont des coefficients [W/m²/K]. ``air_temperature``
        et ``radiative_temperature`` sont en °C. Les puissances internes sont
        stockées en W ; le provider OpenFOAM reçoit ensuite un flux en W/m².
        """
        dtime = scalar(dtime, "s", name="dtime")
        if dtime <= 0:
            raise ValueError("dtime doit être strictement positif et fini")
        h = as_magnitude(h, "W/m^2/K", name="h").reshape(-1)
        ta = as_magnitude(air_temperature, "degC", name="air_temperature").reshape(-1)
        tr = ta if radiative_temperature is None else as_magnitude(
            radiative_temperature, "degC", name="radiative_temperature"
        ).reshape(-1)
        if h.size != self.mapping.zone_ids.size or ta.size != h.size or tr.size != h.size:
            raise ValueError("h, air_temperature et radiative_temperature doivent correspondre au mapping")
        if not np.all(np.isfinite(h)) or not np.all(np.isfinite(ta)) or not np.all(np.isfinite(tr)):
            raise ValueError("Les champs thermiques doivent être finis")
        if np.any(h < 0) or hr < 0 or not np.isfinite(hr):
            raise ValueError("Les coefficients d’échange doivent être positifs et finis")
        zone_ta = np.zeros(N_ZONES)
        np.add.at(zone_ta, self.mapping.zone_ids, ta * self.mapping.areas)
        zone_ta = np.divide(zone_ta, self.zone_area, out=np.zeros(N_ZONES), where=self.zone_area > 0)
        self.model.Ta = zone_ta
        skin_temperature = self.model.Tsk[self.mapping.zone_ids]
        body_power = self.anchor_conductance * (self.surface_temperature - skin_temperature)
        environment_power = (
            h * (self.surface_temperature - ta)
            + hr * (self.surface_temperature - tr)
        ) * self.mapping.areas
        self.model.set_external_heat_flux(
            np.bincount(self.mapping.zone_ids, weights=body_power, minlength=17),
            tissue="skin",
        )
        self.model.simulate(times=1, dtime=dtime)
        # Surface-node energy balance: C dT/dt = -Q_to_body - Q_to_environment.
        self.surface_temperature += dtime * (-body_power - environment_power) / self.capacity
        zone_body_power = np.bincount(
            self.mapping.zone_ids, weights=body_power, minlength=17
        )
        exchange = DistributedSurfaceExchange(
            self.surface_temperature.copy(), ta.copy(), h.copy(),
            environment_power.copy(), body_power.copy(), zone_body_power,
        )
        self.last_exchange = exchange
        return exchange

    def run_transient(self, provider: NodalFieldProvider, *, dtime=1.0, steps=1, hr=0.0):
        """Exécute l’échange distribué avec un provider OpenFOAM Python."""
        exchanges = []
        for _ in range(steps):
            fields = provider.read_nodal_fields()
            exchange = self.step(
                fields["h"], fields["air_temperature"],
                radiative_temperature=fields.get("radiative_temperature"),
                dtime=dtime, hr=hr,
            )
            # externalCoupled attend un flux surfacique [W/m²], alors que
            # environment_power est une puissance nodale [W].
            provider.write_nodal_flux(
                (exchange.environment_power / self.mapping.areas).copy()
            )
            exchanges.append(exchange)
        return exchanges
