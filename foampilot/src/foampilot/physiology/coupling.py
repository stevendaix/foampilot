"""Couplage en mémoire entre une surface CFD et le modèle JOS-3 embarqué."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from .jos3 import BODY_NAMES


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
        areas = np.asarray(self.areas, dtype=float).reshape(-1)
        if zone_ids.size != areas.size:
            raise ValueError("zone_ids et areas doivent avoir la même longueur")
        if np.any((zone_ids < 0) | (zone_ids >= 17)):
            raise ValueError("zone_ids doit être compris entre 0 et 16")
        if np.any(areas <= 0):
            raise ValueError("Les aires nodales doivent être positives")
        object.__setattr__(self, "zone_ids", zone_ids)
        object.__setattr__(self, "areas", areas)
        if self.points is not None:
            points = np.asarray(self.points, dtype=float)
            if points.shape != (zone_ids.size, 3):
                raise ValueError("points doit être de forme (nombre_de_noeuds, 3)")
            object.__setattr__(self, "points", points)

    @property
    def zone_areas(self) -> np.ndarray:
        result = np.zeros(17)
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
        h = np.asarray(h, dtype=float).reshape(-1)
        ts = np.asarray(surface_temperature, dtype=float).reshape(-1)
        ta = np.asarray(air_temperature, dtype=float).reshape(-1)
        if not (h.size == ts.size == ta.size):
            raise ValueError("Les trois champs nodaux doivent avoir la même longueur")
        outward = h * (ts - ta)
        body = -outward if outward_positive else outward
        return outward, body

    def exchange(self, h, surface_temperature, air_temperature) -> ThermalExchange:
        outward, body = self.nodal_flux(
            h, surface_temperature, air_temperature,
            outward_positive=self.outward_positive,
        )
        if body.size != self.mapping.zone_ids.size:
            raise ValueError("Les champs nodaux ne correspondent pas au mapping")
        zone_area = self.mapping.zone_areas
        zone_power = np.zeros(17)
        zone_h = np.zeros(17)
        zone_ta = np.zeros(17)
        np.add.at(zone_power, self.mapping.zone_ids, body * self.mapping.areas)
        np.add.at(zone_h, self.mapping.zone_ids, np.asarray(h) * self.mapping.areas)
        np.add.at(zone_ta, self.mapping.zone_ids, np.asarray(air_temperature) * self.mapping.areas)
        zone_flux_mean = np.divide(
            zone_power, zone_area, out=np.zeros(17), where=zone_area > 0
        )
        zone_h_mean = np.divide(zone_h, zone_area, out=np.zeros(17), where=zone_area > 0)
        zone_air_temperature = np.divide(
            zone_ta, zone_area, out=np.zeros(17), where=zone_area > 0
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
        if anchor_conductance is None:
            # Conductance from each JOS-3 skin node to its inner tissues.
            conductance = np.sum(model._cdt[skin_indices, :], axis=1)
        else:
            conductance = np.asarray(anchor_conductance, dtype=float).reshape(17)
        if np.any(conductance <= 0):
            raise ValueError("Les conductances d’ancrage doivent être positives")
        self.anchor_conductance = conductance[mapping.zone_ids] * area_fraction
        if surface_temperature is None:
            self.surface_temperature = model.Tsk[mapping.zone_ids].copy()
        else:
            surface_temperature = np.asarray(surface_temperature, dtype=float).reshape(-1)
            if surface_temperature.size != mapping.zone_ids.size:
                raise ValueError("surface_temperature ne correspond pas au mapping")
            self.surface_temperature = surface_temperature.copy()
        self.last_exchange: DistributedSurfaceExchange | None = None
        model.set_environment_mode("external_surface")

    def step(self, h, air_temperature, *, dtime=1.0, hr=0.0):
        """Avance le réseau et JOS-3 d’un pas en échangeant tous les points."""
        h = np.asarray(h, dtype=float).reshape(-1)
        ta = np.asarray(air_temperature, dtype=float).reshape(-1)
        if h.size != self.mapping.zone_ids.size or ta.size != h.size:
            raise ValueError("h et air_temperature ne correspondent pas au mapping")
        if np.any(h < 0) or hr < 0:
            raise ValueError("Les coefficients d’échange doivent être positifs")
        zone_ta = np.zeros(17)
        np.add.at(zone_ta, self.mapping.zone_ids, ta * self.mapping.areas)
        zone_ta = np.divide(zone_ta, self.zone_area, out=np.zeros(17), where=self.zone_area > 0)
        self.model.Ta = zone_ta
        skin_temperature = self.model.Tsk[self.mapping.zone_ids]
        body_power = self.anchor_conductance * (self.surface_temperature - skin_temperature)
        environment_power = (h + hr) * (self.surface_temperature - ta) * self.mapping.areas
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
                dtime=dtime, hr=hr,
            )
            provider.write_nodal_flux(exchange.environment_power.copy())
            exchanges.append(exchange)
        return exchanges
