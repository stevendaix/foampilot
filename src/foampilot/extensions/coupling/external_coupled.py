"""File-based temperature coupling compatible with OpenFOAM 13.

The implementation follows the OpenFOAM ``externalCoupledTemperature``
contract and deliberately has no dependency on preCICE.  It is intended to be
used by a MOOSE external participant or by a validation participant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Iterable


class ExternalCouplingTimeout(TimeoutError):
    """Raised when OpenFOAM does not complete a file handshake in time."""


@dataclass(frozen=True)
class CoupledPatchData:
    """One face record exchanged by externalCoupledTemperature.

    ``area`` and ``temperature`` are received from OpenFOAM. ``heat_flux`` and
    ``heat_transfer_coefficient`` are also provided by OpenFOAM and can be used
    by the MOOSE participant to construct its boundary condition.
    """

    patch: str
    area: float
    temperature: float
    heat_flux: float
    heat_transfer_coefficient: float


class ExternalCoupledTemperature:
    """Coordinate one explicit OpenFOAM temperature coupling exchange.

    OpenFOAM writes ``<file>.out`` after removing ``OpenFOAM.lock``. The
    external participant writes ``<file>.in`` and recreates the lock file.
    """

    def __init__(
        self,
        comms_dir: str | Path,
        file_name: str = "temperature",
        wait_interval: float = 0.1,
        timeout: float = 120.0,
    ) -> None:
        if wait_interval <= 0 or timeout <= 0:
            raise ValueError("wait_interval and timeout must be positive")
        self.comms_dir = Path(comms_dir)
        self.file_name = file_name
        self.wait_interval = wait_interval
        self.timeout = timeout

    @property
    def output_file(self) -> Path:
        return self.comms_dir / f"{self.file_name}.out"

    @property
    def input_file(self) -> Path:
        return self.comms_dir / f"{self.file_name}.in"

    @property
    def lock_file(self) -> Path:
        return self.comms_dir / "OpenFOAM.lock"

    def wait_for_openfoam(self) -> list[CoupledPatchData]:
        """Wait until OpenFOAM publishes an ``.out`` file and lock is absent."""

        self._wait_until(lambda: self.output_file.exists() and not self.lock_file.exists())
        return self._read_output(self.output_file)

    def send_temperature_mixed_values(
        self,
        values: Iterable[tuple[float, float, float]],
    ) -> None:
        """Write ``value gradient valueFraction`` records and release OpenFOAM."""

        rows = list(values)
        if not rows:
            raise ValueError("at least one boundary record is required")
        self.comms_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.input_file.with_suffix(self.input_file.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for value, gradient, value_fraction in rows:
                stream.write(f"{value:.16g} {gradient:.16g} {value_fraction:.16g}\n")
        temporary.replace(self.input_file)
        self.lock_file.touch()

    def _wait_until(self, predicate) -> None:
        deadline = time.monotonic() + self.timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise ExternalCouplingTimeout(
                    f"Timed out waiting for OpenFOAM coupling files in {self.comms_dir}"
                )
            time.sleep(self.wait_interval)

    @staticmethod
    def _read_output(path: Path) -> list[CoupledPatchData]:
        records: list[CoupledPatchData] = []
        patch: str | None = None
        tokens: list[float] = []

        def flush() -> None:
            if patch is None:
                if tokens:
                    raise ValueError("OpenFOAM data appeared before a # Patch header")
                return
            if len(tokens) % 4:
                raise ValueError(
                    f"Expected groups of area, temperature, heat flux, htc for {patch}"
                )
            for offset in range(0, len(tokens), 4):
                area, temperature, heat_flux, htc = tokens[offset : offset + 4]
                records.append(
                    CoupledPatchData(patch, area, temperature, heat_flux, htc)
                )

        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("//"):
                continue
            if line.startswith("#"):
                if line.startswith("# Values:"):
                    # OpenFOAM writes all configured patches as one ordered
                    # stream for this boundary condition.
                    patch = "all"
                    continue
                if not line.startswith("# Patch:"):
                    continue
                flush()
                tokens = []
                parts = line.split()
                if len(parts) < 3:
                    raise ValueError(f"Malformed patch header: {line!r}")
                patch = parts[2]
                try:
                    tokens.extend(float(value) for value in parts[3:])
                except ValueError as error:
                    raise ValueError(f"Malformed patch data: {line!r}") from error
                continue
            if patch is None:
                patch = "all"
            tokens.extend(float(value) for value in line.split())

        flush()
        if not records:
            raise ValueError(f"No coupling records found in {path}")
        return records
