"""
Wolf Dynamics tutorial integration base for FoamPilot.

This module defines the common contract to integrate an external tutorial 
into the FoamPilot framework. It enforces the generation of missing OpenFOAM 
dictionaries (e.g., transportProperties) and validates the case structure 
before execution, preventing the use of unsafe third-party shell scripts.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import warnings

from foampilot.tutorials.openfoam13 import (
    OpenFOAM13Environment,
    run_foampilot_case,
    validate_generated_case,
)


class WolfDynamicsTutorialBase:
    """Base class to adapt a Wolf Dynamics tutorial into FoamPilot.
    
    Each specific tutorial should inherit from this class, provide its source
    directory, and implement `write_case` to inject or modify dictionaries 
    required by OpenFOAM 13 (e.g., adding `nu` for incompressible flows).
    """

    def __init__(
        self,
        source_case_path: str | Path,
        target_case_path: str | Path,
        *,
        foamrun_module: str,
        compressible: bool = False,
        end_time: float | int | None = None,
        write_interval: float | int | None = None,
    ):
        self.source_case_path = Path(source_case_path).resolve()
        self.case_path = Path(target_case_path).resolve()
        self.foamrun_module = foamrun_module
        self.compressible = compressible
        self.end_time = end_time
        self.write_interval = write_interval
        self._prepared = False

    def setup_case(self) -> None:
        """Copy the source case to a disposable target directory.
        
        This method is idempotent to allow running `checkMesh` and `foamRun`
        in sequence without deleting the validated case.
        """
        if self._prepared:
            return
        if not self.source_case_path.is_dir():
            raise FileNotFoundError(
                f"Source case not found: {self.source_case_path}"
            )
        if self.case_path.exists():
            shutil.rmtree(self.case_path)
        shutil.copytree(self.source_case_path, self.case_path)
        self._prepared = True

    def write_case(self) -> None:
        """Apply explicit FoamPilot modifications to the case.
        
        By default, this enforces smoke-test controls (endTime, writeInterval)
        if provided. Subclasses must call `super().write_case()` and add their
        own dictionary injections.
        """
        control = self.case_path / "system" / "controlDict"
        if not control.exists():
            return
        content = control.read_text(encoding="utf-8", errors="ignore")
        if self.end_time is not None:
            content = self._replace_scalar_entry(content, "endTime", self.end_time)
        if self.write_interval is not None:
            content = self._replace_scalar_entry(
                content, "writeInterval", self.write_interval
            )
        control.write_text(content, encoding="utf-8")

    def validate(self):
        """Validate the generated case against FoamPilot constraints."""
        return validate_generated_case(
            self.case_path, compressible=self.compressible
        )

    def check_mesh(
        self, environment: OpenFOAM13Environment | None = None
    ) -> None:
        """Run checkMesh and save the log in the case directory."""
        env = environment or OpenFOAM13Environment()
        env.run(
            ["checkMesh"],
            cwd=self.case_path,
            log_path=self.case_path / "log.checkMesh",
        )

    def run(self, environment: OpenFOAM13Environment | None = None) -> None:
        """Validate and run the case using FoamPilot's OpenFOAM 13 boundary."""
        run_foampilot_case(self, environment=environment)

    @staticmethod
    def _replace_scalar_entry(content: str, key: str, value: float | int) -> str:
        pattern = rf"^(?P<prefix>\s*{re.escape(key)}\s+)[^;]+;"
        updated, count = re.subn(
            pattern, rf"\g<prefix>{value};", content, flags=re.MULTILINE
        )
        if count != 1:
            warnings.warn(f"Expected 1 '{key}' entry in controlDict; found {count}")
        return updated
