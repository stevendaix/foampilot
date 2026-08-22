"""Wolf Dynamics tutorial adapters for FoamPilot.

This module provides wrappers to safely generate and run Wolf Dynamics
tutorials (from the Figshare archive) within the FoamPilot OpenFOAM 13
environment. It enforces the minimum FoamPilot contract before launching
``foamRun`` and never executes third-party ``Allrun`` scripts.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil

from foampilot.tutorials.openfoam13 import (
    OpenFOAM13Environment,
    run_foampilot_case,
    validate_generated_case,
)


class WolfDynamicsTutorial:
    """Adapt a Wolf Dynamics OpenFOAM 13 tutorial to the FoamPilot contract.

    The source is copied into a disposable target directory.  No external
    ``Allrun`` script is called.  This keeps the command boundary explicit:
    FoamPilot first validates the copied case, optionally controls its short
    smoke-test horizon, checks its mesh, then invokes ``foamRun``.
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
        """Prepare a disposable case directory without running solvers.

        The method is idempotent within one adapter instance.  This lets the
        wrapper run ``checkMesh`` before ``run_foampilot_case`` without the
        latter deleting the checked copy and its log files.
        """
        if self._prepared:
            return
        if not self.source_case_path.is_dir():
            raise FileNotFoundError(
                f"Wolf Dynamics source case not found: {self.source_case_path}"
            )
        if self.case_path.exists():
            shutil.rmtree(self.case_path)
        shutil.copytree(self.source_case_path, self.case_path)
        self._prepared = True

    def write_case(self) -> None:
        """Apply explicit FoamPilot smoke-test controls when requested."""
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
        """Return FoamPilot's structural validation of the generated case."""
        return validate_generated_case(
            self.case_path, compressible=self.compressible
        )

    def check_mesh(
        self, environment: OpenFOAM13Environment | None = None
    ) -> None:
        """Run OpenFOAM's mesh validator and save its log in the case."""
        (environment or OpenFOAM13Environment()).run(
            ["checkMesh"],
            cwd=self.case_path,
            log_path=self.case_path / "log.checkMesh",
        )

    def run(self, environment: OpenFOAM13Environment | None = None) -> None:
        """Generate, validate and run with FoamPilot's OpenFOAM 13 boundary."""
        run_foampilot_case(self, environment=environment)

    @staticmethod
    def _replace_scalar_entry(content: str, key: str, value: float | int) -> str:
        pattern = rf"^(?P<prefix>\s*{re.escape(key)}\s+)[^;]+;"
        updated, count = re.subn(
            pattern, rf"\g<prefix>{value};", content, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(
                f"Expected exactly one '{key}' entry in system/controlDict; found {count}"
            )
        return updated
