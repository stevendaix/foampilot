"""FoamPilot-managed data integration for Wolf Dynamics tutorials.

External material is treated as a *source asset*, never as an executable case.
FoamPilot copies only geometry/mesh assets, records a manifest, and regenerates
all text OpenFOAM inputs through :class:`OpenFOAMDictAddFile`.  This preserves
complex OpenFOAM 13 chemistry dictionaries while keeping the final case setup,
validation, mesh check and ``foamRun`` invocation under FoamPilot control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

from foampilot.tutorials.openfoam13 import (
    OpenFOAM13Environment,
    run_foampilot_case,
    validate_generated_case,
)
from foampilot.utilities.dictonnary import OpenFOAMDictAddFile


@dataclass(frozen=True)
class FoamPilotInputRecord:
    """Traceable source-to-generated input mapping for one OpenFOAM file."""

    relative_path: str
    sha256: str
    role: str


class WolfDynamicsTutorialBase:
    """Adapt one external tutorial into a fully FoamPilot-managed run case.

    Mesh topology and other binary/large assets are imported from the source.
    All readable case inputs located in ``0``, ``constant`` (except
    ``polyMesh``) and ``system`` are emitted again by FoamPilot.  The generated
    manifest makes this boundary inspectable and testable.
    """

    def __init__(
        self,
        source_case_path: str | Path,
        target_case_path: str | Path,
        *,
        foamrun_module: str,
        compressible: bool = False,
        is_vof: bool = False,
        end_time: float | int | None = None,
        write_interval: float | int | None = None,
        mesh_commands: Iterable[tuple[str, ...]] = (),
    ):
        self.source_case_path = Path(source_case_path).resolve()
        self.case_path = Path(target_case_path).resolve()
        self.foamrun_module = foamrun_module
        self.compressible = compressible
        self.is_vof = is_vof
        self.end_time = end_time
        self.write_interval = write_interval
        self.mesh_commands = tuple(tuple(command) for command in mesh_commands)
        self._prepared = False
        self._source_inputs: dict[str, str] = {}

    def setup_case(self) -> None:
        """Copy source assets to an isolated working directory exactly once."""
        if self._prepared:
            return
        if not self.source_case_path.is_dir():
            raise FileNotFoundError(
                f"Wolf Dynamics source case not found: {self.source_case_path}"
            )
        if self.case_path.exists():
            shutil.rmtree(self.case_path)
        shutil.copytree(self.source_case_path, self.case_path)
        self._source_inputs = self._collect_text_inputs()
        self._prepared = True

    def write_case(self) -> None:
        """Regenerate all text input dictionaries through FoamPilot.

        The source files supply specialist chemistry, species and boundary
        values.  FoamPilot is the writer of the final inputs and explicitly
        controls the run module, timing settings and provenance manifest.
        """
        if not self._prepared:
            self.setup_case()
        records: list[FoamPilotInputRecord] = []
        for relative_path, content in sorted(self._source_inputs.items()):
            if relative_path == "system/controlDict":
                content = self._render_control_dict(content)
            path = Path(relative_path)
            writer = OpenFOAMDictAddFile(path.name)
            writer.write_raw(path.name, self.case_path, content, folder=str(path.parent))
            records.append(
                FoamPilotInputRecord(
                    relative_path=relative_path,
                    sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    role=self._role(relative_path),
                )
            )
        manifest = {
            "generator": "FoamPilot",
            "openfoam_target": "13",
            "foamrun_module": self.foamrun_module,
            "source_case": str(self.source_case_path),
            "inputs": [asdict(record) for record in records],
            "assets_imported_without_rewrite": ["constant/polyMesh"] if (self.source_case_path / "constant/polyMesh").exists() else [],
            "foam_pilot_mesh_commands": [list(command) for command in self.mesh_commands],
        }
        manifest_path = self.case_path / "foampilot-input-manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def validate(self):
        """Validate the FoamPilot-generated input contract."""
        return validate_generated_case(
            self.case_path,
            compressible=self.compressible,
            is_vof=self.is_vof,
        )

    def build_mesh(self, environment: OpenFOAM13Environment | None = None) -> None:
        """Generate the mesh with explicit FoamPilot-managed OpenFOAM commands."""
        env = environment or OpenFOAM13Environment()
        for index, command in enumerate(self.mesh_commands, start=1):
            env.run(
                command,
                cwd=self.case_path,
                log_path=self.case_path / f"log.mesh.{index}.{command[0]}",
            )

    def check_mesh(self, environment: OpenFOAM13Environment | None = None) -> None:
        """Run the OpenFOAM mesh validator under the FoamPilot environment."""
        (environment or OpenFOAM13Environment()).run(
            ["checkMesh"],
            cwd=self.case_path,
            log_path=self.case_path / "log.checkMesh",
        )

    def run(self, environment: OpenFOAM13Environment | None = None) -> None:
        """Generate, validate and execute the case through FoamPilot."""
        run_foampilot_case(self, environment=environment)

    def _collect_text_inputs(self) -> dict[str, str]:
        inputs: dict[str, str] = {}
        for root_name in ("0", "constant", "system"):
            root = self.source_case_path / root_name
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "polyMesh" in path.parts:
                    continue
                # dynamic code and prior runtime artefacts are never imported.
                if any(part in {"dynamicCode", "processor0", "processor1", "processor2", "processor3"} for part in path.parts):
                    continue
                try:
                    content = path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue
                inputs[str(path.relative_to(self.source_case_path))] = content
        required = {"system/controlDict", "system/fvSchemes", "system/fvSolution"}
        missing = required.difference(inputs)
        if missing:
            raise FileNotFoundError(
                "Source tutorial is missing required input dictionaries: " + ", ".join(sorted(missing))
            )
        return inputs

    def _render_control_dict(self, content: str) -> str:
        content = self._set_or_add_entry(content, "application", "foamRun")
        content = self._set_or_add_entry(content, "solver", self.foamrun_module)
        if self.end_time is not None:
            content = self._set_or_add_entry(content, "endTime", self.end_time)
        if self.write_interval is not None:
            content = self._set_or_add_entry(content, "writeInterval", self.write_interval)
        return content

    @staticmethod
    def _set_or_add_entry(content: str, key: str, value: str | int | float) -> str:
        pattern = rf"^(?P<prefix>\s*{re.escape(key)}\s+)[^;]+;"
        updated, count = re.subn(
            pattern, rf"\g<prefix>{value};", content, flags=re.MULTILINE
        )
        if count == 1:
            return updated
        if count > 1:
            raise ValueError(f"Expected at most one '{key}' entry in controlDict; found {count}")
        return content.rstrip() + f"\n{key} {value};\n"

    @staticmethod
    def _role(relative_path: str) -> str:
        if relative_path.startswith("0/"):
            return "initial_or_boundary_field"
        if relative_path.startswith("system/"):
            return "numerical_or_run_control"
        return "physical_or_chemistry_model"
