import os
import subprocess
from pathlib import Path
from typing import Dict


class OpenFOAMEnvironment:
    """OpenFOAM environment discovery and sourcing."""

    def command_environment(self) -> Dict[str, str]:
        environment = os.environ.copy()
        bashrc = environment.get("FOAM_BASHRC", "")
        if not bashrc and environment.get("WM_PROJECT_DIR"):
            bashrc = str(Path(environment["WM_PROJECT_DIR"]) / "etc" / "bashrc")
        if not bashrc:
            return environment
        bashrc_path = Path(bashrc).expanduser()
        if not bashrc_path.is_file():
            raise FileNotFoundError(f"OpenFOAM bashrc not found: {bashrc_path}")
        result = subprocess.run(
            ["bash", "-lc", 'source "$1" >/dev/null 2>&1 && env -0', "foampilot-env", str(bashrc_path)],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to source OpenFOAM bashrc: {bashrc_path}")
        sourced = {}
        for item in result.stdout.split(b"\0"):
            if b"=" in item:
                key, value = item.split(b"=", 1)
                sourced[key.decode()] = value.decode(errors="replace")
        return sourced

    @staticmethod
    def openfoam_version() -> str | None:
        import shutil
        version = os.environ.get("WM_PROJECT_VERSION")
        if version:
            return version
        foam_version = shutil.which("foamVersion")
        if not foam_version:
            return None
        result = subprocess.run([foam_version], capture_output=True, text=True, check=False)
        output = (result.stdout or result.stderr).strip()
        return output.removeprefix("OpenFOAM-") or None

    def require_openfoam(self, major: str | int | None = None) -> str:
        version = self.openfoam_version()
        if not version:
            raise RuntimeError("OpenFOAM is not sourced or foamVersion is unavailable")
        if major is not None and str(version) != str(major):
            raise RuntimeError(f"OpenFOAM {major} is required, found {version}")
        return version
