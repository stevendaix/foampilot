import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from foampilot.openfoam.execution.environment import OpenFOAMEnvironment


class OpenFOAMRunner:
    """Execute OpenFOAM commands, parallel runs, and legacy solvers."""

    def __init__(self, case_path: Path, env: Optional[OpenFOAMEnvironment] = None):
        self.case_path = Path(case_path)
        self._env = env or OpenFOAMEnvironment()

    def _process_environment(self, env: Optional[Dict[str, str]] = None, environment: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        process_environment = self._env.command_environment()
        if env:
            process_environment.update(env)
        if environment:
            process_environment.update(environment)
        return process_environment

    def run_command(
        self,
        cmd: Sequence[str],
        log_filename: str,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        workdir = cwd if cwd is not None else self.case_path
        log_path = self.case_path / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        process_environment = self._process_environment(env=env, environment=environment)
        with log_path.open("w", encoding="utf-8") as log_file:
            return subprocess.run(
                list(cmd), cwd=workdir, env=process_environment, text=True,
                stdout=log_file, stderr=subprocess.STDOUT, check=True,
            )

    def run_external(
        self,
        cmd: Sequence[str],
        log_filename: str,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        return self.run_command(cmd, log_filename, cwd=cwd, env=env)

    def run_command_async(self, cmd: Sequence[str], log_filename: str):
        log_path = self.case_path / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            list(cmd), cwd=self.case_path, env=self._env.command_environment(), text=True,
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        process._foampilot_log_file = log_file
        return process

    def wait_command(self, process, check: bool = True) -> int:
        returncode = process.wait()
        log_file = getattr(process, "_foampilot_log_file", None)
        if log_file is not None:
            log_file.close()
        if check and returncode != 0:
            raise subprocess.CalledProcessError(returncode, process.args)
        return returncode

    def validate_results(self, solver_name: str, log_filename: Optional[str] = None) -> Path:
        log_path = self.case_path / (log_filename or f"log.{solver_name}")
        if not log_path.is_file():
            raise RuntimeError(f"Solver log was not produced: {log_path}")
        if "End" not in log_path.read_text(encoding="utf-8", errors="replace"):
            raise RuntimeError(f"Solver did not finish successfully: {log_path}")
        times = []
        for path in self.case_path.iterdir():
            if not path.is_dir():
                continue
            try:
                float(path.name)
            except ValueError:
                continue
            times.append(path)
        if not times:
            raise RuntimeError("The solver produced no numeric time directory")
        return max(times, key=lambda p: float(p.name))

    def check_solver_module_exists(self, foamrun_module: str) -> bool:
        foam_modules = os.getenv("FOAM_MODULES", "")
        if not foam_modules:
            return shutil.which("foamRun") is not None
        module_dir = Path(foam_modules)
        module_candidates = (
            module_dir / foamrun_module,
            module_dir / f"lib{foamrun_module}.so",
            module_dir / f"lib{foamrun_module}.dylib",
        )
        if not any(candidate.exists() for candidate in module_candidates):
            return False
        return True

    def run_simulation(self, solver_name: str, foamrun_module: str, nb_proc: int = 1, log_filename: Optional[str] = None):
        legacy_solvers = {"overInterDyMFoam", "rhoSimpleFoam", "simpleFoam", "pimpleFoam", "marineFoam"}
        if solver_name in legacy_solvers:
            self._run_legacy_solver(solver_name, nb_proc, log_filename)
            return

        if nb_proc >= 2:
            return self.run_parallel(foamrun_module, nb_proc, log_filename)

        if log_filename is None:
            log_filename = f"log.{solver_name}"

        if not self.check_solver_module_exists(foamrun_module):
            raise RuntimeError(
                f"Solver module '{foamrun_module}' is not available."
            )

        self.run_command(["foamRun", "-solver", foamrun_module], log_filename)

    def _run_legacy_solver(self, solver_name: str, nb_proc: int, log_filename: Optional[str] = None) -> None:
        if log_filename is None:
            log_filename = f"log.{solver_name}"

        if nb_proc >= 2:
            with open(self.case_path / log_filename, "w", encoding="utf-8") as log_file:
                log_file.write("=== decomposePar ===\n")
                subprocess.run(
                    ["decomposePar", "-case", str(self.case_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                log_file.write(f"\n=== mpirun {solver_name} ===\n")
                subprocess.run(
                    ["mpirun", "--oversubscribe", "-np", str(nb_proc), solver_name, "-parallel"],
                    cwd=self.case_path,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                log_file.write("\n=== reconstructPar ===\n")
                subprocess.run(
                    ["reconstructPar", "-case", str(self.case_path)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        else:
            self.run_command([solver_name], log_filename)

    def run_parallel(self, foamrun_module: str, nb_proc: int, log_filename: Optional[str] = None, force_decompose: bool = False):
        if log_filename is None:
            log_filename = f"log.{foamrun_module}"
        log_path = self.case_path / log_filename

        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write("=== decomposePar ===\n")
            subprocess.run(
                ["decomposePar", "-force", "-case", str(self.case_path)] if force_decompose else ["decomposePar", "-case", str(self.case_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== mpirun foamRun ===\n")
            mpi_command = ["mpirun"]
            if os.getenv("FOAMPILOT_MPI_OVERSUBSCRIBE", "1").lower() not in {"0", "false", "no"}:
                mpi_command.append("--oversubscribe")
            mpi_command += ["-np", str(nb_proc), "foamRun", "-solver", foamrun_module, "-parallel"]
            subprocess.run(
                mpi_command,
                cwd=self.case_path,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write("\n=== reconstructPar ===\n")
            subprocess.run(
                ["reconstructPar", "-case", str(self.case_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
