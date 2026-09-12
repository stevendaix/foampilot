from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from foampilot.solver.base_solver import BaseSolver


class BaseSolverExecutionTests(unittest.TestCase):
    def test_openfoam_version_from_environment(self) -> None:
        with patch.dict(os.environ, {"WM_PROJECT_VERSION": "13"}, clear=False):
            self.assertEqual(BaseSolver.openfoam_version(), "13")

    def test_validate_results_returns_latest_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            (case / "0").mkdir()
            (case / "1.1e-06").mkdir()
            (case / "log.foamRun").write_text("ExecutionTime = 0.1 s\nEnd\n", encoding="utf-8")
            solver = BaseSolver(case, "foamRun")
            self.assertEqual(solver.validate_results("log.foamRun").name, "1.1e-06")

    def test_run_external_writes_log_in_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case = Path(tmp)
            solver = BaseSolver(case, "foamRun")
            solver.run_external(["python3", "-c", "print('ok')"], "log.build")
            self.assertIn("ok", (case / "log.build").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
