from __future__ import annotations

from pathlib import Path

from foampilot.solver.base_solver import BaseSolver


def test_base_solver_uses_core_case_layout(tmp_path: Path) -> None:
    solver = BaseSolver(tmp_path / "case", solver_name="incompressibleFluid")
    solver.ensure_dirs()
    assert solver.case_layout.case_path == (tmp_path / "case").resolve()
    assert all((tmp_path / "case" / name).is_dir() for name in ("0", "0.orig", "constant", "system"))
