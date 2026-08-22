from pathlib import Path

from foampilot.report import CFDReportGenerator, SimulationReport


def test_simulation_report_keeps_extracted_settings(tmp_path, monkeypatch):
    log_path = tmp_path / "log.simpleFoam"
    log_path.write_text(
        "Time = 1\n"
        "smoothSolver:  Solving for U, Initial residual = 1, Final residual = 1e-5, No Iterations 2\n"
        "ExecutionTime = 0.5 s  ClockTime = 1 s\n"
        "End\n",
        encoding="utf-8",
    )

    report = SimulationReport(tmp_path)
    monkeypatch.setattr(
        report, "_extract_solver_settings", lambda: {"application": "simpleFoam"}
    )
    monkeypatch.setattr(
        report, "_extract_bc_summary", lambda: {"inlet": {"type": "fixedValue"}}
    )

    content = report.generate_report()

    assert report.solver_settings == {"application": "simpleFoam"}
    assert report.bc_summary == {"inlet": {"type": "fixedValue"}}
    assert "simpleFoam" in content
    assert "fixedValue" in content


def test_typst_report_renders_registered_content(tmp_path):
    generator = CFDReportGenerator(tmp_path)
    generator.add_statistic("Re", 1000, "-", "Reynolds number")
    generator.add_table([[1, 2]], ["A", "B"], "Results")
    generator.add_figure("mesh.png", "Mesh", label="fig_mesh")

    output = generator.save_typst_report()
    content = output.read_text(encoding="utf-8")

    assert output == tmp_path / "report" / "cfd_report.typ"
    assert "Reynolds number" in content
    assert "Results" in content
    assert "mesh.png" in content


def test_latex_report_returns_generated_path(tmp_path):
    generator = CFDReportGenerator(tmp_path)
    output = generator.save_latex_report()

    assert isinstance(output, Path)
    assert output == tmp_path / "report" / "cfd_report.tex"
    assert output.exists()
