from pathlib import Path

import pytest

from foampilot.solver.workflows import OpenFOAMEnvironment, RunWorkflow


class DummySolver:
    foamrun_module = "fluid"

    def __init__(self):
        self.commands = []

    def run_command(self, command, log_filename, environment=None):
        self.commands.append((command, log_filename, environment))


def test_environment_missing_bashrc_has_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="bashrc"):
        OpenFOAMEnvironment(tmp_path / "missing-bashrc").environment()


def test_utility_and_foam_run_use_managed_environment():
    solver = DummySolver()
    workflow = RunWorkflow(solver, environment={"FOAMPILOT_TEST": "1"})

    workflow.utility("blockMesh")
    workflow.foam_run()

    assert solver.commands[0][0] == ["blockMesh"]
    assert solver.commands[0][1] == "log.blockMesh"
    assert solver.commands[0][2]["FOAMPILOT_TEST"] == "1"
    assert solver.commands[1][0] == ["foamRun", "-solver", "fluid"]


def test_parallel_workflow_has_decompose_run_and_reconstruct_steps():
    solver = DummySolver()
    workflow = RunWorkflow(solver, environment={"MPI_BUFFER_SIZE": "20000000"})

    workflow.parallel(4)

    assert [entry[0][0] for entry in solver.commands] == [
        "decomposePar", "mpirun", "reconstructPar"
    ]
    assert solver.commands[1][0][-1] == "-parallel"
    assert solver.commands[0][2]["MPI_BUFFER_SIZE"] == "20000000"


def test_parallel_rejects_one_process():
    with pytest.raises(ValueError, match="at least two"):
        RunWorkflow(DummySolver()).parallel(1)
