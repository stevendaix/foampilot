"""Build and preview the Foampilot solids4foam beam example."""

from pathlib import Path

from foampilot.solids4foam import build_beam_in_cross_flow


if __name__ == "__main__":
    case_path = Path(__file__).resolve().parent / "case"
    _, workflow = build_beam_in_cross_flow(case_path, parallel=False)
    print(workflow.preview())
