import logging
from pathlib import Path
from typing import Optional

from .cad_reconstruction import CADReconstruction


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = Path("/home/steven/foampilot/examples/coa")
    stl = base / "data_preproc" / "tbad_stl_output" / "tbad_TL_walls.stl"
    case_dir = base / "cad_output" / "patient58"
    if not stl.exists():
        raise SystemExit(f"Missing STL: {stl}")
    recon = CADReconstruction(case_dir=case_dir, centerline_spacing_mm=2.0)
    result = recon.run(stl)
    logging.info("Done: %s", result)


if __name__ == "__main__":
    main()
