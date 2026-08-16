import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh
import vtk

from .centerline_extractor import CenterlineExtractor
from .section_extractor import SectionExtractor
from .occ_builder import OCCBuilder

logger = logging.getLogger(__name__)


class CADReconstruction:
    def __init__(self, case_dir: Path, centerline_spacing_mm: float = 2.0):
        self.case_dir = Path(case_dir)
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.centerline_spacing_mm = centerline_spacing_mm
        self.centerline_extractor = CenterlineExtractor(resampling_step_mm=centerline_spacing_mm)
        self.section_extractor = SectionExtractor(spacing_mm=centerline_spacing_mm)
        self.occ_builder = OCCBuilder()

    def run(self, stl_path: Path, labels: Optional[dict] = None):
        labels = labels or {"tl": 1, "fl": 2}
        tl_stl = self.case_dir / "tbad_TL_walls.stl"
        fl_stl = self.case_dir / "tbad_FL_walls.stl"
        for src, dst in [(stl_path, tl_stl), (stl_path, fl_stl)]:
            if not src.exists():
                raise FileNotFoundError(src)
            if src != dst:
                dst.write_bytes(src.read_bytes())

        centerline = self.centerline_extractor.extract(tl_stl)

        # Save centerline as .npy for use in mesh generation
        np.save(self.case_dir / "centerline.npy", centerline)
        logger.info("Centerline saved to %s", self.case_dir / "centerline.npy")

        mesh = trimesh.load(tl_stl)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump(concatenate=True)
        sections = self.section_extractor.extract(mesh, centerline)
        loft_result = self.occ_builder.build_from_sections(sections, case_dir=self.case_dir)

        return {
            "centerline_points": int(centerline.shape[0]),
            "sections": len(sections),
            "loft_result": loft_result,
            "centerline_file": str(self.case_dir / "centerline.npy"),
        }
