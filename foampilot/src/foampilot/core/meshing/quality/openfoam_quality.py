#!/usr/bin/env python3
"""
Parse OpenFOAM checkMesh output and provide quality gates (Gate B).

Runs / reads checkMesh -allGeometry -allTopology logs and extracts:
    - max non-orthogonality, average non-orthogonality
    - P95 / P99 non-orthogonality (from histogram buckets when available)
    - faces > 65°, faces > 70°, faces > 75°
    - max skewness, average skewness
    - boundary skewness (when reported per-patch)
    - min volume, total volume
    - max aspect ratio
    - mesh topology counts (points, faces, cells, patches)

Computes a MeshScore (0-100):
     30 % non-orthogonality
     20 % skewness
     15 % bad cells
     15 % aspect ratio
     10 % volume quality
     10 % surface quality

Quality gates:
    GOOD      : maxNonOrtho < 65° and skewness < 4
    ACCEPTABLE: 65° ≤ maxNonOrtho ≤ 70°
    WARNING   : 70° < maxNonOrtho ≤ 75° or skewness ≥ 4
    BAD       : maxNonOrtho > 75° or skewness ≥ 8

Outputs:
    - quality_report.json
    - quality_report.csv
    - console report

Usage:
    PYTHONPATH=src python3 openfoam_quality.py \\
        --log log.checkMesh \\
        --case /path/to/case \\
        --json quality_report.json \\
        --csv quality_report.csv
"""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RE_NONORTH_MAX = re.compile(
    r"Mesh\s+non-orthogonality\s+Max:\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)
RE_NONORTH_AVG = re.compile(
    r"Mesh\s+non-orthogonality\s+Max:.*?average:\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)
RE_NONORTH_SEVERE_70 = re.compile(
    r"Number\s+of\s+severely\s+non-orthogonal\s*\(\s*>\s*70\s+degrees\s*\)\s+faces:\s*(\d+)",
    re.IGNORECASE,
)
RE_NONORTH_SEVERE_75 = re.compile(
    r"Number\s+of\s+severely\s+non-orthogonal\s*\(\s*>\s*75\s+degrees\s*\)\s+faces:\s*(\d+)",
    re.IGNORECASE,
)
RE_FACES_ABOVE_THRESHOLD = re.compile(
    r"Non-orthogonality\s+faces\s+>\s*(\d+)\s+deg\s*=\s*(\d+)", re.IGNORECASE
)
RE_MAX_SKEWNESS = re.compile(r"Max\s+skewness\s*=\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)
RE_BOUNDARY_SKEWNESS = re.compile(
    r"Max\s+boundary\s+skewness\s*=\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)
RE_PATCH_SKEWNESS = re.compile(
    r"\s+([A-Za-z0-9_\-]+)\s+Max\s+skewness\s*=\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)
RE_MIN_VOLUME = re.compile(r"Min\s+volume\s*=\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)
RE_TOTAL_VOLUME = re.compile(r"Total\s+volume\s*=\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE)
RE_MAX_ASPECT_RATIO = re.compile(
    r"Max\s+aspect\s+ratio\s*=\s*([\d]+(?:\.\d+)?(?:[eE][+-]?\d+)?)", re.IGNORECASE
)
RE_CELLS = re.compile(r"^\s+cells:\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_FACES = re.compile(r"^\s+faces:\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
RE_INTERNAL_FACES = re.compile(
    r"^\s+internal\s+faces:\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE
)
RE_BOUNDARY_PATCHES = re.compile(
    r"^\s+boundary\s+patches:\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE
)
RE_POINTS = re.compile(r"^\s+points:\s+(\d+)\s*$", re.IGNORECASE | re.MULTILINE)

RE_HISTOGRAM_ROW = re.compile(
    r"^\s*(?:<\s*(\d+)|(\d+)\s*-\s*(\d+)|>\s*(\d+))\s+(\d+)\s*$"
)


class CheckMeshParser:
    """Parse OpenFOAM checkMesh log and compute quality metrics."""

    def __init__(self, log_text: str):
        self.text = log_text
        self.metrics: Dict[str, Any] = {}
        self.histograms: Dict[str, List[Tuple[float, float, int]]] = {}
        self._parse()

    def _parse(self) -> None:
        self._parse_scalar_metrics()
        self._parse_mesh_stats()
        self._parse_histograms()

    def _parse_scalar_metrics(self) -> None:
        def _safe_float(pattern: re.Pattern, key: str) -> None:
            m = pattern.search(self.text)
            if m:
                try:
                    self.metrics[key] = float(m.group(1))
                except ValueError:
                    pass

        def _safe_int(pattern: re.Pattern, key: str) -> None:
            m = pattern.search(self.text)
            if m:
                try:
                    self.metrics[key] = int(m.group(1))
                except ValueError:
                    pass

        _safe_float(RE_NONORTH_MAX, "max_non_orthogonality")
        _safe_float(RE_NONORTH_AVG, "avg_non_orthogonality")
        _safe_int(RE_NONORTH_SEVERE_70, "faces_above_70")
        _safe_int(RE_NONORTH_SEVERE_75, "faces_above_75")

        m = RE_FACES_ABOVE_THRESHOLD.search(self.text)
        if m:
            thr = int(m.group(1))
            val = int(m.group(2))
            if thr == 65:
                self.metrics["faces_above_65"] = val
            elif thr == 70:
                self.metrics["faces_above_70"] = val
            elif thr == 75:
                self.metrics["faces_above_75"] = val

        if "faces_above_65" not in self.metrics:
            _safe_int(
                re.compile(
                    r"Non-orthogonality\s+faces\s+>\s*65\s+deg\s*=\s*(\d+)",
                    re.IGNORECASE,
                ),
                "faces_above_65",
            )

        _safe_float(RE_MAX_SKEWNESS, "max_skewness")
        _safe_float(RE_BOUNDARY_SKEWNESS, "boundary_skewness")

        patch_skew = {}
        for pm in RE_PATCH_SKEWNESS.finditer(self.text):
            patch_skew[pm.group(1)] = float(pm.group(2))
        if patch_skew:
            self.metrics["patch_skewness"] = patch_skew
            self.metrics["boundary_skewness"] = max(patch_skew.values())

        _safe_float(RE_MIN_VOLUME, "min_volume")
        _safe_float(RE_TOTAL_VOLUME, "total_volume")
        _safe_float(RE_MAX_ASPECT_RATIO, "max_aspect_ratio")

        m = RE_CELLS.search(self.text)
        if m:
            self.metrics["cell_count"] = int(m.group(1))
        m = RE_FACES.search(self.text)
        if m:
            self.metrics["face_count"] = int(m.group(1))
        m = RE_INTERNAL_FACES.search(self.text)
        if m:
            self.metrics["internal_faces"] = int(m.group(1))
        m = RE_BOUNDARY_PATCHES.search(self.text)
        if m:
            self.metrics["boundary_patches"] = int(m.group(1))
        m = RE_POINTS.search(self.text)
        if m:
            self.metrics["points"] = int(m.group(1))

        concave = re.search(
            r"Concave\s+cells\s+\(using\s+face\s+planes\)\s+found.*?number\s+of\s+cells:\s*(\d+)",
            self.text,
            re.IGNORECASE,
        )
        if concave:
            self.metrics["concave_cells"] = int(concave.group(1))

        determinant = re.search(
            r"Cells\s+with\s+small\s+determinant.*?number\s+of\s+cells:\s*(\d+)",
            self.text,
            re.IGNORECASE,
        )
        if determinant:
            self.metrics["small_determinant_cells"] = int(determinant.group(1))

        failed = re.search(r"Failed\s+(\d+)\s+mesh\s+checks", self.text, re.IGNORECASE)
        if failed:
            self.metrics["failed_checks"] = int(failed.group(1))

    def _parse_mesh_stats(self) -> None:
        pass

    def _parse_histograms(self) -> None:
        lines = self.text.splitlines()
        i = 0
        current_hist: Optional[str] = None
        current_rows: List[Tuple[float, float, int]] = []

        def _save_current() -> None:
            nonlocal current_hist, current_rows
            if current_hist is not None and current_rows:
                self.histograms[current_hist] = current_rows
            current_hist = None
            current_rows = []

        while i < len(lines):
            line = lines[i].strip()
            lowered = line.lower()

            if lowered.startswith("non-orthogonality faces:") or lowered.startswith(
                "skewness faces:"
            ):
                _save_current()
                current_hist = lowered.split()[0].replace(":", "")
                current_rows = []
                i += 1
                continue

            if current_hist is not None:
                m = RE_HISTOGRAM_ROW.match(line)
                if m:
                    if m.group(1) is not None:
                        lo = 0.0
                        hi = float(m.group(1))
                    elif m.group(2) is not None and m.group(3) is not None:
                        lo = float(m.group(2))
                        hi = float(m.group(3))
                    elif m.group(4) is not None:
                        lo = float(m.group(4))
                        hi = float("inf")
                    else:
                        i += 1
                        continue
                    count = int(m.group(5))
                    current_rows.append((lo, hi, count))
                elif line == "" or lowered.startswith("checking") or lowered.startswith("mesh"):
                    _save_current()
            i += 1

        _save_current()

    def _percentile_from_histogram(
        self, hist: List[Tuple[float, float, int]], pct: float
    ) -> Optional[float]:
        total = sum(c for _, _, c in hist)
        if total == 0:
            return None
        target = total * pct / 100.0
        cum = 0
        for lo, hi, count in hist:
            cum += count
            if cum >= target:
                if hi == float("inf"):
                    return lo
                return hi
        return None

    def _count_above_threshold(
        self, hist: List[Tuple[float, float, int]], threshold: float
    ) -> int:
        total = 0
        for lo, hi, count in hist:
            if lo >= threshold:
                total += count
            elif hi > threshold:
                total += count
        return total

    @property
    def p95_non_orthogonality(self) -> Optional[float]:
        if "non-orthogonality" in self.histograms:
            return self._percentile_from_histogram(
                self.histograms["non-orthogonality"], 95.0
            )
        return None

    @property
    def p99_non_orthogonality(self) -> Optional[float]:
        if "non-orthogonality" in self.histograms:
            return self._percentile_from_histogram(
                self.histograms["non-orthogonality"], 99.0
            )
        return None

    @property
    def faces_above_65_from_histogram(self) -> Optional[int]:
        if "non-orthogonality" in self.histograms:
            return self._count_above_threshold(self.histograms["non-orthogonality"], 65.0)
        return None

    @property
    def faces_above_70_from_histogram(self) -> Optional[int]:
        if "non-orthogonality" in self.histograms:
            return self._count_above_threshold(self.histograms["non-orthogonality"], 70.0)
        return None

    @property
    def faces_above_75_from_histogram(self) -> Optional[int]:
        if "non-orthogonality" in self.histograms:
            return self._count_above_threshold(self.histograms["non-orthogonality"], 75.0)
        return None

    @property
    def p95_skewness(self) -> Optional[float]:
        if "skewness" in self.histograms:
            return self._percentile_from_histogram(self.histograms["skewness"], 95.0)
        return None

    @property
    def p99_skewness(self) -> Optional[float]:
        if "skewness" in self.histograms:
            return self._percentile_from_histogram(self.histograms["skewness"], 99.0)
        return None


class QualityGate:
    """Evaluate mesh quality against thresholds and compute MeshScore."""

    THRESHOLDS = {
        "good": {"max_non_ortho": 65.0, "max_skewness": 4.0},
        "acceptable": {"max_non_ortho_hi": 70.0, "max_skewness_hi": 6.0},
        "warning": {"max_non_ortho_hi": 75.0, "max_skewness_hi": 8.0},
    }

    def __init__(self, metrics: Dict[str, Any]):
        self.metrics = metrics

    def gate(self) -> Tuple[str, str]:
        max_non_ortho = self.metrics.get("max_non_orthogonality", None)
        max_skewness = self.metrics.get("max_skewness", None)

        if max_non_ortho is None and max_skewness is None:
            return "UNKNOWN", "Insufficient data"

        non_ortho_bad = max_non_ortho is not None and max_non_ortho > 75.0
        skewness_bad = max_skewness is not None and max_skewness >= 8.0

        if non_ortho_bad or skewness_bad:
            return "BAD", (
                f"maxNonOrtho={max_non_ortho}°, maxSkewness={max_skewness}"
                if max_non_ortho is not None and max_skewness is not None
                else f"maxNonOrtho={max_non_ortho}°"
                if max_non_ortho is not None
                else f"maxSkewness={max_skewness}"
            )

        non_ortho_warning = max_non_ortho is not None and max_non_ortho > 70.0
        skewness_warning = max_skewness is not None and max_skewness >= 6.0

        if non_ortho_warning or skewness_warning:
            return "WARNING", (
                f"maxNonOrtho={max_non_ortho}°, maxSkewness={max_skewness}"
                if max_non_ortho is not None and max_skewness is not None
                else f"maxNonOrtho={max_non_ortho}°"
                if max_non_ortho is not None
                else f"maxSkewness={max_skewness}"
            )

        non_ortho_ok = max_non_ortho is not None and max_non_ortho <= 70.0
        skewness_ok = max_skewness is not None and max_skewness < 6.0

        if non_ortho_ok and skewness_ok:
            if max_non_ortho is not None and max_non_ortho < 65.0 and (max_skewness is None or max_skewness < 4.0):
                return "GOOD", "Within tight tolerances"
            return "ACCEPTABLE", "Within acceptable tolerances"

        return "ACCEPTABLE", "Within acceptable tolerances"

    def _score_non_ortho(self) -> float:
        val = self.metrics.get("max_non_orthogonality", None)
        if val is None:
            return 1.0
        if val < 65.0:
            return 1.0
        if val > 90.0:
            return 0.0
        return max(0.0, 1.0 - (val - 65.0) / 25.0)

    def _score_skewness(self) -> float:
        val = self.metrics.get("max_skewness", None)
        if val is None:
            return 1.0
        if val < 4.0:
            return 1.0
        if val > 12.0:
            return 0.0
        return max(0.0, 1.0 - (val - 4.0) / 8.0)

    def _score_bad_cells(self) -> float:
        cell_count = self.metrics.get("cell_count", 0)
        internal_faces = self.metrics.get("internal_faces", 0)

        bad_faces = 0
        if "faces_above_70" in self.metrics:
            bad_faces += self.metrics["faces_above_70"]
        if "concave_cells" in self.metrics:
            bad_faces += self.metrics["concave_cells"]
        if "small_determinant_cells" in self.metrics:
            bad_faces += self.metrics["small_determinant_cells"]

        denominator = max(cell_count, internal_faces, 1)
        bad_fraction = bad_faces / denominator
        if bad_fraction <= 0.0:
            return 1.0
        if bad_fraction >= 0.05:
            return 0.0
        return max(0.0, 1.0 - bad_fraction / 0.05)

    def _score_aspect_ratio(self) -> float:
        val = self.metrics.get("max_aspect_ratio", None)
        if val is None:
            return 1.0
        if val <= 100.0:
            return 1.0
        if val >= 1000.0:
            return 0.0
        return max(0.0, 1.0 - (val - 100.0) / 900.0)

    def _score_volume(self) -> float:
        min_vol = self.metrics.get("min_volume", None)
        total_vol = self.metrics.get("total_volume", None)
        cell_count = self.metrics.get("cell_count", 0)

        if min_vol is None or total_vol is None or cell_count <= 0 or total_vol <= 0:
            return 1.0

        avg_vol = total_vol / cell_count
        if avg_vol <= 0:
            return 1.0

        ratio = min_vol / avg_vol
        if ratio >= 0.1:
            return 1.0
        if ratio <= 0.0:
            return 0.0
        return max(0.0, ratio / 0.1)

    def _score_surface(self) -> float:
        bnd_skew = self.metrics.get("boundary_skewness", None)
        if bnd_skew is None:
            return 1.0
        if bnd_skew <= 4.0:
            return 1.0
        if bnd_skew >= 20.0:
            return 0.0
        return max(0.0, 1.0 - (bnd_skew - 4.0) / 16.0)

    def mesh_score(self) -> Dict[str, float]:
        scores = {
            "non_ortho": self._score_non_ortho(),
            "skewness": self._score_skewness(),
            "bad_cells": self._score_bad_cells(),
            "aspect_ratio": self._score_aspect_ratio(),
            "volume": self._score_volume(),
            "surface": self._score_surface(),
        }
        weights = {
            "non_ortho": 0.30,
            "skewness": 0.20,
            "bad_cells": 0.15,
            "aspect_ratio": 0.15,
            "volume": 0.10,
            "surface": 0.10,
        }
        total = sum(scores[k] * weights[k] for k in scores) * 100.0
        return {"scores": scores, "total": round(total, 2)}


class OpenFOAMQualityAnalyzer:
    """Analyze OpenFOAM mesh quality from a case directory."""

    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.log_path = self.case_dir / "log.checkMesh"

    def analyze(self) -> Dict[str, Any]:
        if not self.log_path.exists():
            proc = subprocess.run(
                ["checkMesh", "-allGeometry", "-allTopology", "-case", str(self.case_dir)],
                capture_output=True, text=True, timeout=120,
            )
            self.log_path.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="ignore")

        text = self.log_path.read_text(encoding="utf-8", errors="ignore")
        parser = CheckMeshParser(text)
        return build_report(parser)


def build_report(parser: CheckMeshParser) -> Dict[str, Any]:
    gate_status, gate_reason = QualityGate(parser.metrics).gate()
    score_data = QualityGate(parser.metrics).mesh_score()

    faces_above_65 = parser.metrics.get("faces_above_65")
    if faces_above_65 is None:
        faces_above_65 = parser.faces_above_65_from_histogram

    faces_above_70 = parser.metrics.get("faces_above_70")
    if faces_above_70 is None:
        faces_above_70 = parser.faces_above_70_from_histogram

    faces_above_75 = parser.metrics.get("faces_above_75")
    if faces_above_75 is None:
        faces_above_75 = parser.faces_above_75_from_histogram

    report: Dict[str, Any] = {
        "metrics": {
            "max_non_orthogonality": parser.metrics.get("max_non_orthogonality"),
            "avg_non_orthogonality": parser.metrics.get("avg_non_orthogonality"),
            "p95_non_orthogonality": parser.p95_non_orthogonality,
            "p99_non_orthogonality": parser.p99_non_orthogonality,
            "faces_above_65": faces_above_65,
            "faces_above_70": faces_above_70,
            "faces_above_75": faces_above_75,
            "max_skewness": parser.metrics.get("max_skewness"),
            "boundary_skewness": parser.metrics.get("boundary_skewness"),
            "min_volume": parser.metrics.get("min_volume"),
            "total_volume": parser.metrics.get("total_volume"),
            "max_aspect_ratio": parser.metrics.get("max_aspect_ratio"),
            "cell_count": parser.metrics.get("cell_count"),
            "face_count": parser.metrics.get("face_count"),
            "internal_faces": parser.metrics.get("internal_faces"),
            "boundary_patches": parser.metrics.get("boundary_patches"),
            "points": parser.metrics.get("points"),
            "concave_cells": parser.metrics.get("concave_cells"),
            "small_determinant_cells": parser.metrics.get("small_determinant_cells"),
            "failed_checks": parser.metrics.get("failed_checks"),
        },
        "histograms": {
            name: [
                {"lo": lo, "hi": hi if hi != float("inf") else None, "count": cnt}
                for lo, hi, cnt in rows
            ]
            for name, rows in parser.histograms.items()
        },
        "gate": {
            "status": gate_status,
            "reason": gate_reason,
        },
        "mesh_score": score_data,
    }
    return report


def console_report(report: Dict[str, Any], case_name: str = "unknown") -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append(f" OpenFOAM Mesh Quality Report  —  {case_name}")
    lines.append("=" * 70)
    lines.append("")

    lines.append("GATE ASSESSMENT")
    lines.append("-" * 70)
    gate = report["gate"]
    status = gate["status"]
    reason = gate["reason"]
    status_colored = status
    if status == "GOOD":
        status_colored = f"GOOD      ({reason})"
    elif status == "ACCEPTABLE":
        status_colored = f"ACCEPTABLE ({reason})"
    elif status == "WARNING":
        status_colored = f"WARNING    ({reason})"
    elif status == "BAD":
        status_colored = f"BAD        ({reason})"
    lines.append(f"  Status : {status_colored}")
    lines.append("")

    lines.append("KEY METRICS")
    lines.append("-" * 70)
    m = report["metrics"]
    rows = [
        ("Max non-orthogonality", _fmt(m.get("max_non_orthogonality"), "°")),
        ("Avg non-orthogonality", _fmt(m.get("avg_non_orthogonality"), "°")),
        ("P95 non-orthogonality", _fmt(m.get("p95_non_orthogonality"), "°")),
        ("P99 non-orthogonality", _fmt(m.get("p99_non_orthogonality"), "°")),
        ("Faces > 65°", _fmt(m.get("faces_above_65"), "")),
        ("Faces > 70°", _fmt(m.get("faces_above_70"), "")),
        ("Faces > 75°", _fmt(m.get("faces_above_75"), "")),
        ("Max skewness", _fmt(m.get("max_skewness"), "")),
        ("Boundary skewness", _fmt(m.get("boundary_skewness"), "")),
        ("Min volume", _fmt(m.get("min_volume"), "m³")),
        ("Total volume", _fmt(m.get("total_volume"), "m³")),
        ("Max aspect ratio", _fmt(m.get("max_aspect_ratio"), "")),
        ("Cells", _fmt(m.get("cell_count"), "")),
        ("Internal faces", _fmt(m.get("internal_faces"), "")),
        ("Boundary patches", _fmt(m.get("boundary_patches"), "")),
        ("Concave cells", _fmt(m.get("concave_cells"), "")),
        ("Small determinant cells", _fmt(m.get("small_determinant_cells"), "")),
        ("Failed checks", _fmt(m.get("failed_checks"), "")),
    ]
    for label, value in rows:
        lines.append(f"  {label:<30} {value}")

    lines.append("")
    lines.append("MESH SCORE BREAKDOWN")
    lines.append("-" * 70)
    scores = report["mesh_score"]["scores"]
    weights = {
        "non_ortho": 0.30,
        "skewness": 0.20,
        "bad_cells": 0.15,
        "aspect_ratio": 0.15,
        "volume": 0.10,
        "surface": 0.10,
    }
    for key, weight in weights.items():
        val = scores.get(key, 0.0)
        lines.append(f"  {key:<15} {val * 100:6.1f}%  (weight {weight * 100:.0f}%)")
    lines.append(f"  {'TOTAL':<15} {report['mesh_score']['total']:6.1f}%")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _fmt(value: Any, unit: str) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        if value == 0:
            formatted = "0.0"
        elif abs(value) < 1e-3 or abs(value) > 1e5:
            formatted = f"{value:.4e}"
        else:
            formatted = f"{value:.4f}"
        return f"{formatted} {unit}".strip()
    return f"{value} {unit}".strip()


def write_json(report: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return path


def write_csv(report: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    flat: Dict[str, Any] = {}
    for k, v in report["metrics"].items():
        flat[k] = v if v is not None else "N/A"
    flat["gate_status"] = report["gate"]["status"]
    flat["gate_reason"] = report["gate"]["reason"]
    flat["mesh_score_total"] = report["mesh_score"]["total"]
    for k, v in report["mesh_score"]["scores"].items():
        flat[f"score_{k}"] = round(v * 100, 2)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
    return path


def analyze_log(log_path: Path) -> Dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    parser = CheckMeshParser(text)
    return build_report(parser)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse OpenFOAM checkMesh output and compute quality gates."
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to checkMesh log file (e.g. log.checkMesh)",
    )
    parser.add_argument(
        "--case",
        type=Path,
        default=None,
        help="Case directory name for the report header",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Output JSON report path (default: quality_report.json)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV report path (default: quality_report.csv)",
    )
    args = parser.parse_args()

    if not args.log.exists():
        print(f"Error: log file not found: {args.log}", file=sys.stderr)
        return 1

    case_name = args.case.name if args.case else args.log.parent.name
    report = analyze_log(args.log)

    print(console_report(report, case_name))

    json_path = args.json or Path("quality_report.json")
    csv_path = args.csv or Path("quality_report.csv")

    write_json(report, json_path)
    write_csv(report, csv_path)

    print(f"\nJSON report : {json_path}")
    print(f"CSV report  : {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
