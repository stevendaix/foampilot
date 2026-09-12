"""OpenFOAM checkMesh runner and parser."""
from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def run_checkmesh(case_dir: Path) -> Dict[str, object]:
    """Run OpenFOAM checkMesh on a case directory and parse results.

    Args:
        case_dir: Path to the OpenFOAM case (containing constant/polyMesh).

    Returns:
        Dictionary with mesh quality metrics. ``passed`` key is True if
        checkMesh exits 0 and no errors are detected.
    """
    case_dir = Path(case_dir)
    result: Dict[str, object] = {"passed": False, "errors": [], "metrics": {}}

    try:
        proc = subprocess.run(
            ["checkMesh", "-case", str(case_dir)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = proc.stdout + proc.stderr
        result["return_code"] = proc.returncode
        result["passed"] = proc.returncode == 0
    except FileNotFoundError:
        result["errors"].append("checkMesh not found — is OpenFOAM sourced?")
        result["passed"] = False
        return result
    except subprocess.TimeoutExpired:
        result["errors"].append("checkMesh timed out (>120s)")
        result["passed"] = False
        return result

    # Parse key metrics from checkMesh output
    # Supports both legacy ("Number of cells:") and OF13 ("cells:") formats
    patterns = {
        "n_cells": r"(?:Number of )?cells:\s+(\d+)",
        "n_points": r"(?:Number of )?points:\s+(\d+)",
        "n_faces": r"^\s*faces:\s+(\d+)",
        "n_internal_faces": r"(?:Number of )?internal faces:\s+(\d+)",
        "n_boundary_faces": r"(?:Number of )?boundary faces:\s+(\d+)",
        "min_vol": r"Minimum volume:\s+([0-9.eE+-]+)",
        "max_vol": r"Maximum volume:\s+([0-9.eE+-]+)",
        "max_non_ortho": r"Non-orthogonality Max:\s+([0-9.eE+-]+)",
        "max_skewness": r"Max\s+skewness:\s+([0-9.eE+-]+)",
        "max_aspect_ratio": r"Max aspect ratio:\s*=\s*([0-9.eE+-]+)",
    }

    for key, pat in patterns.items():
        m = re.search(pat, output)
        if m:
            try:
                value = m.group(1)
                result["metrics"][key] = (
                    float(value) if "." in value or "e" in value.lower() else int(value)
                )
            except (ValueError, IndexError):
                result["metrics"][key] = value if m.groups() else None

    # Check for errors in output
    error_patterns = [
        r"ERROR",
        r"Fatal error",
        r"Cannot open",
        r"No mesh",
        r"Empty mesh",
    ]
    for ep in error_patterns:
        if re.search(ep, output, re.IGNORECASE):
            result["errors"].append(f"checkMesh reported: {ep}")

    if result["errors"]:
        result["passed"] = False

    # Warn on quality thresholds
    metrics = result["metrics"]
    if "max_non_ortho" in metrics:
        try:
            if float(metrics["max_non_ortho"]) > 70:
                result["errors"].append(
                    f"High non-orthogonality: {metrics['max_non_ortho']}° (threshold 70°)"
                )
        except (TypeError, ValueError):
            pass
    if "max_skewness" in metrics:
        try:
            if float(metrics["max_skewness"]) > 4.0:
                result["errors"].append(
                    f"High skewness: {metrics['max_skewness']} (threshold 4.0)"
                )
        except (TypeError, ValueError):
            pass

    if result["errors"]:
        result["passed"] = False

    result["raw_output"] = output[-2000:] if len(output) > 2000 else output
    return result
