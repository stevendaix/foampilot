#!/usr/bin/env python3
"""
Post-process the neighborhood CFD case.

Usage:
    PYTHONPATH=../../../src python3 postprocess.py --case neighborhood_case
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import run_postprocessing


def main():
    parser = argparse.ArgumentParser(description="Post-process neighborhood CFD case")
    parser.add_argument("--case", default="neighborhood_case", help="Case directory")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <case>/post)")
    parser.add_argument("--speed", type=float, default=10.0, help="Reference wind speed (m/s)")
    args = parser.parse_args()

    case_dir = Path(args.case)
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "post"
    run_postprocessing(case_dir, output_dir, speed=args.speed)


if __name__ == "__main__":
    main()
