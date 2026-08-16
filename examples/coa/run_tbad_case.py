#!/usr/bin/env python3
"""
Simple entry point for TBAD → OpenFOAM pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "foampilot" / "src"))

from run_full_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
