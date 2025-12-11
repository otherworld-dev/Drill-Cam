#!/usr/bin/env python3
"""
DrillCam runner script.

Run this file directly to start DrillCam:
    python run.py
    python run.py --mock
    python run.py --debug
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run
from drillcam.main import main

if __name__ == "__main__":
    sys.exit(main())
