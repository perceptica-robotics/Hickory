"""Compatibility wrapper for the 3D navigation planner.

New code should import from hickory.navigation.planner_3d.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hickory.navigation.planner_3d import *  # noqa: F401,F403
from hickory.navigation.planner_3d import main


if __name__ == "__main__":
    main()
