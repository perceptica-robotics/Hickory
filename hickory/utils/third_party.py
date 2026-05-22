"""Path helpers for vendored third-party dependencies."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY_ROOT = PROJECT_ROOT / "third_party"
FOUNDATIONPOSE_ROOT = THIRD_PARTY_ROOT / "FoundationPose"
SAM3D_ROOT = THIRD_PARTY_ROOT / "sam-3d-objects"


def add_third_party_paths() -> None:
    """Make vendored dependencies importable from their organized location."""
    paths = [
        THIRD_PARTY_ROOT,
        THIRD_PARTY_ROOT / "mps",
        SAM3D_ROOT,
        SAM3D_ROOT / "notebook",
        FOUNDATIONPOSE_ROOT,
    ]
    for path in paths:
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)
