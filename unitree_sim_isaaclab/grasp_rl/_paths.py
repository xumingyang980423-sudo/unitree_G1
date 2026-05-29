"""Repository root paths for grasp_rl (assets, robots, tasks live at repo root)."""
from __future__ import annotations

import os
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PKG_DIR)

# Allow `import grasp_rl` when only this file is loaded first.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def setup_repo_paths() -> str:
    """Set PROJECT_ROOT and ensure repo root is on sys.path."""
    os.environ.setdefault("PROJECT_ROOT", REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    return REPO_ROOT
