from __future__ import annotations

import sys
from pathlib import Path


UNIT_ROOT = Path(__file__).resolve().parent
unit_root_str = str(UNIT_ROOT)
if unit_root_str not in sys.path:
    sys.path.insert(0, unit_root_str)
