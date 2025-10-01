from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TypeAlias, Any

import numpy as np
# Make affctrllib optional
try:  # noqa: SIM105
    from affctrllib import AffComm, AffPosCtrl, AffStateThread  # type: ignore
except Exception:  # noqa: BLE001
    AffComm = AffPosCtrl = AffStateThread = object  # type: ignore

# Optional Raspberry Pi backend types (import only for type-checking/reporting).
# At runtime failures are ignored; we keep the public type broad.
try:  # noqa: SIM105
    from affetto_nn_ctrl.raspberry_pi_hardware import (  # type: ignore
        RPiComm,  # noqa: F401
        RPiPosCtrl,  # noqa: F401
        RPiStateThread,  # noqa: F401
    )
except Exception:  # noqa: BLE001
    pass

ROOT_DIR_PATH = Path(__file__).parent.parent.parent
SRC_DIR_PATH = ROOT_DIR_PATH / "src"
APPS_DIR_PATH = ROOT_DIR_PATH / "apps"
TESTS_DIR_PATH = ROOT_DIR_PATH / "tests"
DEFAULT_BASE_DIR_PATH = ROOT_DIR_PATH / "data"
DEFAULT_CONFIG_PATH = ROOT_DIR_PATH / "config" / "affetto.toml"

DEFAULT_DURATION = 10.0  # sec
DEFAULT_SEED = None
DEFAULT_N_REPEAT = 1
DEFAULT_TIME_HOME = 10

# Controller tuple kept broad to allow alternate backend implementations.
CONTROLLER_T: TypeAlias = tuple[Any, Any, Any]
RefFuncType: TypeAlias = Callable[[float], np.ndarray]

# Local Variables:
# jinx-local-words: "src"
# End:

# affetto_nn_ctrl package root

# Existing exports (trimmed for brevity)...
from .hw import IntegratedConfig, IntegratedSession, compute_valves, RandomWalk2Ch
from .control.trajectory_presets import TrajectoryPreset, build_preset, PRESETS

__all__ = [
    'IntegratedConfig', 'IntegratedSession', 'compute_valves', 'RandomWalk2Ch',
    'TrajectoryPreset', 'build_preset', 'PRESETS'
]
