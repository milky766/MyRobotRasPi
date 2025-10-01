"""Hardware abstraction subpackage.

Modules:
- gpio: chip-select and GPIO helpers
- dac8564: DAC driver
- max11632: ADC driver (pressure)
- ldc1614: inductance/tension sensor
- encoder: quadrature encoder polling
- scheduler: deterministic frame scheduler utilities
- modes: valve command generation (sine/random walk)
- integrated_session: unified data acquisition & actuation loop
"""
from .gpio import CsLine  # noqa: F401
from .dac8564 import Dac8564  # noqa: F401
from .max11632 import Max11632  # noqa: F401
from .ldc1614 import LDC1614  # noqa: F401
from .encoder import EncoderSimple  # noqa: F401
from .modes import compute_valves, RandomWalk2Ch  # noqa: F401
from .scheduler import FrameScheduler  # noqa: F401
from .integrated_session import IntegratedConfig, IntegratedSession  # noqa: F401

# Re-export hardware level session & config
from .integrated_session import IntegratedConfig, IntegratedSession
from .modes import compute_valves, RandomWalk2Ch

# Future: trajectory presets / PID integration will be added here
__all__ = [
    'IntegratedConfig', 'IntegratedSession', 'compute_valves', 'RandomWalk2Ch'
]
