# Minimal stub of affctrllib for running myrobot backend without native libs.
from __future__ import annotations
import time
from typing import Any

# Provide a minimal PTP motion profile class used by control_utility.get_back_home_position
class PTP:
    """Simple point-to-point profile: linear interpolation with constant velocity.

    q(t) and dq(t) return numpy arrays sized to the provided DOF.
    This is a lightweight stub for testing and does not attempt to match
    production motion profiles.
    """
    def __init__(self, q0, q1, duration, profile_name: str | None = None):
        import numpy as _np
        self._q0 = _np.asarray(q0, dtype=float)
        self._q1 = _np.asarray(q1, dtype=float)
        self._duration = float(duration) if duration and duration > 0 else 1.0
        self._dq_const = (_np.asarray(self._q1 - self._q0, dtype=float) / self._duration)

    def q(self, t: float):
        import numpy as _np
        tau = 0.0 if t <= 0.0 else (1.0 if t >= self._duration else t / self._duration)
        return _np.asarray(self._q0 + (self._q1 - self._q0) * tau, dtype=float)

    def dq(self, t: float):
        import numpy as _np
        if t <= 0.0 or t >= self._duration:
            return _np.zeros_like(self._q0, dtype=float)
        return _np.asarray(self._dq_const, dtype=float)

class Logger:
    def __init__(self) -> None:
        self._filename: str | None = None
    def set_labels(self, *args: Any) -> None:
        pass
    def set_filename(self, filename: str | None) -> None:
        self._filename = str(filename) if filename is not None else None
    def erase_data(self) -> None:
        pass
    def store(self, *args: Any, **kwargs: Any) -> None:
        pass
    def dump(self, quiet: bool = False) -> None:
        pass

class Timer:
    def __init__(self, rate: float | None = None) -> None:
        self.rate = rate
        self._t0: float | None = None
        self._last: float | None = None
        self._acc: float = 0.0
    def start(self) -> None:
        now = time.perf_counter()
        self._t0 = now
        self._last = now
        self._acc = 0.0
    def elapsed_time(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.perf_counter() - self._t0
    def accumulated_time(self) -> float:
        if self.rate and self.rate > 0:
            self._acc += 1.0 / self.rate
            return self._acc
        return self.elapsed_time()
    def block(self) -> None:
        if not self.rate or self._last is None:
            return
        next_t = self._last + 1.0 / self.rate
        now = time.perf_counter()
        sleep_s = max(0.0, next_t - now)
        if sleep_s > 0:
            time.sleep(sleep_s)
        self._last = next_t

class AffComm:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass
    def create_command_socket(self) -> None:
        pass
    def send_commands(self, ca, cb) -> None:  # noqa: ANN001
        pass
    def close_command_socket(self) -> None:
        pass

class AffPosCtrl:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.dof = 13
        self.freq = kwargs.get("freq", 30.0)
    def set_inactive_joints(self, *args: Any, **kwargs: Any) -> None:
        pass
    def update(self, *args: Any, **kwargs: Any):  # noqa: ANN001
        import numpy as np
        return np.zeros(self.dof), np.zeros(self.dof)

class AffStateThread:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.freq = kwargs.get("freq", 30.0)
        self.q = [50.0] * 13
    def prepare(self) -> None:
        pass
    def start(self) -> None:
        pass
    def get_raw_states(self):  # noqa: ANN201
        import numpy as np
        return np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13)
    def get_states(self):  # noqa: ANN201
        import numpy as np
        return np.zeros(13), np.zeros(13), np.zeros(13), np.zeros(13)
    def join(self) -> None:
        pass
