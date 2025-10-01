from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


class Timer:
    _rate: float | None
    _period: float | None
    _period_ns: int
    _n_step: int
    _time_started_ns: int
    _time_last_blocked_ns: int
    _time_ns_func: Callable[[], int]

    def __init__(self, rate: float | None = None, period: float | None = None) -> None:
        self._rate = None
        self._period = None
        self._period_ns = 0
        if rate is not None:
            self.rate = rate
        if period is not None:
            self.period = period
        self._n_step = 0
        self._time_started_ns = 0
        self._time_ns_func = time.time_ns

    def set_rate(self, rate: float) -> None:
        self._rate = rate
        self._period = 1.0 / rate
        self._period_ns = int(self._period * 1e9)

    @property
    def rate(self) -> float:
        if self._rate is None:
            msg = "Timer: rate is not set"
            raise ValueError(msg)
        return self._rate

    @rate.setter
    def rate(self, rate: float) -> None:
        self.set_rate(rate)

    def set_period(self, period: float) -> None:
        self._period = period
        self._period_ns = int(self._period * 1e9)
        self._rate = 1.0 / period

    @property
    def period(self) -> float:
        if self._period is None:
            msg = "Timer: period is not set"
            raise ValueError(msg)
        return self._period

    @period.setter
    def period(self, period: float) -> None:
        self.set_period(period)

    @property
    def period_ns(self) -> int:
        return self._period_ns

    @property
    def n_step(self) -> int:
        return self._n_step

    def start(self) -> None:
        self._time_started_ns = self._time_ns_func()
        self._time_last_blocked_ns = self._time_started_ns

    def reset(self) -> None:
        self._n_step = 0
        self.start()

    def elapsed_time(self) -> float:
        return (self._time_ns_func() - self._time_started_ns) * 1e-9

    def elapsed_time_ns(self) -> int:
        return self._time_ns_func() - self._time_started_ns

    def accumulated_time(self) -> float:
        return self.period * self._n_step

    def accumulated_time_ns(self) -> int:
        t = self.period_ns * self._n_step
        self._n_step += 1
        return t

    def sleep(self) -> None:
        time.sleep(self._period if self._period is not None else 0)

    def block(self) -> None:
        try:
            elapsed_since_last_blocked = self._time_ns_func() - self._time_last_blocked_ns
        except AttributeError:
            msg = "Timer.start() must be called before Timer.block()"
            raise RuntimeError(msg) from None

        time_to_sleep = self._period_ns - elapsed_since_last_blocked
        if time_to_sleep > 0:
            time.sleep(time_to_sleep * 1e-9)
        else:
            msg = f"It took longer than specified period at t={self.elapsed_time()}"
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        self._time_last_blocked_ns = self._time_ns_func()
        self._n_step += 1
