from __future__ import annotations


class PeriodicRunner:
    _dt: float
    _freq: float
    _n_steps: int

    def __init__(self, dt: float | None = None, freq: float | None = None) -> None:
        self.set_frequency(dt=dt, freq=freq)
        self._n_steps = 0

    def set_frequency(self, dt: float | None = None, freq: float | None = None) -> None:
        if dt is not None and freq is not None:
            msg = "Unable to specify DT and FREQ simultaneously"
            raise ValueError(msg)
        if dt is not None:
            self._dt = dt
            self._freq = 1.0 / dt
        elif freq is not None:
            self._freq = freq
            self._dt = 1.0 / freq

    def set_dt(self, dt: float) -> None:
        self.set_frequency(dt=dt)

    @property
    def dt(self) -> float:
        return self._dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self.set_dt(dt)

    def set_freq(self, freq: float) -> None:
        self.set_frequency(freq=freq)

    @property
    def freq(self) -> float:
        return self._freq

    @freq.setter
    def freq(self, freq: float) -> None:
        self.set_freq(freq)

    @property
    def n_steps(self) -> int:
        return self._n_steps

    def update(self) -> None:
        self._n_steps += 1
