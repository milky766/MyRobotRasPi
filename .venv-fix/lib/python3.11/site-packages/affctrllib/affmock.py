# ruff: noqa: T201,NPY002
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._sockutil import Socket
from .affetto import Affetto
from .timer import Timer

if TYPE_CHECKING:
    from pathlib import Path


class Sinusoidal:
    _amplitude: float
    _period: float
    _base: float
    _phase: float
    _omega: float

    def __init__(self, amplitude: float, period: float, base: float, phase: float = 0.0) -> None:
        self._amplitude = amplitude
        self._period = period
        self._base = base
        self._phase = phase
        self._omega = 2.0 * np.pi / self._period

    def __call__(self, t: float, phi: np.ndarray) -> np.ndarray:
        return self._amplitude * np.sin(t * self._omega - self._phase - phi) + self._base

    @property
    def omega(self) -> float:
        return self._omega


sin_q = Sinusoidal(amplitude=50, period=5, base=50)
sin_pa = Sinusoidal(amplitude=300, period=2.5, base=300, phase=0.25 * np.pi)
sin_pb = Sinusoidal(amplitude=300, period=2.5, base=300, phase=0.5 * np.pi)


def generate_pseudo_sensory_data_string(t: float, dof: int = 13) -> str:
    phi = 0.5 * np.pi / (dof - 1) * np.arange(dof)
    q = sin_q(t, phi) + np.random.normal(0.0, 2, size=(dof,))
    pa = sin_pa(t, phi) + np.random.normal(0.0, 12, size=(dof,))
    pb = sin_pb(t, phi) + np.random.normal(0.0, 12, size=(dof,))
    A = np.vstack((q, pa, pb))
    return " ".join([f"{x[0]:.0f} {x[1]:.1f} {x[2]:.1f}" for x in A.T])


class AffMock(Affetto):
    config_path: Path | None  # type: ignore[assignment]
    command_socket: Socket  # local
    sensory_socket: Socket  # remote
    sensor_rate: float

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.command_socket = Socket()
        self.sensory_socket = Socket()
        super().__init__(config_path)

    def __repr__(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}()"

    def load_config(self, config_dict: dict[str, Any]) -> None:
        super().load_config(config_dict)
        self.load_mock_config()

    def load_mock_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.mock_config = c["mock"]
        self.command_socket.addr = self.mock_config["local"]
        self.sensory_socket.addr = self.mock_config["remote"]
        self.sensor_rate = self.mock_config["sensor"]["rate"]

    def start(self, rate: float | None = None, *, quiet: bool = False) -> None:
        self.sensory_socket.create()
        if rate is None:
            rate = self.sensor_rate
        timer = Timer(rate=rate)
        timer.start()
        while True:
            t = timer.elapsed_time()
            msg = generate_pseudo_sensory_data_string(t, self.dof)
            sz = self.sensory_socket.sendto(msg.encode())
            if not quiet:
                print(f"t={t:.2f}: sent <{msg}> to {self.sensory_socket.addr} ({sz} bytes)")
            timer.block()


# Local Variables:
# jinx-local-words: "noqa"
# End:
