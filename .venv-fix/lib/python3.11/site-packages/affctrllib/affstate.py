# ruff: noqa: SIM105,ERA001

from __future__ import annotations

import itertools
import sys
import warnings
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

import numpy as np

from ._periodic_runner import PeriodicRunner
from .affcomm import AffComm, unzip_array_as_ndarray
from .affetto import Affetto
from .filter import Butterworth, Filter
from .logger import Logger
from .timer import Timer

if TYPE_CHECKING:
    from pathlib import Path


class AffState(Affetto, PeriodicRunner):
    state_config: dict[str, Any]
    _filter_list: list[Filter | Butterworth | None]
    _raw_data: list[float] | list[int] | np.ndarray
    _data_ndarray: np.ndarray
    _filtered_data: list[np.ndarray]
    _q_prev: np.ndarray
    _dq: np.ndarray

    DEFAULT_FREQ: float = 100

    def __init__(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
        *,
        butterworth: bool = False,
    ) -> None:
        super().__init__(config)
        PeriodicRunner.__init__(self)

        self.set_frequency(dt=dt, freq=freq)
        self._filter_list = [Filter(), Filter(), Filter()]
        self._idled = False

        if not hasattr(self, "_freq"):
            self.set_freq(self.DEFAULT_FREQ)
            msg = f"Sensor frequency is not provided, set to default: {self._freq}"
            warnings.warn(msg, stacklevel=2)

        if butterworth:
            cutoff = 10
            order = 5
            self._filter_list = [
                Butterworth(cutoff, self.freq, order),
                Butterworth(cutoff, self.freq, order),
                Butterworth(cutoff, self.freq, order),
            ]

    def load_config(self, config_dict: dict[str, Any]) -> None:
        super().load_config(config_dict)
        self.load_state_config()

    def load_state_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.state_config = c["state"]
        dt = self.state_config.get("dt", None)
        freq = self.state_config.get("freq", None)
        self.set_frequency(dt=dt, freq=freq)

    @property
    def raw_data(self) -> list[float] | list[int] | np.ndarray:
        return self._raw_data

    @property
    def raw_q(self) -> np.ndarray:
        return self._data_ndarray[0]

    @property
    def raw_dq(self) -> np.ndarray:
        return self._dq

    @property
    def raw_pa(self) -> np.ndarray:
        return self._data_ndarray[1]

    @property
    def raw_pb(self) -> np.ndarray:
        return self._data_ndarray[2]

    @property
    def q(self) -> np.ndarray:
        return self._filtered_data[0]

    @property
    def dq(self) -> np.ndarray:
        return self._dq

    @property
    def pa(self) -> np.ndarray:
        return self._filtered_data[1]

    @property
    def pb(self) -> np.ndarray:
        return self._filtered_data[2]

    def update(self, raw_data: list[float] | list[int] | np.ndarray) -> None:  # type: ignore[override]
        self._raw_data = raw_data
        self._data_ndarray = unzip_array_as_ndarray(self._raw_data, ncol=3)
        # Process input signal filtering.
        self._filtered_data = [
            f.update(d) if f is not None else d for f, d in zip(self._filter_list, self._data_ndarray, strict=False)
        ]
        # Calculate time derivative of q.
        try:
            self._dq = (self.q - self._q_prev) / self.dt
        except AttributeError:
            self._dq = np.zeros(shape=self.q.shape)
        self._q_prev = self.q
        PeriodicRunner.update(self)

    def idled(self) -> bool:
        return self._idled

    def idle(
        self,
        acom: AffComm,
        n_sample: int = 100,
        freq_tol: float = 1,
        *,
        no_error: bool = False,
        quiet: bool = False,
    ) -> None:
        spinner = itertools.cycle(["-", "/", "|", "\\"])
        if not quiet:
            sys.stdout.write("Idling sensory module... ")
            sys.stdout.flush()
        if not acom.sensory_socket.is_created():
            acom.create_sensory_socket()
        timer = Timer()
        received_time_series = []
        timer.start()
        for i in range(n_sample):
            sarr = acom.receive_as_list()  # type: ignore[var-annotated]
            received_time_series.append(timer.elapsed_time())
            self.update(sarr)
            if not quiet and i % 10 == 0:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                sys.stdout.write("\b")
        time_series = np.array(received_time_series)
        dt_series = np.subtract(time_series[1:], time_series[:-1])
        estimated_freq = 1.0 / np.mean(dt_series)
        if not no_error and abs(self.freq - estimated_freq) > freq_tol:
            msg = "Specified sampling frequency is probably incorrect:\n"
            msg += f"  {estimated_freq:.3f} (estimated) vs {self.freq:.3f} (specified)"
            raise RuntimeError(msg)
        if not quiet:
            sys.stdout.write("done.\n")
        self._idled = True


class AffStateThread(Thread):
    _acom: AffComm
    _astate: AffState
    _lock: Lock
    _stopped: Event
    _idled: Event
    _timer: Timer
    _current_time: float
    _logger: Logger

    def __init__(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
        output: str | Path | None = None,
        *,
        logging: bool = True,
        butterworth: bool = False,
    ) -> None:
        self._acom = AffComm(config)
        self._astate = AffState(config, dt=dt, freq=freq, butterworth=butterworth)
        self._lock = Lock()
        self._stopped = Event()
        self._idled = Event()
        self._timer = Timer(rate=self._astate.freq)
        self._current_time = 0
        if logging:
            self._create_logger(output)
        Thread.__init__(self)

        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def _create_logger(self, output: str | Path | None) -> Logger:
        self._logger = Logger(output)
        self._logger.set_labels(
            "t",
            # raw data
            [f"rq{i}" for i in range(self._astate.dof)],
            [f"rdq{i}" for i in range(self._astate.dof)],
            [f"rpa{i}" for i in range(self._astate.dof)],
            [f"rpb{i}" for i in range(self._astate.dof)],
            # estimated states
            [f"q{i}" for i in range(self._astate.dof)],
            [f"dq{i}" for i in range(self._astate.dof)],
            [f"pa{i}" for i in range(self._astate.dof)],
            [f"pb{i}" for i in range(self._astate.dof)],
        )
        return self._logger

    def prepare(
        self,
        n_sample: int = 100,
        freq_tol: float = 1,
        *,
        no_error: bool = False,
        quiet: bool = False,
    ) -> None:
        self._astate.idle(self._acom, n_sample, freq_tol, no_error=no_error, quiet=quiet)
        self._idled.set()

    def prepared(self) -> bool:
        return self._astate.idled() and self._idled.is_set()

    def wait_for_idling(self, timeout: float | None = None) -> bool:
        return self._idled.wait(timeout)

    def run(self) -> None:
        if not self.prepared():
            warnings.warn("Skipped idling process for sensory module", stacklevel=2)

        # Start timer.
        with self._lock:
            self._timer.start()

        # Start the main loop.
        while not self._stopped.is_set():
            with self._lock:
                t = self._timer.elapsed_time()
            sarr = self._acom.receive_as_list()  # type: ignore[var-annotated]
            with self._lock:
                self._current_time = t
                self._astate.update(sarr)
                s = self._astate
                rq, rdq, rpa, rpb = s.raw_q, s.raw_dq, s.raw_pa, s.raw_pb
                q, dq, pa, pb = s.q, s.dq, s.pa, s.pb
            try:
                self._logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb)
            except AttributeError:
                pass
            # with self._lock:
            #     self._timer.block()

        # Close socket after having left the loop.
        self._acom.close_sensory_socket()

    def join(self, timeout: float | None = None) -> None:
        self.stop()
        Thread.join(self, timeout)

    def stop(self) -> None:
        try:
            if self._logger.fpath is not None:
                self._logger.dump()
        except AttributeError:
            pass
        self._stopped.set()

    def reset_timer(self) -> None:
        with self._lock:
            self._timer.reset()

    @property
    def dof(self) -> int:
        with self._lock:
            return self._astate.dof

    @property
    def current_time(self) -> float:
        with self._lock:
            return self._current_time

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def dt(self) -> float:
        with self._lock:
            return self._astate.dt

    @dt.setter
    def dt(self, dt: float) -> None:
        with self._lock:
            self._astate.dt = dt

    @property
    def freq(self) -> float:
        with self._lock:
            return self._astate.freq

    @freq.setter
    def freq(self, freq: float) -> None:
        with self._lock:
            self._astate.freq = freq

    @property
    def n_steps(self) -> int:
        with self._lock:
            return self._astate.n_steps

    def get_raw_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            s = self._astate
            return (
                np.copy(s.raw_q),
                np.copy(s.raw_dq),
                np.copy(s.raw_pa),
                np.copy(s.raw_pb),
            )

    def get_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            s = self._astate
            return np.copy(s.q), np.copy(s.dq), np.copy(s.pa), np.copy(s.pb)

    @property
    def raw_data(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.raw_data)

    @property
    def raw_q(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.raw_q)

    @property
    def raw_dq(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.raw_dq)

    @property
    def raw_pa(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.raw_pa)

    @property
    def raw_pb(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.raw_pb)

    @property
    def q(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.q)

    @property
    def dq(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.dq)

    @property
    def pa(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.pa)

    @property
    def pb(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._astate.pb)


# Local Variables:
# jinx-local-words: "dt noqa"
# End:
