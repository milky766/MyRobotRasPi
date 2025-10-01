# ruff: noqa: SIM105
from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Sequence
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from ._periodic_runner import PeriodicRunner
from .affcomm import AffComm
from .affetto import Affetto
from .affstate import AffStateThread
from .logger import Logger
from .timer import Timer

if TYPE_CHECKING:
    from pathlib import Path

JointT = TypeVar("JointT", int, float, np.ndarray)


class AffCtrl(Affetto, PeriodicRunner, Generic[JointT]):
    ctrl_config: dict[str, Any]
    _input_range: tuple[float, float]
    _scale_gain: float
    _inactive_joints: np.ndarray

    DEFAULT_FREQ: float = 30
    DEFAULT_INACTIVE_PRESSURE: float = 0

    def __init__(
        self,
        config_path: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
    ) -> None:
        self.reset_inactive_joints()
        super().__init__(config_path)
        PeriodicRunner.__init__(self)
        self.set_frequency(dt=dt, freq=freq)

        if not hasattr(self, "_freq"):
            self.set_freq(self.DEFAULT_FREQ)
            msg = f"Control frequency is not provided, set to default: {self._freq}"
            warnings.warn(msg, stacklevel=2)

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return ""

    def load_config(self, config_dict: dict[str, Any]) -> None:
        super().load_config(config_dict)
        self.load_ctrl_config()

    def load_ctrl_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.ctrl_config = c["ctrl"]
        dt = self.ctrl_config.get("dt", None)
        freq = self.ctrl_config.get("freq", None)
        self.set_frequency(dt=dt, freq=freq)
        try:
            self.input_range = (self.ctrl_config["input_range"][0], self.ctrl_config["input_range"][1])
        except KeyError:
            pass
        self.load_inactive_joints()

    def load_inactive_joints(self, ctrl_config: dict[str, Any] | None = None) -> None:
        if ctrl_config is None:
            ctrl_config = self.ctrl_config
        if "inactive_joints" in ctrl_config:
            for inactive_joint in ctrl_config["inactive_joints"]:
                index = inactive_joint["index"]
                try:
                    self.add_inactive_joints(index, inactive_joint["pressure"])
                except KeyError:
                    self.add_inactive_joints(index)

    def set_input_range(self, input_range: tuple[float, float]) -> None:
        self._input_range = input_range
        self._scale_gain = 255.0 / (self._input_range[1] - self._input_range[0])

    @property
    def input_range(self) -> tuple[float, float]:
        return self._input_range

    @input_range.setter
    def input_range(self, input_range: tuple[float, float]) -> None:
        self.set_input_range(input_range)

    @property
    def scale_gain(self) -> float:
        return self._scale_gain

    def scale(self, u1: JointT, u2: JointT) -> tuple[JointT, JointT]:
        try:
            c1 = np.clip(u1, self._input_range[0], self._input_range[1])
            c2 = np.clip(u2, self._input_range[0], self._input_range[1])
            return (
                self._scale_gain * (c1 - self._input_range[0]),
                self._scale_gain * (c2 - self._input_range[0]),
            )  # type: ignore[reportReturnType]
        except AttributeError:
            return (u1, u2)

    @property
    def inactive_joints(self) -> np.ndarray:
        return self._inactive_joints

    @property
    def inactive_joints_index(self) -> list[int]:
        index = []
        for i, _, _ in self._inactive_joints:
            index.append(i)
        return index

    def _expand_as_index_range(self, pattern: str) -> list[int]:
        indices: list[int] = []
        _range = pattern.split("-")
        if len(_range) > 1:
            try:
                beg = int(_range[0])
            except ValueError:
                beg = 0
            try:
                end = int(_range[1]) + 1
            except ValueError:
                end = self.dof
            indices.extend(range(beg, end))
        return indices

    def _expand_as_index(self, pattern: str) -> list[int]:
        index: list[int] = []
        for p in pattern.split(","):
            if "-" in p:
                index.extend(self._expand_as_index_range(p))
            else:
                try:
                    index.extend([int(p)])
                except ValueError:
                    pass
        return index

    def _make_index_list(self, pattern: int | Sequence[int] | str | None) -> list[int]:
        index: list[int]
        if pattern is None:
            index = []
        elif isinstance(pattern, str):
            index = self._expand_as_index(pattern)
        elif isinstance(pattern, Sequence):
            index = list(pattern)  # type: ignore[arg-type]
        else:
            index = [int(pattern)]
        return index

    def _make_inactive_joints_array(
        self,
        pattern: int | Sequence[int] | str,
        pressure: float | None = None,
    ) -> np.ndarray:
        if pressure is None:
            pressure = self.DEFAULT_INACTIVE_PRESSURE
        index_list = self._make_index_list(pattern)
        arr = np.full((len(index_list), 3), pressure)
        arr[:, 0] = index_list
        return arr

    def set_inactive_joints(
        self,
        pattern: int | Sequence[int] | str,
        pressure: float | None = None,
    ) -> None:
        self._inactive_joints = self._make_inactive_joints_array(pattern, pressure)

    def add_inactive_joints(
        self,
        pattern: int | Sequence[int] | str,
        pressure: float | None = None,
    ) -> None:
        index_list = self._make_index_list(pattern)
        removing_index = []
        for i, inactive_joint in enumerate(self._inactive_joints):
            if inactive_joint[0] in index_list:
                removing_index.append(i)
        self._inactive_joints = np.delete(self._inactive_joints, removing_index, axis=0)
        self._inactive_joints = np.append(
            self._inactive_joints,
            self._make_inactive_joints_array(pattern, pressure),
            axis=0,
        )

    def reset_inactive_joints(self) -> None:
        self._inactive_joints = np.empty((0, 3), dtype=float)

    @property
    def active_joints_index(self) -> list[int]:
        index = list(range(self.dof))
        for i, _, _ in self._inactive_joints:
            index.remove(i)
        return index

    def set_active_joints(
        self,
        pattern: int | Sequence[int] | str | None = None,
        inactive_pressure: float | None = None,
    ) -> None:
        index_list = self._make_index_list(pattern)
        inactive_index = [e for e in range(self.dof) if e not in index_list]
        self.set_inactive_joints(inactive_index, inactive_pressure)

    def add_active_joints(
        self,
        pattern: int | Sequence[int] | str,
    ) -> None:
        index_list = self._make_index_list(pattern)
        removing_index = []
        for i, inactive_joint in enumerate(self._inactive_joints):
            if inactive_joint[0] in index_list:
                removing_index.append(i)
        self._inactive_joints = np.delete(self._inactive_joints, removing_index, axis=0)

    def mask(self, u1: JointT, u2: JointT) -> tuple[JointT, JointT]:
        mask = self.inactive_joints[:, 0].astype(int)
        try:
            np.put(u1, mask, self.inactive_joints[:, 1])  # type: ignore[reportArgumentType,arg-type]
            np.put(u2, mask, self.inactive_joints[:, 2])  # type: ignore[reportArgumentType,arg-type]
        except TypeError:
            # Pass over TypeError, which will be raised when u1 and u2
            # are not instance of np.ndarray.
            pass
        return (u1, u2)

    def update(  # type: ignore[override]
        self,
        _: float,
        u1: JointT,
        u2: JointT,
    ) -> tuple[JointT, JointT]:
        PeriodicRunner.update(self)
        u1, u2 = self.mask(u1, u2)
        return self.scale(u1, u2)


class AffCtrlThread(Thread):
    _acom: AffComm
    _actrl: AffCtrl
    _astate: AffStateThread
    _astate_created_inside: bool
    _lock: Lock
    _stopped: Event
    _timer: Timer
    _current_time: float
    _ctrl_input: Callable[[float], tuple[np.ndarray, np.ndarray]]
    _logger: Logger

    def __init__(
        self,
        astate: AffStateThread | None = None,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
        output: str | Path | None = None,
        sensor_dt: float | None = None,
        sensor_freq: float | None = None,
        *,
        logging: bool = True,
        butterworth: bool = False,
    ) -> None:
        self._acom = AffComm(config)
        self._acom.create_command_socket()
        self._astate_created_inside = False
        if astate is not None:
            self._astate = astate
        else:
            self._astate = self._create_state_estimator(config, dt=sensor_dt, freq=sensor_freq, butterworth=butterworth)
            self._astate_created_inside = True
        self._actrl = AffCtrl(config, dt, freq)
        self._lock = Lock()
        self._stopped = Event()
        self._timer = Timer(rate=self._actrl.freq)
        self._current_time = 0
        self.reset_ctrl_input()
        if logging:
            self._create_logger(output)
        Thread.__init__(self)

        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def _create_state_estimator(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
        *,
        butterworth: bool = False,
    ) -> AffStateThread:
        return AffStateThread(config, dt=dt, freq=freq, logging=False, output=None, butterworth=butterworth)

    def _create_logger(self, output: str | Path | None) -> Logger:
        self._logger = Logger(output)
        self._logger.set_labels(
            "t",
            # raw data
            [f"rq{i}" for i in range(self._actrl.dof)],
            [f"rdq{i}" for i in range(self._actrl.dof)],
            [f"rpa{i}" for i in range(self._actrl.dof)],
            [f"rpb{i}" for i in range(self._actrl.dof)],
            # estimated states
            [f"q{i}" for i in range(self._actrl.dof)],
            [f"dq{i}" for i in range(self._actrl.dof)],
            [f"pa{i}" for i in range(self._actrl.dof)],
            [f"pb{i}" for i in range(self._actrl.dof)],
            # command data
            [f"ca{i}" for i in range(self._actrl.dof)],
            [f"cb{i}" for i in range(self._actrl.dof)],
        )
        return self._logger

    def run(self) -> None:
        # Since idling process may take several seconds to finish,
        # interaction with control thread should be started after
        # idling process has finished by using
        # AffCtrlThread.wait_for_idling().

        # Start state estimator thread if not started.
        if not self._astate.is_alive():
            if not self._astate.prepared():
                self._astate.prepare()
            self._astate.start()

        # Start timer.
        with self._lock:
            self._timer.start()

        # Start the main loop.
        while not self._stopped.is_set():
            with self._lock:
                t = self._timer.elapsed_time()
            rq, rdq, rpa, rpb = self._astate.get_raw_states()
            q, dq, pa, pb = self._astate.get_states()
            with self._lock:
                self._current_time = t
                u1, u2 = self._ctrl_input(t)
                ca, cb = self._actrl.update(t, u1, u2)
            self._acom.send_commands(ca, cb)
            try:
                self._logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb)
            except AttributeError:
                pass
            with self._lock:
                self._timer.block()

        # Close socket after having left the loop.
        self._acom.close_command_socket()

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
        time.sleep(0.1)
        if self._astate_created_inside:
            self._astate.join()

    def wait_for_idling(self, timeout: float | None = None) -> bool:
        return self._astate.wait_for_idling(timeout)

    def reset_timer(self) -> None:
        self._astate.reset_timer()
        with self._lock:
            self._timer.reset()

    @property
    def dof(self) -> int:
        with self._lock:
            return self._actrl.dof

    @property
    def current_time(self) -> float:
        with self._lock:
            return self._current_time

    @property
    def state(self) -> AffStateThread:
        return self._astate

    @property
    def logger(self) -> Logger:
        return self._logger

    @property
    def dt(self) -> float:
        with self._lock:
            return self._actrl.dt

    @dt.setter
    def dt(self, dt: float) -> None:
        with self._lock:
            self._actrl.dt = dt

    @property
    def freq(self) -> float:
        with self._lock:
            return self._actrl.freq

    @freq.setter
    def freq(self, freq: float) -> None:
        with self._lock:
            self._actrl.freq = freq

    @property
    def n_steps(self) -> int:
        with self._lock:
            return self._actrl.n_steps

    @property
    def inactive_joints(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._actrl.inactive_joints)

    @property
    def inactive_joints_index(self) -> list[int]:
        with self._lock:
            return self._actrl.inactive_joints_index.copy()

    def set_inactive_joints(
        self,
        pattern: int | Sequence[int] | str,
        pressure: float | None = None,
    ) -> None:
        with self._lock:
            self._actrl.set_inactive_joints(pattern, pressure)

    def add_inactive_joints(
        self,
        pattern: int | Sequence[int] | str,
        pressure: float | None = None,
    ) -> None:
        with self._lock:
            self._actrl.add_inactive_joints(pattern, pressure)

    def reset_inactive_joints(self) -> None:
        with self._lock:
            self._actrl.reset_inactive_joints()

    def set_active_joints(
        self,
        pattern: int | Sequence[int] | str | None = None,
        pressure: float | None = None,
    ) -> None:
        with self._lock:
            self._actrl.set_active_joints(pattern, pressure)

    def add_active_joints(
        self,
        pattern: int | Sequence[int] | str,
    ) -> None:
        with self._lock:
            self._actrl.add_active_joints(pattern)

    @property
    def active_joints_index(self) -> list[int]:
        with self._lock:
            return self._actrl.active_joints_index.copy()

    def set_ctrl_input(self, ctrl_input_func: Callable[[float], tuple[np.ndarray, np.ndarray]]) -> None:
        with self._lock:
            self._ctrl_input = ctrl_input_func

    def reset_ctrl_input(self) -> None:
        dof = self._actrl.dof
        self.set_ctrl_input(
            lambda _: (np.zeros((dof,)), np.zeros((dof,))),
        )


# Local Variables:
# jinx-local-words: "Ctrl JointT arg ctrl dt ndarray noqa np"
# End:
