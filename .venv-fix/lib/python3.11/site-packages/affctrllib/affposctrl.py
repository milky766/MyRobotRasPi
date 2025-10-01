# ruff: noqa: SIM105

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

from .affctrl import AffCtrl, AffCtrlThread
from .affstate import AffStateThread

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from .logger import Logger

JointT = TypeVar("JointT", int, float, np.ndarray)


class Feedback(ABC, Generic[JointT]):
    _kP: JointT
    _kD: JointT
    _kI: JointT
    _stiff: JointT
    _accum_qerr: JointT
    _scheme_name: str

    def __init__(self, **kwargs: JointT) -> None:
        if "kP" in kwargs:
            self.kP = kwargs["kP"]
        if "kD" in kwargs:
            self.kD = kwargs["kD"]
        if "kI" in kwargs:
            self.kI = kwargs["kI"]
        self._scheme_name = ""

    @property
    def scheme_name(self) -> str:
        return self._scheme_name

    @property
    def kP(self) -> JointT:
        return self._kP

    @kP.setter
    def kP(self, kP: JointT) -> None:
        self._kP = kP

    @property
    def kD(self) -> JointT:
        return self._kD

    @kD.setter
    def kD(self, kD: JointT) -> None:
        self._kD = kD

    @property
    def kI(self) -> JointT:
        return self._kI

    @kI.setter
    def kI(self, kI: JointT) -> None:
        self._kI = kI

    @property
    def stiff(self) -> JointT:
        return self._stiff

    @stiff.setter
    def stiff(self, stiff: JointT) -> None:
        self._stiff = stiff

    def positional_feedback(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> JointT:
        _ = t  # avoid typing error
        qerr = qdes - q
        try:
            self._accum_qerr = self._accum_qerr + qerr
        except AttributeError:
            self._accum_qerr = qerr

        # NOTE: operator '*' in np.array does multiplications in
        # element-wise way.
        return self.kP * qerr + self.kD * (dqdes - dq) + self.kI * self._accum_qerr

    @abstractmethod
    def update(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        pa: JointT,
        pb: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]: ...


class FeedbackPID(Feedback[JointT]):
    _stiff: JointT

    def __init__(self, **kwargs: JointT) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if "stiff" in kwargs:
            self.stiff = kwargs["stiff"]
        self._scheme_name = "pid"

    def update(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        pa: JointT,
        pb: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        _, _ = pa, pb  # avoid typing error
        e = self.positional_feedback(t, q, dq, qdes, dqdes)
        return (self.stiff + e, self.stiff - e)


class FeedbackPIDF(Feedback[JointT]):
    _stiff: JointT
    _press_gain: JointT

    def __init__(self, **kwargs: JointT) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        if "stiff" in kwargs:
            self.stiff = kwargs["stiff"]
        if "press_gain" in kwargs:
            self.press_gain = kwargs["press_gain"]
        self._scheme_name = "pidf"

    @property
    def press_gain(self) -> JointT:
        return self._press_gain

    @press_gain.setter
    def press_gain(self, press_gain: JointT) -> None:
        self._press_gain = press_gain

    def update(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        pa: JointT,
        pb: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        try:
            d = self.press_gain * (pa - pb)
        except AttributeError:
            d = pa - pb
        e = self.positional_feedback(t, q, dq, qdes, dqdes)
        return (self.stiff + e - d, self.stiff - e + d)


AFFPOSCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES = {
    "PID": FeedbackPID,
    "pid": FeedbackPID,
    "PIDF": FeedbackPIDF,
    "pidf": FeedbackPIDF,
}

AFFPOSCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY = {
    "FeedbackPID": "pid",
    "FeedbackPIDF": "pidf",
}


class AffPosCtrl(AffCtrl[JointT]):
    _feedback_scheme: Feedback

    DEFAULT_FREQ: float = 30
    DEFAULT_INACTIVE_PRESSURE: float = 0

    def __init__(
        self,
        config_path: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
    ) -> None:
        super().__init__(config_path, dt, freq)

    @property
    def feedback_scheme(self) -> Feedback:
        return self._feedback_scheme

    def load_config(self, config_dict: dict[str, Any]) -> None:
        super().load_config(config_dict)
        self.load_ctrl_config()

    def load_ctrl_config(self, config: dict[str, Any] | None = None) -> None:
        super().load_ctrl_config(config)
        try:
            scheme = self.ctrl_config["scheme"]
        except KeyError:
            scheme = "pid"
        self.load_feedback_scheme(scheme)

    @classmethod
    def _load_feedback_scheme_find_array(
        cls,
        ctrl_config: dict[str, Any],
        scheme_key: str,
        param_key: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            kwargs[param_key] = np.array(ctrl_config[scheme_key][param_key], dtype=float)
        except KeyError:
            pass
        return kwargs

    def load_feedback_scheme(self, scheme: str, ctrl_config: dict[str, Any] | None = None) -> None:
        if ctrl_config is None:
            ctrl_config = self.ctrl_config
        scheme_class = AFFPOSCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES[scheme]
        scheme_key = AFFPOSCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY[scheme_class.__name__]
        kwargs: dict[str, Any] = {}
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "kP", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "kD", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "kI", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "stiff", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "press_gain", kwargs)
        self._feedback_scheme = scheme_class(**kwargs)  # type: ignore[abstract]

    def update(  # type: ignore[override]
        self,
        t: float,
        q: JointT,
        dq: JointT,
        pa: JointT,
        pb: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        u1, u2 = self.feedback_scheme.update(t, q, dq, pa, pb, qdes, dqdes)
        return super().update(t, u1, u2)  # type: ignore[return-value]


class AffPosCtrlThread(AffCtrlThread):
    _actrl: AffPosCtrl
    _qdes_func: Callable[[float], np.ndarray]
    _dqdes_func: Callable[[float], np.ndarray]

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
        super().__init__(
            astate,
            config,
            dt,
            freq,
            output,
            sensor_dt,
            sensor_freq,
            logging=logging,
            butterworth=butterworth,
        )
        del self._actrl
        self._actrl = AffPosCtrl(config, dt, freq)  # type: ignore[reportIncompatibleVariableOverride]
        self.reset_trajectory()

    def _create_state_estimator(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
        *,
        butterworth: bool = False,
    ) -> AffStateThread:
        return AffStateThread(config, dt=dt, freq=freq, logging=False, output=None, butterworth=butterworth)

    def _create_logger(self, output: str | Path) -> Logger:  # type: ignore[override]
        super()._create_logger(output)
        self._logger.extend_labels(
            [f"qdes{i}" for i in range(self._actrl.dof)],
            [f"dqdes{i}" for i in range(self._actrl.dof)],
        )
        return self._logger

    def run(self) -> None:
        # Since idling process may take several seconds to finish,
        # interaction with control thread should be started after
        # idling process has finished by using
        # AffPosCtrlThread.wait_for_idling().

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
                qdes = self._qdes_func(t)
                dqdes = self._dqdes_func(t)
                ca, cb = self._actrl.update(t, q, dq, pa, pb, qdes, dqdes)
            self._acom.send_commands(ca, cb)
            try:
                self._logger.store(t, rq, rdq, rpa, rpb, q, dq, pa, pb, ca, cb, qdes, dqdes)
            except AttributeError:
                pass
            with self._lock:
                self._timer.block()

        # Close socket after having left the loop.
        self._acom.close_command_socket()

    @property
    def feedback_scheme(self) -> Feedback:
        with self._lock:
            return self._actrl.feedback_scheme

    def load_feedback_scheme(self, scheme: str, ctrl_config: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._actrl.load_feedback_scheme(scheme, ctrl_config)

    def set_qdes_func(self, qdes_func: Callable[[float], np.ndarray]) -> None:
        with self._lock:
            self._qdes_func = qdes_func

    def set_dqdes_func(self, dqdes_func: Callable[[float], np.ndarray]) -> None:
        with self._lock:
            self._dqdes_func = dqdes_func

    def set_trajectory(
        self,
        qdes_func: Callable[[float], np.ndarray],
        dqdes_func: Callable[[float], np.ndarray],
    ) -> None:
        with self._lock:
            self._qdes_func = qdes_func
            self._dqdes_func = dqdes_func

    def reset_trajectory(self, q0: float | np.ndarray = 0) -> None:
        dof = self._actrl.dof
        self.set_trajectory(
            lambda _: np.full((dof,), q0),
            lambda _: np.zeros((dof,)),
        )


# Local Variables:
# jinx-local-words: "Ctrl FeedbackPID FeedbackPIDF JointT Pos arg dqdes kD kI kP noqa np pid pidf qdes"
# End:
