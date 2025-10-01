# ruff: noqa: RET505,ANN003

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

JointT = TypeVar("JointT", int, float, np.ndarray)
TOL = 1e-12


class Profile(ABC, Generic[JointT]):
    _q0: JointT
    _qF: JointT
    _T: float
    _t0: float

    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float) -> None:
        self._q0 = q0
        self._qF = qF
        self._T = T
        self._t0 = t0

    @property
    def q0(self) -> JointT:
        return self._q0

    @property
    def qF(self) -> JointT:
        return self._qF

    @property
    def T(self) -> float:
        return self._T

    @property
    def t0(self) -> float:
        return self._t0

    @abstractmethod
    def s(self, t: float) -> JointT | float: ...

    @abstractmethod
    def ds(self, t: float) -> JointT | float: ...

    @abstractmethod
    def dds(self, t: float) -> JointT | float: ...

    def q(self, t: float) -> JointT | float:
        s = self.s(t)
        return self._q0 * (1.0 - s) + self._qF * s

    def dq(self, t: float) -> JointT | float:
        return (self._qF - self._q0) * self.ds(t)

    def ddq(self, t: float) -> JointT | float:
        return (self._qF - self._q0) * self.dds(t)


class StepProfile(Profile, Generic[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float) -> None:
        Profile.__init__(self, q0, qF, T, t0)

    def s(self, t: float) -> float:
        _ = t
        return 1.0

    def ds(self, t: float) -> float:
        _ = t
        return 0

    def dds(self, t: float) -> float:
        _ = t
        return 0


class ConstVelocityProfile(Profile, Generic[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float) -> None:
        Profile.__init__(self, q0, qF, T, t0)

    def s(self, t: float) -> float:
        t_rel = t - self._t0
        return t_rel / self._T

    def ds(self, t: float) -> float:
        _ = t
        return 1.0 / self._T

    def dds(self, t: float) -> float:
        _ = t
        return 0


class TriangularVelocityProfile(Profile, Generic[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float) -> None:
        Profile.__init__(self, q0, qF, T, t0)
        self._s_coeff = 2.0 / (T * T)
        self._ds_coeff = 4.0 / (T * T)

    def s(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._s_coeff * t_rel * t_rel
        elif t_rel <= self._T:
            return 1.0 - self._s_coeff * (self._T - t_rel) * (self._T - t_rel)
        else:
            return 1

    def ds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._ds_coeff * t_rel
        elif t_rel <= self._T:
            return self._ds_coeff * (self._T - t_rel)
        else:
            return 0

    def dds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._ds_coeff
        elif t_rel <= self._T:
            return -self._ds_coeff
        else:
            return 0


class TrapezoidalVelocityProfile(Profile, Generic[JointT]):
    _vmax: JointT | float | np.ndarray
    _tb: JointT | float | np.ndarray
    _vM: JointT | float | np.ndarray
    _a: JointT | float | np.ndarray
    _zeros: JointT | float | np.ndarray
    _ones: JointT | float | np.ndarray

    def __init__(
        self,
        q0: JointT,
        qF: JointT,
        T: float,
        t0: float,
        vmax: JointT | float | None = None,
        tb: JointT | float | None = None,
    ) -> None:
        Profile.__init__(self, q0, qF, T, t0)
        if vmax is not None:
            self.set_vmax(vmax)
        elif tb is not None:
            self.set_tb(tb)
        else:
            self.set_vmax(0)
        if isinstance(vmax, np.ndarray):
            self._zeros = np.zeros(shape=vmax.shape)
            self._ones = np.ones(shape=vmax.shape)
        else:
            self._zeros = 0
            self._ones = 1

    @property
    def vmax(self) -> JointT | float | np.ndarray:
        return self._vmax

    def set_vmax(self, vmax: JointT | float) -> None:
        if isinstance(self.q0, np.ndarray) and isinstance(vmax, int | float):
            _vmax = np.full(self.q0.shape, vmax)
        else:
            _vmax = vmax  # type: ignore[assignment]
        self._vM = np.absolute(_vmax / (self.qF - self.q0))
        self._vM = np.where(self._vM < TOL, 3.0 / (2.0 * self.T), self._vM)

        # Raise error when vM is too small.
        if np.any(self._vM <= 1.0 / self.T):
            i = np.flatnonzero(self._vM <= 1.0 / self.T)[0]
            try:
                v = _vmax[i]  # type: ignore[reportIndexIssue]
            except (TypeError, IndexError):
                v = _vmax
            msg = f"Specified Vmax for q[{i}] is too small "
            msg += f"to reach desired position: {v}"
            raise ValueError(msg)

        # Emit warning when vM is too large.
        if np.any(self._vM > 2.0 / self.T):
            v = 2.0 / self.T
            indices = np.flatnonzero(self._vM > v)
            vM = np.where(self._vM > v, v, self._vM)
            for i in indices:
                msg = f"Specified Vmax for q[{i}] is truncated: "
                try:
                    v1, v2 = self._vM[i], vM[i]
                except IndexError:
                    v1, v2 = self._vM, vM
                msg += f"{v1} -> {v2}"
                warnings.warn(msg, ResourceWarning, stacklevel=2)
            self._vM = vM

        self._vmax = self._vM * (self.qF - self.q0)
        self._tb = self.T - 1.0 / self._vM
        self._a = self._vM / self.tb

    @property
    def tb(self) -> JointT | float | np.ndarray:
        return self._tb

    def set_tb(self, tb: JointT | float) -> None:
        if isinstance(self.q0, np.ndarray) and isinstance(tb, int | float):
            self._tb = np.full(self.q0.shape, tb)
        else:
            self._tb = tb
        self._tb = np.where(self._tb < TOL, self.T / 3, self._tb)

        # Emit warning when tb is too large.
        if np.any(self._tb > 0.5 * self.T):
            half_T = 0.5 * self.T
            indices = np.flatnonzero(self._tb > half_T)
            Tb = np.where(self._tb > half_T, half_T, self._tb)
            for i in indices:
                msg = f"Specified Tb for q[{i}] is reduced: "
                try:
                    tb1, tb2 = self._tb[i], Tb[i]
                except IndexError:
                    tb1, tb2 = self._tb, Tb
                msg += f"{tb1} -> {tb2}"
                warnings.warn(msg, ResourceWarning, stacklevel=2)
            self._tb = Tb

        self._vM = 1.0 / (self.T - self.tb)
        self._vmax = self._vM * (self.qF - self.q0)
        self._a = self._vM / self.tb

    def s(self, t: float) -> np.ndarray | float:
        t_rel = t - self.t0
        if t_rel < 0:
            return self._zeros
        elif t_rel >= self.T:
            return self._ones
        return np.where(
            t_rel < self.tb,
            0.5 * self._a * t_rel * t_rel,
            np.where(
                (self.tb <= t_rel) & (t_rel < (self.T - self.tb)),
                self._vM * (t_rel - 0.5 * self.tb),
                np.where(
                    ((self.T - self.tb) <= t_rel) & (t_rel < self.T),
                    1.0 - 0.5 * self._a * (self.T - t_rel) * (self.T - t_rel),
                    self._zeros,
                ),
            ),
        )

    def ds(self, t: float) -> np.ndarray | float:
        t_rel = t - self.t0
        if t_rel < 0 or t_rel >= self.T:
            return self._zeros
        return np.where(
            t_rel < self.tb,
            self._a * t_rel,
            np.where(
                (self.tb <= t_rel) & (t_rel < (self.T - self.tb)),
                self._vM,
                np.where(
                    ((self.T - self.tb) <= t_rel) & (t_rel < self.T),
                    self._a * (self.T - t_rel),
                    self._zeros,
                ),
            ),
        )

    def dds(self, t: float) -> np.ndarray | float:
        t_rel = t - self.t0
        if t_rel < 0 or t_rel >= self.T:
            return self._zeros
        return np.where(
            t_rel < self.tb,
            self._a,
            np.where(
                (self.tb <= t_rel) & (t_rel < self.T - self.tb),
                self._zeros,
                np.where(
                    ((self.T - self.tb) <= t_rel) & (t_rel < self.T),
                    -self._a,
                    self._zeros,
                ),
            ),
        )


class SinusoidalVelocityProfile(Profile[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float) -> None:
        super().__init__(q0, qF, T, t0)  # type: ignore[arg-type]
        self._omega = 2.0 * np.pi / T
        self._s_coeff = -1.0 / (2.0 * np.pi)
        self._ds_coeff = 1.0 / T
        self._dds_coeff = self._omega / T

    def s(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel > self._T:
            return 1
        return self._s_coeff * np.sin(self._omega * t_rel) + t_rel / self.T

    def ds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        return self._ds_coeff * (1.0 - np.cos(self._omega * t_rel))

    def dds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        return self._dds_coeff * np.sin(self._omega * t_rel)


class FifthDegreePolynomialProfile(Profile, Generic[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float) -> None:
        Profile.__init__(self, q0, qF, T, t0)
        self._ds_coeff = 30.0 / T
        self._dds_coeff = 60.0 / (T * T)

    def s(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel > self._T:
            return 1
        t1 = t_rel / self._T
        t2 = t1 * t1
        t3 = t2 * t1
        return t3 * (6.0 * t2 - 15.0 * t1 + 10.0)

    def ds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        t1 = t_rel / self._T
        t2 = t1 * t1
        return t2 * (t1 - 1.0) * (t1 - 1.0) * self._ds_coeff

    def dds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        t1 = t_rel / self._T
        return t1 * (2.0 * t1 - 1.0) * (t1 - 1.0) * self._dds_coeff


PTP_ACCEPTABLE_PROFILE_NAMES = {
    "step": StepProfile,
    "constant velocity": ConstVelocityProfile,
    "const velocity": ConstVelocityProfile,
    "constant": ConstVelocityProfile,
    "const": ConstVelocityProfile,
    "con": ConstVelocityProfile,
    "triangular velocity": TriangularVelocityProfile,
    "triangular": TriangularVelocityProfile,
    "tri": TriangularVelocityProfile,
    "trapezoidal velocity": TrapezoidalVelocityProfile,
    "trapezoidal": TrapezoidalVelocityProfile,
    "trapez": TrapezoidalVelocityProfile,
    "tra": TrapezoidalVelocityProfile,
    "sinusoidal velocity": SinusoidalVelocityProfile,
    "sinusoidal": SinusoidalVelocityProfile,
    "sin": SinusoidalVelocityProfile,
    "5th-degree polynomial": FifthDegreePolynomialProfile,
    "5th degree polynomial": FifthDegreePolynomialProfile,
    "5th-degree": FifthDegreePolynomialProfile,
    "5th degree": FifthDegreePolynomialProfile,
    "5th": FifthDegreePolynomialProfile,
}


class PTP(Generic[JointT]):
    _profile: Profile

    def __init__(
        self,
        q0: JointT,
        qF: JointT,
        T: float,
        t0: float = 0,
        profile_name: str = "triangular",
        **kwargs,
    ) -> None:
        self.select_profile(q0, qF, T, t0, profile_name, **kwargs)

    @property
    def q0(self) -> JointT:
        return self.profile.q0

    @property
    def qF(self) -> JointT:
        return self.profile.qF

    @property
    def T(self) -> float:
        return self.profile.T

    @property
    def t0(self) -> float:
        return self.profile.t0

    @property
    def profile(self) -> Profile:
        return self._profile

    def select_profile(self, q0: JointT, qF: JointT, T: float, t0: float, profile_name: str, **kwargs) -> None:
        if profile_name not in PTP_ACCEPTABLE_PROFILE_NAMES:
            msg = f"Invalid profile name: {profile_name}"
            raise ValueError(msg)
        self._profile = PTP_ACCEPTABLE_PROFILE_NAMES[profile_name](q0, qF, T, t0, **kwargs)

    def q(self, t: float) -> JointT | float:
        return self.profile.q(t)

    def dq(self, t: float) -> JointT | float:
        return self.profile.dq(t)

    def ddq(self, t: float) -> JointT | float:
        return self.profile.ddq(t)


# Local Variables:
# jinx-local-words: "JointT Vmax arg const noqa tra trapez tri vM"
# End:
