from __future__ import annotations
"""RandomTrajectory minimal port."""
import numpy as np
from typing import List, Tuple, Any
from dataclasses import dataclass

try:
    from affctrllib import PTP  # type: ignore
except Exception:  # pragma: no cover
    PTP = None  # type: ignore

from affetto_nn_ctrl.random_utility import get_rng
from affetto_nn_ctrl._typing import NoDefault, no_default

MIN_UPDATE_Q_DELTA = 1e-4

@dataclass
class RandomTrajectory:
    active_joints: List[int]
    t0: float
    q0: np.ndarray
    update_t_range_list: List[Tuple[float,float]]
    update_q_range_list: List[Tuple[float,float]]
    update_q_limit_list: List[Tuple[float,float]]
    update_profile: str
    async_update: bool
    rng: Any

    def __init__(self, active_joints: List[int], t0: float, q0: np.ndarray,
                 update_t_range: Tuple[float,float] | List[Tuple[float,float]],
                 update_q_range: Tuple[float,float] | List[Tuple[float,float]],
                 update_q_limit: Tuple[float,float] | List[Tuple[float,float]],
                 update_profile: str = 'trapez', seed: int | NoDefault | None = no_default,
                 *, async_update: bool=False):
        if PTP is None:
            raise RuntimeError('PTP unavailable (affctrllib not installed)')
        self.active_joints = active_joints
        self.t0 = t0
        self.q0 = q0.copy()
        self.update_t_range_list = self._expand(active_joints, update_t_range)
        self.update_q_range_list = self._expand(active_joints, update_q_range)
        self.update_q_limit_list = self._expand(active_joints, update_q_limit)
        self.update_profile = update_profile
        self.async_update = async_update
        self.rng = get_rng(seed)
        self.reset_updater()

    @staticmethod
    def _expand(active: List[int], spec: Tuple[float,float] | List[Tuple[float,float]]):
        if isinstance(spec, tuple):
            lo, hi = min(spec), max(spec)
            return [(lo,hi) for _ in active]
        if len(spec) != len(active):
            raise ValueError('length mismatch')
        return [(min(a), max(a)) for a in spec]

    def _new_duration(self, r: Tuple[float,float]):
        lo, hi = min(r), max(r)
        return float(self.rng.uniform(lo, hi))

    def _new_position(self, q0: float, q_range: Tuple[float,float], q_limit: Tuple[float,float]):
        dmin, dmax = min(q_range), max(q_range)
        qmin, qmax = min(q_limit), max(q_limit)
        while True:
            delta = self.rng.uniform(dmin, dmax)
            qdes = q0 + self.rng.choice([-1,1]) * delta
            if qdes < qmin:
                qdes = qmin + (qmin - qdes)
            elif qdes > qmax:
                qdes = qmax - (qdes - qmax)
            qdes = max(min(qmax, qdes), qmin)
            if abs(qdes - q0) > MIN_UPDATE_Q_DELTA:
                return qdes

    def _init_sync(self, t0: float, q0: np.ndarray):
        active_q0 = q0[self.active_joints]
        dur = self._new_duration(self.update_t_range_list[0])
        qdes = [self._new_position(active_q0[i], self.update_q_range_list[i], self.update_q_limit_list[i]) for i in range(len(self.active_joints))]
        return PTP(active_q0, np.array(qdes), dur, t0, profile_name=self.update_profile)

    def _init_async(self, t0: float, q0: np.ndarray):
        ptps = []
        for i, j in enumerate(self.active_joints):
            qstart = q0[j]
            dur = self._new_duration(self.update_t_range_list[i])
            qdes = self._new_position(qstart, self.update_q_range_list[i], self.update_q_limit_list[i])
            ptps.append(PTP(qstart, qdes, dur, t0, profile_name=self.update_profile))
        return ptps

    def reset_updater(self, t0: float | None=None, q0: np.ndarray | None=None):
        if t0 is not None: self.t0 = t0
        if q0 is not None: self.q0 = q0.copy()
        if self.async_update:
            self.async_updater = self._init_async(self.t0, self.q0)
        else:
            self.sync_updater = self._init_sync(self.t0, self.q0)

    def _roll_sync(self, t: float):
        ptp = self.sync_updater
        if ptp.t0 + ptp.T < t:
            new_t0 = ptp.t0 + ptp.T
            new_q0 = ptp.qF
            dur = self._new_duration(self.update_t_range_list[0])
            qdes = [self._new_position(new_q0[i], self.update_q_range_list[i], self.update_q_limit_list[i]) for i in range(len(self.active_joints))]
            self.sync_updater = PTP(new_q0, np.array(qdes), dur, new_t0, profile_name=self.update_profile)

    def _roll_async(self, t: float):
        for i, ptp in enumerate(self.async_updater):
            if ptp.t0 + ptp.T < t:
                new_t0 = ptp.t0 + ptp.T
                qstart = ptp.qF
                dur = self._new_duration(self.update_t_range_list[i])
                qdes = self._new_position(qstart, self.update_q_range_list[i], self.update_q_limit_list[i])
                self.async_updater[i] = PTP(qstart, qdes, dur, new_t0, profile_name=self.update_profile)

    def qdes(self, t: float):
        if self.async_update:
            self._roll_async(t)
            q = self.q0.copy()
            q[self.active_joints] = [ptp.q(t) for ptp in self.async_updater]
            return q
        self._roll_sync(t)
        q = self.q0.copy()
        q[self.active_joints] = self.sync_updater.q(t)
        return q

    def dqdes(self, t: float):
        if self.async_update:
            self._roll_async(t)
            dq = np.zeros_like(self.q0)
            dq[self.active_joints] = [ptp.dq(t) for ptp in self.async_updater]
            return dq
        self._roll_sync(t)
        dq = np.zeros_like(self.q0)
        dq[self.active_joints] = self.sync_updater.dq(t)
        return dq

__all__ = ['RandomTrajectory']
