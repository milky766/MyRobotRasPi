from __future__ import annotations
from typing import Any, Callable
import numpy as np
import os
import time

try:
    from affetto_nn_ctrl.control_utility import RandomTrajectory
except Exception:
    RandomTrajectory = None

# Prefer affetto's PTP if available to match profile semantics exactly
try:
    from affctrllib import PTP  # type: ignore
except Exception:
    try:
        from affctrllib.ptp import PTP  # type: ignore
    except Exception:
        PTP = None


class LocalRandomTrajectory:
    def __init__(self, seed: int, t_range: tuple[float, float], q_range: tuple[float, float]):
        self.rng = np.random.RandomState(seed)
        self.t_min, self.t_max = float(t_range[0]), float(t_range[1])
        self.q_min, self.q_max = float(q_range[0]), float(q_range[1])
        self.reset_updater()

    def reset_updater(self):
        self.t0 = 0.0
        self.next_switch = 0.0
        self.current_q = (self.q_min + self.q_max) / 2.0

    def get_qdes_func(self) -> Callable[[float], np.ndarray]:
        def f(t: float):
            if t >= self.next_switch:
                dwell = self.rng.uniform(self.t_min, self.t_max)
                self.current_q = float(self.rng.uniform(self.q_min, self.q_max))
                self.next_switch = t + dwell
            return np.array([self.current_q], dtype=float)
        return f


# --- New: Local Trapezoidal PTP implementation ---
class _PTPTrapezoidSegment:
    """Single PTP segment with trapezoidal (or triangular) velocity profile.
    Units are degrees, seconds, deg/s, deg/s^2.
    The profile durations (tb, tc) are fixed by T; a and vmax are scaled so that
    the end position reaches exactly qf at t=T.
    """

    def __init__(self, qs: float, qf: float, T: float, a_max: float = 200.0, vmax_max: float = 150.0):
        self.qs = float(qs)
        self.qf = float(qf)
        self.T = max(1e-6, float(T))
        self.a_max = max(1e-6, float(a_max))
        self.vmax_max = max(1e-6, float(vmax_max))
        self._prepare()

    def _prepare(self):
        D = self.qf - self.qs
        self._sign = 1.0 if D >= 0 else -1.0
        L = abs(D)
        # Raw durations from constraints
        tb_raw = min(self.vmax_max / self.a_max, self.T / 2.0)
        tc_raw = max(0.0, self.T - 2.0 * tb_raw)
        vmax_raw = self.a_max * tb_raw
        # Distance achievable with raw a, vmax over (tb_raw, tc_raw)
        dist_raw = self.a_max * tb_raw * tb_raw + vmax_raw * tc_raw
        if dist_raw <= 1e-9 or L <= 1e-12:
            # Degenerate; hold position
            self.a = 0.0
            self.tb = 0.0
            self.tc = self.T
            self.vmax = 0.0
            return
        # Scale a and vmax together to hit L exactly while keeping durations
        s = L / dist_raw
        self.a = self.a_max * s
        self.tb = tb_raw
        self.tc = tc_raw
        self.vmax = vmax_raw * s

    def eval(self, tau: float) -> tuple[float, float]:
        """Evaluate position and velocity at elapsed time tau in [0, T].
        Returns (position, velocity) in (deg, deg/s).
        """
        if tau <= 0.0:
            return self.qs, 0.0
        if tau >= self.T:
            return self.qf, 0.0
        t = tau
        # handle degenerate case: no accel/decel, constant (or zero-motion)
        if self.tb <= 0.0:
            # constant velocity motion (or zero if vmax==0)
            pos = self.qs + self._sign * self.vmax * t
            vel = self._sign * self.vmax
            return pos, vel
        # Accelerate phase
        if t < self.tb:
            dq = 0.5 * self.a * t * t
            vel = self.a * t
            return self.qs + self._sign * dq, self._sign * vel
        # Cruise phase
        t2 = t - self.tb
        if t2 < self.tc:
            dq = 0.5 * self.a * self.tb * self.tb + self.vmax * t2
            vel = self.vmax
            return self.qs + self._sign * dq, self._sign * vel
        # Decelerate phase
        t3 = t2 - self.tc
        # position increment during decel
        dq = 0.5 * self.a * self.tb * self.tb + self.vmax * self.tc + self.vmax * t3 - 0.5 * self.a * t3 * t3
        dq = min(dq, abs(self.qf - self.qs))
        # instantaneous velocity during decel
        vel = max(self.vmax - self.a * t3, 0.0)
        return self.qs + self._sign * dq, self._sign * vel


class LocalPTPRandomTrajectory:
    """Random PTP trajectory that generates step or trapezoidal segments locally.

    - q_range is absolute limits (min, max) in degrees.
    - update_q_delta is a non-negative delta magnitude range (min, max). Sign is chosen internally.
    - profile: 'trapezoidal' for smooth motion, 'step' for instantaneous changes.
    """

    def __init__(
        self,
        seed: int,
        t_range: tuple[float, float],
        q_range: tuple[float, float],
        q0: Any | None = None,
        update_q_delta: tuple[float, float] | None = None,
        profile: str = 'trapezoidal',
        a_max: float = 200.0,
        vmax_max: float = 150.0,
    ):
        self.rng = np.random.RandomState(seed)
        self.t_min, self.t_max = float(t_range[0]), float(t_range[1])
        self.q_min, self.q_max = float(q_range[0]), float(q_range[1])
        self.profile = profile
        self.a_max = float(a_max)
        self.vmax_max = float(vmax_max)
        span = max(0.0, self.q_max - self.q_min)
        if update_q_delta is None:
            self.delta_min, self.delta_max = 0.0, span
        else:
            self.delta_min, self.delta_max = float(update_q_delta[0]), float(update_q_delta[1])
        self.q_current = float(q0[0]) if (q0 is not None and np.size(q0) >= 1) else (self.q_min + self.q_max) / 2.0
        self.t_seg_start = 0.0
        self.seg_T = 0.0
        # allow any sentinel or local segment object; typing as Any keeps flexibility
        self._segment: Any = None

    def _choose_target(self, q: float) -> float:
        # Pick delta magnitude, then sign that keeps within limits; if both OK, choose randomly.
        dmag = float(self.rng.uniform(self.delta_min, self.delta_max))
        plus_ok = (q + dmag) <= self.q_max
        minus_ok = (q - dmag) >= self.q_min
        if plus_ok and minus_ok:
            sign = 1.0 if self.rng.rand() < 0.5 else -1.0
        elif plus_ok:
            sign = 1.0
        elif minus_ok:
            sign = -1.0
        else:
            # Delta too large; clamp to nearest limit
            return max(self.q_min, min(self.q_max, q))
        return max(self.q_min, min(self.q_max, q + sign * dmag))

    def reset_updater(self, t0: float = 0.0, q0: Any | None = None):
        self.t_seg_start = float(t0)
        if q0 is not None and np.size(q0) >= 1:
            self.q_current = float(q0[0])
        self._start_new_segment(self.t_seg_start, self.q_current)

    def _start_new_segment(self, t_start: float, q_start: float):
        self.t_seg_start = float(t_start)
        q_target = self._choose_target(q_start)
        self.seg_T = float(self.rng.uniform(self.t_min, self.t_max))
        if self.profile == 'step' or q_target == q_start:
            # step: no continuous segment object
            self._segment = None
            if hasattr(self, '_aff_ptp'):
                try:
                    delattr(self, '_aff_ptp')
                except Exception:
                    pass
            self.q_next = q_target
        else:
            # If affetto PTP is available, use it so dq/dq generation matches affetto exactly
            if PTP is not None:
                try:
                    # create affetto PTP with absolute t0=self.t_seg_start
                    self._aff_ptp = PTP(q_start, q_target, self.seg_T, self.t_seg_start, profile_name=self.profile)
                    # mark that an external PTP should be used; use a non-None sentinel for _segment
                    self._segment = object()
                    self.q_next = q_target
                    return
                except Exception:
                    # fallback to local segment on any error
                    try:
                        if hasattr(self, '_aff_ptp'):
                            delattr(self, '_aff_ptp')
                    except Exception:
                        pass
            # fallback local trapezoid implementation
            self._segment = _PTPTrapezoidSegment(q_start, q_target, self.seg_T, self.a_max, self.vmax_max)
            self.q_next = q_target

    def set_update_q_range(self, active_joints, update_q_range):
        try:
            self.delta_min, self.delta_max = float(update_q_range[0]), float(update_q_range[1])
        except Exception:
            pass

    def get_qdes_func(self) -> Callable[[float], np.ndarray]:
        def f(t: float):
            # Advance segment if needed
            while (t - self.t_seg_start) >= self.seg_T:
                # End of segment
                self.q_current = self.q_next if hasattr(self, 'q_next') else self.q_current
                # Start next segment with start time aligned at end of previous
                self._start_new_segment(self.t_seg_start + self.seg_T, self.q_current)
            tau = max(0.0, t - self.t_seg_start)
            if self.profile == 'step' or self._segment is None:
                q = self.q_current if tau < 1e-12 else self.q_next
                v = 0.0
            else:
                # If using affetto PTP instance, call its q/dq with absolute time
                if hasattr(self, '_aff_ptp') and self._aff_ptp is not None:
                    try:
                        q = float(self._aff_ptp.q(t))
                        v = float(self._aff_ptp.dq(t))
                    except Exception:
                        # fallback to local segment eval
                        res = self._segment.eval(tau) if hasattr(self._segment, 'eval') else None
                        if isinstance(res, (tuple, list, np.ndarray)):
                            q = float(res[0])
                            try:
                                v = float(res[1])
                            except Exception:
                                v = 0.0
                        else:
                            q = float(res) if res is not None else float(self.q_current)
                            v = 0.0
                else:
                    res = self._segment.eval(tau)
                    if isinstance(res, (tuple, list, np.ndarray)):
                        q = float(res[0])
                        try:
                            v = float(res[1])
                        except Exception:
                            v = 0.0
                    else:
                        q = float(res)
                        v = 0.0
            return np.array([q, v], dtype=float)
        return f


class AffettoRTAdapter:
    def __init__(self, rt_inner: Any, active_joints: list[int] | None = None):
        self._rt = rt_inner
        # 1DOF想定の this project では active_joints の先頭を対象とする
        try:
            self._jid = int(active_joints[0]) if active_joints else 0
        except Exception:
            self._jid = 0

    def get_qdes_func(self):
        # affetto の qdes/dqdes を束ねて [q, dq] を返す関数を作る
        qf = self._rt.get_qdes_func()
        dqf = getattr(self._rt, 'get_dqdes_func', None)
        if dqf is not None:
            dqf = self._rt.get_dqdes_func()
        def f(t: float):
            q = qf(t)
            try:
                qj = float(q[self._jid])
            except Exception:
                qj = float(q)
            if dqf is not None:
                dq = dqf(t)
                try:
                    dj = float(dq[self._jid])
                except Exception:
                    dj = float(dq)
            else:
                dj = 0.0
            return np.array([qj, dj], dtype=float)
        return f

    def reset_updater(self, *args, **kwargs):
        try:
            return self._rt.reset_updater(*args, **kwargs)
        except TypeError:
            return self._rt.reset_updater()

    def set_update_q_range(self, active_joints, update_q_range):
        if hasattr(self._rt, 'set_update_q_range'):
            try:
                return self._rt.set_update_q_range(active_joints, update_q_range)
            except Exception:
                return None
        return None


def make_random_trajectory(seed: int | None, active_joints: list[int], q0: Any, t_range: tuple[float, float], q_range: tuple[float, float], profile: str = 'trapezoidal', update_q_delta: tuple[float, float] | None = None):
    # q_range passed to this factory is treated as absolute min/max by our scripts
    # but affetto's RandomTrajectory expects update_q_range to be a *delta* range
    # (amount to change from current q). To keep behavior consistent between the
    # Local fallback and the affetto implementation, convert absolute q_range to
    # a delta range covering the full span and pass the absolute limits as
    # update_q_limit.
    if RandomTrajectory is None:
        # Use local implementation with trapezoidal profile support
        return LocalPTPRandomTrajectory(
            seed=seed,
            t_range=t_range,
            q_range=q_range,
            q0=q0,
            update_q_delta=update_q_delta,
            profile=profile,
        )
    # If caller didn't specify a seed (None), create a non-deterministic seed
    if seed is None:
        # use os.urandom to get unpredictable seed
        try:
            seed_val = int.from_bytes(os.urandom(4), 'little')
        except Exception:
            seed_val = int(time.time() * 1000) & 0xFFFFFFFF
    else:
        seed_val = int(seed)

    if RandomTrajectory is None:
        # Use local implementation with trapezoidal profile support
        return LocalPTPRandomTrajectory(
            seed=seed_val,
            t_range=t_range,
            q_range=q_range,
            q0=q0,
            update_q_delta=update_q_delta,
            profile=profile,
        )
    try:
        # compute delta range that allows movement across the absolute q_range
        qmin, qmax = float(q_range[0]), float(q_range[1])
        delta_span = max(0.0, qmax - qmin)
        # If caller provided an explicit delta range, use it directly so the
        # RandomTrajectory generation is exactly the affetto implementation.
        if update_q_delta is not None:
            update_q_range_delta = (float(update_q_delta[0]), float(update_q_delta[1]))
        else:
            # Default behavior: provide a delta covering the full span (0..delta_span).
            # This mirrors the common usage where update_q_range is the allowed positive
            # delta; callers who want symmetric deltas should pass update_q_delta.
            update_q_range_delta = (0.0, float(delta_span))
        rt_obj = RandomTrajectory(
            active_joints=active_joints,
            t0=0.0,
            q0=q0,
            update_t_range=t_range,
            update_q_range=update_q_range_delta,
            update_q_limit=q_range,
            update_profile=profile,
            seed=seed_val,
            async_update=True,
        )
        return AffettoRTAdapter(rt_obj, active_joints)
    except Exception:
        # Fallback to local PTP implementation if affetto call fails
        return LocalPTPRandomTrajectory(
            seed=seed_val,
            t_range=t_range,
            q_range=q_range,
            q0=q0,
            update_q_delta=update_q_delta,
            profile=profile,
        )
