from __future__ import annotations
from typing import Any, Tuple


class SimplePID:
    def __init__(self, kp: float, ki: float, kd: float):
        # Use same semantics as pid_tune.py: integrator stored as 'i', derivative on measurement,
        # and conditional anti-windup when umin/umax are provided.
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i = 0.0
        self.prev_y = None
        # expose last computed P/I/D components for external logging/diagnostics
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def reset(self) -> None:
        self.i = 0.0
        self.prev_y = None
        self.last_p = 0.0
        self.last_i = 0.0
        self.last_d = 0.0

    def step(self, sp: float, y: float, dt: float, umin: float | None = None, umax: float | None = None, dq_des: float | None = None, dq_meas: float | None = None) -> float:
        """PID step.

        - If both dq_des and dq_meas are provided, use the affetto-style derivative term kD*(dq_des - dq_meas).
        - If only dq_des is provided, approximate dq_meas by derivative on measurement to avoid setpoint kick.
        - If dq_des is not provided, fall back to derivative-on-measurement.
        """
        if dt <= 0:
            return 0.0
        # error and proportional term
        e = sp - y
        p = self.kp * e
        # derivative term
        d = 0.0
        if dq_des is not None:
            # prefer externally measured velocity when available
            if dq_meas is not None:
                try:
                    d = self.kd * (float(dq_des) - float(dq_meas))
                except Exception:
                    d = 0.0
            else:
                # approximate measured velocity from encoder differences
                measured_dq = 0.0
                if self.prev_y is not None and dt > 0:
                    measured_dq = (y - self.prev_y) / dt
                try:
                    d = self.kd * (float(dq_des) - float(measured_dq))
                except Exception:
                    d = 0.0
        else:
            # fallback: derivative on measurement (avoid setpoint kick)
            if self.prev_y is not None and dt > 0:
                d = - self.kd * (y - self.prev_y) / dt
            else:
                d = 0.0
        # candidate integrator advance (note: ki multiplies the integrated error)
        i_next = self.i + self.ki * e * dt
        u_unclamped = p + i_next + d
        # apply saturation with conditional anti-windup if limits provided
        if umin is not None and umax is not None:
            # clamp and detect saturation
            u = max(umin, min(umax, u_unclamped))
            saturated_high = u >= umax - 1e-9
            saturated_low = u <= umin + 1e-9
            # inhibit integration if it would push further into saturation
            if (saturated_high and e > 0) or (saturated_low and e < 0):
                i_next = self.i
                u = max(umin, min(umax, p + i_next + d))
        else:
            u = u_unclamped
        # commit integrator and measurement
        self.i = i_next
        self.prev_y = y
        # store components for external inspection
        try:
            self.last_p = float(p)
        except Exception:
            self.last_p = 0.0
        try:
            self.last_i = float(i_next)
        except Exception:
            self.last_i = 0.0
        try:
            self.last_d = float(d)
        except Exception:
            self.last_d = 0.0
        return u


class RobotController:
    def __init__(self, dac: Any, encoder: Any, kp: float, ki: float, kd: float, center: float = 50.0, span_pct: float = 40.0, min_pct: float = 5.0, max_pct: float = 95.0):
        self.dac = dac
        self.encoder = encoder
        self.pid = SimplePID(kp, ki, kd)
        self.center = center
        self.span_pct = span_pct
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.last_u = 0.0

    def reset(self):
        self.pid.reset()
        self.last_u = 0.0

    def _u_limits(self) -> tuple[float,float]:
        safe_hi = min(self.max_pct, 95.0)
        lo = self.min_pct
        hi = safe_hi
        cB = self.center
        umin = max(2.0 * (lo - cB), -2.0 * (hi - cB))
        umax = min(2.0 * (hi - cB), -2.0 * (lo - cB))
        return umin, umax

    def update(self, q_des: float, current_angle: float | None, dt: float, dq_des: float | None = None, dq_meas: float | None = None) -> tuple[float,float,float]:
        """Compute PID using position error and velocity-error (affetto style when available),
        then return (a_pct, b_pct, u).

        - If dq_des and dq_meas are provided, D-term uses kD*(dq_des - dq_meas).
        - No velocity feed-forward is applied; control is pure feedback from the PID implementation.
        """
        y = 0.0 if current_angle is None else float(current_angle)
        umin, umax = self._u_limits()
        # Pass dq_des and dq_meas into the PID so the D-term becomes velocity-error when available
        u = self.pid.step(q_des, y, dt, umin=umin, umax=umax, dq_des=dq_des, dq_meas=dq_meas)
        # Map to valve percentages around center: affetto風の対向バルブ対称コマンド
        delta_pct = 0.5 * u
        a_pct = self.center + delta_pct
        b_pct = self.center - delta_pct
        # apply safety clamps
        a_pct = max(0.0, min(95.0, a_pct))
        b_pct = max(0.0, min(95.0, b_pct))
        self.last_u = u
        return a_pct, b_pct, u


def create_controller(dac, encoder, kp: float = 0.1, ki: float = 0.04, kd: float = 0.01, center: float = 50.0, span_pct: float = 40.0, min_pct: float = 5.0, max_pct: float = 95.0):
    return RobotController(dac, encoder, kp, ki, kd, center, span_pct, min_pct, max_pct)
