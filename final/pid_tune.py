#!/usr/bin/env python3
from __future__ import annotations
"""Interactive / batch PID gain tuning helper.

Modes:
  1. step: Single step response (q0 -> q1 at t_change)
  2. multistep: Sequence of levels for observing steady error & drift
  3. relay: Relay (on/off) induced oscillation for Ku, Tu estimation
  4. analyze: Offline metrics from an existing CSV (rise, overshoot, settling)
  5. autotune: Run relay test, estimate Ku/Tu, and propose PID gains
  6. zn_limit: Ziegler–Nichols ultimate sensitivity method (P-only sweep to find Ku, Tu)
  7. zn_step: Ziegler–Nichols step-response identification (open-loop)
S
Examples:
  uv run python apps/pid_tune.py step --step-q1 40 --total 8 --kp 1.5 --encoder -v
  uv run python apps/pid_tune.py multistep --levels 0 30 0 50 0 70 --dwell 3 --kp 2.0 --encoder
  uv run python apps/pid_tune.py relay --amp 25 --bias 60 --target 30 --encoder -v
  uv run python apps/pid_tune.py analyze data/integrated_sensor/integrated_log_20250817_123000.csv
  uv run python apps/pid_tune.py autotune --amp 25 --bias 60 --target 30 --rule zn_pid --encoder -v
  uv run python apps/pid_tune.py zn_limit --kp-start 1 --kp-max 100 --dwell 10 --target 30 --encoder -v
  uv run python apps/pid_tune.py zn_step --u-step 10 --total 5 --encoder -v

CSV columns produced: ms,valve_a_pct,valve_b_pct,q_des,pid_u,enc_deg

PIDゲインのチューニング:
uv run python apps/pid_tune.py step --step-q0 0 --step-q1 20 --total 8 --kp 0.1 --ki 0.04 --kd 0.01 --encoder --zero-at-start --pre-check -v
(このゲインがベスト)→
uv run python apps/pid_tune.py step --step-q0 0 --step-q1 20 --total 8 --kp 0.4 --ki 0.02 --kd 0.02 --encoder --zero-at-start -v
こちらにすると応答性が良くなるがオーバーシュートと振動に関して少し悪化


決定したゲインの性能評価（正弦波）:
uv run python apps/pid_tune.py evaluate --traj sin --amp 20 --offset 20 --freq 0.25 --t-total 8 --kp 0.1 --ki 0.04 --kd 0.01 --encoder --zero-at-start --pre-check -v

決定したゲインの性能評価(ランダムステップ）:
uv run python apps/pid_tune.py evaluate --traj random --amp 20 --offset 20 --dwell 1.0 --t-total 8 --kp 0.1 --ki 0.04 --kd 0.01 --encoder --zero-at-start --pre-check -v
"""
import argparse, os, time, csv, math, pathlib, json
from collections import deque
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
import numpy as np

# Replace IntegratedSession usage with direct hardware I/O matching apps/integrated_sensor_sin_python.py
try:
    import spidev  # type: ignore
except Exception:  # noqa: BLE001
    spidev = None  # type: ignore
try:
    import gpiod  # type: ignore
    from gpiod.line import Direction, Value  # type: ignore
except Exception:  # noqa: BLE001
    gpiod = None  # type: ignore
    Direction = Value = None  # type: ignore
# Optional encoder (reuse shared implementation)
try:
    from affetto_nn_ctrl.hw.encoder import EncoderSimple  # type: ignore
except Exception:  # noqa: BLE001
    EncoderSimple = None  # type: ignore

# SPI/DAC constants (can be overridden via CLI)
SPI_BUS = 0
SPI_DEV = 0               # /dev/spidev0.0
SPI_MAX_HZ = 1_000_000
SPI_MODE_DAC = 0b01       # MODE1 for DAC8564
GPIO_CS_DAC = 19          # Manual CS GPIO (active low)

# Safety cap: do not open valves beyond this percentage (overridable via CLI)
VALVE_MAX_PCT = 100.0

# Default encoder pins (match integrated_sensor_sin_python.py)
ENC_CHIP = '/dev/gpiochip4'
ENC_A = 14
ENC_B = 4

class CsLine:
    """Minimal CS line using libgpiod v2 (same as integrated_sensor_sin_python)."""
    def __init__(self, pin: int):
        if gpiod is None:
            raise RuntimeError('python3-libgpiod is required')
        self.pin = pin
        self.chip = gpiod.Chip('/dev/gpiochip0')
        ls = gpiod.LineSettings()
        if Direction is not None:
            ls.direction = Direction.OUTPUT
        if Value is not None:
            ls.output_value = Value.ACTIVE
        self.req = self.chip.request_lines(consumer='dac_cs', config={pin: ls})
        try:
            self.set(1)  # inactive (high)
        except Exception:
            pass
    def set(self, val: int):
        if Value is not None:
            self.req.set_values({self.pin: (Value.ACTIVE if val else Value.INACTIVE)})
        else:  # pragma: no cover
            self.req.set_values({self.pin: val})  # type: ignore[arg-type]
    def close(self):  # pragma: no cover
        try:
            self.set(1)
        except Exception:
            pass
        try:
            self.req.release()
        except Exception:
            pass
        try:
            self.chip.close()
        except Exception:
            pass

class Dac8564:
    """Direct DAC8564 access with manual CS per transfer (aligned with working script)."""
    def __init__(self, bus: int, dev: int, cs_gpio: int):
        if spidev is None:
            raise RuntimeError('spidev is required')
        self.spi = spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        self.cs = CsLine(cs_gpio)
        self.opened = False
    def open(self):
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            self.spi.max_speed_hz = SPI_MAX_HZ
            self.spi.mode = SPI_MODE_DAC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_and_init()
    def close(self):  # pragma: no cover
        if self.opened:
            try:
                self.spi.close()
            except Exception:
                pass
        try:
            self.cs.close()
        except Exception:
            pass
    def _xfer(self, data: list[int]):
        if not self.opened:
            raise RuntimeError('SPI not opened yet')
        self.spi.mode = SPI_MODE_DAC
        self.cs.set(0)
        self.spi.xfer2(list(data))
        self.cs.set(1)
    @staticmethod
    def _pct_to_code(pct: float) -> int:
        if pct < 0.0: pct = 0.0
        # Enforce safety cap
        if pct > VALVE_MAX_PCT: pct = VALVE_MAX_PCT
        return int(pct * 65535.0 / 100.0 + 0.5)
    @staticmethod
    def _cmd(ch: int) -> int:
        # 0x10 (write input) with LDAC low -> immediate update
        return 0x10 | (ch << 1)
    def set_channels(self, a_pct: float, b_pct: float):
        a_code = self._pct_to_code(a_pct)
        b_code = self._pct_to_code(b_pct)
        a = [self._cmd(0), (a_code >> 8) & 0xFF, a_code & 0xFF]
        b = [self._cmd(1), (b_code >> 8) & 0xFF, b_code & 0xFF]
        self._xfer(a); self._xfer(b)
    def _reset_and_init(self):
        # Reset and power config (exactly as in working script)
        self._xfer([0x28,0x00,0x01])
        time.sleep(0.001)
        self._xfer([0x38,0x00,0x01])

# Simple PID controller (local)
class SimplePID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.i = 0.0
        self.prev_y = None
    def reset(self) -> None:
        self.i = 0.0
        self.prev_y = None
    def step(self, sp: float, y: float, dt: float, umin: float | None = None, umax: float | None = None) -> float:
        """Parallel PID with derivative on measurement and conditional anti-windup.
        If umin/umax are provided, the integrator is inhibited when saturated in the
        direction that would increase saturation."""
        e = sp - y
        p = self.kp * e
        d = 0.0
        if self.prev_y is not None and dt > 0:
            d = - self.kd * (y - self.prev_y) / dt  # derivative on measurement (no setpoint kick)
        # candidate integrator advance
        i_next = self.i + self.ki * e * dt
        u_unclamped = p + i_next + d
        if umin is not None and umax is not None:
            u = _clamp(u_unclamped, umin, umax)
            saturated_high = u >= umax - 1e-9
            saturated_low = u <= umin + 1e-9
            # inhibit integration if pushing further into saturation
            if (saturated_high and e > 0) or (saturated_low and e < 0):
                i_next = self.i
                u = _clamp(p + i_next + d, umin, umax)
        else:
            u = u_unclamped
        self.i = i_next
        self.prev_y = y
        return u

# ---- small helper to poll sensors between frames ----

def _sleep_with_poll(enc, dt: float) -> None:
    """Sleep for dt seconds while polling encoder frequently.
    Poll at ~200µs (or finer for small dt) to avoid missing quadrature edges,
    matching the tight polling cadence used in integrated_sensor_sin_python.
    """
    end = time.perf_counter() + max(0.0, dt)
    # target inner poll period ~200µs, adapt with dt as upper bound
    base = 0.0002
    step = min(base, max(0.00005, dt/20.0))
    while True:
        now = time.perf_counter()
        if now >= end:
            break
        try:
            if enc is not None:
                enc.poll()
        except Exception:
            pass
        rem = end - now
        # sleep a bit but keep high poll rate near the end
        time.sleep(step if rem > step else max(0.0, rem - 1e-05))

# ---- helpers for opening encoder, CSV, clamp, and verbose printing ----

def _open_encoder(enable: bool, chip: str, pin_a: int, pin_b: int):
    if not enable or EncoderSimple is None:
        return None
    try:
        return EncoderSimple(chip, pin_a, pin_b)  # type: ignore[call-arg]
    except Exception as e:
        print(f"[ERROR] Failed to open encoder: {e}")
        return None

def _csv_open(path_hint: str | None):
    if path_hint:
        csv_path = path_hint
        # Handle cases where the hint might be just a filename
        base_dir = os.path.dirname(path_hint)
        if not base_dir:
            base_dir = os.path.join('data', 'integrated_sensor')
            csv_path = os.path.join(base_dir, path_hint)
    else:
        base_dir = os.path.join('data', 'integrated_sensor')
        ts = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(base_dir, f'pid_tune_{ts}.csv')

    os.makedirs(base_dir, exist_ok=True)
    f = open(csv_path, 'w', buffering=1)
    f.write('ms,valve_a_pct,valve_b_pct,q_des,pid_u,enc_deg\n')
    return f, csv_path

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _clamp_valve(x: float, lo: float, hi: float) -> float:
    """Clamp valve duty with safety max applied."""
    safe_hi = min(hi, VALVE_MAX_PCT)
    return _clamp(x, lo, safe_hi)

# New: compute controller output limits from valve bounds to prevent windup

def _u_limits(min_pct: float, max_pct: float, center: float, bias: float = 0.0) -> tuple[float, float]:
    """Return (umin, umax) for u such that both valves remain within [min_pct, safe_max]."""
    safe_hi = min(max_pct, VALVE_MAX_PCT)
    cB = center + bias
    lo = min_pct
    hi = safe_hi
    umin = max(2.0 * (lo - cB), -2.0 * (hi - cB))
    umax = min(2.0 * (hi - cB), -2.0 * (lo - cB))
    return (umin, umax)

def _print_verbose(t_rel: float, a_pct: float, b_pct: float, q_des: float, angle: float, u: float) -> None:
    # Only print every 0.2 seconds to avoid flooding the terminal
    if int(t_rel * 5) % 1 == 0:  # Print every 0.2s
        print(f"t={t_rel:6.3f}s A={a_pct:6.1f}% B={b_pct:6.1f}% q_des={q_des:7.3f} enc={angle:7.3f} u={u:8.3f}")

# New: Pre-run check to observe encoder response
def _run_pre_check(dac: Dac8564, enc, invert: bool, zero_deg: float, ppr: int, center: float, min_pct: float, max_pct: float) -> None:
    if enc is None:
        return
    print("\n[INFO] Starting pre-run check to observe encoder response...")
    
    duration = 3.0
    max_diff = 25.0
    interval_s = 0.02  # 50Hz loop

    # Test movement 1: Ramp up positive pressure
    print("\n[INFO] Pre-check: Ramping positive pressure.")
    t0 = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0
        if elapsed > duration:
            break
        
        diff = (elapsed / duration) * max_diff
        a_pct = _clamp_valve(center + diff, min_pct, max_pct)
        b_pct = _clamp_valve(center - diff, min_pct, max_pct)
        dac.set_channels(a_pct, b_pct)
        
        angle = _read_angle(enc, invert, zero_deg, ppr)
        print(f"\r[Pre-check] Angle: {angle:7.3f} deg, Diff: {diff:5.1f}", end="")
        _sleep_with_poll(enc, interval_s)
    
    # Test movement 2: Ramp up negative pressure
    print("\n[INFO] Pre-check: Ramping negative pressure.")
    t0 = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0
        if elapsed > duration:
            break
            
        diff = (elapsed / duration) * max_diff
        a_pct = _clamp_valve(center - diff, min_pct, max_pct)
        b_pct = _clamp_valve(center + diff, min_pct, max_pct)
        dac.set_channels(a_pct, b_pct)

        angle = _read_angle(enc, invert, zero_deg, ppr)
        print(f"\r[Pre-check] Angle: {angle:7.3f} deg, Diff: {-diff:5.1f}", end="")
        _sleep_with_poll(enc, interval_s)

    # Test movement 3: Ramp back to zero
    print("\n[INFO] Pre-check: Ramping back to zero.")
    t0 = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0
        if elapsed > duration:
            break
            
        diff = max_diff * (1.0 - (elapsed / duration))
        # Assuming last state was negative pressure
        a_pct = _clamp_valve(center - diff, min_pct, max_pct)
        b_pct = _clamp_valve(center + diff, min_pct, max_pct)
        dac.set_channels(a_pct, b_pct)

        angle = _read_angle(enc, invert, zero_deg, ppr)
        print(f"\r[Pre-check] Angle: {angle:7.3f} deg, Diff: {-diff:5.1f}", end="")
        _sleep_with_poll(enc, interval_s)


    print("\n[INFO] Pre-run check finished. Returning to zero and settling.")
    dac.set_channels(0.0, 0.0)
    time.sleep(2.0)
    # NOTE: We no longer re-zero here. The pre-check is for observation only
    # and should not affect the control reference frame.
    print("\n[INFO] Pre-check complete. Starting PID tuning run...")


# ---- encoder angle helpers (zeroing and reading) ----

def _capture_zero(enc, invert: bool, ppr: int) -> float:
    """Sample encoder briefly and return averaged angle (after optional invert) as zero offset."""
    if enc is None:
        return 0.0
    t0 = time.perf_counter()
    samples: list[float] = []
    while time.perf_counter() - t0 < 0.08:
        try:
            enc.poll()
            a = enc.degrees(ppr)
            if invert:
                a = -a
            samples.append(a)
        except Exception:
            pass
        time.sleep(0.002)
    if not samples:
        try:
            enc.poll()
            a = enc.degrees(ppr)
            if invert:
                a = -a
            return a
        except Exception:
            return 0.0
    return sum(samples) / len(samples)

def _read_angle(enc, invert: bool, zero_deg: float, ppr: int) -> float:
    a = 0.0
    if enc is not None:
        try:
            enc.poll()
            a = enc.degrees(ppr)
        except Exception as e:
            print(f"[ERROR] Failed to read encoder: {e}")
            a = 0.0
    if invert:
        a = -a
    return a - zero_deg

# -------- Performance Metrics Calculation --------

@dataclass
class PIDPerformanceMetrics:
    """A dataclass to hold various PID performance metrics."""
    rise_time_10_90: float | None = None
    overshoot_percent: float | None = None
    settling_time_2_percent: float | None = None
    peak_time: float | None = None
    peak_value: float | None = None
    steady_state_error: float | None = None
    rmse: float | None = None
    iae: float | None = None
    itae: float | None = None
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    csv_path: str | None = None


def compute_rmse(error: np.ndarray) -> float:
    """Computes the Root Mean Squared Error."""
    return np.sqrt(np.mean(error**2))

def compute_iae(error: np.ndarray, t: np.ndarray) -> float:
    """Computes the Integral of Absolute Error."""
    return np.trapz(np.abs(error), t)

def compute_itae(error: np.ndarray, t: np.ndarray) -> float:
    """Computes the Integral of Time-weighted Absolute Error."""
    return np.trapz(t * np.abs(error), t)

def compute_overshoot_percent(y: np.ndarray, setpoint: float, step_amp: float) -> float | None:
    """Computes the percentage overshoot."""
    if step_amp == 0: return 0.0
    peak = np.max(y) if step_amp > 0 else np.min(y)
    overshoot = peak - setpoint
    return (overshoot / step_amp) * 100.0 if step_amp != 0 else 0.0

def compute_rise_time_10_90(t: np.ndarray, y: np.ndarray, q0: float, q1: float) -> float | None:
    """Computes the 10% to 90% rise time."""
    step_amp = q1 - q0
    if step_amp == 0: return None
    
    th_10 = q0 + 0.1 * step_amp
    th_90 = q0 + 0.9 * step_amp

    if step_amp > 0:
        t10_indices = np.where(y >= th_10)[0]
        t90_indices = np.where(y >= th_90)[0]
    else: # Negative step
        t10_indices = np.where(y <= th_10)[0]
        t90_indices = np.where(y <= th_90)[0]

    if t10_indices.size == 0 or t90_indices.size == 0:
        return None
        
    t10 = t[t10_indices[0]]
    t90 = t[t90_indices[0]]
    return t90 - t10

def compute_settling_time(t: np.ndarray, y: np.ndarray, setpoint: float, step_amp: float, t_change: float, tolerance_pct: float = 0.02) -> float | None:
    """Computes the settling time within a given tolerance percentage."""
    if step_amp == 0: return 0.0
    tolerance = tolerance_pct * abs(step_amp)
    settled_mask = np.abs(y - setpoint) <= tolerance
    
    # Find the last time the system was outside the tolerance band after the step change
    unsettled_indices = np.where(~settled_mask & (t >= t_change))[0]
    if unsettled_indices.size == 0:
        # If it never leaves the tolerance band after the step, it's settled immediately
        return 0.0
        
    last_unsettled_time = t[unsettled_indices[-1]]
    return last_unsettled_time - t_change

def compute_peak_time(t: np.ndarray, y: np.ndarray, t_change: float) -> float | None:
    """Finds the time to the first peak after the step change."""
    y_after_step = y[t >= t_change]
    t_after_step = t[t >= t_change]
    if y_after_step.size == 0: return None
    
    peak_idx = np.argmax(np.abs(y_after_step - y_after_step[0]))
    return t_after_step[peak_idx] - t_change

def calculate_performance_metrics(csv_path: str, t_change: float) -> PIDPerformanceMetrics | None:
    """Analyzes a CSV from a step response to calculate performance metrics."""
    try:
        df = pd.read_csv(csv_path)
        if 'ms' not in df.columns or 'q_des' not in df.columns or 'enc_deg' not in df.columns:
            print(f"[WARN] Metrics: CSV '{csv_path}' missing required columns (ms, q_des, enc_deg).")
            return None

        t = df['ms'].to_numpy() / 1000.0
        q_des = df['q_des'].to_numpy()
        y = df['enc_deg'].to_numpy()

        # Find step parameters
        q0 = q_des[0]
        q1_candidates = q_des[t >= t_change]
        if q1_candidates.size == 0:
            print(f"[WARN] Metrics: No data found after t_change={t_change}s.")
            return None
        q1 = q1_candidates[0]
        step_amp = q1 - q0

        # Filter data to the relevant analysis window (from step change onwards)
        analysis_mask = t >= t_change
        t_analysis = t[analysis_mask]
        y_analysis = y[analysis_mask]
        q_des_analysis = q_des[analysis_mask]
        
        if t_analysis.size < 2:
            print("[WARN] Metrics: Not enough data points for analysis after step change.")
            return None

        error = q_des_analysis - y_analysis
        
        # Calculate metrics
        metrics = PIDPerformanceMetrics()
        metrics.rise_time_10_90 = compute_rise_time_10_90(t, y, q0, q1)
        metrics.overshoot_percent = compute_overshoot_percent(y_analysis, q1, step_amp)
        metrics.settling_time_2_percent = compute_settling_time(t, y, q1, step_amp, t_change)
        metrics.peak_time = compute_peak_time(t, y, t_change)
        if metrics.peak_time is not None:
            peak_full_idx = np.argmin(np.abs(t - (t_change + metrics.peak_time)))
            metrics.peak_value = y[peak_full_idx]
        metrics.steady_state_error = np.mean(error[-max(1, int(len(error)*0.1)):]) # Avg of last 10%
        metrics.rmse = compute_rmse(error)
        metrics.iae = compute_iae(error, t_analysis)
        metrics.itae = compute_itae(error, t_analysis)
        metrics.csv_path = csv_path

        return metrics

    except Exception as e:
        print(f"[ERROR] Failed to calculate performance metrics for '{csv_path}': {e}")
        return None

def save_metrics(metrics: PIDPerformanceMetrics, gains: dict, summary_file: str = 'pid_tune_results.csv'):
    """Saves metrics to a JSON file and appends a summary to a CSV."""
    if not metrics.csv_path:
        print("[ERROR] Cannot save metrics, CSV path is missing.")
        return

    # Save detailed metrics to JSON
    json_path = metrics.csv_path.replace('.csv', '.metrics.json')
    try:
        metrics_dict = asdict(metrics)
        metrics_dict.update(gains)
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved detailed metrics to: {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save metrics JSON to '{json_path}': {e}")

    # Append summary to master CSV
    summary_path = os.path.join(os.path.dirname(metrics.csv_path), '..', summary_file)
    try:
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                header = ['timestamp', 'kp', 'ki', 'kd', 'rmse', 'overshoot_pct', 'rise_time_10_90', 'settling_time_2_pct', 'steady_state_error', 'csv_path']
                writer.writerow(header)
            
            # Write data row
            row = [
                time.strftime('%Y-%m-%d %H:%M:%S'),
                gains.get('kp'),
                gains.get('ki'),
                gains.get('kd'),
                f"{metrics.rmse:.4f}" if metrics.rmse is not None else "",
                f"{metrics.overshoot_percent:.2f}" if metrics.overshoot_percent is not None else "",
                f"{metrics.rise_time_10_90:.4f}" if metrics.rise_time_10_90 is not None else "",
                f"{metrics.settling_time_2_percent:.4f}" if metrics.settling_time_2_percent is not None else "",
                f"{metrics.steady_state_error:.4f}" if metrics.steady_state_error is not None else "",
                os.path.basename(metrics.csv_path)
            ]
            writer.writerow(row)
        print(f"Appended summary to: {summary_path}")
    except Exception as e:
        print(f"[ERROR] Failed to append summary to '{summary_path}': {e}")


# -------- Offline analysis --------

def analyze_csv(path: str):
    if not os.path.isfile(path):
        print('[ERROR] File not found:', path); return
    with open(path) as f:
        r=csv.DictReader(f)
        t=[]; qd=[]; y=[]
        for row in r:
            try:
                tt=float(row['ms'])/1000.0
                q=row.get('q_des',''); yv=row.get('enc_deg','')
                if q: qd.append((tt,float(q)))
                if yv: y.append((tt,float(yv)))
                t.append(tt)
            except: pass
    if not qd or not y:
        print('[WARN] Need q_des & enc_deg.')
        return
    q0=qd[0][1]
    idx_change=None; q1=q0
    for tt,q in qd:
        if q!=q0:
            idx_change=tt; q1=q; break
    if idx_change is None or q1==q0:
        print('[WARN] No step detected.')
        return
    amp=q1-q0
    # rise time 10-90
    yvals=y
    def first(th):
        for tt,v in yvals:
            if (amp>0 and v>=q0+th*amp) or (amp<0 and v<=q0+th*amp): return tt
        return None
    t10=first(0.1); t90=first(0.9)
    peak=max(v for _,v in yvals) if amp>0 else min(v for _,v in yvals)
    overshoot=(peak-q1)/amp*100 if amp!=0 else 0
    # settling 2%
    up=q1+0.02*abs(amp); lo=q1-0.02*abs(amp)
    t_set=None
    for tt,v in reversed(yvals):
        if lo<=v<=up: t_set=tt
        else: break
    print(f'Step at {idx_change:.3f}s to {q1:.2f}')
    if t10 and t90: print(f'Rise(10-90)={(t90-t10):.3f}s')
    print(f'Overshoot={overshoot:.1f}% (peak {peak:.2f})')
    if t_set: print(f'Settling(±2%)={(t_set-idx_change):.3f}s')

    # New: Run full metrics calculation
    print("\n--- Detailed Performance Metrics ---")
    metrics = calculate_performance_metrics(path, t_change=idx_change)
    if metrics:
        for key, value in asdict(metrics).items():
            if value is not None and key != 'csv_path':
                print(f"{key:<25}: {value:.4f}")
    print("----------------------------------")

    # After analysis, optionally plot
    try:
        df = pd.read_csv(path)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        if 'ms' in df.columns:
            t = df['ms'].astype(float)/1000.0
        else:
            t = pd.Series(range(len(df)))
        if 'enc_deg' in df.columns:
            ax1.plot(t, df['enc_deg'], label='enc_deg')
        if 'q_des' in df.columns:
            ax1.plot(t, df['q_des'], label='q_des')
        ax1.set_ylabel('Angle [deg]'); ax1.legend(); ax1.grid(True)
        if 'pid_u' in df.columns:
            ax2.plot(t, df['pid_u'], label='u')
        else:
            ax2.plot(t, df.iloc[:, -3], label='u')
        ax2.set_ylabel('u'); ax2.set_xlabel('Time [s]'); ax2.grid(True)
        out = path.replace('.csv', '.png')
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print('Saved plot:', out)
    except Exception:
        pass

# -------- Run modes (direct hardware) --------

def build_parser():
    ap=argparse.ArgumentParser(description='PID tuning helper')
    def common(p):
        p.add_argument('--min', type=float, default=10)
        p.add_argument('--max', type=float, default=100)
        p.add_argument('--interval-ms', type=int, default=20)
        p.add_argument('--total', type=float, default=10.0)
        p.add_argument('--kp', type=float, default=2.0)
        p.add_argument('--ki', type=float, default=0.0)
        p.add_argument('--kd', type=float, default=0.0)
        p.add_argument('--encoder', action='store_true')
        # Flip encoder sign by default; allow turning off with --no-encoder-invert
        p.add_argument('--encoder-invert', dest='encoder_invert', action='store_true', default=True, help='Invert encoder sign (default: on)')
        p.add_argument('--no-encoder-invert', dest='encoder_invert', action='store_false', help='Do not invert encoder sign')
        # New: control polarity (flip actuation sign)
        p.add_argument('--control-invert', dest='control_invert', action='store_true', default=False, help='Invert control polarity (flip u sign)')
        p.add_argument('--no-control-invert', dest='control_invert', action='store_false', help='Do not invert control polarity')
        # New: encoder angle definition parameters
        p.add_argument('--ppr', type=int, default=2048, help='Encoder PPR per channel (same as integrated_sensor_sin_python)')
        p.add_argument('--zero-at-start', dest='zero_at_start', action='store_true', default=True, help='Capture current angle as zero at start (default: on)')
        p.add_argument('--no-zero-at-start', dest='zero_at_start', action='store_false', help='Do not capture current angle as zero at start')
        p.add_argument('--zero-deg', type=float, default=None, help='Explicit zero offset in degrees (applied after invert)')
        # New: Pre-run check
        p.add_argument('--pre-check', dest='pre_check', action='store_true', default=True, help='Run a pre-check sequence to observe encoder response (default: on)')
        p.add_argument('--no-pre-check', dest='pre_check', action='store_false', help='Do not run a pre-check sequence')
        # New: hardware configuration
        p.add_argument('--spi-bus', type=int, default=SPI_BUS, help='SPI bus index (default 0)')
        p.add_argument('--spi-dev', type=int, default=SPI_DEV, help='SPI device index (default 0 => /dev/spidev0.0)')
        p.add_argument('--cs-gpio', type=int, default=GPIO_CS_DAC, help='Manual CS GPIO pin (default 19)')
        p.add_argument('--enc-chip', default=ENC_CHIP, help='Encoder gpiochip path (default /dev/gpiochip4)')
        p.add_argument('--enc-a', type=int, default=ENC_A, help='Encoder A pin offset on chip')
        p.add_argument('--enc-b', type=int, default=ENC_B, help='Encoder B pin offset on chip')
        # New: safety cap override
        p.add_argument('--valve-max-pct', type=float, default=VALVE_MAX_PCT, help='Safety cap for valve open percentage')
        p.add_argument('--csv', default=None)
        p.add_argument('-v','--verbose', action='store_true')
    sub=ap.add_subparsers(dest='mode', required=True)
    sp=sub.add_parser('step', help='Single step response'); common(sp)
    sp.add_argument('--step-q0', type=float, default=0.0)
    sp.add_argument('--step-q1', type=float, default=50.0)
    sp.add_argument('--t-change', type=float, default=2.0)
    mp=sub.add_parser('multistep', help='Multiple step sequence'); common(mp)
    mp.add_argument('--levels', type=float, nargs='+', required=True)
    mp.add_argument('--dwell', type=float, default=3.0)
    rp=sub.add_parser('relay', help='Relay oscillation test'); common(rp)
    rp.add_argument('--amp', type=float, default=20.0)
    rp.add_argument('--bias', type=float, default=0.0)
    rp.add_argument('--deadband', type=float, default=1.0)
    rp.add_argument('--target', type=float, default=0.0)
    at=sub.add_parser('autotune', help='Auto-tune PID via relay (Åström–Hägglund)'); common(at)
    at.add_argument('--amp', type=float, default=20.0, help='Relay differential amplitude (u)')
    at.add_argument('--bias', type=float, default=0.0, help='Bias added to both valves')
    at.add_argument('--deadband', type=float, default=1.0, help='Error deadband for switching')
    at.add_argument('--target', type=float, default=0.0, help='Target (encoder units)')
    at.add_argument('--rule', default='zn_pid', choices=['zn_pid','zn_pi','tl_pid','pessen'], help='Tuning rule')
    at.add_argument('--settle-sec', type=float, default=3.0, help='Ignore peaks before this time')
    at.add_argument('--cycles', type=int, default=6, help='Number of peaks to collect (max)')
    # New: automatic features
    at.add_argument('--preposition', action='store_true', help='Bring encoder near target before relay')
    at.add_argument('--pre-timeout', type=float, default=6.0, help='Preposition timeout seconds')
    at.add_argument('--pre-bias', type=float, default=0.0, help='Bias applied during preposition')
    at.add_argument('--pre-max-u', type=float, default=30.0, help='Max control effort during preposition')
    at.add_argument('--auto-amp', action='store_true', help='Automatically ramp relay amplitude until switching starts')
    at.add_argument('--amp-min', type=float, default=10.0, help='Minimum relay amplitude when auto ramping')
    at.add_argument('--amp-max', type=float, default=50.0, help='Maximum relay amplitude when auto ramping')
    at.add_argument('--amp-step', type=float, default=5.0, help='Amplitude increment when auto ramping')
    at.add_argument('--save', default=None, help='Optional path to save tuned gains as JSON')
    # New: correct Ku for relay hysteresis (deadband)
    at.add_argument('--hysteresis-correct', action='store_true', help='Apply deadband hysteresis correction to Ku')
    # New: ZN ultimate sensitivity (P-only sweep)
    zl=sub.add_parser('zn_limit', help='ZN ultimate sensitivity: sweep P-gain to find Ku, Tu'); common(zl)
    zl.add_argument('--kp-start', type=float, default=0.5)
    zl.add_argument('--kp-max', type=float, default=300.0)
    zl.add_argument('--kp-step', type=float, default=0.5)
    zl.add_argument('--dwell', type=float, default=5.0, help='Seconds to run per Kp')
    zl.add_argument('--target', type=float, default=0.0)
    zl.add_argument('--tol', type=float, default=0.05, help='Peak ratio tolerance for sustained oscillation')
    zl.add_argument('--settle-sec', type=float, default=1.5)
    zl.add_argument('--peaks', type=int, default=6)
    zl.add_argument('--rule', default='zn_pid', choices=['zn_pid','zn_pi','tl_pid','pessen'])
    zl.add_argument('--save', default=None)
    # New: ZN step-response identification (open-loop)
    zs=sub.add_parser('zn_step', help='ZN step-response identification: estimate K,L,T'); common(zs)
    zs.add_argument('--u-step', type=float, default=10.0, help='Open-loop differential u to apply')
    zs.add_argument('--bias', type=float, default=0.0)
    zs.add_argument('--pre-wait', type=float, default=1.0, help='Seconds to measure baseline before step')
    zs.add_argument('--rule', default='zn_step_pid', choices=['zn_step_pid','zn_step_pi','zn_step_p'])
    zs.add_argument('--smooth', type=int, default=5, help='SMA window for slope estimate (samples)')
    zs.add_argument('--save', default=None)
    an=sub.add_parser('analyze', help='Analyze existing CSV'); an.add_argument('path')
    pl=sub.add_parser('plot', help='Plot existing CSV'); pl.add_argument('path')

    # New: evaluation mode to test tracking on trajectories (sine / random)
    ev = sub.add_parser('evaluate', help='Evaluate tracking on trajectories (sin/random)')
    common(ev)
    ev.add_argument('--traj', choices=['sin','random'], default='sin', help='Trajectory type')
    ev.add_argument('--amp', type=float, default=30.0, help='Trajectory amplitude (deg)')
    ev.add_argument('--offset', type=float, default=0.0, help='Trajectory offset (deg)')
    ev.add_argument('--freq', type=float, default=0.2, help='Frequency for sin [Hz]')
    ev.add_argument('--dwell', type=float, default=0.5, help='Random step dwell time [s]')
    ev.add_argument('--seed', type=int, default=None, help='Seed for random trajectory')
    ev.add_argument('--t-total', type=float, default=None, help='Optional override for total run time (s)')
    return ap

# Replace IntegratedSession usage with direct hardware I/O matching apps/integrated_sensor_sin_python.py
try:
    import spidev  # type: ignore
except Exception:  # noqa: BLE001
    spidev = None  # type: ignore
try:
    import gpiod  # type: ignore
    from gpiod.line import Direction, Value  # type: ignore
except Exception:  # noqa: BLE001
    gpiod = None  # type: ignore
    Direction = Value = None  # type: ignore
# Optional encoder (reuse shared implementation)
try:
    from affetto_nn_ctrl.hw.encoder import EncoderSimple  # type: ignore
except Exception:  # noqa: BLE001
    EncoderSimple = None  # type: ignore

# SPI/DAC constants (can be overridden via CLI)
SPI_BUS = 0
SPI_DEV = 0               # /dev/spidev0.0
SPI_MAX_HZ = 1_000_000
SPI_MODE_DAC = 0b01       # MODE1 for DAC8564
GPIO_CS_DAC = 19          # Manual CS GPIO (active low)

# Safety cap: do not open valves beyond this percentage (overridable via CLI)
VALVE_MAX_PCT = 100.0

# Default encoder pins (match integrated_sensor_sin_python.py)
ENC_CHIP = '/dev/gpiochip4'
ENC_A = 14
ENC_B = 4

class CsLine:
    """Minimal CS line using libgpiod v2 (same as integrated_sensor_sin_python)."""
    def __init__(self, pin: int):
        if gpiod is None:
            raise RuntimeError('python3-libgpiod is required')
        self.pin = pin
        self.chip = gpiod.Chip('/dev/gpiochip0')
        ls = gpiod.LineSettings()
        if Direction is not None:
            ls.direction = Direction.OUTPUT
        if Value is not None:
            ls.output_value = Value.ACTIVE
        self.req = self.chip.request_lines(consumer='dac_cs', config={pin: ls})
        try:
            self.set(1)  # inactive (high)
        except Exception:
            pass
    def set(self, val: int):
        if Value is not None:
            self.req.set_values({self.pin: (Value.ACTIVE if val else Value.INACTIVE)})
        else:  # pragma: no cover
            self.req.set_values({self.pin: val})  # type: ignore[arg-type]
    def close(self):  # pragma: no cover
        try:
            self.set(1)
        except Exception:
            pass
        try:
            self.req.release()
        except Exception:
            pass
        try:
            self.chip.close()
        except Exception:
            pass

class Dac8564:
    """Direct DAC8564 access with manual CS per transfer (aligned with working script)."""
    def __init__(self, bus: int, dev: int, cs_gpio: int):
        if spidev is None:
            raise RuntimeError('spidev is required')
        self.spi = spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        self.cs = CsLine(cs_gpio)
        self.opened = False
    def open(self):
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            self.spi.max_speed_hz = SPI_MAX_HZ
            self.spi.mode = SPI_MODE_DAC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_and_init()
    def close(self):  # pragma: no cover
        if self.opened:
            try:
                self.spi.close()
            except Exception:
                pass
        try:
            self.cs.close()
        except Exception:
            pass
    def _xfer(self, data: list[int]):
        if not self.opened:
            raise RuntimeError('SPI not opened yet')
        self.spi.mode = SPI_MODE_DAC
        self.cs.set(0)
        self.spi.xfer2(list(data))
        self.cs.set(1)
    @staticmethod
    def _pct_to_code(pct: float) -> int:
        if pct < 0.0: pct = 0.0
        # Enforce safety cap
        if pct > VALVE_MAX_PCT: pct = VALVE_MAX_PCT
        return int(pct * 65535.0 / 100.0 + 0.5)
    @staticmethod
    def _cmd(ch: int) -> int:
        # 0x10 (write input) with LDAC low -> immediate update
        return 0x10 | (ch << 1)
    def set_channels(self, a_pct: float, b_pct: float):
        a_code = self._pct_to_code(a_pct)
        b_code = self._pct_to_code(b_pct)
        a = [self._cmd(0), (a_code >> 8) & 0xFF, a_code & 0xFF]
        b = [self._cmd(1), (b_code >> 8) & 0xFF, b_code & 0xFF]
        self._xfer(a); self._xfer(b)
    def _reset_and_init(self):
        # Reset and power config (exactly as in working script)
        self._xfer([0x28,0x00,0x01])
        time.sleep(0.001)
        self._xfer([0x38,0x00,0x01])

# Simple PID controller (local)
class SimplePID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.i = 0.0
        self.prev_y = None
    def reset(self) -> None:
        self.i = 0.0
        self.prev_y = None
    def step(self, sp: float, y: float, dt: float, umin: float | None = None, umax: float | None = None) -> float:
        """Parallel PID with derivative on measurement and conditional anti-windup.
        If umin/umax are provided, the integrator is inhibited when saturated in the
        direction that would increase saturation."""
        e = sp - y
        p = self.kp * e
        d = 0.0
        if self.prev_y is not None and dt > 0:
            d = - self.kd * (y - self.prev_y) / dt  # derivative on measurement (no setpoint kick)
        # candidate integrator advance
        i_next = self.i + self.ki * e * dt
        u_unclamped = p + i_next + d
        if umin is not None and umax is not None:
            u = _clamp(u_unclamped, umin, umax)
            saturated_high = u >= umax - 1e-9
            saturated_low = u <= umin + 1e-9
            # inhibit integration if pushing further into saturation
            if (saturated_high and e > 0) or (saturated_low and e < 0):
                i_next = self.i
                u = _clamp(p + i_next + d, umin, umax)
        else:
            u = u_unclamped
        self.i = i_next
        self.prev_y = y
        return u

# ---- small helper to poll sensors between frames ----

def _sleep_with_poll(enc, dt: float) -> None:
    """Sleep for dt seconds while polling encoder frequently.
    Poll at ~200µs (or finer for small dt) to avoid missing quadrature edges,
    matching the tight polling cadence used in integrated_sensor_sin_python.
    """
    end = time.perf_counter() + max(0.0, dt)
    # target inner poll period ~200µs, adapt with dt as upper bound
    base = 0.0002
    step = min(base, max(0.00005, dt/20.0))
    while True:
        now = time.perf_counter()
        if now >= end:
            break
        try:
            if enc is not None:
                enc.poll()
        except Exception:
            pass
        rem = end - now
        # sleep a bit but keep high poll rate near the end
        time.sleep(step if rem > step else max(0.0, rem - 1e-05))

# ---- helpers for opening encoder, CSV, clamp, and verbose printing ----

def _open_encoder(enable: bool, chip: str, pin_a: int, pin_b: int):
    if not enable or EncoderSimple is None:
        return None
    try:
        return EncoderSimple(chip, pin_a, pin_b)  # type: ignore[call-arg]
    except Exception as e:
        print(f"[ERROR] Failed to open encoder: {e}")
        return None

def _csv_open(path_hint: str | None):
    if path_hint:
        csv_path = path_hint
        # Handle cases where the hint might be just a filename
        base_dir = os.path.dirname(path_hint)
        if not base_dir:
            base_dir = os.path.join('data', 'integrated_sensor')
            csv_path = os.path.join(base_dir, path_hint)
    else:
        base_dir = os.path.join('data', 'integrated_sensor')
        ts = time.strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(base_dir, f'pid_tune_{ts}.csv')

    os.makedirs(base_dir, exist_ok=True)
    f = open(csv_path, 'w', buffering=1)
    f.write('ms,valve_a_pct,valve_b_pct,q_des,pid_u,enc_deg\n')
    return f, csv_path

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def _clamp_valve(x: float, lo: float, hi: float) -> float:
    """Clamp valve duty with safety max applied."""
    safe_hi = min(hi, VALVE_MAX_PCT)
    return _clamp(x, lo, safe_hi)

# New: compute controller output limits from valve bounds to prevent windup

def _u_limits(min_pct: float, max_pct: float, center: float, bias: float = 0.0) -> tuple[float, float]:
    """Return (umin, umax) for u such that both valves remain within [min_pct, safe_max]."""
    safe_hi = min(max_pct, VALVE_MAX_PCT)
    cB = center + bias
    lo = min_pct
    hi = safe_hi
    umin = max(2.0 * (lo - cB), -2.0 * (hi - cB))
    umax = min(2.0 * (hi - cB), -2.0 * (lo - cB))
    return (umin, umax)

def _print_verbose(t_rel: float, a_pct: float, b_pct: float, q_des: float, angle: float, u: float) -> None:
    # Only print every 0.2 seconds to avoid flooding the terminal
    if int(t_rel * 5) % 1 == 0:  # Print every 0.2s
        print(f"t={t_rel:6.3f}s A={a_pct:6.1f}% B={b_pct:6.1f}% q_des={q_des:7.3f} enc={angle:7.3f} u={u:8.3f}")

# New: Pre-run check to observe encoder response
def _run_pre_check(dac: Dac8564, enc, invert: bool, zero_deg: float, ppr: int, center: float, min_pct: float, max_pct: float) -> None:
    if enc is None:
        return
    print("\n[INFO] Starting pre-run check to observe encoder response...")
    
    duration = 3.0
    max_diff = 25.0
    interval_s = 0.02  # 50Hz loop

    # Test movement 1: Ramp up positive pressure
    print("\n[INFO] Pre-check: Ramping positive pressure.")
    t0 = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0
        if elapsed > duration:
            break
        
        diff = (elapsed / duration) * max_diff
        a_pct = _clamp_valve(center + diff, min_pct, max_pct)
        b_pct = _clamp_valve(center - diff, min_pct, max_pct)
        dac.set_channels(a_pct, b_pct)
        
        angle = _read_angle(enc, invert, zero_deg, ppr)
        print(f"\r[Pre-check] Angle: {angle:7.3f} deg, Diff: {diff:5.1f}", end="")
        _sleep_with_poll(enc, interval_s)
    
    # Test movement 2: Ramp up negative pressure
    print("\n[INFO] Pre-check: Ramping negative pressure.")
    t0 = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0
        if elapsed > duration:
            break
            
        diff = (elapsed / duration) * max_diff
        a_pct = _clamp_valve(center - diff, min_pct, max_pct)
        b_pct = _clamp_valve(center + diff, min_pct, max_pct)
        dac.set_channels(a_pct, b_pct)

        angle = _read_angle(enc, invert, zero_deg, ppr)
        print(f"\r[Pre-check] Angle: {angle:7.3f} deg, Diff: {-diff:5.1f}", end="")
        _sleep_with_poll(enc, interval_s)

    # Test movement 3: Ramp back to zero
    print("\n[INFO] Pre-check: Ramping back to zero.")
    t0 = time.perf_counter()
    while True:
        loop_start = time.perf_counter()
        elapsed = loop_start - t0
        if elapsed > duration:
            break
            
        diff = max_diff * (1.0 - (elapsed / duration))
        # Assuming last state was negative pressure
        a_pct = _clamp_valve(center - diff, min_pct, max_pct)
        b_pct = _clamp_valve(center + diff, min_pct, max_pct)
        dac.set_channels(a_pct, b_pct)

        angle = _read_angle(enc, invert, zero_deg, ppr)
        print(f"\r[Pre-check] Angle: {angle:7.3f} deg, Diff: {-diff:5.1f}", end="")
        _sleep_with_poll(enc, interval_s)


    print("\n[INFO] Pre-run check finished. Returning to zero and settling.")
    dac.set_channels(0.0, 0.0)
    time.sleep(2.0)
    # NOTE: We no longer re-zero here. The pre-check is for observation only
    # and should not affect the control reference frame.
    print("\n[INFO] Pre-check complete. Starting PID tuning run...")


# ---- encoder angle helpers (zeroing and reading) ----

def _capture_zero(enc, invert: bool, ppr: int) -> float:
    """Sample encoder briefly and return averaged angle (after optional invert) as zero offset."""
    if enc is None:
        return 0.0
    t0 = time.perf_counter()
    samples: list[float] = []
    while time.perf_counter() - t0 < 0.08:
        try:
            enc.poll()
            a = enc.degrees(ppr)
            if invert:
                a = -a
            samples.append(a)
        except Exception:
            pass
        time.sleep(0.002)
    if not samples:
        try:
            enc.poll()
            a = enc.degrees(ppr)
            if invert:
                a = -a
            return a
        except Exception:
            return 0.0
    return sum(samples) / len(samples)

def _read_angle(enc, invert: bool, zero_deg: float, ppr: int) -> float:
    a = 0.0
    if enc is not None:
        try:
            enc.poll()
            a = enc.degrees(ppr)
        except Exception as e:
            print(f"[ERROR] Failed to read encoder: {e}")
            a = 0.0
    if invert:
        a = -a
    return a - zero_deg

# -------- Performance Metrics Calculation --------

@dataclass
class PIDPerformanceMetrics:
    """A dataclass to hold various PID performance metrics."""
    rise_time_10_90: float | None = None
    overshoot_percent: float | None = None
    settling_time_2_percent: float | None = None
    peak_time: float | None = None
    peak_value: float | None = None
    steady_state_error: float | None = None
    rmse: float | None = None
    iae: float | None = None
    itae: float | None = None
    kp: float | None = None
    ki: float | None = None
    kd: float | None = None
    csv_path: str | None = None


def compute_rmse(error: np.ndarray) -> float:
    """Computes the Root Mean Squared Error."""
    return np.sqrt(np.mean(error**2))

def compute_iae(error: np.ndarray, t: np.ndarray) -> float:
    """Computes the Integral of Absolute Error."""
    return np.trapz(np.abs(error), t)

def compute_itae(error: np.ndarray, t: np.ndarray) -> float:
    """Computes the Integral of Time-weighted Absolute Error."""
    return np.trapz(t * np.abs(error), t)

def compute_overshoot_percent(y: np.ndarray, setpoint: float, step_amp: float) -> float | None:
    """Computes the percentage overshoot."""
    if step_amp == 0: return 0.0
    peak = np.max(y) if step_amp > 0 else np.min(y)
    overshoot = peak - setpoint
    return (overshoot / step_amp) * 100.0 if step_amp != 0 else 0.0

def compute_rise_time_10_90(t: np.ndarray, y: np.ndarray, q0: float, q1: float) -> float | None:
    """Computes the 10% to 90% rise time."""
    step_amp = q1 - q0
    if step_amp == 0: return None
    
    th_10 = q0 + 0.1 * step_amp
    th_90 = q0 + 0.9 * step_amp

    if step_amp > 0:
        t10_indices = np.where(y >= th_10)[0]
        t90_indices = np.where(y >= th_90)[0]
    else: # Negative step
        t10_indices = np.where(y <= th_10)[0]
        t90_indices = np.where(y <= th_90)[0]

    if t10_indices.size == 0 or t90_indices.size == 0:
        return None
        
    t10 = t[t10_indices[0]]
    t90 = t[t90_indices[0]]
    return t90 - t10

def compute_settling_time(t: np.ndarray, y: np.ndarray, setpoint: float, step_amp: float, t_change: float, tolerance_pct: float = 0.02) -> float | None:
    """Computes the settling time within a given tolerance percentage."""
    if step_amp == 0: return 0.0
    tolerance = tolerance_pct * abs(step_amp)
    settled_mask = np.abs(y - setpoint) <= tolerance
    
    # Find the last time the system was outside the tolerance band after the step change
    unsettled_indices = np.where(~settled_mask & (t >= t_change))[0]
    if unsettled_indices.size == 0:
        # If it never leaves the tolerance band after the step, it's settled immediately
        return 0.0
        
    last_unsettled_time = t[unsettled_indices[-1]]
    return last_unsettled_time - t_change

def compute_peak_time(t: np.ndarray, y: np.ndarray, t_change: float) -> float | None:
    """Finds the time to the first peak after the step change."""
    y_after_step = y[t >= t_change]
    t_after_step = t[t >= t_change]
    if y_after_step.size == 0: return None
    
    peak_idx = np.argmax(np.abs(y_after_step - y_after_step[0]))
    return t_after_step[peak_idx] - t_change

def calculate_performance_metrics(csv_path: str, t_change: float) -> PIDPerformanceMetrics | None:
    """Analyzes a CSV from a step response to calculate performance metrics."""
    try:
        df = pd.read_csv(csv_path)
        if 'ms' not in df.columns or 'q_des' not in df.columns or 'enc_deg' not in df.columns:
            print(f"[WARN] Metrics: CSV '{csv_path}' missing required columns (ms, q_des, enc_deg).")
            return None

        t = df['ms'].to_numpy() / 1000.0
        q_des = df['q_des'].to_numpy()
        y = df['enc_deg'].to_numpy()

        # Find step parameters
        q0 = q_des[0]
        q1_candidates = q_des[t >= t_change]
        if q1_candidates.size == 0:
            print(f"[WARN] Metrics: No data found after t_change={t_change}s.")
            return None
        q1 = q1_candidates[0]
        step_amp = q1 - q0

        # Filter data to the relevant analysis window (from step change onwards)
        analysis_mask = t >= t_change
        t_analysis = t[analysis_mask]
        y_analysis = y[analysis_mask]
        q_des_analysis = q_des[analysis_mask]
        
        if t_analysis.size < 2:
            print("[WARN] Metrics: Not enough data points for analysis after step change.")
            return None

        error = q_des_analysis - y_analysis
        
        # Calculate metrics
        metrics = PIDPerformanceMetrics()
        metrics.rise_time_10_90 = compute_rise_time_10_90(t, y, q0, q1)
        metrics.overshoot_percent = compute_overshoot_percent(y_analysis, q1, step_amp)
        metrics.settling_time_2_percent = compute_settling_time(t, y, q1, step_amp, t_change)
        metrics.peak_time = compute_peak_time(t, y, t_change)
        if metrics.peak_time is not None:
            peak_full_idx = np.argmin(np.abs(t - (t_change + metrics.peak_time)))
            metrics.peak_value = y[peak_full_idx]
        metrics.steady_state_error = np.mean(error[-max(1, int(len(error)*0.1)):]) # Avg of last 10%
        metrics.rmse = compute_rmse(error)
        metrics.iae = compute_iae(error, t_analysis)
        metrics.itae = compute_itae(error, t_analysis)
        metrics.csv_path = csv_path

        return metrics

    except Exception as e:
        print(f"[ERROR] Failed to calculate performance metrics for '{csv_path}': {e}")
        return None

def save_metrics(metrics: PIDPerformanceMetrics, gains: dict, summary_file: str = 'pid_tune_results.csv'):
    """Saves metrics to a JSON file and appends a summary to a CSV."""
    if not metrics.csv_path:
        print("[ERROR] Cannot save metrics, CSV path is missing.")
        return

    # Save detailed metrics to JSON
    json_path = metrics.csv_path.replace('.csv', '.metrics.json')
    try:
        metrics_dict = asdict(metrics)
        metrics_dict.update(gains)
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"Saved detailed metrics to: {json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save metrics JSON to '{json_path}': {e}")

    # Append summary to master CSV
    summary_path = os.path.join(os.path.dirname(metrics.csv_path), '..', summary_file)
    try:
        file_exists = os.path.isfile(summary_path)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            # Write header if new file
            if not file_exists:
                header = ['timestamp', 'kp', 'ki', 'kd', 'rmse', 'overshoot_pct', 'rise_time_10_90', 'settling_time_2_pct', 'steady_state_error', 'csv_path']
                writer.writerow(header)
            
            # Write data row
            row = [
                time.strftime('%Y-%m-%d %H:%M:%S'),
                gains.get('kp'),
                gains.get('ki'),
                gains.get('kd'),
                f"{metrics.rmse:.4f}" if metrics.rmse is not None else "",
                f"{metrics.overshoot_percent:.2f}" if metrics.overshoot_percent is not None else "",
                f"{metrics.rise_time_10_90:.4f}" if metrics.rise_time_10_90 is not None else "",
                f"{metrics.settling_time_2_percent:.4f}" if metrics.settling_time_2_percent is not None else "",
                f"{metrics.steady_state_error:.4f}" if metrics.steady_state_error is not None else "",
                os.path.basename(metrics.csv_path)
            ]
            writer.writerow(row)
        print(f"Appended summary to: {summary_path}")
    except Exception as e:
        print(f"[ERROR] Failed to append summary to '{summary_path}': {e}")


# -------- Offline analysis --------

def analyze_csv(path: str):
    if not os.path.isfile(path):
        print('[ERROR] File not found:', path); return
    with open(path) as f:
        r=csv.DictReader(f)
        t=[]; qd=[]; y=[]
        for row in r:
            try:
                tt=float(row['ms'])/1000.0
                q=row.get('q_des',''); yv=row.get('enc_deg','')
                if q: qd.append((tt,float(q)))
                if yv: y.append((tt,float(yv)))
                t.append(tt)
            except: pass
    if not qd or not y:
        print('[WARN] Need q_des & enc_deg.')
        return
    q0=qd[0][1]
    idx_change=None; q1=q0
    for tt,q in qd:
        if q!=q0:
            idx_change=tt; q1=q; break
    if idx_change is None or q1==q0:
        print('[WARN] No step detected.')
        return
    amp=q1-q0
    # rise time 10-90
    yvals=y
    def first(th):
        for tt,v in yvals:
            if (amp>0 and v>=q0+th*amp) or (amp<0 and v<=q0+th*amp): return tt
        return None
    t10=first(0.1); t90=first(0.9)
    peak=max(v for _,v in yvals) if amp>0 else min(v for _,v in yvals)
    overshoot=(peak-q1)/amp*100 if amp!=0 else 0
    # settling 2%
    up=q1+0.02*abs(amp); lo=q1-0.02*abs(amp)
    t_set=None
    for tt,v in reversed(yvals):
        if lo<=v<=up: t_set=tt
        else: break
    print(f'Step at {idx_change:.3f}s to {q1:.2f}')
    if t10 and t90: print(f'Rise(10-90)={(t90-t10):.3f}s')
    print(f'Overshoot={overshoot:.1f}% (peak {peak:.2f})')
    if t_set: print(f'Settling(±2%)={(t_set-idx_change):.3f}s')

    # New: Run full metrics calculation
    print("\n--- Detailed Performance Metrics ---")
    metrics = calculate_performance_metrics(path, t_change=idx_change)
    if metrics:
        for key, value in asdict(metrics).items():
            if value is not None and key != 'csv_path':
                print(f"{key:<25}: {value:.4f}")
    print("----------------------------------")

    # After analysis, optionally plot
    try:
        df = pd.read_csv(path)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        if 'ms' in df.columns:
            t = df['ms'].astype(float)/1000.0
        else:
            t = pd.Series(range(len(df)))
        if 'enc_deg' in df.columns:
            ax1.plot(t, df['enc_deg'], label='enc_deg')
        if 'q_des' in df.columns:
            ax1.plot(t, df['q_des'], label='q_des')
        ax1.set_ylabel('Angle [deg]'); ax1.legend(); ax1.grid(True)
        if 'pid_u' in df.columns:
            ax2.plot(t, df['pid_u'], label='u')
        else:
            ax2.plot(t, df.iloc[:, -3], label='u')
        ax2.set_ylabel('u'); ax2.set_xlabel('Time [s]'); ax2.grid(True)
        out = path.replace('.csv', '.png')
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print('Saved plot:', out)
    except Exception:
        pass

# -------- Run modes (direct hardware) --------

def build_parser():
    ap=argparse.ArgumentParser(description='PID tuning helper')
    def common(p):
        p.add_argument('--min', type=float, default=10)
        p.add_argument('--max', type=float, default=100)
        p.add_argument('--interval-ms', type=int, default=20)
        p.add_argument('--total', type=float, default=10.0)
        p.add_argument('--kp', type=float, default=2.0)
        p.add_argument('--ki', type=float, default=0.0)
        p.add_argument('--kd', type=float, default=0.0)
        p.add_argument('--encoder', action='store_true')
        # Flip encoder sign by default; allow turning off with --no-encoder-invert
        p.add_argument('--encoder-invert', dest='encoder_invert', action='store_true', default=True, help='Invert encoder sign (default: on)')
        p.add_argument('--no-encoder-invert', dest='encoder_invert', action='store_false', help='Do not invert encoder sign')
        # New: control polarity (flip actuation sign)
        p.add_argument('--control-invert', dest='control_invert', action='store_true', default=False, help='Invert control polarity (flip u sign)')
        p.add_argument('--no-control-invert', dest='control_invert', action='store_false', help='Do not invert control polarity')
        # New: encoder angle definition parameters
        p.add_argument('--ppr', type=int, default=2048, help='Encoder PPR per channel (same as integrated_sensor_sin_python)')
        p.add_argument('--zero-at-start', dest='zero_at_start', action='store_true', default=True, help='Capture current angle as zero at start (default: on)')
        p.add_argument('--no-zero-at-start', dest='zero_at_start', action='store_false', help='Do not capture current angle as zero at start')
        p.add_argument('--zero-deg', type=float, default=None, help='Explicit zero offset in degrees (applied after invert)')
        # New: Pre-run check
        p.add_argument('--pre-check', dest='pre_check', action='store_true', default=True, help='Run a pre-check sequence to observe encoder response (default: on)')
        p.add_argument('--no-pre-check', dest='pre_check', action='store_false', help='Do not run a pre-check sequence')
        # New: hardware configuration
        p.add_argument('--spi-bus', type=int, default=SPI_BUS, help='SPI bus index (default 0)')
        p.add_argument('--spi-dev', type=int, default=SPI_DEV, help='SPI device index (default 0 => /dev/spidev0.0)')
        p.add_argument('--cs-gpio', type=int, default=GPIO_CS_DAC, help='Manual CS GPIO pin (default 19)')
        p.add_argument('--enc-chip', default=ENC_CHIP, help='Encoder gpiochip path (default /dev/gpiochip4)')
        p.add_argument('--enc-a', type=int, default=ENC_A, help='Encoder A pin offset on chip')
        p.add_argument('--enc-b', type=int, default=ENC_B, help='Encoder B pin offset on chip')
        # New: safety cap override
        p.add_argument('--valve-max-pct', type=float, default=VALVE_MAX_PCT, help='Safety cap for valve open percentage')
        p.add_argument('--csv', default=None)
        p.add_argument('-v','--verbose', action='store_true')
    sub=ap.add_subparsers(dest='mode', required=True)
    sp=sub.add_parser('step', help='Single step response'); common(sp)
    sp.add_argument('--step-q0', type=float, default=0.0)
    sp.add_argument('--step-q1', type=float, default=50.0)
    sp.add_argument('--t-change', type=float, default=2.0)
    mp=sub.add_parser('multistep', help='Multiple step sequence'); common(mp)
    mp.add_argument('--levels', type=float, nargs='+', required=True)
    mp.add_argument('--dwell', type=float, default=3.0)
    rp=sub.add_parser('relay', help='Relay oscillation test'); common(rp)
    rp.add_argument('--amp', type=float, default=20.0)
    rp.add_argument('--bias', type=float, default=0.0)
    rp.add_argument('--deadband', type=float, default=1.0)
    rp.add_argument('--target', type=float, default=0.0)
    at=sub.add_parser('autotune', help='Auto-tune PID via relay (Åström–Hägglund)'); common(at)
    at.add_argument('--amp', type=float, default=20.0, help='Relay differential amplitude (u)')
    at.add_argument('--bias', type=float, default=0.0, help='Bias added to both valves')
    at.add_argument('--deadband', type=float, default=1.0, help='Error deadband for switching')
    at.add_argument('--target', type=float, default=0.0, help='Target (encoder units)')
    at.add_argument('--rule', default='zn_pid', choices=['zn_pid','zn_pi','tl_pid','pessen'], help='Tuning rule')
    at.add_argument('--settle-sec', type=float, default=3.0, help='Ignore peaks before this time')
    at.add_argument('--cycles', type=int, default=6, help='Number of peaks to collect (max)')
    # New: automatic features
    at.add_argument('--preposition', action='store_true', help='Bring encoder near target before relay')
    at.add_argument('--pre-timeout', type=float, default=6.0, help='Preposition timeout seconds')
    at.add_argument('--pre-bias', type=float, default=0.0, help='Bias applied during preposition')
    at.add_argument('--pre-max-u', type=float, default=30.0, help='Max control effort during preposition')
    at.add_argument('--auto-amp', action='store_true', help='Automatically ramp relay amplitude until switching starts')
    at.add_argument('--amp-min', type=float, default=10.0, help='Minimum relay amplitude when auto ramping')
    at.add_argument('--amp-max', type=float, default=50.0, help='Maximum relay amplitude when auto ramping')
    at.add_argument('--amp-step', type=float, default=5.0, help='Amplitude increment when auto ramping')
    at.add_argument('--save', default=None, help='Optional path to save tuned gains as JSON')
    # New: correct Ku for relay hysteresis (deadband)
    at.add_argument('--hysteresis-correct', action='store_true', help='Apply deadband hysteresis correction to Ku')
    # New: ZN ultimate sensitivity (P-only sweep)
    zl=sub.add_parser('zn_limit', help='ZN ultimate sensitivity: sweep P-gain to find Ku, Tu'); common(zl)
    zl.add_argument('--kp-start', type=float, default=0.5)
    zl.add_argument('--kp-max', type=float, default=300.0)
    zl.add_argument('--kp-step', type=float, default=0.5)
    zl.add_argument('--dwell', type=float, default=5.0, help='Seconds to run per Kp')
    zl.add_argument('--target', type=float, default=0.0)
    zl.add_argument('--tol', type=float, default=0.05, help='Peak ratio tolerance for sustained oscillation')
    zl.add_argument('--settle-sec', type=float, default=1.5)
    zl.add_argument('--peaks', type=int, default=6)
    zl.add_argument('--rule', default='zn_pid', choices=['zn_pid','zn_pi','tl_pid','pessen'])
    zl.add_argument('--save', default=None)
    # New: ZN step-response identification (open-loop)
    zs=sub.add_parser('zn_step', help='ZN step-response identification: estimate K,L,T'); common(zs)
    zs.add_argument('--u-step', type=float, default=10.0, help='Open-loop differential u to apply')
    zs.add_argument('--bias', type=float, default=0.0)
    zs.add_argument('--pre-wait', type=float, default=1.0, help='Seconds to measure baseline before step')
    zs.add_argument('--rule', default='zn_step_pid', choices=['zn_step_pid','zn_step_pi','zn_step_p'])
    zs.add_argument('--smooth', type=int, default=5, help='SMA window for slope estimate (samples)')
    zs.add_argument('--save', default=None)
    an=sub.add_parser('analyze', help='Analyze existing CSV'); an.add_argument('path')
    pl=sub.add_parser('plot', help='Plot existing CSV'); pl.add_argument('path')

    # New: evaluation mode to test tracking on trajectories (sine / random)
    ev = sub.add_parser('evaluate', help='Evaluate tracking on trajectories (sin/random)')
    common(ev)
    ev.add_argument('--traj', choices=['sin','random'], default='sin', help='Trajectory type')
    ev.add_argument('--amp', type=float, default=30.0, help='Trajectory amplitude (deg)')
    ev.add_argument('--offset', type=float, default=0.0, help='Trajectory offset (deg)')
    ev.add_argument('--freq', type=float, default=0.2, help='Frequency for sin [Hz]')
    ev.add_argument('--dwell', type=float, default=0.5, help='Random step dwell time [s]')
    ev.add_argument('--seed', type=int, default=None, help='Seed for random trajectory')
    ev.add_argument('--t-total', type=float, default=None, help='Optional override for total run time (s)')
    return ap

# Pre-position helper to drive angle near target before relay

def _preposition_to_target(dac: Dac8564, enc, setpoint: float, center: float, bias: float, max_u: float, deadband: float, interval_ms: int, timeout_s: float, invert: bool, zero_deg: float, ppr: int, verbose: bool=False, control_invert: bool=False) -> None:
    if enc is None:
        return
    kp_pre = 4.0  # gentle proportional push
    t0 = time.perf_counter()
    within_since = None
    dt_sleep = max(0.005, interval_ms/1000.0)
    while True:
        now = time.perf_counter()
        if now - t0 > timeout_s:
            break
        try:
            enc.poll()
        except Exception:
            pass
        # Inline angle calculation to avoid dependency ordering
        try:
            angle = enc.degrees(ppr)
        except Exception:
            angle = 0.0
        if invert:
            angle = -angle
        angle = angle - zero_deg
        err = setpoint - angle
        if abs(err) <= deadband:
            if within_since is None:
                within_since = now
            if now - within_since >= 0.3:
                break
        else:
            within_since = None
        u = _clamp(kp_pre * err, -max_u, max_u)
        if control_invert:
            u = -u
        a_pct = center + u/2.0 + bias
        b_pct = center - u/2.0 + bias
        a_pct = _clamp_valve(a_pct, 0.0, 100.0)
        b_pct = _clamp_valve(b_pct, 0.0, 100.0)
        dac.set_channels(a_pct, b_pct)
        if verbose:
            print(f"[pre] err={err:7.3f} angle={angle:7.3f} u={u:6.2f} A={a_pct:5.1f}% B={b_pct:5.1f}%")
        _sleep_with_poll(enc, dt_sleep)

# ---- Run modes (direct hardware) --------

def run_step(a):
    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    
    # Log encoder status
    if enc is None:
        print("[WARN] Encoder not available - running without feedback")
    else:
        print(f"[INFO] Encoder opened: chip={a.enc_chip}, pins A={a.enc_a}, B={a.enc_b}")
        # Test initial read
        try:
            enc.poll()
            initial_angle = enc.degrees(a.ppr)
            print(f"[INFO] Initial encoder reading: {initial_angle:.3f} degrees")
        except Exception as e:
            print(f"[ERROR] Initial encoder read failed: {e}")
    
    # Determine zero reference: match integrated_sensor_sin_python by default (no zeroing unless requested)
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")
        

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    pid = SimplePID(a.kp, a.ki, a.kd)
    # Reset integrator after pre-positioning to avoid initial bias
    pid.reset()
    umin, umax = _u_limits(a.min, a.max, center, 0.0)
    
    # Initialize valve states for recurrence relation approach
    valve_a_state = 50.0  # Initial valve A percentage
    valve_b_state = 50.0  # Initial valve B percentage
    
    start=time.perf_counter(); last = start
    current = a.step_q0
    t_change = a.t_change
    desired_dt = a.interval_ms / 1000.0
    try:
        while True:
            loop_start = time.perf_counter()
            t_rel = loop_start - start
            if a.total>0 and t_rel>a.total: break
            if t_rel>=t_change: current = a.step_q1
            
            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            dt_pid = max(1e-6, loop_start - last)
            last = loop_start
            
            u = pid.step(current, angle, dt_pid, umin, umax)
            if a.control_invert:
                u = -u
            
            # Recurrence relation approach: integrate control signal into valve states
            # Use moderate coefficient for good tracking with stability
            # Increased from 0.1 to reduce steady-state error
            valve_a_state += u 
            valve_b_state -= u 
            
            # Apply clamping to output values but preserve internal states
            a_pct = _clamp_valve(valve_a_state, a.min, a.max)
            b_pct = _clamp_valve(valve_b_state, a.min, a.max)
            
            # Anti-windup: only update states if they haven't saturated
            if a.min <= valve_a_state <= a.max:
                pass  # valve_a_state already updated above
            else:
                valve_a_state = a_pct  # clamp only when saturated
            
            if a.min <= valve_b_state <= a.max:
                pass  # valve_b_state already updated above
            else:
                valve_b_state = b_pct  # clamp only when saturated
            
            dac.set_channels(a_pct, b_pct)
            # CSV/print
            f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},{current:.3f},{u:.4f},{angle:.3f}\n")
            # Always print verbose output to monitor real-time behavior
            _print_verbose(t_rel, a_pct, b_pct, current, angle, u)
            
            loop_end = time.perf_counter()
            sleep_duration = desired_dt - (loop_end - loop_start)
            _sleep_with_poll(enc, sleep_duration)

    finally:
        try: dac.set_channels(0.0,0.0)
        except Exception: pass
        try: dac.close()
        except Exception: pass
        try: f.close()
        except Exception: pass
    print('CSV:', csv_path)
    # New: Automatically analyze the results
    print("\n--- Post-run Analysis ---")
    metrics = calculate_performance_metrics(csv_path, t_change=a.t_change)
    if metrics:
        gains = {'kp': a.kp, 'ki': a.ki, 'kd': a.kd}
        save_metrics(metrics, gains)
        # Add to global list for final summary
        if 'results' not in globals():
            globals()['results'] = []
        results.append(metrics)
    
    # New: Automatically plot the results
    print("\n--- Generating Plot ---")
    try:
        plot_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to generate plot: {e}")


def run_multistep(a):
    levels=a.levels
    total=a.dwell*len(levels)
    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    # Determine zero reference: match integrated_sensor_sin_python by default (no zeroing unless requested)
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")

        if a.pre_check:
            center = 0.5*(a.min + a.max)
            _run_pre_check(dac, enc, a.encoder_invert, zero_deg, a.ppr, center, a.min, a.max)

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    pid = SimplePID(a.kp, a.ki, a.kd)
    umin, umax = _u_limits(a.min, a.max, center, 0.0)
    
    start=time.perf_counter(); last = start
    desired_dt = a.interval_ms / 1000.0
    try:
        while True:
            loop_start = time.perf_counter()
            t_rel = loop_start - start
            if t_rel>=total: break
            idx=int(t_rel//a.dwell)
            if idx>=len(levels): idx=len(levels)-1
            current=levels[idx]

            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            dt_pid = max(1e-6, loop_start - last)
            last = loop_start

            u = pid.step(current, angle, dt_pid, umin, umax)
            if a.control_invert:
                u = -u
            a_pct = center + u/2.0; b_pct = center - u/2.0
            a_pct = _clamp_valve(a_pct, a.min, a.max); b_pct = _clamp_valve(b_pct, a.min, a.max)
            dac.set_channels(a_pct,b_pct)
            f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},{current:.3f},{u:.4f},{angle:.3f}\n")
            # Always print verbose output to monitor real-time behavior  
            _print_verbose(t_rel, a_pct, b_pct, current, angle, u)

            loop_end = time.perf_counter()
            sleep_duration = desired_dt - (loop_end - loop_start)
            _sleep_with_poll(enc, sleep_duration)
    finally:
        try: dac.set_channels(0.0,0.0)
        except Exception: pass
        try: dac.close()
        except Exception: pass
        try: f.close()
        except Exception: pass
    print('CSV:', csv_path)


def run_relay(a):
    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    # Determine zero reference: match integrated_sensor_sin_python by default (no zeroing unless requested)
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")

        if a.pre_check:
            center = 0.5*(a.min + a.max)
            _run_pre_check(dac, enc, a.encoder_invert, zero_deg, a.ppr, center, a.min, a.max)

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    setpoint=a.target; sign=1.0
    start=time.perf_counter()
    desired_dt = a.interval_ms / 1000.0
    try:
        while True:
            loop_start = time.perf_counter()
            t_rel = loop_start - start
            if a.total>0 and t_rel>a.total:
                break

            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            err=setpoint-angle
            if err>a.deadband:
                sign = +1.0
            elif err<-a.deadband:
                sign = -1.0
            u=sign*a.amp
            if a.control_invert:
                u = -u
            a_pct=center+u/2.0+a.bias; b_pct=center-u/2.0+a.bias
            a_pct=_clamp_valve(a_pct,a.min,a.max); b_pct=_clamp_valve(b_pct,a.min,a.max)
            dac.set_channels(a_pct,b_pct)
            f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},{setpoint:.3f},{u:.4f},{angle:.3f}\n")
            # Always print verbose output to monitor real-time behavior
            _print_verbose(t_rel, a_pct, b_pct, setpoint, angle, u)
            
            loop_end = time.perf_counter()
            sleep_duration = desired_dt - (loop_end - loop_start)
            _sleep_with_poll(enc, sleep_duration)
    finally:
        try:
            dac.set_channels(0.0,0.0)
        except Exception:
            pass
        try:
            dac.close()
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass
    print('CSV:', csv_path)

# ---- Auto-tune (relay) ----

def _compute_pid_from_ku_tu(ku: float, tu: float, rule: str) -> tuple[float,float,float]:
    if ku<=0 or tu<=0: return (0.0,0.0,0.0)
    if rule=='zn_pid':
        Kp=0.6*ku; Ti=0.5*tu; Td=0.125*tu
        Ki=Kp/Ti; Kd=Kp*Td
        return (Kp, Ki, Kd)
    if rule=='zn_pi':
        Kp=0.45*ku; Ti=0.83*tu
        Ki=Kp/Ti; Kd=Kp*Td
        return (Kp, Ki, 0.0)
    if rule=='tl_pid':  # Tyreus–Luyben (more conservative)
        Kp=0.33*ku; Ti=2.2*tu; Td=0.168*tu
        Ki=Kp/Ti; Kd=Kp*Td
        return (Kp, Ki, Kd)
    if rule=='pessen':  # Pessen Integral Rule (alternative ZN refinement)
        Kp=0.7*ku; Ti=0.4*tu; Td=0.15*tu
        Ki=Kp/Ti; Kd=Kp*Td
        return (Kp, Ki, Kd)
    return (0.0,0.0,0.0)

# ZN step-response formulas (FOPDT, tangent method)

def _compute_pid_from_step(K: float, L: float, T: float, rule: str) -> tuple[float,float,float]:
    if K<=0 or L<=0 or T<=0:
        return (0.0,0.0,0.0)
    if rule=='zn_step_p':
        Kp = T/(K*L); return (Kp, 0.0, 0.0)
    if rule=='zn_step_pi':
        # Per Ziegler–Nichols step-response table: Kp=0.9*T/(K*L), Ti=3.3*L
        Kp = 0.9*T/(K*L); Ti = 3.3*L; Ki = Kp/Ti; return (Kp, Ki, 0.0)
    # zn_step_pid
    Kp = 1.2*T/(K*L); Ti = 2.0*L; Td = 0.5*L
    Ki = Kp/Ti; Kd = Kp*Td
    return (Kp, Ki, Kd)

def run_autotune(a):
    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    # Determine zero reference
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")

        if a.pre_check:
            center = 0.5*(a.min + a.max)
            _run_pre_check(dac, enc, a.encoder_invert, zero_deg, a.ppr, center, a.min, a.max)

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    setpoint=a.target
    umin, umax = _u_limits(a.min, a.max, center, a.bias)

    if a.preposition:
        print(f"[INFO] Pre-positioning to {setpoint:.2f}...")
        _preposition_to_target(dac, enc, setpoint, center, a.bias, a.pre_max_u, a.deadband, a.interval_ms, a.pre_timeout, a.encoder_invert, zero_deg, a.ppr, a.verbose, a.control_invert)

    # Relay test with optional auto amplitude ramp
    amp = max(a.amp, a.amp_min) if a.auto_amp else a.amp
    used_amp = amp
    sign=1.0
    start=time.perf_counter()
    pts: deque[tuple[float,float]] = deque(maxlen=3)
    max_peaks: list[tuple[float,float]] = []
    min_peaks: list[tuple[float,float]] = []
    switch_times: list[float] = []
    last_switch_state = 0
    # improved auto-amp: keep ramping if switching stalls
    last_ramp_check = start
    last_switch_count = 0
    try:
        while True:
            t_rel=time.perf_counter()-start
            if a.total>0 and t_rel>a.total: break
            # auto ramp amplitude periodically; if no new switches since last check, bump amplitude
            if a.auto_amp and (time.perf_counter()-last_ramp_check) > 2.0:
                if amp < a.amp_max and len(switch_times) <= last_switch_count:
                    amp = min(a.amp_max, amp + a.amp_step)
                    used_amp = amp
                    if a.verbose:
                        print(f"[auto-amp] increasing relay amp -> {amp:.1f}")
                last_ramp_check = time.perf_counter()
                last_switch_count = len(switch_times)
            if enc is not None:
                try: enc.poll()
                except Exception: pass
            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            err=setpoint-angle
            if err>a.deadband: sign=1.0
            elif err<-a.deadband: sign=-1.0
            u=sign*amp
            if a.control_invert:
                u = -u
            a_pct=center+u/2.0+a.bias; b_pct=center-u/2.0+a.bias
            a_pct=_clamp_valve(a_pct,a.min,a.max); b_pct=_clamp_valve(b_pct,a.min,a.max)
            dac.set_channels(a_pct,b_pct)
            f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},{setpoint:.3f},{u:.4f},{angle:.3f}\n")
            # Always print verbose output to monitor real-time behavior
            _print_verbose(t_rel, a_pct, b_pct, setpoint, angle, u)
            # peaks after settle
            pts.append((t_rel, angle))
            if len(pts)==3 and t_rel>=a.settle_sec:
                (t0,y0),(t1,y1),(t2,y2)=pts
                if y1>y0 and y1>y2:
                    max_peaks.append((t1,y1))
                if y1<y0 and y1<y2:
                    min_peaks.append((t1,y1))
            # detect switch by sign flip
            curr_state = 1 if sign>0 else -1
            if curr_state != last_switch_state:
                if last_switch_state != 0:
                    if len(switch_times)==0 or (t_rel - switch_times[-1])>0.05:
                        switch_times.append(t_rel)
                last_switch_state = curr_state
            if len(max_peaks)>=a.cycles and len(min_peaks)>=a.cycles:
                break
            _sleep_with_poll(enc, a.interval_ms/1000.0)
    finally:
        try: dac.set_channels(0.0,0.0)
        except Exception: pass
        try: dac.close()
        except Exception: pass
        try: f.close()
        except Exception: pass

    # Estimate Tu from peaks or switch times
    if len(max_peaks)>=2:
        periods=[max_peaks[i+1][0]-max_peaks[i][0] for i in range(len(max_peaks)-1)]
    elif len(switch_times)>=2:
        half=[switch_times[i+1]-switch_times[i] for i in range(len(switch_times)-1)]
        periods=[2.0*sum(half)/len(half)] if half else []
    else:
        periods=[]
    amps=[]
    for i in range(min(len(max_peaks), len(min_peaks))):
        amps.append(abs(max_peaks[i][1]-min_peaks[i][1])/2.0)
    a_y = sum(amps)/len(amps) if amps else 0.0
    Tu = sum(periods)/len(periods) if periods else 0.0
    if a_y<=0 or Tu<=0:
        print('[ERROR] Failed to detect stable oscillation. Try adjusting --amp/--deadband/--settle-sec.')
        return
    d = used_amp
    Ku = 4.0*d/(math.pi*a_y)
    # Optional: correct Ku for relay deadband hysteresis
    if getattr(a, 'hysteresis_correct', False) and a.deadband>0 and a_y>0:
        ratio = a.deadband / a_y
        if ratio < 1.0:
            Ku *= math.sqrt(max(0.0, 1.0 - ratio*ratio))
    Kp,Ki,Kd = _compute_pid_from_ku_tu(Ku, Tu, a.rule)
    print('Auto-tune result:')
    print(f'  a (output amp) = {a_y:.3f}, Tu = {Tu:.3f}s, Ku = {Ku:.3f}')
    print(f'  Rule={a.rule} -> Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}')
    # Optional: save JSON
    if a.save:
        try:
            out = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'target': setpoint,
                'deadband': a.deadband,
                'relay_amp_used': used_amp,
                'osc_amp': a_y,
                'Tu': Tu,
                'Ku': Ku,
                'rule': a.rule,
                'Kp': Kp,
                'Ki': Ki,
                'Kd': Kd,
            }
            out_path = a.save
            base = os.path.dirname(out_path)
            if base:
                os.makedirs(base, exist_ok=True)
            with open(out_path, 'w') as jf:
                json.dump(out, jf, indent=2)
            print('Saved:', out_path)
        except Exception as e:
            print('[WARN] Failed to save JSON:', e)

def run_zn_limit(a):
    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    # Determine zero reference
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")

        if a.pre_check:
            center = 0.5*(a.min + a.max)
            _run_pre_check(dac, enc, a.encoder_invert, zero_deg, a.ppr, center, a.min, a.max)

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    setpoint=a.target
    k_val = a.kp_start
    last_best_k = None
    pid = SimplePID(k_val, 0.0, 0.0)
    umin, umax = _u_limits(a.min, a.max, center, 0.0)
    desired_dt = a.interval_ms / 1000.0
    Ku = None
    Tu = None

    try:
        while k_val <= a.kp_max:
            print(f"\n[INFO] Testing Kp = {k_val:.2f}")
            pts: deque[tuple[float,float]] = deque(maxlen=3)
            max_peaks: list[tuple[float,float]] = []
            min_peaks: list[tuple[float,float]] = []
            start = time.perf_counter(); last = start
            # run for dwell seconds
            while True:
                loop_start = time.perf_counter()
                t_rel = loop_start - start
                if t_rel >= a.dwell: break
                if enc is not None:
                    try: enc.poll()
                    except Exception: pass
                angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
                dt_pid = max(1e-6, loop_start - last)
                last = loop_start

                u = pid.step(setpoint, angle, dt_pid, umin, umax)
                if a.control_invert:
                    u = -u
                a_pct=center+u/2.0; b_pct=center-u/2.0
                a_pct=_clamp_valve(a_pct,a.min,a.max); b_pct=_clamp_valve(b_pct,a.min,a.max)
                dac.set_channels(a_pct,b_pct)
                f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},{setpoint:.3f},{u:.4f},{angle:.3f}\n")
                # Always print verbose output to monitor real-time behavior
                _print_verbose(t_rel,a_pct,b_pct,setpoint,angle,u)
                pts.append((t_rel, angle))
                if len(pts)==3 and t_rel>=a.settle_sec:
                    (t0,y0),(t1,y1),(t2,y2)=pts
                    if y1>y0 and y1>y2: max_peaks.append((t1,y1))
                    if y1<y0 and y1<y2: min_peaks.append((t1,y1))
                # sleep remainder of interval
                elapsed = time.perf_counter() - loop_start
                _sleep_with_poll(enc, max(0.0, desired_dt - elapsed))
                last = time.perf_counter()
            # evaluate oscillation sustainment
            peak_pairs = min(len(max_peaks), len(min_peaks))
            if peak_pairs >= 3:
                # use max/min peak values to compute amplitudes
                amps = [abs(max_peaks[i][1] - min_peaks[i][1]) / 2.0 for i in range(peak_pairs)]
                ratios = [amps[i+1]/amps[i] for i in range(len(amps)-1) if amps[i]>1e-9]
                if ratios and all(abs(r-1.0) <= a.tol for r in ratios[-3:]):
                    # sustained oscillation detected
                    Ku = k_val
                    if len(max_peaks) >= 2:
                        Ts = [max_peaks[i+1][0]-max_peaks[i][0] for i in range(len(max_peaks)-1)]
                        Tu = sum(Ts)/len(Ts)
                    elif len(min_peaks) >= 2:
                        Ts = [min_peaks[i+1][0]-min_peaks[i][0] for i in range(len(min_peaks)-1)]
                        Tu = sum(Ts)/len(Ts)
                    break
            k_val += a.kp_step
        if Ku is None or Tu is None or Tu<=0:
            print('[ERROR] Failed to find Ku/Tu. Try adjusting kp range/step or dwell/settle.')
            return
        Kp,Ki,Kd = _compute_pid_from_ku_tu(Ku, Tu, a.rule)
        print('ZN ultimate result:')
        print(f'  Ku={Ku:.3f}, Tu={Tu:.3f}s -> {a.rule}: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}')
        if a.save:
            out = {
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
                'method': 'zn_limit',
                'Ku': Ku, 'Tu': Tu, 'rule': a.rule,
                'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
            }
            base = os.path.dirname(a.save);
            (os.makedirs(base, exist_ok=True) if base else None)
            with open(a.save, 'w') as jf: json.dump(out, jf, indent=2)
            print('Saved:', a.save)
    finally:
        try: dac.set_channels(0.0,0.0)
        except Exception: pass
        try: dac.close()
        except Exception: pass
        try: f.close()
        except Exception: pass
    print('CSV:', csv_path)

# --- ZN step-response identification (open-loop) ---

def run_zn_step(a):
    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    # Determine zero reference
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")

        if a.pre_check:
            center = 0.5*(a.min + a.max)
            _run_pre_check(dac, enc, a.encoder_invert, zero_deg, a.ppr, center, a.min, a.max)

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    dt_desired = a.interval_ms/1000.0
    times: list[float] = []
    ys: list[float] = []
    try:
        # baseline
        t0 = time.perf_counter();
        while time.perf_counter() - t0 < a.pre_wait:
            if enc is not None:
                try: enc.poll()
                except Exception: pass
            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            dac.set_channels(center, center)  # no differential
            t_rel = time.perf_counter() - t0
            f.write(f"{int(t_rel*1000)},{center:.1f},{center:.1f},0.0,0.0,{angle:.3f}\n")
            times.append(t_rel); ys.append(angle)
            _sleep_with_poll(enc, dt_desired)
        # apply open-loop step (respect control polarity)
        step_start = time.perf_counter()
        u_step_eff = a.u_step if not a.control_invert else -a.u_step
        while time.perf_counter() - step_start < a.total:
            if enc is not None:
                try: enc.poll()
                except Exception: pass
            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            a_pct = _clamp_valve(center+ u_step_eff/2.0 + a.bias, a.min, a.max)
            b_pct = _clamp_valve(center- u_step_eff/2.0 + a.bias, a.min, a.max)
            dac.set_channels(a_pct, b_pct)
            t_rel = (time.perf_counter() - step_start) + a.pre_wait
            f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},0.0,{u_step_eff:.4f},{angle:.3f}\n")
            times.append(t_rel); ys.append(angle)
            _sleep_with_poll(enc, dt_desired)
    finally:
        try: dac.set_channels(0.0,0.0)
        except Exception: pass
        try: dac.close()
        except Exception: pass
        try: f.close()
        except Exception: pass
    # analyze step response
   
    if len(times) < 10:
        print('[ERROR] Not enough data for step identification.')
        return
    y0 = sum(ys[:max(1, int(a.pre_wait/dt_desired)//2)])/max(1, int(a.pre_wait/dt_desired)//2)
    y_ss = sum(ys[-max(3, int(1.0/dt_desired)):])/max(3, int(1.0/dt_desired))
    dy = [0.0]*len(ys)
    # simple moving average smoothing
    w = max(1, int(a.smooth))
    y_smooth = []
    for i in range(len(ys)):
        i0 = max(0, i-w+1); seg = ys[i0:i+1]
        y_smooth.append(sum(seg)/len(seg))
    for i in range(1, len(ys)):
        dt_i = (times[i]-times[i-1]) if times[i]>times[i-1] else dt_desired
        dy[i] = (y_smooth[i]-y_smooth[i-1]) / dt_i
    idx = max(range(len(dy)), key=lambda k: dy[k])
    R = dy[idx]
    t_inf = times[idx]
    y_inf = y_smooth[idx]
    if R <= 1e-9:
        print('[ERROR] Slope too small to identify.')
        return
    T = (y_ss - y0) / R
    L = t_inf - (y_inf - y0) / R
    Kproc = (y_ss - y0) / u_step_eff if abs(u_step_eff) > 1e-9 else 0.0
    print(f'ZN step id: K={Kproc:.5f}, L={L:.5f}s, T={T:.5f}s (R={R:.5f})')
    Kp,Ki,Kd = _compute_pid_from_step(Kproc, L, T, a.rule)
    print(f'  Rule={a.rule} -> Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}')
    if a.save:
        out = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'method': 'zn_step',
            'K': Kproc, 'L': L, 'T': T, 'R': R,
            'u_step': u_step_eff, 'rule': a.rule,
            'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
        }
        base = os.path.dirname(a.save);
        (os.makedirs(base, exist_ok=True) if base else None)
        with open(a.save, 'w') as jf: json.dump(out, jf, indent=2)
        print('Saved:', a.save)
    print('CSV:', csv_path)

def plot_csv(path: str):
    # Load CSV and generate plot
    try:
        df = pd.read_csv(path)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        if 'ms' in df.columns:
            t = df['ms'].astype(float) / 1000.0
        else:
            t = pd.Series(range(len(df)))
        if 'enc_deg' in df.columns:
            ax1.plot(t, df['enc_deg'], label='enc_deg')
        if 'q_des' in df.columns:
            ax1.plot(t, df['q_des'], label='q_des')
        ax1.set_ylabel('Angle [deg]'); ax1.legend(); ax1.grid(True)
        if 'pid_u' in df.columns:
            ax2.plot(t, df['pid_u'], label='u')
        else:
            ax2.plot(t, df.iloc[:, -3], label='u')
        ax2.set_ylabel('Control u'); ax2.set_xlabel('Time [s]'); ax2.grid(True)
        out = path.replace('.csv', '.png')
        fig.tight_layout(); fig.savefig(out, dpi=150)
        print('Saved plot:', out)
    except Exception as e:
        print('Plot failed:', e)

def compute_tracking_metrics(csv_path: str) -> PIDPerformanceMetrics | None:
    """Compute tracking metrics (RMSE, IAE, ITAE, steady error) for a trajectory CSV."""
    try:
        df = pd.read_csv(csv_path)
        if 'ms' not in df.columns or 'q_des' not in df.columns or 'enc_deg' not in df.columns:
            print(f"[WARN] Tracking metrics: CSV '{csv_path}' missing required columns.")
            return None
        t = df['ms'].to_numpy() / 1000.0
        q_des = df['q_des'].to_numpy()
        y = df['enc_deg'].to_numpy()
        error = q_des - y
        metrics = PIDPerformanceMetrics()
        metrics.rmse = compute_rmse(error)
        metrics.iae = compute_iae(error, t)
        metrics.itae = compute_itae(error, t)
        # steady-state error estimate: mean of last 10% of samples
        n = len(error)
        tail = max(1, int(n*0.1))
        metrics.steady_state_error = float(np.mean(error[-tail:]))
        metrics.csv_path = csv_path
        return metrics
    except Exception as e:
        print(f"[ERROR] Failed to compute tracking metrics for '{csv_path}': {e}")
        return None


def run_evaluate(a):
    """Run a closed-loop tracking test following a sine or random trajectory using current PID gains."""
    # allow overriding total via --t-total
    total_time = a.total if (not hasattr(a, 't_total') or a.t_total is None) else a.t_total
    if hasattr(a, 't_total') and a.t_total is not None:
        try: total_time = float(a.t_total)
        except Exception: pass

    dac = Dac8564(a.spi_bus, a.spi_dev, a.cs_gpio); dac.open()
    enc = _open_encoder(a.encoder, a.enc_chip, a.enc_a, a.enc_b)
    # zeroing
    zero_deg = 0.0
    if enc is not None:
        if a.zero_deg is not None:
            zero_deg = float(a.zero_deg)
        elif a.zero_at_start:
            print("[INFO] Preparing to zero encoder. Position arm at zero point...")
            time.sleep(1.0)
            zero_deg = _capture_zero(enc, a.encoder_invert, a.ppr)
            print(f"[INFO] Encoder zeroed. Initial offset: {zero_deg:.3f} degrees.")
        if a.pre_check:
            center = 0.5*(a.min + a.max)
            _run_pre_check(dac, enc, a.encoder_invert, zero_deg, a.ppr, center, a.min, a.max)

    f, csv_path = _csv_open(a.csv)
    center = 0.5*(a.min + a.max)
    pid = SimplePID(a.kp, a.ki, a.kd)
    umin, umax = _u_limits(a.min, a.max, center, 0.0)

    # initialize internal valve states to the center (use same smoothing as run_step)
    valve_a_state = center
    valve_b_state = center
    # smoothing gain controlling how rapidly valve states follow u (matches run_step)

    # prepare random generator if needed
    rng = np.random.RandomState(a.seed) if a.seed is not None else np.random.RandomState(int(time.time() % 2**31))
    # random step target initialization
    last_step_t = 0.0
    current_target = a.offset

    start = time.perf_counter(); last = start
    desired_dt = a.interval_ms / 1000.0
    try:
        while True:
            loop_start = time.perf_counter()
            t_rel = loop_start - start
            if total_time>0 and t_rel>total_time: break

            # determine desired q_des based on trajectory
            if a.traj == 'sin':
                q_des = a.offset + a.amp * math.sin(2.0 * math.pi * a.freq * t_rel)
            else:  # random step trajectory
                if t_rel - last_step_t >= a.dwell:
                    last_step_t = t_rel
                    current_target = a.offset + (rng.rand()*2.0 - 1.0) * a.amp
                # linear interpolation to avoid instantaneous jumps: interpolate over small transition window
                q_des = current_target

            angle = _read_angle(enc, a.encoder_invert, zero_deg, a.ppr)
            dt_pid = max(1e-6, loop_start - last)
            last = loop_start

            u = pid.step(q_des, angle, dt_pid, umin, umax)
            if a.control_invert:
                u = -u

            # --- UPDATED: apply same recurrence/state update as run_step (smoothing) ---
            valve_a_state += u 
            valve_b_state -= u 

            a_pct = _clamp_valve(valve_a_state, a.min, a.max)
            b_pct = _clamp_valve(valve_b_state, a.min, a.max)

            # Anti-windup: if internal state exceeded limits, snap it to clamped value
            if not (a.min <= valve_a_state <= a.max):
                valve_a_state = a_pct
            if not (a.min <= valve_b_state <= a.max):
                valve_b_state = b_pct

            dac.set_channels(a_pct, b_pct)
            f.write(f"{int(t_rel*1000)},{a_pct:.1f},{b_pct:.1f},{q_des:.3f},{u:.4f},{angle:.3f}\n")
            _print_verbose(t_rel, a_pct, b_pct, q_des, angle, u)

            _sleep_with_poll(enc, max(0.0, desired_dt - (time.perf_counter() - loop_start)))
    finally:
        try: dac.set_channels(0.0,0.0)
        except Exception: pass
        try: dac.close()
        except Exception: pass
        try: f.close()
        except Exception: pass

    print('CSV:', csv_path)
    print('\n--- Tracking Performance Metrics ---')
    metrics = compute_tracking_metrics(csv_path)
    if metrics:
        gains = {'kp': a.kp, 'ki': a.ki, 'kd': a.kd}
        save_metrics(metrics, gains, summary_file='pid_tune_tracking_results.csv')
        print(f"RMSE={metrics.rmse:.4f}, IAE={metrics.iae:.4f}, ITAE={metrics.itae:.4f}, steady_err={metrics.steady_state_error:.4f}")
    # plot
    try:
        plot_csv(csv_path)
    except Exception as e:
        print(f"[ERROR] Failed to generate plot: {e}")
    
    # New: Print summary if any results were collected
    if 'results' in globals() and results:
        print("\n--- Run Summary ---")
        for r in results:
            print(f"Kp={r.kp:<5} Ki={r.ki:<5} Kd={r.kd:<5} -> RMSE={r.rmse:.3f}, Overshoot={r.overshoot_percent:.2f}%, Rise={r.rise_time_10_90:.3f}s, Settle={r.settling_time_2_percent:.3f}s")
    
    return 0

def main():
    ap = build_parser()
    args = ap.parse_args()

    # Allow runtime override of safety cap
    global VALVE_MAX_PCT
    VALVE_MAX_PCT = getattr(args, 'valve_max_pct', VALVE_MAX_PCT)

    mode = getattr(args, 'mode', None)
    if mode == 'step':
        return run_step(args)
    if mode == 'multistep':
        return run_multistep(args)
    if mode == 'relay':
        return run_relay(args)
    if mode == 'autotune':
        return run_autotune(args)
    if mode == 'zn_limit':
        return run_zn_limit(args)
    if mode == 'zn_step':
        return run_zn_step(args)
    if mode == 'analyze':
        analyze_csv(args.path)
        return 0
    if mode == 'plot':
        plot_csv(args.path)
        return 0
    if mode == 'evaluate':
        return run_evaluate(args)

    print('[ERROR] Unknown mode:', mode)
    return 2

if __name__ == '__main__':
    raise SystemExit(main())
