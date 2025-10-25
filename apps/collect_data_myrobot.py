#!/usr/bin/env python3
"""
Minimal data collection tailored for your robot.
Uses communication pieces from `integrated_sensor_sin_python.py` (DAC/ADC/Encoder/LDC)
and a RandomTrajectory generator from `affetto_nn_ctrl.control_utility` to produce
reference updates. Produces CSV logs compatible with affetto dataset conventions.

Run example: 
uv run python apps/collect_data_myrobot.py --traj trapezoidal --speed slow -T 10 -n 1 -o data/myrobot
"""


from __future__ import annotations

# If a project virtualenv '.venv-fix' exists, re-exec this script with that python to ensure
# plotting and other dependencies installed into the venv are available at runtime.
try:
    import os, sys, pathlib
    _VENV_PY = pathlib.Path(__file__).resolve().parents[1] / '.venv-fix' / 'bin' / 'python'
    if _VENV_PY.exists():
        try:
            _curr = pathlib.Path(sys.executable).resolve()
            if _curr != _VENV_PY.resolve():
                # Replace current process with the venv python running the same script
                os.execv(str(_VENV_PY), [str(_VENV_PY), str(pathlib.Path(__file__).resolve())] + sys.argv[1:])
        except Exception:
            pass
except Exception:
    pass

import argparse
import math
import os
import sys
import time
import pathlib
from datetime import datetime
import numpy as np
from typing import Protocol, runtime_checkable, Any, cast, Callable
import traceback
import csv
import subprocess

# make local src importable (affetto modules)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
# Ensure project root is on sys.path so top-level package `apps` is importable
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Try to make a colocated affetto project importable (affetto-nn-ctrl, affetto_nn_ctrl, affetto)
# This lets the script directly use affetto's RandomTrajectory/PTP implementations if the user
# placed the affetto project next to this repo (as described by the user).
_affetto_dir_candidates = ['affetto-nn-ctrl', 'affetto_nn_ctrl', 'affetto']
for _d in _affetto_dir_candidates:
    _p = _PROJECT_ROOT / _d
    try:
        if _p.exists():
            _p_str = str(_p)
            if _p_str not in sys.path:
                sys.path.insert(0, _p_str)
            _p_src = _p / 'src'
            if _p_src.exists():
                _p_src_str = str(_p_src)
                if _p_src_str not in sys.path:
                    sys.path.insert(0, _p_src_str)
            break
    except Exception:
        # best-effort: ignore any filesystem issues and continue
        pass

# import refactored helpers from myrobot_lib
from apps.myrobot_lib import create_controller, make_random_trajectory, plot_csv as lib_plot_csv  # type: ignore
from apps.myrobot_lib.hardware import open_devices, close_devices  # type: ignore
from apps.myrobot_lib.logger import DataLogger, make_header, TerminalPrinter  # type: ignore
from apps.myrobot_lib import config as cfg
# import adapter to wrap affetto RandomTrajectory when forcing affetto
try:
    from apps.myrobot_lib.trajectory import AffettoRTAdapter  # type: ignore
except Exception:
    AffettoRTAdapter = None

# I2C / LDC constants (match integrated_sensor_sin_python)
I2C_BUS = 1
LDC_ADDRS = [0x1A, 0x1B, 0x2B, 0x2A, 0x24, 0x14, 0x15]
TENSION_POLL_INTERVAL = 0.2

try:
    from affetto_nn_ctrl.control_utility import RandomTrajectory
except Exception:  # pragma: no cover - fall back
    RandomTrajectory = None

# reuse hardware helpers from integrated_sensor_sin_python if available
try:
    from apps.myrobot_lib.dac import Dac8564  # type: ignore
except Exception:
    Dac8564 = None

try:
    from apps.myrobot_lib.adc import Max11632  # type: ignore
except Exception:
    Max11632 = None

try:
    from apps.myrobot_lib.encoder import EncoderSimple  # type: ignore
except Exception:
    EncoderSimple = None

try:
    from apps.myrobot_lib.ldc import LDC1614  # type: ignore
except Exception:
    LDC1614 = None

RUN = True

def create_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def clamp_pct(p: float) -> float:
    if p < 0.0:
        return 0.0
    if p > 95.0:
        return 95.0
    return p


# Local SimplePID/RobotController implementations have been moved to `apps.myrobot_lib.controller`.
# Use `create_controller` imported from that package instead of the in-file definitions.


def _init_csv(base_dir: str, filename_hint: str | None, ldc_addrs: list[int], has_enc: bool, has_adc: bool):
    os.makedirs(base_dir, exist_ok=True)
    ts = create_timestamp()
    filename = filename_hint or f'myrobot_motion_{ts}.csv'
    # Ensure base_dir and filename are converted to native path strings to avoid
    # mixing bytes/PathLike types (static type-checkers can complain otherwise).
    base_dir_str = os.fspath(base_dir)
    if os.path.isabs(filename):
        csv_path = filename
    else:
        csv_path = os.path.join(base_dir_str, filename)
    csv_path = str(csv_path)
    # Use larger buffer and csv.writer to minimize per-row Python-side formatting cost
    f = open(csv_path, 'w', buffering=8192, newline='')
    csv_writer = csv.writer(f)
    # Write header in exact order requested by user
    header = ['ms','ca','cb','caper','cbper','q','qdes','dq','dqdes','pa','pb','Ta','Tb','pid_u']
    csv_writer.writerow(header)
    return f, csv_path, header, csv_writer


def build_valve_from_angle(q_des: float, q_min: float, q_max: float, center: float = 50.0, span_pct: float = 40.0):
    """
    Deprecated simple mapping kept for compatibility. Prefer using RobotController (create_controller) which
    implements PID-based angle control and maps PID output to valve percentages.
    """
    mid = (q_min + q_max) / 2.0
    if q_max == q_min:
        rel = 0.0
    else:
        rel = (q_des - mid) / (q_max - q_min)
    a = center + rel * span_pct
    b = center - rel * span_pct
    return clamp_pct(a), clamp_pct(b)


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
    """Read current encoder angle, apply invert and zero offset."""
    a = 0.0
    if enc is not None:
        try:
            enc.poll()
            a = enc.degrees(ppr)
        except Exception:
            a = 0.0
    if invert:
        a = -a
    return a - (zero_deg if zero_deg is not None else 0.0)


def _clamp_valve(x: float, lo: float, hi: float) -> float:
    """Clamp valve percentage between lo and hi."""
    if x is None:
        return lo
    return lo if x < lo else hi if x > hi else x

def run_once(
    duration: float,
    rt: TrajectoryLike,
    dac: DACLike | None,
    adc: ADCLike | None,
    enc: EncoderLike | None,
    ldc_sensors: list[LDCLike] | None,
    base_dir: str,
    filename: str | None,
    q_range: tuple[float, float],
    controller: ControllerLike | None = None,
    loop_interval_ms: int = 50,
    verbose: int = 0,
    enc_invert: bool = True,
    enc_zero_deg: float | None = None,
    enc_ppr: int = 2048,
    enc_zero_at_start: bool = False,
    # new valve parameters
    center: float = 50.0,
    min_valve: float = 20.0,
    max_valve: float = 100.0,
    incremental_valve: bool = True,
    incr_gain: float = 1.0,
    incr_leak: float = 0.0,
) -> str:
    # update CSV to include q_des and pid_u; get csv_writer to reduce per-loop formatting
    csv_f, csv_path, header, csv_writer = _init_csv(base_dir, filename, [s.addr for s in (ldc_sensors or [])] if ldc_sensors else [], enc is not None, adc is not None)
    # Start a background CSV writer thread to avoid blocking the real-time loop on disk I/O
    try:
        import queue as _queue
        import threading as _threading
        import csv as _csv
    except Exception:
        _queue = None
        _threading = None
        _csv = None
    if _queue is not None and _threading is not None:
        # Bounded queue prevents unbounded memory growth if disk I/O stalls.
        # Real-time loop will never block on enqueue; dropped rows are counted.
        row_queue = _queue.Queue(maxsize=1000)
        row_drops = 0
        def _csv_writer_thread(q, f):
            writer = _csv.writer(f)
            last_flush = time.time()
            cnt = 0
            while True:
                try:
                    item = q.get(timeout=0.5)
                except Exception:
                    # Periodic idle flush to ensure data reaches disk even when queue is empty.
                    if time.time() - last_flush > 1.0:
                        try:
                            f.flush()
                        except Exception:
                            pass
                        last_flush = time.time()
                    continue
                # Sentinel to request thread exit
                if item is None:
                    # Drain remaining items quickly
                    while True:
                        try:
                            rem = q.get_nowait()
                        except _queue.Empty:
                            break
                        if rem is None:
                            break
                        try:
                            writer.writerow(rem)
                        except Exception:
                            pass
                    try:
                        f.flush()
                    except Exception:
                        pass
                    break
                # Normal row write
                try:
                    writer.writerow(item)
                except Exception:
                    pass
                cnt += 1
                # Flush periodically (either count-based or time-based)
                if cnt >= 50 or time.time() - last_flush > 1.0:
                    try:
                        f.flush()
                    except Exception:
                        pass
                    cnt = 0
                    last_flush = time.time()
        csv_thread = _threading.Thread(target=_csv_writer_thread, args=(row_queue, csv_f), daemon=True)
        csv_thread.start()
    else:
        row_queue = None
    # If requested, capture encoder zero offset at start
    if enc is not None and enc_zero_at_start:
        try:
            enc_zero_deg = _capture_zero(enc, enc_invert, enc_ppr)
            print(f'[INFO] Captured encoder zero: {enc_zero_deg:.3f} deg', flush=True)
        except Exception:
            enc_zero_deg = 0.0
    if enc_zero_deg is None:
        enc_zero_deg = 0.0
    # initialize valve state for incremental update mode
    valve_a_state = float(center)
    valve_b_state = float(center)
    # If csv header doesn't include q_des/pid_u, append them to the first line (best-effort)
    # Reopen header is out of scope here; we'll include q_des and pid_u in rows consistently.
    t0 = time.perf_counter()
    interval_s = loop_interval_ms / 1000.0
    next_tick = t0
    last_adc = None
    last_tension_values: list[float | None] = [None] * (len(ldc_sensors) if ldc_sensors else 0)
    last_time = t0
    last_enc_deg = None
    # tracked/measured velocity (low-pass filtered)
    last_dq_meas = 0.0
    # encoder failure tracking: attempt automatic reinitialization after N failures
    enc_poll_failures = 0
    MAX_ENC_POLL_FAILURES = 3
    # If threshold exceeded, abort the run instead of reinitializing devices mid-run
    abort_run = False

    # Watchdog/計測: 各フェーズの処理時間を測定し、遅延時に内訳を警告として出力
    SLOW_DT_THRESH = 0.20   # 1ループ合計がこの秒数を超えたら警告
    SLOW_PHASE_THRESH = 0.05 # 個別フェーズがこの秒数を超えたら要注意

    def _reinit_encoder() -> None:
        nonlocal enc, last_enc_deg, enc_poll_failures
        # NOTE: During a run we should not attempt to recreate the encoder in-place.
        # If poll failures exceed the configured threshold, abort the run explicitly
        # so the outer control flow can handle device reopen/cleanup between runs.
        raise RuntimeError('Encoder poll failures exceeded threshold — aborting run')
    try:
        loop_counter = 0
        while True:
            now = time.perf_counter()
            if now < next_tick:
                if enc is not None:
                    try:
                        enc.poll()
                        enc_poll_failures = 0
                    except Exception:
                        enc_poll_failures += 1
                        if enc_poll_failures >= MAX_ENC_POLL_FAILURES:
                            try:
                                _reinit_encoder()
                            except RuntimeError as e:
                                try:
                                    print(f'[ERROR] {e}', flush=True)
                                except Exception:
                                    pass
                                abort_run = True
                                break
                time.sleep(0.0001)
                continue
            # ループ先頭で計測開始
            iter_start = time.perf_counter()

            t = now - t0
            if duration > 0 and t > duration:
                break
            # cadence
            while now - next_tick > interval_s:
                next_tick += interval_s
            next_tick += interval_s
            loop_counter += 1

            # --- 計測用の局所変数を初期化 ---
            enc_time = traj_time = ctrl_time = dac_time = adc_time = ldc_time = csv_time = 0.0

            # sample encoder early for controller
            # preserve previous encoder reading so we can compute measured velocity
            prev_enc_deg = last_enc_deg
            enc_deg = None
            if enc is not None:
                _t0 = time.perf_counter()
                try:
                    # perform poll/read here and handle failures explicitly
                    enc.poll()
                    raw_a = enc.degrees(enc_ppr)
                    if enc_invert:
                        raw_a = -raw_a
                    enc_deg = raw_a - (enc_zero_deg if enc_zero_deg is not None else 0.0)
                    last_enc_deg = enc_deg
                    enc_poll_failures = 0
                except Exception:
                    enc_poll_failures += 1
                    if enc_poll_failures >= MAX_ENC_POLL_FAILURES:
                        try:
                            _reinit_encoder()
                        except RuntimeError as e:
                            try:
                                print(f'[ERROR] {e}', flush=True)
                            except Exception:
                                pass
                            abort_run = True
                            break
                    enc_deg = last_enc_deg
                finally:
                    enc_time += (time.perf_counter() - _t0)
            # desired
            _t0 = time.perf_counter()
            qdes_func = rt.get_qdes_func()
            qvec = qdes_func(t)
            traj_time += (time.perf_counter() - _t0)
            # Support both [q] and [q, dq] formats
            if qvec is None or len(qvec) == 0:
                q_des = 0.0
                dq_des = 0.0
            else:
                try:
                    q_des = float(qvec[0])
                except Exception:
                    q_des = float(qvec)
                try:
                    dq_des = float(qvec[1])
                except Exception:
                    dq_des = 0.0
            # Ensure trajectory generator cannot command targets outside the configured q_range
            try:
                q_min, q_max = q_range
                if q_des < q_min:
                    q_des = float(q_min)
                elif q_des > q_max:
                    q_des = float(q_max)
            except Exception:
                pass
            # compute dt
            dt = now - last_time if last_time is not None else interval_s
            last_time = now
            # compute measured velocity from encoder (deg/s) with simple low-pass filtering
            dq_meas = 0.0
            try:
                if enc is not None and prev_enc_deg is not None and enc_deg is not None and dt and dt > 1e-9:
                    raw_dq = (float(enc_deg) - float(prev_enc_deg)) / float(dt)
                    # simple exponential smoothing to suppress jitter
                    alpha = 1.0
                    last_dq_meas = alpha * raw_dq + (1.0 - alpha) * last_dq_meas
                    dq_meas = last_dq_meas
                else:
                    dq_meas = last_dq_meas
            except Exception:
                dq_meas = last_dq_meas
            pid_u = 0.0
            # PID component placeholders
            pid_p = 0.0
            pid_i = 0.0
            pid_d = 0.0
            if controller is not None:
                _t0 = time.perf_counter()
                try:
                    # Pass dq_meas so the controller can use affetto-style D-term: kD*(dq_des - dq_meas)
                    a_pct, b_pct, pid_u = controller.update(q_des, enc_deg, dt, dq_des=dq_des, dq_meas=dq_meas)
                except Exception as e:
                    print(f'[ERROR] controller.update failed: {e}', flush=True)
                    traceback.print_exc()
                    a_pct, b_pct = build_valve_from_angle(q_des, q_range[0], q_range[1])
                    pid_u = 0.0
                else:
                    # extract PID component values if controller exposes them
                    try:
                        pid_obj = getattr(controller, 'pid', None)
                        if pid_obj is not None:
                            pid_p = float(getattr(pid_obj, 'last_p', 0.0))
                            pid_i = float(getattr(pid_obj, 'last_i', 0.0))
                            pid_d = float(getattr(pid_obj, 'last_d', 0.0))
                    except Exception:
                        pid_p = pid_i = pid_d = 0.0
                    if incremental_valve:
                        # Maintain a symmetric offset around the valve center instead of
                        # decaying absolute valve percentages. This keeps the center
                        # (e.g. 50) as the neutral command and applies pid_u as a
                        # signed offset: a = center + offset, b = center - offset.
                        # - valve_offset starts at 0.0
                        # - incr_leak decays the offset toward 0
                        # - incr_gain scales pid_u contribution; integrate over dt
                        try:
                            if not isinstance(valve_a_state, float) or not isinstance(valve_b_state, float):
                                valve_offset = 0.0
                            else:
                                valve_offset = (valve_a_state - float(center))
                        except Exception:
                            valve_offset = 0.0
                        try:
                            if incr_leak and float(incr_leak) > 0.0 and dt is not None:
                                decay = max(0.0, 1.0 - float(incr_leak) * dt)
                                valve_offset *= decay
                        except Exception:
                            pass
                        try:
                            valve_offset += float(incr_gain) * float(pid_u) 
                        except Exception:
                            valve_offset += float(incr_gain) * float(pid_u)
                        a_pct = float(center) + valve_offset
                        b_pct = float(center) - valve_offset
                        a_clamped = min(max(a_pct, min_valve), max_valve)
                        b_clamped = min(max(b_pct, min_valve), max_valve)
                        valve_offset = (a_clamped - float(center))
                        a_pct = a_clamped
                        b_pct = b_clamped
                        valve_a_state = a_pct
                        valve_b_state = b_pct
                finally:
                    ctrl_time += (time.perf_counter() - _t0)
            else:
                a_pct, b_pct = build_valve_from_angle(q_des, q_range[0], q_range[1])
            # ensure valves commanded (real hardware required — stop on error)
            if dac is None:
                raise RuntimeError('DAC not initialized — cannot command valves')
            # let exceptions propagate so failures stop the run and are visible
            _t0 = time.perf_counter()
            dac.set_channels(a_pct, b_pct)
            dac_time += (time.perf_counter() - _t0)
            # ADC read once per loop if available
            if adc is not None:
                _t0 = time.perf_counter()
                try:
                    last_adc = adc.read_pair()
                except Exception:
                    last_adc = None
                finally:
                    adc_time += (time.perf_counter() - _t0)
            # tension sensors
            if ldc_sensors:
                _t0 = time.perf_counter()
                for i, s in enumerate(ldc_sensors):
                    try:
                        last_tension_values[i] = s.read_ch0_induct_uH()
                    except Exception:
                        last_tension_values[i] = None
                ldc_time += (time.perf_counter() - _t0)
            # write
            elapsed_ms = int(t * 1000)
            # Store raw DAC output voltage in `ca`/`cb` (user requested). Keep `caper`/`cbper` as percent.
            # Assume DAC full-scale is 5.0 V.
            try:
                ca_volt = float(a_pct) * 5.0 / 100.0
            except Exception:
                ca_volt = 0.0
            try:
                cb_volt = float(b_pct) * 5.0 / 100.0
            except Exception:
                cb_volt = 0.0
            # ADC values
            pa_val = ''
            pb_val = ''
            if adc is not None and last_adc is not None:
                try:
                    r0, v0, k0, r1, v1, k1 = last_adc
                    pa_val = k0
                    pb_val = k1
                except Exception:
                    pa_val = ''
                    pb_val = ''
            # LDC tension values
            Ta_val = ''
            Tb_val = ''
            try:
                if ldc_sensors:
                    for i, s in enumerate(ldc_sensors):
                        addr = getattr(s, 'addr', None)
                        val = last_tension_values[i] if i < len(last_tension_values) else None
                        if addr == 0x2A:
                            Ta_val = val if val is not None else ''
                        elif addr == 0x2B:
                            Tb_val = val if val is not None else ''
            except Exception:
                Ta_val = Tb_val = ''

            # Build row using raw numeric values where possible to avoid expensive per-loop formatting
            _t0 = time.perf_counter()
            row = [elapsed_ms, ca_volt, cb_volt, a_pct, b_pct, ('' if enc_deg is None else enc_deg), q_des, dq_meas, dq_des, pa_val, pb_val, Ta_val, Tb_val, pid_u]
            try:
                # enqueue row for background writer when available to avoid blocking
                if row_queue is not None:
                    try:
                        # never block the real-time loop; drop row when queue is full
                        row_queue.put_nowait(row)
                    except Exception:
                        # queue.Full or other put failure -> count dropped rows
                        try:
                            row_drops += 1
                        except Exception:
                            pass
                        try:
                            # Log infrequently to avoid spamming
                            if row_drops % 100 == 0:
                                print(f'[WARN] CSV row dropped: {row_drops}', flush=True)
                        except Exception:
                            pass
                else:
                    csv_writer.writerow(row)
            except Exception:
                # fallback to string join if writer fails for some reason
                csv_f.write(','.join([str(x) for x in row]) + '\n')
            finally:
                csv_time += (time.perf_counter() - _t0)

            # Terminal logging only when enabled (default: off). Keep minimal formatting.
            if verbose:
                try:
                    if loop_counter == 1:
                        print(','.join(header), flush=True)
                    print(','.join([str(x) for x in row]), flush=True)
                except Exception:
                    pass

            # ループ末尾で合計時間を評価し、遅延があれば内訳を警告出力
            iter_elapsed = time.perf_counter() - iter_start
            if (iter_elapsed > SLOW_DT_THRESH) or (
                enc_time > SLOW_PHASE_THRESH or traj_time > SLOW_PHASE_THRESH or ctrl_time > SLOW_PHASE_THRESH or
                dac_time > SLOW_PHASE_THRESH or adc_time > SLOW_PHASE_THRESH or ldc_time > SLOW_PHASE_THRESH or csv_time > SLOW_PHASE_THRESH
            ):
                try:
                    print(
                        '[WARN] slow loop: ' +
                        f'dt={iter_elapsed*1000:.1f}ms ' +
                        f'enc={enc_time*1000:.1f} adc={adc_time*1000:.1f} ldc={ldc_time*1000:.1f} ' +
                        f'dac={dac_time*1000:.1f} ctrl={ctrl_time*1000:.1f} traj={traj_time*1000:.1f} csv={csv_time*1000:.1f}',
                        flush=True
                    )
                except Exception:
                    pass
    finally:
        # Do not close hardware here — allow main() to reuse/open the same devices across repeats.
        try:
            if dac is not None:
                dac.set_channels(0.0, 0.0)
        except Exception:
            pass
        # If background CSV writer is running, signal it to exit and wait briefly
        try:
            if row_queue is not None:
                try:
                    # attempt to enqueue sentinel; wait briefly so writer can finish flushing
                    try:
                        row_queue.put(None, timeout=1.0)
                    except Exception:
                        try:
                            # best-effort blocking put as fallback
                            row_queue.put(None)
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    csv_thread.join(timeout=5.0)
                except Exception:
                    pass
        except Exception:
            pass
        # Only close the CSV file for this run
        if csv_f is not None:
            try:
                csv_f.close()
            except Exception:
                pass
            if csv_path:
                try:
                    print(f'[INFO] CSV saved: {csv_path}', flush=True)
                except Exception:
                    pass
    return csv_path


# replace local plot_csv implementation with call to library
def plot_csv(csv_path: str) -> str:
    return lib_plot_csv(csv_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Collect random motion data for MyRobot')
    parser.add_argument('-T', '--duration', default=10.0, type=float)
    # Allow t-range to be omitted so it can be inferred from --speed presets
    parser.add_argument('-t', '--t-range', nargs=2, type=float, default=None,
                        help='Optional update time range (two floats). If omitted, inferred from --speed')
    # Default random angle generation range changed to -10..30 degrees
    parser.add_argument('-q', '--q-range', nargs=2, type=float, default=(-20, 30))
    parser.add_argument('-Q', '--q-limit', nargs=2, type=float, default=cfg.DEFAULT_Q_LIMIT)
    parser.add_argument('-p', '--profile', default='trapezoidal')
    parser.add_argument('-n', '--n-repeat', default=3, type=int)
    parser.add_argument('-s', '--seed', default=None, type=int, help='Random seed (omit for non-deterministic runs)')
    parser.add_argument('-o', '--output', default='data/myrobot', help='output directory')
    parser.add_argument('--output-prefix', default='myrobot_motion', help='file prefix')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Enable terminal per-loop logging (default: off)')
    parser.add_argument('--interval-ms', type=float, default=cfg.DEFAULT_LOOP_INTERVAL_MS, help='Loop interval in ms (default 33.333333 -> 30Hz)')
    # PID and valve mapping parameters
    # Updated default PID gains
    parser.add_argument('--kp', type=float, default=0.15, help='PID Kp')
    parser.add_argument('--ki', type=float, default=0.25, help='PID Ki')
    parser.add_argument('--kd', type=float, default=0.01, help='PID Kd')
    parser.add_argument('--center', type=float, default=60.0, help='Valve center percent')
    parser.add_argument('--span', type=float, default=cfg.VALVE_SPAN, help='Valve span percent used for mapping')
    parser.add_argument('--min-valve', type=float, default=cfg.VALVE_MIN, help='Minimum valve percent')
    parser.add_argument('--max-valve', type=float, default=cfg.VALVE_MAX, help='Maximum valve percent')
    # Encoder options (from pid_tune.py): ppr and zeroing
    parser.add_argument('--ppr', type=int, default=cfg.ENCODER_PPR, help='Encoder PPR per channel (default 2048)')
    parser.add_argument('--zero-at-start', dest='zero_at_start', action='store_true', default=True, help='Capture current angle as zero at start (default: on)')
    parser.add_argument('--no-zero-at-start', dest='zero_at_start', action='store_false', help='Do not capture current angle as zero at start')
    parser.add_argument('--zero-deg', type=float, default=None, help='Explicit zero offset in degrees (applied after invert)')
    parser.add_argument('--encoder-invert', dest='encoder_invert', action='store_true', default=True, help='Invert encoder sign (default: on)')
    parser.add_argument('--no-encoder-invert', dest='encoder_invert', action='store_false', help='Do not invert encoder sign')
    # trajectory options (pid_tune evaluate compatibility)
    parser.add_argument('--traj', choices=['random','sin','step','trapezoidal'], default='random', help='Trajectory type (random matches pid_tune evaluate)')
    parser.add_argument('--amp', type=float, default=None, help='Amplitude for random trajectory (matches pid_tune --amp)')
    parser.add_argument('--offset', type=float, default=None, help='Offset for random trajectory (matches pid_tune --offset)')
    parser.add_argument('--dwell', type=float, default=None, help='Dwell time for random trajectory (matches pid_tune --dwell)')
    parser.add_argument('--update-q-delta', nargs=2, type=float, default=None,
                        help='Delta magnitude range in degrees (min max, non-negative). Sign is chosen internally. Example: 0 15')
    # new arguments for incremental valve update
    parser.add_argument('--incremental-valve', dest='incremental_valve', action='store_true', default=True, help='Enable incremental valve updates (default: on)')
    parser.add_argument('--no-incremental-valve', dest='incremental_valve', action='store_false', help='Disable incremental valve updates')
    parser.add_argument('--incr-gain', type=float, default=1.0, help='Incremental update gain (scales PID output)')
    parser.add_argument('--incr-leak', type=float, default=0.0, help='Incremental accumulator leak rate per second (0 disables)')
    # Speed presets to quickly choose fast/middle/slow collection cadence similar to affetto
    parser.add_argument('--speed', choices=['fast', 'middle', 'slow'], default='middle',
                        help='Preset cadence: fast/middle/slow. Used to infer -t/--dwell if not provided')
    parser.add_argument('--force-affetto', dest='force_affetto', action='store_true', default=False, help='Force use of colocated affetto RandomTrajectory if available')

    # Batch automation options: run full matrix {step,trapezoidal} x {fast,middle,slow}
    parser.add_argument('--batch', action='store_true', default=False, help='Run full batch of predefined profile/speed combinations')
    parser.add_argument('--batch-runs', type=int, default=100, help='Number of repeats per combination when --batch is used')
    parser.add_argument('--batch-duration', type=float, default=100.0, help='Duration (s) per run when --batch is used')
    parser.add_argument('--batch-quiet', action='store_true', default=False, help='Suppress child process output during batch')
    return parser.parse_args()


def main():
    args = parse_args()

    # Batch orchestration: invoke this script repeatedly for the full matrix of
    # profiles and speeds. Child invocations reuse the same script and thus
    # perform encoder zeroing per run. When --batch-quiet is set child stdout/stderr
    # are suppressed to reduce terminal I/O load.
    try:
        if getattr(args, 'batch', False):
            import subprocess, sys
            profiles = ['step', 'trapezoidal']
            speeds = ['fast', 'middle', 'slow']
            script = os.path.abspath(__file__)
            for profile in profiles:
                for speed in speeds:
                    print(f"[INFO] Batch start: profile={profile} speed={speed} runs={args.batch_runs} duration={args.batch_duration}s", flush=True)
                    cmd = [sys.executable, script,
                           '--traj', profile,
                           '--speed', speed,
                           '-T', str(float(args.batch_duration)),
                           '-n', str(int(args.batch_runs)),
                           '-o', args.output,
                           '--output-prefix', args.output_prefix,
                           # ensure encoder zeroing remains enabled in children
                           '--zero-at-start']
                    # propagate a small set of commonly adjusted options
                    for opt_name, flag in (('min_valve', '--min-valve'), ('max_valve', '--max-valve'), ('center', '--center'), ('span', '--span'), ('kp', '--kp'), ('ki', '--ki'), ('kd', '--kd'), ('ppr', '--ppr')):
                        v = getattr(args, opt_name, None)
                        if v is not None:
                            cmd += [flag, str(v)]
                    try:
                        if args.batch_quiet:
                            ret = subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            ret = subprocess.call(cmd)
                        print(f"[INFO] Batch done: profile={profile} speed={speed} exit={ret}", flush=True)
                    except Exception as e:
                        print(f"[ERROR] Batch failed for profile={profile} speed={speed}: {e!r}", flush=True)
            print('[INFO] Batch complete', flush=True)
            return 0
    except Exception:
        # if batch orchestration fails, continue to normal single-run behavior
        pass

    # If user asked for a 'step' trajectory, prefer step profile unless explicitly overridden
    try:
        if getattr(args, 'traj', None) == 'step':
            args.profile = 'step'
    except Exception:
        pass
    base_dir = args.output
    duration = args.duration

    # Preset time ranges for quick speed selection (seconds) — align with affetto presets
    SPEED_PRESETS = {
        'fast': (0.1, 1.0),
        'middle': (1.0, 2.0),
        'slow': (2.0, 4.0),
    }

    # instantiate hardware using myrobot_lib helper
    try:
        dac, adc, enc, ldc_sensors = open_devices(i2c_bus=I2C_BUS, ldc_addrs=LDC_ADDRS)
        print('[INFO] DAC opened', flush=True)
    except Exception as e:
        print(f'[ERROR] {e}', flush=True)
        return 1

    if adc is not None:
        print('[INFO] ADC opened', flush=True)
    if enc is not None:
        print('[INFO] Encoder enabled', flush=True)
    if ldc_sensors:
        print(f'[INFO] LDC sensors: {len(ldc_sensors)}', flush=True)

    # Determine encoder zero and starting angle to build q0 consistent with runtime
    enc_zero_deg_main = args.zero_deg
    try:
        if enc is not None:
            if args.zero_at_start:
                enc_zero_deg_main = _capture_zero(enc, args.encoder_invert, args.ppr)
                try:
                    print(f'[INFO] Captured encoder zero (main): {enc_zero_deg_main:.3f} deg', flush=True)
                except Exception:
                    pass
            if enc_zero_deg_main is None:
                enc_zero_deg_main = 0.0
            start_angle = _read_angle(enc, args.encoder_invert, enc_zero_deg_main, args.ppr)
        else:
            start_angle = 0.0
    except Exception:
        if enc_zero_deg_main is None:
            enc_zero_deg_main = 0.0
        start_angle = 0.0

    # create controller wrapper using PID gains from args
    controller = cast(ControllerLike, create_controller(dac, enc, kp=args.kp, ki=args.ki, kd=args.kd, center=args.center, span_pct=args.span, min_pct=args.min_valve, max_pct=args.max_valve))

    # Prepare logger
    header = make_header(adc is not None, [getattr(s, 'addr', 0) for s in ldc_sensors], enc is not None)
    logger = DataLogger(base_dir, args.output_prefix, header)
    tprinter = TerminalPrinter(args.verbose)

    # Print connected sensors summary similar to integrated_sensor_sin_python.py
    try:
        ldc_addrs_str = ','.join([f'0x{getattr(s, "addr", 0):02X}' for s in ldc_sensors]) if ldc_sensors else 'none'
        print(f"[INFO] Sensors: ADC={'yes' if adc is not None else 'no'} LDC={len(ldc_sensors)}({ldc_addrs_str}) Encoder={'yes' if enc is not None else 'no'}", flush=True)
    except Exception:
        pass

    # Prepare RandomTrajectory (single DOF)
    # Use the refactored factory from apps.myrobot_lib which returns an object with get_qdes_func() and reset_updater().
    q0 = np.array([start_angle], dtype=float)
    active_joints = [0]
    # If user provided -t/--t-range use it, otherwise infer from --speed preset
    if args.t_range is None:
        update_t_range = SPEED_PRESETS.get(args.speed, SPEED_PRESETS['middle'])
    else:
        update_t_range = (float(args.t_range[0]), float(args.t_range[1]))
    update_q_range = (float(args.q_range[0]), float(args.q_range[1]))
    update_q_limit = (float(args.q_limit[0]), float(args.q_limit[1]))

    # Ensure starting q0 lies inside the absolute limits used by the generator
    q0_clamped_val = min(max(float(start_angle), update_q_limit[0]), update_q_limit[1])
    if abs(q0_clamped_val - float(start_angle)) > 1e-6:
        try:
            print(f"[INFO] q0 clamped from {start_angle:.3f} to {q0_clamped_val:.3f} within q_limit {update_q_limit}", flush=True)
        except Exception:
            pass
    q0 = np.array([q0_clamped_val], dtype=float)
    
    try:
        # Use affetto's RandomTrajectory for random/step/trapezoidal profiles so
        # generation matches affetto exactly. Allow the user to override the
        # update_q_range (delta magnitude) via --update-q-delta or compute it
        # from limits around the current q0.
        if args.traj in ('random', 'step', 'trapezoidal'):
            # Determine absolute low/high for generation based on amp/offset or q_range
            qmin, qmax = float(args.q_range[0]), float(args.q_range[1])
            inferred_amp = (qmax - qmin) / 2.0
            inferred_offset = (qmax + qmin) / 2.0
            if args.amp is None and args.offset is None:
                low = qmin
                high = qmax
            else:
                amp = args.amp if args.amp is not None else inferred_amp
                offset = args.offset if args.offset is not None else inferred_offset
                low = offset - amp
                high = offset + amp
            low = max(low, qmin)
            high = min(high, qmax)
            # Determine t_range (dwell) to pass to affetto
            if args.dwell is not None:
                t_range_pass = (float(args.dwell), float(args.dwell))
            else:
                t_range_pass = update_t_range
            # Map traj to affetto profile name
            profile_map = {
                'random': 'trapezoidal',  # shape is trapezoidal; randomness is in targets
                'step': 'step',
                'trapezoidal': 'trapezoidal',
            }
            profile_str = profile_map.get(args.traj, 'trapezoidal')
            # Prepare delta magnitude range (non-negative). RandomTrajectory chooses the sign internally.
            if getattr(args, 'update_q_delta', None) is not None:
                try:
                    d0 = abs(float(args.update_q_delta[0]))
                    d1 = abs(float(args.update_q_delta[1]))
                    lo_mag, hi_mag = (d0, d1) if d0 <= d1 else (d1, d0)
                    update_q_delta = (float(lo_mag), float(hi_mag))
                except Exception:
                    d = max(abs(float(args.update_q_delta[0])), abs(float(args.update_q_delta[1])))
                    update_q_delta = (0.0, float(d))
            else:
                # Infer max magnitude from available room to limits around current q0
                left = abs(float(q0[0]) - float(low))
                right = abs(float(high) - float(q0[0]))
                amp_mag = max(left, right)
                update_q_delta = (0.0, float(amp_mag))
            # If user requested to force affetto, try to instantiate affetto's RandomTrajectory directly
            if getattr(args, 'force_affetto', False) and RandomTrajectory is not None and AffettoRTAdapter is not None:
                try:
                    qmin, qmax = float(args.q_range[0]), float(args.q_range[1])
                    # affetto expects update_q_range as positive delta range
                    update_q_range_delta = (float(update_q_delta[0]), float(update_q_delta[1]))
                    rt_obj = RandomTrajectory(
                        active_joints=active_joints,
                        t0=0.0,
                        q0=q0,
                        update_t_range=t_range_pass,
                        update_q_range=update_q_range_delta,
                        update_q_limit=(qmin, qmax),
                        update_profile=profile_str,
                        seed=args.seed,
                        async_update=True,
                    )
                    rt = cast(TrajectoryLike, AffettoRTAdapter(rt_obj, active_joints))
                except Exception:
                    # fall back to factory below
                    rt = cast(TrajectoryLike, make_random_trajectory(
                        seed=args.seed,
                        active_joints=active_joints,
                        q0=q0,
                        t_range=t_range_pass,
                        q_range=update_q_range,
                        profile=profile_str,
                        update_q_delta=update_q_delta,
                    ))
            else:
                rt = cast(TrajectoryLike, make_random_trajectory(
                    seed=args.seed,
                    active_joints=active_joints,
                    q0=q0,
                    t_range=t_range_pass,
                    q_range=update_q_range,
                    profile=profile_str,
                    update_q_delta=update_q_delta,
                ))
        elif args.traj == 'sin':
            # Local sine trajectory implementation
            class _SineTraj:
                def __init__(self, q0_val: float, low: float, high: float, amp: float | None, offset: float | None, period: float):
                    self.low = low
                    self.high = high
                    # default center is current q0; allow override
                    self.offset = float(q0_val) if offset is None else float(offset)
                    # clamp amplitude not to exceed limits
                    max_amp = max(0.0, min(self.offset - self.low, self.high - self.offset))
                    if amp is None:
                        # default to 40% of available room
                        a = 0.4 * max_amp
                    else:
                        a = float(amp)
                    self.amp = max(0.0, min(a, max_amp))
                    self.period = max(0.2, float(period))
                def get_qdes_func(self):
                    import math as _m
                    def f(t: float):
                        q = self.offset + self.amp * _m.sin(2.0 * _m.pi * (t / self.period))
                        # clamp just in case
                        q = min(max(q, self.low), self.high)
                        return np.array([q], dtype=float)
                    return f
                def reset_updater(self, *a, **k):
                    pass
            # choose period: use dwell if specified; else the midpoint of preset range
            if args.dwell is not None:
                period = float(args.dwell)
            else:
                period = float(sum(update_t_range) / 2.0)
            rt = cast(TrajectoryLike, _SineTraj(q0[0], update_q_range[0], update_q_range[1], args.amp, args.offset, period))
        else:
            # Fallback to affetto generator with trapezoidal profile
            rt = cast(TrajectoryLike, make_random_trajectory(
                seed=args.seed,
                active_joints=active_joints,
                q0=q0,
                t_range=update_t_range,
                q_range=update_q_range,
                profile='trapezoidal',
            ))
    except Exception as e:
        # worst-case fallback: simple constant trajectory
        try:
            print(f"[WARN] Trajectory build failed ({e!r}); falling back to constant q0.", flush=True)
        except Exception:
            pass
        class DummyRT:
            def __init__(self, val=0.0):
                self._val = val
            def get_qdes_func(self):
                return lambda t: np.array([self._val], dtype=float)
            def reset_updater(self, *a, **k):
                pass
        rt = cast(TrajectoryLike, DummyRT(q0[0]))

    # --- Prepare dated directory layout and config files ---
    try:
        date_str = datetime.now().strftime('%Y%m%d')
        # map internal 'trapezoidal' to folder 'trapezoid'
        if getattr(args, 'profile', None) == 'trapezoidal':
            profile_folder = 'trapezoid'
        else:
            profile_folder = getattr(args, 'profile', 'other')
        speed_folder = getattr(args, 'speed', 'middle')
        csv_date_root = os.path.join(args.output, 'hirai', 'csv', date_str, profile_folder, speed_folder)
        graph_date_root = os.path.join(args.output, 'hirai', 'graph', date_str, profile_folder, speed_folder)
        os.makedirs(csv_date_root, exist_ok=True)
        os.makedirs(graph_date_root, exist_ok=True)
        # write config tomls under csv/.../config
        try:
            config_dir = os.path.join(csv_date_root, 'config')
            os.makedirs(config_dir, exist_ok=True)
            myrobot_toml_path = os.path.join(config_dir, 'myrobot.toml')
            init_toml_path = os.path.join(config_dir, 'init.toml')
            # myrobot.toml captures run parameters
            try:
                ldc_addrs = [f"0x{getattr(s, 'addr', 0):02X}" for s in ldc_sensors] if ldc_sensors else []
            except Exception:
                ldc_addrs = []
            myrobot_toml = []
            myrobot_toml.append('[run]')
            myrobot_toml.append(f"duration = {float(duration):.6f}")
            myrobot_toml.append(f"interval_ms = {float(args.interval_ms):.6f}")
            try:
                if args.seed is None:
                    myrobot_toml.append("seed = null")
                else:
                    myrobot_toml.append(f"seed = {int(args.seed)}")
            except Exception:
                myrobot_toml.append("seed = null")
            myrobot_toml.append(f"traj = \"{getattr(args, 'traj', 'random')}\"")
            myrobot_toml.append(f"profile = \"{profile_folder}\"")
            myrobot_toml.append(f"speed = \"{speed_folder}\"")
            myrobot_toml.append(f"t_range = [{float(update_t_range[0]):.6f}, {float(update_t_range[1]):.6f}]")
            myrobot_toml.append(f"q_range = [{float(update_q_range[0]):.6f}, {float(update_q_range[1]):.6f}]")
            myrobot_toml.append(f"q_limit = [{float(update_q_limit[0]):.6f}, {float(update_q_limit[1]):.6f}]")
            myrobot_toml.append('')
            myrobot_toml.append('[pid]')
            myrobot_toml.append(f"kp = {float(args.kp):.9f}")
            myrobot_toml.append(f"ki = {float(args.ki):.9f}")
            myrobot_toml.append(f"kd = {float(args.kd):.9f}")
            myrobot_toml.append('')
            myrobot_toml.append('[valve]')
            myrobot_toml.append(f"center = {float(args.center):.6f}")
            myrobot_toml.append(f"span = {float(args.span):.6f}")
            myrobot_toml.append(f"min = {float(args.min_valve):.6f}")
            myrobot_toml.append(f"max = {float(args.max_valve):.6f}")
            myrobot_toml.append(f"incremental = {bool(args.incremental_valve)}")
            myrobot_toml.append(f"incr_gain = {float(args.incr_gain):.9f}")
            myrobot_toml.append(f"incr_leak = {float(args.incr_leak):.9f}")
            myrobot_toml.append('')
            myrobot_toml.append('[encoder]')
            myrobot_toml.append(f"ppr = {int(args.ppr)}")
            myrobot_toml.append(f"invert = {bool(args.encoder_invert)}")
            myrobot_toml.append(f"zero_at_start = {bool(args.zero_at_start)}")
            myrobot_toml.append(f"zero_deg = {('null' if args.zero_deg is None else f'{float(args.zero_deg):.6f}')}")
            myrobot_toml.append('')
            myrobot_toml.append('[sensors]')
            myrobot_toml.append(f"adc = {('true' if adc is not None else 'false')}")
            myrobot_toml.append(f"encoder = {('true' if enc is not None else 'false')}")
            # Safely format ldc_addrs without nested escaped quotes which break f-strings
            try:
                items = ", ".join([f'"{a}"' for a in ldc_addrs])
            except Exception:
                items = ""
            myrobot_toml.append("ldc_addrs = [" + items + "]")
            with open(myrobot_toml_path, 'w') as f:
                f.write('\n'.join(myrobot_toml) + '\n')
            # init.toml captures initial conditions
            init_toml = []
            init_toml.append('[init]')
            init_toml.append(f"enc_zero_deg = {float(enc_zero_deg_main if enc_zero_deg_main is not None else 0.0):.6f}")
            try:
                init_toml.append(f"q0 = {float(q0[0]):.6f}")
            except Exception:
                init_toml.append("q0 = 0.000000")
            init_toml.append(f"timestamp = \"{datetime.now().isoformat()}\"")
            with open(init_toml_path, 'w') as f:
                f.write('\n'.join(init_toml) + '\n')
        except Exception:
            pass
        # determine next run index based on existing CSV files
        def _scan_next_idx(dir_path: str) -> int:
            import re as _re
            max_idx = -1
            try:
                for name in os.listdir(dir_path):
                    if name.startswith('motion_data_') and name.endswith('.csv'):
                        m = _re.match(r"motion_data_(\d{3,})\.csv", name)
                        if m:
                            try:
                                idx = int(m.group(1))
                                if idx > max_idx:
                                    max_idx = idx
                            except Exception:
                                pass
            except Exception:
                return 0
            return max_idx + 1 if max_idx >= 0 else 0
        base_run_index = _scan_next_idx(csv_date_root)
    except Exception:
        # fallback if anything goes wrong
        date_str = datetime.now().strftime('%Y%m%d')
        profile_folder = 'trapezoid' if getattr(args, 'profile', None) == 'trapezoidal' else getattr(args, 'profile', 'other')
        speed_folder = getattr(args, 'speed', 'middle')
        csv_date_root = os.path.join(args.output, 'hirai', 'csv', date_str, profile_folder, speed_folder)
        graph_date_root = os.path.join(args.output, 'hirai', 'graph', date_str, profile_folder, speed_folder)
        os.makedirs(csv_date_root, exist_ok=True)
        os.makedirs(graph_date_root, exist_ok=True)
        base_run_index = 0

    # run repeats
    for i in range(args.n_repeat):
        print(f'[INFO] Run {i+1}/{args.n_repeat}', flush=True)
        # Recreate trajectory per-run with the same seed to ensure deterministic q_des
        try:
            if args.traj in ('random', 'step', 'trapezoidal'):
                # Prefer affetto generator only if explicitly forced, but run it synchronously
                if getattr(args, 'force_affetto', False) and RandomTrajectory is not None and AffettoRTAdapter is not None:
                    try:
                        qmin, qmax = float(args.q_range[0]), float(args.q_range[1])
                        update_q_delta_local = locals().get('update_q_delta', None)
                        if update_q_delta_local is not None:
                            uq0 = float(update_q_delta_local[0]); uq1 = float(update_q_delta_local[1])
                            update_q_range_delta = (uq0, uq1)
                        else:
                            # fallback to inferred range
                            update_q_range_delta = (0.0, float(max(abs(q0[0]-qmin), abs(qmax-q0[0]))))
                        rt_obj = RandomTrajectory(
                            active_joints=active_joints,
                            t0=0.0,
                            q0=q0,
                            update_t_range=t_range_pass,
                            update_q_range=update_q_range_delta,
                            update_q_limit=(qmin, qmax),
                            update_profile=profile_str,
                            seed=args.seed,
                            async_update=False,
                        )
                        rt = cast(TrajectoryLike, AffettoRTAdapter(rt_obj, active_joints))
                    except Exception:
                        rt = cast(TrajectoryLike, make_random_trajectory(
                            seed=args.seed,
                            active_joints=active_joints,
                            q0=q0,
                            t_range=t_range_pass,
                            q_range=update_q_range,
                            profile=profile_str,
                            update_q_delta=locals().get('update_q_delta', None),
                        ))
                else:
                    rt = cast(TrajectoryLike, make_random_trajectory(
                        seed=args.seed,
                        active_joints=active_joints,
                        q0=q0,
                        t_range=t_range_pass,
                        q_range=update_q_range,
                        profile=profile_str,
                        update_q_delta=locals().get('update_q_delta', None),
                    ))
            elif args.traj == 'sin':
                # Recreate simple deterministic sine trajectory per-run
                class _SineTrajLocal:
                    def __init__(self, q0_val, low, high, amp, offset, period):
                        import math as _m
                        self.low = low; self.high = high
                        self.offset = float(q0_val) if offset is None else float(offset)
                        max_amp = max(0.0, min(self.offset - self.low, self.high - self.offset))
                        a = (0.4 * max_amp) if amp is None else float(amp)
                        self.amp = max(0.0, min(a, max_amp))
                        self.period = max(0.2, float(period))
                    def get_qdes_func(self):
                        import math as _m
                        def f(t: float):
                            q = self.offset + self.amp * _m.sin(2.0 * _m.pi * (t / self.period))
                            q = min(max(q, self.low), self.high)
                            return np.array([q], dtype=float)
                        return f
                    def reset_updater(self, *a, **k):
                        pass
                period = float(args.dwell) if args.dwell is not None else float(sum(t_range_pass)/2.0)
                rt = cast(TrajectoryLike, _SineTrajLocal(q0[0], update_q_range[0], update_q_range[1], args.amp, args.offset, period))
        except Exception:
            # If recreation fails, keep existing rt
            pass

        # --- Reset controller state and recapture encoder zero for each run ---
        # Recreate controller per-run to ensure internal state (integrator/filters) is clean
        try:
            controller = cast(ControllerLike, create_controller(
                dac, enc,
                kp=args.kp, ki=args.ki, kd=args.kd,
                center=args.center, span_pct=args.span,
                min_pct=args.min_valve, max_pct=args.max_valve
            ))
        except Exception:
            # If recreation fails, keep existing controller instance (if any)
            try:
                pass
            except Exception:
                pass
        # Ensure valves are set to neutral and allow system to settle before taking encoder zero
        try:
            # Read encoder value BEFORE commanding valves so we can compare
            pre_valve_enc_raw = None
            if enc is not None:
                try:
                    enc.poll()
                    a = enc.degrees(args.ppr)
                    if args.encoder_invert:
                        a = -a
                    pre_valve_enc_raw = a
                    print(f'[DEBUG] Pre-valve encoder raw (run {i+1}): {pre_valve_enc_raw:.3f} deg', flush=True)
                except Exception:
                    pre_valve_enc_raw = None
            if dac is not None:
                try:
                    # set both channels to valve center percent (neutral)
                    dac.set_channels(float(args.center), float(args.center))
                except Exception:
                    pass
            # Wait a short time to let mechanical vibrations decay before sampling zero
            try:
                time.sleep(2.0)
            except Exception:
                pass
            # Capture encoder zero at valve center and compute offset from pre-valve reading
            if enc is not None and args.zero_at_start:
                try:
                    zero_at_center = _capture_zero(enc, args.encoder_invert, args.ppr)
                    enc_zero_deg_main = zero_at_center
                    try:
                        if pre_valve_enc_raw is not None:
                            offset = zero_at_center - pre_valve_enc_raw
                            print(f'[INFO] Captured encoder zero for run {i+1}: {enc_zero_deg_main:.3f} deg (offset from pre-valve: {offset:.3f} deg)', flush=True)
                        else:
                            print(f'[INFO] Captured encoder zero for run {i+1}: {enc_zero_deg_main:.3f} deg', flush=True)
                    except Exception:
                        pass
                except Exception:
                    # keep previous zero on failure
                    pass
        except Exception:
            pass
        # Ensure trajectory restarts cleanly from current q0 and t0=0 for each run
        try:
            rt.reset_updater(t0=0.0, q0=q0)
        except Exception:
            try:
                rt.reset_updater()
            except Exception:
                pass
        # Diagnostic print: show resolved trajectory parameters to help debug degenerate ranges
        try:
            # print only if trajectory was built from RandomTrajectory branch
            info = {
                'q0': float(q0[0]),
                'q_limit': update_q_limit,
                'q_range': update_q_range,
            }
            # use locals().get to avoid static-analysis warnings about possibly-unbound names
            _l = locals()
            _low = _l.get('low', None)
            if _low is not None:
                try:
                    info['low'] = float(_low)
                except Exception:
                    pass
            _high = _l.get('high', None)
            if _high is not None:
                try:
                    info['high'] = float(_high)
                except Exception:
                    pass
            _upd = _l.get('update_q_delta', None)
            if _upd is not None:
                try:
                    info['update_q_delta'] = tuple(_upd)
                except Exception:
                    pass
            _trp = _l.get('t_range_pass', None)
            if _trp is not None:
                try:
                    info['t_range_pass'] = tuple(_trp)
                except Exception:
                    pass
            print(f"[DEBUG] Trajectory params: {info}", flush=True)
        except Exception:
            pass
        # Use timestamped filename per execution so files are unique across runs
        # Build output directory: hirai/csv|graph/YYYYMMDD/{step,trapezoid}/{fast,middle,slow}/
        csv_root = csv_date_root
        graph_root = graph_date_root
        # Determine sequential index NNN within 000-099, avoid collision by seeking next free slot
        try:
            desired = base_run_index + i
        except Exception:
            desired = i
        try:
            idx = desired % 100
            attempts = 0
            while attempts < 100:
                name_csv_try = os.path.join(csv_root, f"motion_data_{idx:03d}.csv")
                if not os.path.exists(name_csv_try):
                    break
                idx = (idx + 1) % 100
                attempts += 1
        except Exception:
            idx = desired % 100
        per_run_filename = f"motion_data_{idx:03d}.csv"
        csv_path = run_once(
            duration,
            rt,
            dac,
            adc,
            enc,
            ldc_sensors,
            csv_root,
            per_run_filename,
            tuple(args.q_range),
            controller=controller,
            loop_interval_ms=args.interval_ms,
            verbose=args.verbose,
            enc_invert=args.encoder_invert,
            enc_zero_deg=enc_zero_deg_main,
            enc_ppr=args.ppr,
            # Zero was captured in main; avoid double capture here
            enc_zero_at_start=False,
            center=args.center,
            min_valve=args.min_valve,
            max_valve=args.max_valve,
            incremental_valve=args.incremental_valve,
            incr_gain=args.incr_gain,
            incr_leak=args.incr_leak,
        )
        # Generate plot and move/rename it into graph_root with motion_data_NNN.png
        try:
            png_path = plot_csv(csv_path)
            if png_path:
                try:
                    target = os.path.join(graph_root, f"motion_data_{idx:03d}.png")
                    # ensure target dir exists
                    os.makedirs(graph_root, exist_ok=True)
                    # move/rename to desired name
                    if os.path.abspath(png_path) != os.path.abspath(target):
                        try:
                            os.replace(png_path, target)
                        except Exception:
                            # as fallback, copy via read/write
                            try:
                                with open(png_path, 'rb') as src, open(target, 'wb') as dst:
                                    dst.write(src.read())
                            except Exception:
                                pass
                    print(f'[INFO] Plot saved: {target}', flush=True)
                except Exception:
                    # fallback: keep original
                    print(f'[WARN] Failed to move plot to {graph_root}; kept at {png_path}', flush=True)
        except Exception:
            pass
        # Short pause between runs to allow hardware to settle / reset
        try:
            time.sleep(3.0)
        except Exception:
            pass
        

    # cleanup hardware opened in main()
    try:
        close_devices(dac, adc, enc, ldc_sensors)
    except Exception:
        pass

    print('[INFO] All runs finished', flush=True)
    return 0


# Protocols describing the minimal hardware APIs used by this script.
@runtime_checkable
class DACLike(Protocol):
    def open(self) -> None: ...
    def set_channels(self, a: float, b: float) -> None: ...
    def close(self) -> None: ...

@runtime_checkable
class ADCLike(Protocol):
    def open(self) -> None: ...
    def read_pair(self) -> tuple[int, float, float, int, float, float]: ...
    def close(self) -> None: ...

@runtime_checkable
class EncoderLike(Protocol):
    def poll(self) -> None: ...
    def degrees(self) -> float: ...
    def close(self) -> None: ...

@runtime_checkable
class LDCLike(Protocol):
    addr: int
    def init(self) -> bool: ...
    def read_ch0_induct_uH(self) -> float: ...
    def close(self) -> None: ...

@runtime_checkable
class ControllerLike(Protocol):
    def update(self, q_des: float, current_angle: float | None, dt: float, dq_des: float | None = None, dq_meas: float | None = None) -> tuple[float, float, float]: ...
    def reset(self) -> None: ...

@runtime_checkable
class TrajectoryLike(Protocol):
    def get_qdes_func(self) -> Callable[[float], np.ndarray]: ...
    def reset_updater(self, *args, **kwargs) -> None: ...


if __name__ == '__main__':
    raise SystemExit(main())
