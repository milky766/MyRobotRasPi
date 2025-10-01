#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
import time
from typing import Any, Callable, cast

import numpy as np

# Workspace paths for local imports
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Add affetto-nn-ctrl/src to path so we can import affetto_nn_ctrl without installing the package
_AFFETTO_SRC = _PROJECT_ROOT / 'affetto-nn-ctrl' / 'src'
if _AFFETTO_SRC.exists() and str(_AFFETTO_SRC) not in sys.path:
    sys.path.insert(0, str(_AFFETTO_SRC))

# myrobot_lib helpers
from apps.myrobot_lib.hardware import open_devices, close_devices  # type: ignore
from apps.myrobot_lib.controller import create_controller  # type: ignore
from apps.myrobot_lib import config as cfg  # type: ignore
from apps.myrobot_lib.plotter import plot_csv as lib_plot_csv  # type: ignore

# affetto model loader (optional)
try:
    from affetto_nn_ctrl.model_utility import load_trained_model  # type: ignore
except Exception:
    load_trained_model = None

# ---- Reference CSV utilities ----

def _load_reference_csv(path: str) -> tuple[Callable[[float], float], Callable[[float], float], float]:
    """Load a reference CSV and return (qdes(t), dqdes(t), duration_sec).
    Accept columns: ms or time(s), qdes or q_des or q or enc_deg (deg). If dqdes missing, estimate by finite diff.
    """
    ts: list[float] = []
    qd: list[float] = []
    dq_col: list[float] | None = None
    with open(path, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        # detect time key
        first_row = None
        # determine which q key to use by inspecting header / first row keys
        possible_q_keys = ('qdes', 'q_des', 'q', 'enc_deg', 'enc', 'encdeg')
        possible_dq_keys = ('dqdes', 'dq_des', 'dq')
        q_key_found = None
        dq_key_found = None
        for row in r:
            if first_row is None:
                first_row = row
                # pick q key from header/row
                for k in possible_q_keys:
                    if k in row:
                        q_key_found = k
                        break
                # pick dq key if present
                for k in possible_dq_keys:
                    if k in row:
                        dq_key_found = k
                        break
                if dq_key_found is not None:
                    dq_col = []
            # time
            if 'ms' in row and row['ms'] != '':
                t = float(row['ms']) / 1000.0
            elif 'time' in row and row['time'] != '':
                t = float(row['time'])
            else:
                # If time is absent, assume uniform loop interval from config
                t = (len(ts)) * (cfg.DEFAULT_LOOP_INTERVAL_MS / 1000.0)
            # qdes / reference angle
            q = 0.0
            if q_key_found is not None and q_key_found in row and row[q_key_found] != '':
                try:
                    q = float(row[q_key_found])
                except Exception:
                    q = 0.0
            else:
                # fallback: try explicit keys
                for k in ('qdes', 'q_des', 'q', 'enc_deg', 'enc', 'encdeg'):
                    if k in row and row[k] != '':
                        try:
                            q = float(row[k])
                        except Exception:
                            q = 0.0
                        break
            ts.append(t)
            qd.append(q)
            if dq_col is not None:
                val = 0.0
                try:
                    for k in possible_dq_keys:
                        if k in row and row[k] != '':
                            val = float(row[k])
                            break
                except Exception:
                    val = 0.0
                dq_col.append(val)
    if not ts:
        raise RuntimeError(f'Empty reference: {path}')
    T = ts[-1] - ts[0]
    # linear interpolation builders
    def _lin_builder(xs: list[float], ys: list[float]) -> Callable[[float], float]:
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        def f(t: float) -> float:
            if t <= x[0]:
                return float(y[0])
            if t >= x[-1]:
                return float(y[-1])
            return float(np.interp(t, x, y))
        return f
    q_func = _lin_builder(ts, qd)
    if dq_col is not None and any(abs(v) > 0 for v in dq_col):
        dq_func = _lin_builder(ts, dq_col)
    else:
        # numeric diff of qdes with simple slope estimate
        xs = np.asarray(ts, dtype=float)
        ys = np.asarray(qd, dtype=float)
        # precompute slopes between points
        slopes = np.zeros_like(ys)
        for i in range(1, len(xs)):
            dt = xs[i] - xs[i-1]
            if dt > 0:
                slopes[i] = (ys[i] - ys[i-1]) / dt
            else:
                slopes[i] = slopes[i-1]
        def dqf(t: float) -> float:
            if t <= xs[0]:
                return float(slopes[1] if len(slopes) > 1 else 0.0)
            if t >= xs[-1]:
                return 0.0
            # find right index
            idx = int(np.searchsorted(xs, t, side='right'))
            idx = max(1, min(idx, len(slopes) - 1))
            return float(slopes[idx])
        dq_func = dqf
    return q_func, dq_func, float(T)


def _capture_zero(enc, invert: bool, ppr: int) -> float:
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
 Default: r               a = -a
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
        except Exception:
            a = 0.0
    if invert:
        a = -a
    return a - (zero_deg if zero_deg is not None else 0.0)


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def plot_csv(csv_path: str) -> str:
    return lib_plot_csv(csv_path)


def parse_args():
    p = argparse.ArgumentParser(description='Playback reference trajectory on MyRobot with model or PID')
    p.add_argument('reference', help='Reference CSV to follow')
    p.add_argument('-m', '--model', help='Trained model .joblib (if omitted, PID is used)')
    p.add_argument('-T', '--duration', type=float, default=None, help='Override duration (sec). ef length')
    p.add_argument('--interval-ms', type=float, default=cfg.DEFAULT_LOOP_INTERVAL_MS, help='Loop interval in ms (default 33.333)')
    # PID/valve
    p.add_argument('--kp', type=float, default=0.15)
    p.add_argument('--ki', type=float, default=0.25)
    p.add_argument('--kd', type=float, default=0.01)
    p.add_argument('--center', type=float, default=60.0)
    p.add_argument('--min-valve', type=float, default=cfg.VALVE_MIN)
    p.add_argument('--max-valve', type=float, default=cfg.VALVE_MAX)
    # Encoder
    p.add_argument('--ppr', type=int, default=cfg.ENCODER_PPR)
    p.add_argument('--zero-at-start', dest='zero_at_start', action='store_true', default=True)
    p.add_argument('--no-zero-at-start', dest='zero_at_start', action='store_false')
    p.add_argument('--zero-deg', type=float, default=None)
    p.add_argument('--encoder-invert', dest='encoder_invert', action='store_true', default=True)
    p.add_argument('--no-encoder-invert', dest='encoder_invert', action='store_false')
    # Output
    p.add_argument('-o', '--output', default='data/myrobot/hirai/track', help='Output directory for CSV/plots')
    p.add_argument('--prefix', default='tracked_motion', help='Filename prefix')
    # Model output scale
    p.add_argument('--model-scale', choices=['volt','percent'], default='volt', help='Scale of model outputs (default: volt)')
    # Debug
    p.add_argument('--debug-model-io', action='store_true', help='Print model input/output for first few steps')
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Load reference
    q_func, dq_func, ref_T = _load_reference_csv(args.reference)
    duration = float(args.duration) if args.duration is not None else float(ref_T)

    # Open hardware
    try:
        dac, adc, enc, ldc_sensors = open_devices(i2c_bus=cfg.I2C_BUS, ldc_addrs=cfg.LDC_ADDRS)
        print('[INFO] DAC opened', flush=True)
    except Exception as e:
        print(f'[ERROR] {e}', flush=True)
        return 1
    if adc is not None:
        print('[INFO] ADC opened', flush=True)
    if enc is not None:
        print('[INFO] Encoder enabled', flush=True)
    if ldc_sensors:
        addrs = ','.join([f"0x{getattr(s,'addr',0):02X}" for s in ldc_sensors])
        print(f'[INFO] LDC sensors: {len(ldc_sensors)}({addrs})', flush=True)

    # Initialize valves to center and capture encoder zero
    enc_zero_deg = args.zero_deg
    pre_val = None
    try:
        if enc is not None:
            pre_val = _read_angle(enc, args.encoder_invert, 0.0, args.ppr)
        # Center valves first (60/60 by default)
        if dac is not None:
            dac.set_channels(args.center, args.center)
        time.sleep(0.15)
        if enc is not None and args.zero_at_start:
            enc_zero_deg = _capture_zero(enc, args.encoder_invert, args.ppr)
            off = 0.0 if pre_val is None else (enc_zero_deg - pre_val)
            print(f'[INFO] Captured encoder zero: {enc_zero_deg:.3f} deg (offset from pre-valve: {off:.3f})', flush=True)
        if enc_zero_deg is None:
            enc_zero_deg = 0.0
    except Exception:
        if enc_zero_deg is None:
            enc_zero_deg = 0.0

    # Controller setup (for PID fallback and for logging pid_u)
    controller = create_controller(dac, enc, kp=args.kp, ki=args.ki, kd=args.kd, center=args.center, span_pct=cfg.VALVE_SPAN, min_pct=args.min_valve, max_pct=args.max_valve)

    # Load trained model if provided
    model = None
    model_active = []
    adapter = None
    if args.model:
        if load_trained_model is None:
            print('[WARN] affetto_nn_ctrl not available; using PID', flush=True)
        else:
            try:
                model = load_trained_model(args.model)
                adapter = getattr(model, 'adapter', None)
                model_active = list(getattr(adapter, 'params', getattr(adapter, '_params', None)).active_joints) if adapter is not None else [0]
                print(f'[INFO] Trained model loaded: {args.model}', flush=True)
                print(f'[INFO] Active joints: {model_active}', flush=True)

                # --- Remap for single-joint hardware ---
                # If the loaded model/adapter expects a different joint index (e.g. [5])
                # but this hardware controls a single joint, remap active_joints to [0]
                try:
                    params = getattr(adapter, 'params', None) or getattr(adapter, '_params', None)
                    if params is not None:
                        aj = getattr(params, 'active_joints', None)
                        if aj is not None and any(int(i) != 0 for i in aj):
                            print('[INFO] Adapter active_joints not compatible with single-joint hardware; remapping to [0]', flush=True)
                            try:
                                # try updating params in-place
                                params.active_joints = [0]
                            except Exception:
                                # fallback: just override local model_active
                                pass
                            model_active = [0]
                except Exception:
                    # non-fatal; continue and allow PID fallback if model fails at runtime
                    pass

            except Exception as e:
                print(f'[WARN] Failed to load model ({e}); using PID', flush=True)
                model = None

    # CSV setup
    os.makedirs(args.output, exist_ok=True)
    # derive filename stem from reference
    ref_name = pathlib.Path(args.reference).stem
    ts = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.output, f"{args.prefix}_{ref_name}_{ts}.csv")
    f = open(csv_path, 'w', buffering=1, newline='')
    writer = csv.writer(f)
    header = ['ms','ca','cb','caper','cbper','q','qdes','dq','dqdes','pa','pb','Ta','Tb','pid_u']
    writer.writerow(header)

    # Loop timing
    dt = float(args.interval_ms) / 1000.0
    t0 = time.perf_counter()
    next_tick = t0

    last_time = t0
    last_q = None
    last_adc = None
    # tension cache
    last_tension = {}

    # For model inputs, prepare base arrays sized by max joint index + 1
    if model_active:
        n_states = max(model_active) + 1
    else:
        n_states = 1

    printed_debug = 0

    try:
        while True:
            now = time.perf_counter()
            if now < next_tick:
                # light poll of encoder to keep it fresh
                try:
                    if enc is not None:
                        enc.poll()
                except Exception:
                    pass
                time.sleep(0.0001)
                continue
            t = now - t0
            if duration is not None and t >= duration:
                break
            # keep cadence
            while now - next_tick > dt:
                next_tick += dt
            next_tick += dt

            # Read encoder
            q = _read_angle(enc, args.encoder_invert, enc_zero_deg, args.ppr) if enc is not None else 0.0
            # Estimate measured dq (deg/s)
            dt_meas = now - last_time if last_time is not None else dt
            dq_meas = 0.0
            if last_q is not None and dt_meas > 1e-6:
                dq_meas = (q - last_q) / dt_meas
            last_q = q
            last_time = now

            # Reference
            qdes = float(q_func(t))
            dqdes = float(dq_func(t))

            # ADC (pressure) once per loop
            pa_val = ''
            pb_val = ''
            if adc is not None:
                try:
                    last_adc = adc.read_pair()
                    if last_adc is not None:
                        _, _, k0, _, _, k1 = last_adc
                        pa_val = k0
                        pb_val = k1
                except Exception:
                    last_adc = None

            # Tension (LDC)
            Ta_val = ''
            Tb_val = ''
            if ldc_sensors:
                for s in ldc_sensors:
                    try:
                        addr = getattr(s, 'addr', None)
                        val = s.read_ch0_induct_uH()
                        last_tension[addr] = val
                    except Exception:
                        pass
                # map addresses 0x2A (Ta) and 0x2B (Tb)
                Ta_val = last_tension.get(0x2A, '')
                Tb_val = last_tension.get(0x2B, '')

            # Control: either model or PID
            a_pct = args.center
            b_pct = args.center
            pid_u = 0.0
            if model is not None and adapter is not None:
                # Build states/refs as dicts expected by adapter
                q_arr = np.zeros(n_states, dtype=float)
                dq_arr = np.zeros(n_states, dtype=float)
                pa_arr = np.zeros(n_states, dtype=float)
                pb_arr = np.zeros(n_states, dtype=float)
                # fill our joint (use the first active joint index)
                jid = model_active[0] if model_active else 0
                q_arr[jid] = q
                dq_arr[jid] = dq_meas
                try:
                    pa_arr[jid] = float(pa_val) if pa_val != '' else 0.0
                    pb_arr[jid] = float(pb_val) if pb_val != '' else 0.0
                except Exception:
                    pass
                states = {'q': q_arr, 'dq': dq_arr, 'pa': pa_arr, 'pb': pb_arr}

                def _qdes_vec(tt: float) -> np.ndarray:
                    v = np.zeros(n_states, dtype=float)
                    v[jid] = float(q_func(tt))
                    return v

                def _dqdes_vec(tt: float) -> np.ndarray:
                    v = np.zeros(n_states, dtype=float)
                    v[jid] = float(dq_func(tt))
                    return v

                refs = {'qdes': lambda tt: _qdes_vec(tt), 'dqdes': lambda tt: _dqdes_vec(tt)}
                try:
                    X = adapter.make_model_input(t, states, refs)
                    y = model.predict(X)
                    # base inputs arrays (percent or volts is adapter-agnostic)
                    base_inputs = {'ca': np.zeros(n_states, dtype=float), 'cb': np.zeros(n_states, dtype=float)}
                    ca_all, cb_all = adapter.make_ctrl_input(y, base_inputs)
                    ca_val = float(ca_all[jid])
                    cb_val = float(cb_all[jid])
                    if args.model_scale == 'volt':
                        # convert to percent for hardware (assume 0-5V -> 0-100%)
                        a_pct = clamp(ca_val * 20.0, args.min_valve, args.max_valve)
                        b_pct = clamp(cb_val * 20.0, args.min_valve, args.max_valve)
                        ca_volt = ca_val
                        cb_volt = cb_val
                    else:
                        a_pct = clamp(ca_val, args.min_valve, args.max_valve)
                        b_pct = clamp(cb_val, args.min_valve, args.max_valve)
                        ca_volt = a_pct * 0.05
                        cb_volt = b_pct * 0.05
                    # optional debug
                    if args.debug_model_io and printed_debug < 20:
                        print(f"[DBG] t={t:6.3f} X={X.flatten().tolist()} y={np.ravel(y).tolist()} -> a%={a_pct:.2f} b%={b_pct:.2f}", flush=True)
                        printed_debug += 1
                except Exception as e:
                    if args.debug_model_io:
                        print(f'[WARN] model failed ({e}); falling back to PID at t={t:.3f}', flush=True)
                    a_pct = args.center
                    b_pct = args.center
                    # fall through to PID as safety
                    a_pct, b_pct, pid_u = controller.update(qdes, q, dt, dq_des=dqdes, dq_meas=dq_meas)
                    a_pct = clamp(a_pct, args.min_valve, args.max_valve)
                    b_pct = clamp(b_pct, args.min_valve, args.max_valve)
                    ca_volt = a_pct * 0.05
                    cb_volt = b_pct * 0.05
            else:
                # PID fallback
                a_pct, b_pct, pid_u = controller.update(qdes, q, dt, dq_des=dqdes, dq_meas=dq_meas)
                a_pct = clamp(a_pct, args.min_valve, args.max_valve)
                b_pct = clamp(b_pct, args.min_valve, args.max_valve)
                ca_volt = a_pct * 0.05
                cb_volt = b_pct * 0.05

            # Command DAC (percent assumed by driver shim)
            dac.set_channels(a_pct, b_pct)

            # Write CSV row
            ms = int(t * 1000)
            row = [
                ms,
                ca_volt, cb_volt,
                a_pct, b_pct,
                q, qdes,
                dq_meas, dqdes,
                pa_val, pb_val,
                Ta_val, Tb_val,
                pid_u,
            ]
            writer.writerow(row)
    finally:
        try:
            dac.set_channels(0.0, 0.0)
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass
        close_devices(dac, adc, enc, ldc_sensors)
        print(f'[INFO] CSV saved: {csv_path}', flush=True)
        try:
            out_plot = plot_csv(csv_path)
            if out_plot:
                print(f'[INFO] Plot saved: {out_plot}', flush=True)
        except Exception:
            pass

    return 0


if __name__ == '__main__':
    sys.exit(main())
