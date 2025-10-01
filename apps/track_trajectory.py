#!/usr/bin/env python
from __future__ import annotations

# Rewritten to use MyRobot hardware stack (apps/myrobot_lib) instead of affetto controller.
# Core algorithm is preserved: load reference trajectory -> generate qdes/dqdes over time ->
# real-time loop to track with PID -> log data -> repeat per reference file.
#env PYTHONPATH=/home/hosodalab2/Desktop/MyRobot uv run python -u apps/track_trajectory.py data/myrobot_model_MixAll/trained_model.joblib -r data/recorded_trajectory/csv/reference_trajectory_5.csv -T 20 -n 1 -v


import argparse
import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import sys
import threading
from collections import deque
import shutil

# Ensure project root and src are on sys.path so local packages (apps.myrobot_lib) can be imported
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# Also add affetto-nn-ctrl/src if present so affetto_nn_ctrl can be imported without installation
_AFFETTO_SRC = _PROJECT_ROOT / 'affetto-nn-ctrl' / 'src'
if _AFFETTO_SRC.exists() and str(_AFFETTO_SRC) not in sys.path:
    sys.path.insert(0, str(_AFFETTO_SRC))

# MyRobot helpers
from apps.myrobot_lib.hardware import open_devices, close_devices
from apps.myrobot_lib.controller import create_controller as create_myrobot_controller
from apps.myrobot_lib.logger import DataLogger, make_header
from apps.myrobot_lib.plotter import plot_csv
from apps.myrobot_lib import config as cfg

# --- Affetto dependency stub and model loader ---
import sys
import types
from pathlib import Path as _Path

def _ensure_affctrllib_stub() -> None:
    try:
        import affctrllib  # type: ignore
        _ = affctrllib  # noqa: F401
    except Exception:
        mod = types.ModuleType("affctrllib")
        class Logger:  # minimal stub
            def __init__(self, *a, **k):
                pass
        class Timer:  # minimal stub
            def __init__(self, *a, **k):
                pass
            def start(self):
                pass
        mod.Logger = Logger  # type: ignore[attr-defined]
        mod.Timer = Timer  # type: ignore[attr-defined]
        sys.modules["affctrllib"] = mod


def _load_trained_model(path: str):
    _ensure_affctrllib_stub()
    # If the environment's site-packages (e.g. .venv-fix) isn't on sys.path, try to add it
    try:
        # Inject local project paths so affetto_nn_ctrl is importable
        try:
            proj_root = _Path(__file__).resolve().parent.parent
            candidates = [
                proj_root / "src",
                proj_root / "affetto-nn-ctrl" / "src",
            ]
            for c in candidates:
                cs = str(c)
                if c.exists() and cs not in sys.path:
                    sys.path.insert(0, cs)
        except Exception:
            pass
        from affetto_nn_ctrl.model_utility import load_trained_model  # lazy import after stub
        return load_trained_model(path)
    except Exception as e:
        # attempt to discover .venv-fix site-packages under project root
        try:
            root = _Path.cwd()
            venv_root = root / ".venv-fix" / "lib"
            if venv_root.exists():
                for p in venv_root.glob("**/site-packages"):
                    sp = str(p)
                    if sp not in sys.path:
                        sys.path.insert(0, sp)
                        break
        except Exception:
            pass
        # retry import
        from affetto_nn_ctrl.model_utility import load_trained_model  # lazy import after stub
        return load_trained_model(path)

if TYPE_CHECKING:
    from typing import List, Tuple


DEFAULT_N_REPEAT = 10


# --- Local reference loader/interpolator (avoid pyplotutil/scipy dependency) ---
class Reference:
    def __init__(self, csv_path: Path, active_joints: list[int] | None = None, smoothness: float | None = None) -> None:  # noqa: ARG002
        self.path = Path(csv_path)
        if not self.path.exists():
            msg = f"Reference CSV not found: {csv_path}"
            raise FileNotFoundError(msg)
        # Load columns. Accept common keys:
        #  - time: 'ms' (milliseconds) or 't' (seconds)
        #  - angle: prefer 'qdes'/'q_des', else measured 'q'/'enc_deg'
        ts_ms: list[float] = []
        q_col: list[float] = []
        try:
            with self.path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # time
                    t_ms = row.get("ms")
                    if t_ms is None:
                        t_s = row.get("t")
                        t = float(t_s) if t_s not in (None, "") else 0.0
                        ts_ms.append(1000.0 * t)
                    else:
                        ts_ms.append(float(t_ms))
                    # angle
                    q_val = None
                    for key in ("qdes", "q_des", "q", "enc_deg"):
                        v = row.get(key)
                        if v not in (None, ""):
                            try:
                                q_val = float(v)
                                break
                            except Exception:
                                pass
                    if q_val is None:
                        q_val = float("nan")
                    q_col.append(q_val)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Failed to load reference CSV: {csv_path}: {e}") from e
        # sanitize
        t_arr = np.asarray(ts_ms, dtype=float) / 1000.0
        q_arr = np.asarray(q_col, dtype=float)
        # drop NaNs conservatively
        mask = np.isfinite(t_arr) & np.isfinite(q_arr)
        t_arr = t_arr[mask]
        q_arr = q_arr[mask]
        if t_arr.size < 2:
            raise RuntimeError(f"Reference CSV too short: {csv_path}")
        # Ensure strictly increasing time
        order = np.argsort(t_arr)
        self.t = t_arr[order]
        self.q = q_arr[order]
        # Precompute dq via central differences
        dq = np.gradient(self.q, self.t, edge_order=1)
        self.dq = dq
        self.duration = float(self.t[-1] - self.t[0]) if self.t.size > 0 else 0.0
        # Build simple linear interpolators using numpy.interp
        def _interp1(x: np.ndarray, xp: np.ndarray, fp: np.ndarray, left: float, right: float) -> np.ndarray:
            return np.interp(x, xp, fp, left=left, right=right)

        self._q_at: Callable[[float], float] = lambda tt: float(_interp1(np.array([tt]), self.t, self.q, self.q[0], self.q[-1])[0])
        self._dq_at: Callable[[float], float] = lambda tt: float(_interp1(np.array([tt]), self.t, self.dq, self.dq[0], 0.0)[0])

    def get_qdes_func(self) -> Callable[[float], np.ndarray]:
        return lambda t: np.array([self._q_at(t)], dtype=float)

    def get_dqdes_func(self) -> Callable[[float], np.ndarray]:
        return lambda t: np.array([self._dq_at(t)], dtype=float)


# --- Tracking primitive using MyRobot hardware ---

def track_motion_trajectory(
    dac,
    adc,
    enc,
    ldc_sensors,
    controller,
    reference: Reference,
    duration: float,
    data_logger: DataLogger,
    sweep_logger: DataLogger | None = None,
    header_text: str = "",
    loop_interval_ms: float = cfg.DEFAULT_LOOP_INTERVAL_MS,
    *,
    enc_ppr: int = cfg.ENCODER_PPR,
    enc_invert: bool = True,
    enc_zero_deg: float = 0.0,
    target_joint: int | None = None,
    verbose: int = 0,
) -> str:
    qdes_func = reference.get_qdes_func()
    dqdes_func = reference.get_dqdes_func()

    # open CSV file for this run
    csv_path = data_logger.open_file()

    # control loop
    t0 = time.perf_counter()
    next_tick = t0
    dt_prev = loop_interval_ms / 1000.0
    # debug prints limited to a few iterations
    _debug_left = 10

    # Detect trained model (optional)
    trained_model = None
    try:
        trained_model = getattr(controller, "trained_model", None)
    except Exception:
        trained_model = None

    # If model present, prepare dimensionality helpers (match training adapter spec)
    model_active_joints = None
    target_joint_id = 0
    dof = 1
    include_tension =True
    angle_unit = "deg"  # default safety; adapter may not define this
    # NEW: defaults for adapter params used in manual 7D feature build
    include_dqdes = False
    preview_time = 0.0
    joints: list[int] = []
    if trained_model is not None:
        # Extract adapter params used in training
        try:
            params = getattr(trained_model, "adapter", None)
            params = getattr(params, "params", None)
        except Exception:
            params = None
        try:
            model_active_joints = list(getattr(params, "active_joints", [])) if params is not None else []
            if not model_active_joints:
                model_active_joints = [0]
        except Exception:
            model_active_joints = [0]
        try:
            dof = int(getattr(params, "dof", max(model_active_joints) + 1)) if params is not None else (max(model_active_joints) + 1)
        except Exception:
            dof = max(model_active_joints) + 1
        try:
            include_tension = bool(getattr(params, "include_tension", False)) if params is not None else False
        except Exception:
            include_tension = False
        try:
            angle_unit = str(getattr(params, "angle_unit", "deg")) if params is not None else "deg"
        except Exception:
            angle_unit = "deg"
        # NEW: include_dqdes/preview_time/joints for manual feature vector
        try:
            include_dqdes = bool(getattr(params, "include_dqdes", False)) if params is not None else False
        except Exception:
            include_dqdes = False
        try:
            preview_time = float(getattr(params, "preview_step", 0)) * float(getattr(params, "dt", 0.0)) if params is not None else 0.0
        except Exception:
            preview_time = 0.0
        try:
            joints = list(getattr(params, "active_joints", [])) if params is not None else []
            if not joints:
                joints = [0]
        except Exception:
            joints = [0]
        # choose target joint id (global id expected by adapter)
        try:
            if target_joint is None:
                target_joint_id = int(model_active_joints[0])
            else:
                target_joint_id = int(target_joint)
        except Exception:
            target_joint_id = int(model_active_joints[0])

        # helper: convert degrees->adapter unit (rad if requested)
        def _a_angle(x: float) -> float:
            try:
                return float(np.deg2rad(x)) if angle_unit.lower().startswith("rad") else float(x)
            except Exception:
                return float(x)
        def _a_vel(x: float) -> float:
            try:
                return float(np.deg2rad(x)) if angle_unit.lower().startswith("rad") else float(x)
            except Exception:
                return float(x)

        # full vector length must be adapter dof
        n_full = int(dof)
        def _to_full_vec_at_index(x: float) -> np.ndarray:
            v = np.zeros((n_full,), dtype=float)
            v[target_joint_id] = float(x)
            return v
        # Reference funcs expanded to full dims with value only at target joint id (converted unit)
        def qdes_vec_func(tt: float) -> np.ndarray:
            return _to_full_vec_at_index(_a_angle(qdes_func(tt)[0]))
        def dqdes_vec_func(tt: float) -> np.ndarray:
            return _to_full_vec_at_index(_a_vel(dqdes_func(tt)[0]))
    else:
        n_full = 1
        qdes_vec_func = None  # type: ignore[assignment]
        dqdes_vec_func = None  # type: ignore[assignment]
        include_dqdes = False
        preview_time = 0.0
        joints = [0]

    # Expected feature dimension learned in training (if known) – used only for validation
    try:
        expected_n_features = getattr(controller, "_expected_n_features", None)
    except Exception:
        expected_n_features = None

    # helper to read encoder degrees using collect_data_myrobot method
    def _read_enc_deg() -> float:
        return _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)

    last_q_meas = _read_enc_deg()

    # If a separate sweep_logger was provided, run ROM sweep first
    if sweep_logger is not None:
        try:
            sweep_csv_path = sweep_logger.open_file()
        except Exception:
            sweep_csv_path = None
    else:
        sweep_csv_path = None

    # Always perform the sweep for hardware settling/verification, but only record if sweep_logger provided
    try:
        _perform_rom_sweep(dac, adc, enc, ldc_sensors, enc_zero_deg, enc_invert, enc_ppr, (sweep_logger if sweep_logger is not None else None), t0, verbose)
    except Exception:
        pass

    # If a sweep CSV was created, plot it
    if sweep_csv_path:
        try:
            plot_csv(sweep_csv_path)
        except Exception:
            pass

    # Re-center valves -> re-capture zero -> reset timebase
    try:
        if dac is not None:
            center_valve_pct = 60.0
            try:
                dac.set_channels(center_valve_pct, center_valve_pct)
            except Exception:
                pass
            time.sleep(1.0)
            try:
                enc_zero_new = _capture_zero(enc, enc_invert, enc_ppr)
                enc_zero_deg = enc_zero_new
                print(f"[INFO] Re-captured encoder zero after sweep: {enc_zero_deg:.3f} deg", flush=True)
            except Exception:
                pass
        # Reset time base
        t0 = time.perf_counter()
        next_tick = t0
        last_q_meas = _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)
    except Exception:
        pass
    # Start continuous encoder poller to avoid missing readings
    poller = EncoderPoller(enc, enc_invert, enc_zero_deg, enc_ppr, interval_s=0.00015, verbose=verbose)
    try:
        poller.start()
    except Exception:
        poller = None  # fallback if thread cannot start
    last_poll_ts = time.perf_counter()

    while True:
        now = time.perf_counter()
        # maintain cadence
        if now < next_tick:
            time.sleep(max(0.0, next_tick - now))
            now = time.perf_counter()
        # Read encoder from continuous poller (no misses)
        if poller is not None:
            stats = poller.get_stats_since(last_poll_ts)
            if stats is not None:
                enc_min, enc_max, enc_last, last_ts = stats
                q_meas = enc_last
                last_poll_ts = last_ts
                if verbose >= 2:
                    print(f"[ENC] t={now - t0:6.3f}s enc(last)={enc_last:.6f} (min={enc_min:.3f}, max={enc_max:.3f})", flush=True)
            else:
                q_meas = _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)
        else:
            # fallback
            q_meas = _read_angle(enc, enc_invert, enc_zero_deg, enc_ppr)

        t = now - t0
        if duration > 0 and t >= duration:
            break
        next_tick += loop_interval_ms / 1000.0

        # desired (degrees for logging; adapter unit conversion happens in qdes_vec_func)
        qdes = float(qdes_func(t)[0])
        dqdes = float(dqdes_func(t)[0])
        
        # measured encoder (degrees) - moved after q_meas is determined above
        if 'q_meas' not in locals():
            q_meas = _read_enc_deg()
            
        # Verbose encoder confirmation (first few loops)
        if verbose:
            try:
                # Add encoder synchronization debug info
                valve_state = f"a={a_pct:.1f}% b={b_pct:.1f}%" if 'a_pct' in locals() and 'b_pct' in locals() else "valve=unknown"
                print(f"[ENC] t={t:6.3f}s q_meas={q_meas:.6f} deg (Δq={q_meas-last_q_meas:.6f}) {valve_state}", flush=True)
            except Exception:
                pass
        dq_meas = (q_meas - last_q_meas) / dt_prev if dt_prev > 0 else 0.0
        last_q_meas = q_meas

        # ADC first so pa/pb are available to model
        adc_vals: list[float | int] = []
        pa = pb = 0.0
        if adc is not None and hasattr(adc, "read_pair"):
            try:
                raw0, volt0, kpa0, raw1, volt1, kpa1 = adc.read_pair()
                adc_vals = [raw0, volt0, kpa0, raw1, volt1, kpa1]
                pa, pb = float(kpa0), float(kpa1)
            except Exception:
                adc_vals = []
                pa = pb = 0.0

        # LDC (optional) — read before model to provide Ta/Tb
        ldc_vals: list[float] = []
        ta = tb = 0.0
        if ldc_sensors:
            for s in ldc_sensors:
                try:
                    v = float(s.read_ch0_induct_uH())
                except Exception:
                    v = float("nan")
                ldc_vals.append(v)
                try:
                    addr = getattr(s, "addr", 0)
                    if addr == 0x2A:
                        ta = float(v) if np.isfinite(v) else ta
                    elif addr == 0x2B:
                        tb = float(v) if np.isfinite(v) else tb
                except Exception:
                    pass

        # Decide valve command: model-first, PID fallback
        pid_u = None
        if trained_model is not None:
            # Convert measured angle/vel to adapter unit
            if 'angle_unit' in locals():
                q_meas_a = float(np.deg2rad(q_meas)) if angle_unit.lower().startswith('rad') else float(q_meas)
                dq_meas_a = float(np.deg2rad(dq_meas)) if angle_unit.lower().startswith('rad') else float(dq_meas)
            else:
                q_meas_a = float(q_meas)
                dq_meas_a = float(dq_meas)
            # Build states as full-size vectors (only target joint is non-zero)
            q_vec = np.zeros((n_full,), dtype=float); q_vec[target_joint_id] = q_meas_a
            dq_vec = np.zeros((n_full,), dtype=float); dq_vec[target_joint_id] = dq_meas_a
            pa_vec = np.zeros((n_full,), dtype=float); pa_vec[target_joint_id] = float(pa)
            pb_vec = np.zeros((n_full,), dtype=float); pb_vec[target_joint_id] = float(pb)
            ta_vec = np.zeros((n_full,), dtype=float); tb_vec = np.zeros((n_full,), dtype=float)
            if include_tension:
                ta_vec[target_joint_id] = float(ta)
                tb_vec[target_joint_id] = float(tb)

            # ALWAYS build manual feature vector to match training schema
            try:
                # Active joints slice
                jnts = joints if joints else [target_joint_id]
                qj = q_vec[jnts]
                dqj = dq_vec[jnts]
                paj = pa_vec[jnts]
                pbj = pb_vec[jnts]
                feats = [qj, dqj, paj, pbj]
                if include_tension:
                    taj = ta_vec[jnts]
                    tbj = tb_vec[jnts]
                    feats.extend([taj, tbj])
                # previewed references
                qdes_prev = qdes_vec_func(t + preview_time)[jnts]
                feats.append(qdes_prev)
                if include_dqdes:
                    dqdes_prev = dqdes_vec_func(t + preview_time)[jnts]
                    feats.append(dqdes_prev)
                X = np.atleast_2d(np.concatenate(feats))
                # Validate dimension
                if expected_n_features is not None and X.shape[1] != int(expected_n_features):
                    if verbose:
                        print(f"[WARN] Manual feature size mismatch: X.shape={X.shape} != expected {expected_n_features}. Falling back to PID.", flush=True)
                    raise RuntimeError("manual-feature-size-mismatch")
                if _debug_left > 0 and verbose:
                    print(f"[DEBUG] Built manual X (with tension={include_tension}) shape={X.shape}", flush=True)
            except Exception:
                # On any issue, fall back to PID
                a_pct, b_pct, pid_u = controller.update(qdes, q_meas, dt_prev, dq_des=dqdes, dq_meas=dq_meas)
                # send to hardware and log happen later
                # write row and continue
                # log and continue to next loop iteration
                ms = int(round(t * 1000.0))
                row: list[float | int | str] = [ms, a_pct, b_pct, qdes, (pid_u if pid_u is not None else "")]
                if adc_vals:
                    row.extend(adc_vals)
                if ldc_vals:
                    row.extend(ldc_vals)
                row.append(q_meas)
                data_logger.write_row(row)
                continue

            # Predict
            y = trained_model.predict(X)
            # Normalize predict output to 2D numpy array
            y_arr = np.asarray(y)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(1, -1)

            # If model returns two values, interpret them as [a_pct, b_pct] in percent for the target joint.
            # Otherwise fall back to adapter.make_ctrl_input if available.
            a_pct = b_pct = None
            if y_arr.shape[1] == 2:
                # Model outputs raw control values in range 0-5V for ca/cb.
                # Scale to percent (0-100) by (raw / 5.0 * 100.0). Clip raw to 0-5 to avoid unexpected values.
                try:
                    raw_a = float(y_arr[0, 0])
                except Exception:
                    raw_a = 0.0
                try:
                    raw_b = float(y_arr[0, 1])
                except Exception:
                    raw_b = 0.0
                
                # Debug: print raw model outputs to understand the scale
                if _debug_left > 0 and verbose:
                    print(f"[DEBUG] Raw model outputs: raw_a={raw_a:.6f}, raw_b={raw_b:.6f}", flush=True)
                
                # Clamp to 0-5V range and convert to percentage
                raw_a = max(0.0, min(5.0, raw_a))
                raw_b = max(0.0, min(5.0, raw_b))
                a_pct = (raw_a / 5.0) * 100.0
                b_pct = (raw_b / 5.0) * 100.0
            else:
                ca = np.zeros(n_full, dtype=float)
                cb = np.zeros(n_full, dtype=float)
                try:
                    ca, cb = trained_model.adapter.make_ctrl_input(y, {"ca": ca, "cb": cb})
                    a_pct = float(ca[target_joint_id]); b_pct = float(cb[target_joint_id])
                except Exception:
                    a_pct = 0.0; b_pct = 0.0

            # Determine clamp bounds: prefer controller attributes if present, else config constants
            try:
                min_pct = float(getattr(controller, "min_pct", 20.0))
                max_pct = float(getattr(controller, "max_pct", cfg.VALVE_MAX))
            except Exception:
                min_pct = 20.0; max_pct = float(cfg.VALVE_MAX)

            # Clamp within allowed range
            a_pct = max(min_pct, min(max_pct, float(a_pct)))
            b_pct = max(min_pct, min(max_pct, float(b_pct)))

            # short debug print of model I/O (report model y and final percents)
            if _debug_left > 0 and verbose:
                try:
                    act = getattr(trained_model.adapter.params, "active_joints", None)
                except Exception:
                    act = None
                try:
                    x_shape = getattr(X, "shape", None)
                except Exception:
                    x_shape = None
                print(f"[DEBUG] active_joints={act} dof={dof} include_tension={include_tension} angle_unit={angle_unit} X_shape={x_shape} y_shape={getattr(y,'shape',None)}", flush=True)
                try:
                    print(f"[DEBUG] y_sample={np.asarray(y).ravel()[:8]}", flush=True)
                    if y_arr.shape[1] == 2:
                        print(f"[DEBUG] raw_model (0-5V): a={raw_a:.3f} b={raw_b:.3f} -> scaled (%): a={a_pct:.3f} b={b_pct:.3f}", flush=True)
                    print(f"[DEBUG] final a_pct={a_pct:.3f} b_pct={b_pct:.3f} (clamped to [{min_pct:.1f}, {max_pct:.1f}])", flush=True)
                    print(f"[DEBUG] q_des={qdes:.3f} q_meas={q_meas:.3f} error={qdes-q_meas:.3f} deg", flush=True)
                except Exception:
                    pass
                _debug_left -= 1
        else:
            a_pct, b_pct, pid_u = controller.update(qdes, q_meas, dt_prev, dq_des=dqdes, dq_meas=dq_meas)

        # send to hardware
        try:
            if dac is not None and hasattr(dac, "set_channels"):
                # Ensure percentages are numeric and in 0..100 range
                try:
                    a_pct_send = float(a_pct)
                except Exception:
                    a_pct_send = 0.0
                try:
                    b_pct_send = float(b_pct)
                except Exception:
                    b_pct_send = 0.0
                a_pct_send = max(0.0, min(100.0, a_pct_send))
                b_pct_send = max(0.0, min(100.0, b_pct_send))
                # If DAC provides a pct->code helper, compute codes for debug and verification
                try:
                    pct_to_code = getattr(dac, "_pct_to_code", None)
                    if callable(pct_to_code):
                        a_code = int(pct_to_code(a_pct_send))
                        b_code = int(pct_to_code(b_pct_send))
                        if verbose:
                            print(f"[DEBUG] DAC codes -> a_code={a_code} b_code={b_code} (from a_pct={a_pct_send:.3f} b_pct={b_pct_send:.3f})", flush=True)
                except Exception:
                    pass
                # Send percent values to DAC (driver will map to codes)
                dac.set_channels(a_pct_send, b_pct_send)
        except Exception:
            pass

        # log
        ms = int(round(t * 1000.0))
        row: list[float | int | str] = [ms, a_pct, b_pct, qdes, (pid_u if pid_u is not None else "")]
        if adc_vals:
            row.extend(adc_vals)
        if ldc_vals:
            row.extend(ldc_vals)
        row.append(q_meas)
        data_logger.write_row(row)

    # Stop poller before returning
    try:
        if poller is not None:
            poller.stop()
    except Exception:
        pass

    # end loop — return path
    return csv_path


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track a reference trajectory using MyRobot hardware (model-first; PID fallback).",
    )
    # Input
    parser.add_argument(
        "model",
        help="Path to trained model (.joblib)",
    )
    parser.add_argument(
        "-r",
        "--reference-files",
        nargs="+",
        required=True,
        help="Path(s) to reference trajectory CSV(s).",
    )
    # Parameters
    parser.add_argument(
        "-T",
        "--duration",
        type=float,
        help="Time duration to perform trajectory tracking. If omitted, use reference duration.",
    )
    parser.add_argument(
        "-n",
        "--n-repeat",
        default=DEFAULT_N_REPEAT,
        type=int,
        help="Number of iterations to track each reference trajectory.",
    )
    # Output
    parser.add_argument(
        "-o",
        "--output",
        default="data/myrobot/track",
        help="Output directory to store tracked motion files.",
    )
    parser.add_argument(
        "--output-prefix",
        default="tracked_trajectory",
        help="Filename prefix that will be added to tracked motion files.",
    )
    # Controller params (PID fallback)
    parser.add_argument("--kp", type=float, default=cfg.DEFAULT_KP)
    parser.add_argument("--ki", type=float, default=cfg.DEFAULT_KI)
    parser.add_argument("--kd", type=float, default=cfg.DEFAULT_KD)
    parser.add_argument("--center", type=float, default=cfg.VALVE_CENTER)
    parser.add_argument("--span", type=float, default=cfg.VALVE_SPAN)
    parser.add_argument("--min-valve", type=float, default=20.0, help="Minimum valve percent (default 20)")
    parser.add_argument("--max-valve", type=float, default=cfg.VALVE_MAX)
    # Encoder options (align with collect_data_myrobot)
    parser.add_argument("--ppr", type=int, default=cfg.ENCODER_PPR, help="Encoder PPR per channel")
    parser.add_argument("--zero-at-start", dest="zero_at_start", action="store_true", default=True, help="Capture encoder zero at start")
    parser.add_argument("--no-zero-at-start", dest="zero_at_start", action="store_false")
    parser.add_argument("--zero-deg", type=float, default=None, help="Explicit encoder zero offset (deg)")
    parser.add_argument("--encoder-invert", dest="encoder_invert", action="store_true", default=True, help="Invert encoder sign")
    parser.add_argument("--no-encoder-invert", dest="encoder_invert", action="store_false")
    # Model/adapter mapping options
    parser.add_argument("--target-joint", type=int, default=None, help="Joint id in adapter.active_joints that corresponds to this hardware joint")
    # Others
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Enable verbose console output.",
    )
    parser.add_argument(
        "--interval-ms",
        type=float,
        default=cfg.DEFAULT_LOOP_INTERVAL_MS,
        help="Loop interval in milliseconds (dt = interval_ms/1000).",
    )
    return parser.parse_args()


# --- Encoder utility functions (matching collect_data_myrobot.py exactly) ---
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


def _poll_until_last(enc, enc_invert: bool, enc_zero: float | None, enc_ppr: int, poll_until: float, verbose: int = 0) -> float | None:
    """Poll encoder until poll_until and return last measured angle (matches ROM sweep semantics).

    Uses enc.poll(); enc.degrees(ppr), applies invert and zero offset, polls at ~150µs.
    """
    if enc is None:
        return None
    last_v = None
    while time.perf_counter() < poll_until:
        try:
            enc.poll()
            a = enc.degrees(enc_ppr)
            if enc_invert:
                a = -a
            a = a - (enc_zero if enc_zero is not None else 0.0)
            last_v = a
            if verbose >= 2 and last_v is not None:
                print(f"[POLL] enc_deg={last_v:.3f}", flush=True)
        except Exception:
            pass
        time.sleep(0.00015)
    return last_v


class EncoderPoller:
    """Background encoder poller to avoid missing readings between frames.

    Polls enc at high frequency (~150us), applies invert and zero, and stores
    timestamped samples in a ring buffer. The main loop can then fetch the
    most recent value (and min/max) since the last frame.
    """
    def __init__(self, enc, invert: bool, zero_deg: float | None, ppr: int, interval_s: float = 0.00015, maxlen: int = 20000, verbose: int = 0) -> None:
        self.enc = enc
        self.invert = bool(invert)
        self.zero = float(zero_deg) if zero_deg is not None else 0.0
        self.ppr = int(ppr)
        self.interval_s = float(interval_s)
        self.verbose = int(verbose)
        self._buf: deque[tuple[float, float]] = deque(maxlen=int(maxlen))
        self._last: tuple[float, float] | None = None
        self._lock = threading.Lock()
        self._running = False
        self._th: threading.Thread | None = None

    def update_zero(self, zero_deg: float | None) -> None:
        self.zero = float(zero_deg) if zero_deg is not None else 0.0

    def _read_once(self) -> float | None:
        if self.enc is None:
            return None
        try:
            self.enc.poll()
            a = self.enc.degrees(self.ppr)
        except Exception:
            return None
        if self.invert:
            a = -a
        return a - self.zero

    def _loop(self) -> None:
        while self._running:
            ts = time.perf_counter()
            v = self._read_once()
            if v is not None:
                with self._lock:
                    self._buf.append((ts, v))
                    self._last = (ts, v)
                    if self.verbose >= 3:
                        print(f"[ENC-POLLER] ts={ts:.6f} v={v:.3f}", flush=True)
            # keep cadence
            try:
                time.sleep(self.interval_s)
            except Exception:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._th = threading.Thread(target=self._loop, name="EncoderPoller", daemon=True)
        self._th.start()

    def stop(self, timeout: float | None = 1.0) -> None:
        self._running = False
        th = self._th
        if th is not None:
            try:
                th.join(timeout=timeout)
            except Exception:
                pass
            self._th = None

    def get_last(self) -> tuple[float, float] | None:
        with self._lock:
            return None if self._last is None else (self._last[1], self._last[0])

    def get_stats_since(self, since_ts: float) -> tuple[float, float, float, float] | None:
        """Return (min, max, last_value, last_ts) for samples newer than since_ts.
        If no new samples are available, return the latest sample if present.
        """
        with self._lock:
            if not self._buf:
                return None
            # collect samples newer than since_ts
            vals: list[tuple[float, float]] = [item for item in self._buf if item[0] > since_ts]
            if not vals:
                # no fresh sample, return the most recent
                last_ts, last_v = self._buf[-1]
                return (last_v, last_v, last_v, last_ts)
            ts_list = [ts for ts, _ in vals]
            v_list = [v for _, v in vals]
            vmin = float(min(v_list))
            vmax = float(max(v_list))
            last_ts = float(ts_list[-1])
            last_v = float(v_list[-1])
            return (vmin, vmax, last_v, last_ts)


def _perform_rom_sweep(dac, adc, enc, ldc_sensors, enc_zero: float, enc_invert: bool, enc_ppr: int, data_logger: DataLogger | None, t0: float, verbose: int = 0) -> None:
    """Perform range-of-motion sweep and record every step into the provided DataLogger.

    This function polls the encoder at high frequency (matches ROM sweep semantics)
    and writes one CSV row per step into the provided DataLogger if provided.
    If data_logger is None the sweep still executes (valve commands & polling)
    but no CSV rows are written or plotted.
    """
    if dac is None or enc is None:
        return

    print("[INFO] Performing range-of-motion sweep: 60/60 -> 20/100 -> 100/20 -> 60/60", flush=True)

    # Sweep sequence (unchanged)
    ramps = [((60.0, 60.0), (20.0, 100.0)), ((20.0, 100.0), (100.0, 20.0)), ((100.0, 20.0), (60.0, 60.0))]

    # Timing: 5s total, 50ms steps
    total_sweep_time = 5.0
    step_dt = 0.05
    per_ramp_time = max(0.5, total_sweep_time / len(ramps))

    for ramp_idx, ((start_a, start_b), (end_a, end_b)) in enumerate(ramps):
        steps = max(1, int(per_ramp_time / step_dt))
        for s in range(steps + 1):
            frac = float(s) / float(steps)
            a_pct = start_a + (end_a - start_a) * frac
            b_pct = start_b + (end_b - start_b) * frac

            # Set valve command
            try:
                dac.set_channels(a_pct, b_pct)
            except Exception:
                pass

            # High-frequency polling during this step — keep encoder polled but DO NOT
            # accumulate min/max. Capture the last reading for logging.
            step_start = time.perf_counter()
            poll_until = step_start + step_dt
            last_v = None
            while time.perf_counter() < poll_until:
                try:
                    enc.poll()
                    a = enc.degrees(enc_ppr)
                    if enc_invert:
                        a = -a
                    a = a - (enc_zero if enc_zero is not None else 0.0)
                    last_v = a
                    if verbose >= 2 and last_v is not None:
                        print(f"[SWEEP-POLL] a={a_pct:.1f}% b={b_pct:.1f}% enc_deg={last_v:.3f}", flush=True)
                except Exception:
                    pass
                time.sleep(0.00015)  # ~150µs polling interval

            # Read ADC once for this step (if available)
            adc_vals: list[float | int] = []
            pa = pb = 0.0
            if adc is not None and hasattr(adc, "read_pair"):
                try:
                    raw0, volt0, kpa0, raw1, volt1, kpa1 = adc.read_pair()
                    adc_vals = [raw0, volt0, kpa0, raw1, volt1, kpa1]
                    pa, pb = float(kpa0), float(kpa1)
                except Exception:
                    adc_vals = []

            # Read LDC sensors once for this step (if available)
            ldc_vals: list[float] = []
            if ldc_sensors:
                for sdev in ldc_sensors:
                    try:
                        v = float(sdev.read_ch0_induct_uH())
                    except Exception:
                        v = float("nan")
                    ldc_vals.append(v)

            # Build CSV row consistent with main loop format only if a data_logger was provided:
            # [ms, a_pct, b_pct, qdes, pid_u, <adc vals...>, <ldc vals...>, enc_deg]
            ms = int(round((time.perf_counter() - t0) * 1000.0))
            row: list[float | int | str] = [ms, a_pct, b_pct, "", ""]
            if adc_vals:
                row.extend(adc_vals)
            if ldc_vals:
                row.extend(ldc_vals)
            row.append(last_v if last_v is not None else "")

            # Write only if a DataLogger was provided (user requested recording)
            if data_logger is not None:
                try:
                    data_logger.write_row(row)
                except Exception:
                    pass

            # Step summary (no min/max)
            if verbose >= 1:
                if last_v is not None:
                    print(f"[SWEEP] ramp{ramp_idx+1} step{s}/{steps} a={a_pct:.1f}% b={b_pct:.1f}% enc={last_v:.3f}", flush=True)
                else:
                    print(f"[SWEEP] ramp{ramp_idx+1} step{s}/{steps} a={a_pct:.1f}% b={b_pct:.1f}% enc=none", flush=True)

            # Ensure step timing
            remaining = poll_until - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)

    # Intentionally DO NOT return valves to center here. Leave the last
    # commanded valve percentages in place so the main loop continues from
    # the sweep endpoint without changing air pressure.
    return


def main() -> None:
    args = parse()

    # Ensure project root and src are on sys.path so local packages can be imported
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _SRC_PATH = _PROJECT_ROOT / 'src'
    if str(_SRC_PATH) not in sys.path:
        sys.path.insert(0, str(_SRC_PATH))
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    # Try adding affetto-nn-ctrl/src so affetto_nn_ctrl is importable without installation
    _AFFETTO_SRC = _PROJECT_ROOT / 'affetto-nn-ctrl' / 'src'
    if _AFFETTO_SRC.exists() and str(_AFFETTO_SRC) not in sys.path:
        sys.path.insert(0, str(_AFFETTO_SRC))

    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Open hardware
    dac, adc, enc, ldc_sensors = open_devices(i2c_bus=cfg.I2C_BUS, ldc_addrs=cfg.LDC_ADDRS)
    print("[INFO] DAC opened", flush=True)
    if adc is not None:
        print("[INFO] ADC opened", flush=True)
    if enc is not None:
        print("[INFO] Encoder enabled", flush=True)
    if ldc_sensors:
        addrs = ",".join(f"0x{getattr(s, 'addr', 0):02X}" for s in ldc_sensors)
        print(f"[INFO] LDC sensors: {len(ldc_sensors)} ({addrs})", flush=True)

    # Build controller
    ctrl = create_myrobot_controller(
        dac,
        enc,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        center=args.center,
        span_pct=args.span,
        min_pct=args.min_valve,
        max_pct=args.max_valve,
    )

    # --- Startup sequence following collect_data_myrobot pattern ---
    enc_zero = args.zero_deg
    try:
        # Set initial valves to center (60%) and settle
        if dac is not None:
            center_valve_pct = 60.0
            dac.set_channels(center_valve_pct, center_valve_pct)
            print(f"[INFO] Initial valves set to {center_valve_pct:.1f}% for encoder zero capture", flush=True)
        time.sleep(1.0)
        
        # Capture encoder zero using collect_data_myrobot method
        if args.zero_at_start and enc is not None and enc_zero is None:
            try:
                enc_zero = _capture_zero(enc, args.encoder_invert, args.ppr)
                print(f"[INFO] Captured encoder zero: {enc_zero:.3f} deg", flush=True)
            except Exception:
                enc_zero = 0.0
        if enc_zero is None:
            enc_zero = 0.0
        
        # NOTE: ROM sweep execution is performed per-reference inside track_motion_trajectory.
        # Do not run the sweep here at startup to avoid duplicate sweeps and to keep
        # sweep-recording optional.
        
        # Small settle after potential startup actions
        time.sleep(0.5)
    
    except Exception:
        if enc_zero is None:
            enc_zero = 0.0

    # Load trained model and attach to controller (for use inside loop)
    trained_model = None
    expected_n_features = None
    try:
        trained_model = _load_trained_model(args.model)
        # Reset adapter internal state if available (for delay/preview features)
        try:
            if hasattr(trained_model.adapter, 'reset'):
                trained_model.adapter.reset()
        except Exception:
            pass
        setattr(ctrl, "trained_model", trained_model)
        try:
            aj = getattr(trained_model.adapter.params, "active_joints", None)
            dof = getattr(trained_model.adapter.params, "dof", None)
            inc_t = getattr(trained_model.adapter.params, "include_tension", None)
            ang_u = getattr(trained_model.adapter.params, "angle_unit", None)
        except Exception:
            aj = dof = inc_t = ang_u = None
        # Inspect pipeline expected input feature size (if available)
        try:
            from sklearn.pipeline import Pipeline  # type: ignore
            pipe = getattr(trained_model, "model", None)
            if isinstance(pipe, Pipeline):
                scaler = pipe.steps[0][1]
                expected_n_features = getattr(scaler, "n_features_in_", None)
        except Exception:
            expected_n_features = None
        # Keep expected feature size on controller for use in loop
        try:
            setattr(ctrl, "_expected_n_features", expected_n_features)
        except Exception:
            pass
        print(
            f"[INFO] Model loaded: {getattr(trained_model, 'model_name', type(trained_model).__name__)} (active_joints={aj}, dof={dof}, include_tension={inc_t}, angle_unit={ang_u}, expected_n_features={expected_n_features})",
            flush=True,
        )
    except Exception as e:
        print(f"[WARN] Failed to load model '{args.model}': {e}. Using PID fallback.", flush=True)

    # For each reference file
    ref_paths = [Path(p) for p in args.reference_files]
    for i, ref_path in enumerate(ref_paths, start=1):
        # Load reference
        ref = Reference(ref_path)
        duration = float(args.duration) if args.duration is not None else float(ref.duration)
        # Create per-reference subdir to mimic original layout
        ref_dir = output_dir / f"reference_{i:03d}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        # Centralized destinations for tracked outputs (CSV and graphs)
        base_dest = Path('/home/hosodalab2/Desktop/MyRobot/data/tracked_trajectory')
        csv_dest = base_dest / 'csv'
        graph_dest = base_dest / 'graph'
        csv_dest.mkdir(parents=True, exist_ok=True)
        graph_dest.mkdir(parents=True, exist_ok=True)

        # Prepare logger header (ADC/LDC/ENC presence)
        header = make_header(has_adc=adc is not None, ldc_addrs=[getattr(s, 'addr', 0) for s in ldc_sensors], has_enc=enc is not None)

        # Determine starting index based on existing files in centralized CSV folder
        def _next_idx(d: Path) -> int:
            existing = sorted([p for p in d.glob(f"{args.output_prefix}_*.csv")])
            max_idx = 0
            for p in existing:
                try:
                    stem = p.stem  # e.g., tracked_trajectory_001
                    suffix = stem.rsplit("_", 1)[-1]
                    max_idx = max(max_idx, int(suffix))
                except Exception:
                    pass
            return max_idx + 1

        base_idx = _next_idx(csv_dest)

        for j in range(args.n_repeat):
            # Reset controller state per run
            try:
                ctrl.reset()
            except Exception:
                pass
            # Prepare logger
            # Write tracked CSV directly into centralized csv_dest with sequential numbering
            current_idx = base_idx + j
            output_name = f"{args.output_prefix}_{current_idx}"
            logger = DataLogger(str(csv_dest), output_name, header)
            # Do NOT create a separate sweep logger — perform sweep but do not record CSV/plot
            sweep_logger = None
            # Track
            header_text = f"[Ref:{i}/{len(ref_paths)}(Cnt:{j + 1}/{args.n_repeat})] Tracking..."
            csv_path = track_motion_trajectory(
                dac,
                adc,
                enc,
                ldc_sensors,
                ctrl,
                ref,
                duration,
                logger,
                sweep_logger,
                header_text=header_text,
                loop_interval_ms=float(args.interval_ms),
                enc_ppr=int(args.ppr),
                enc_invert=bool(args.encoder_invert),
                enc_zero_deg=float(enc_zero or 0.0),
                target_joint=(None if args.target_joint is None else int(args.target_joint)),
                verbose=int(args.verbose),
            )
            print(f"[INFO] Motion file saved: {csv_path}")
            # Plot (attempt) and then copy CSV+PNG to central folder
            try:
                # Set plot title so it indicates which trained model / reference was used
                try:
                    import os as _os
                    model_path = Path(args.model)
                    model_folder = model_path.parent.name if model_path.parent.name else model_path.stem
                    _os.environ['PLOT_TITLE'] = f"Model: {model_folder}/{model_path.name}  Ref: {ref_path.name}"
                except Exception:
                    pass
                plot_csv(csv_path)
            except Exception:
                pass

            try:
                # The CSV is already in the centralized csv_dest folder
                src_csv = Path(csv_path)
                
                # Copy PNG (assume plot_csv created a .png alongside CSV) into graph folder
                try:
                    png_src = src_csv.with_suffix('.png')
                    if png_src.exists():
                        shutil.copy2(str(png_src), str(graph_dest / png_src.name))
                        print(f"[INFO] Plot saved: {graph_dest / png_src.name}")
                        # Remove the original PNG from CSV folder
                        png_src.unlink()
                        print(f"[INFO] Removed original PNG from CSV folder: {png_src}")
                except Exception as e:
                    print(f"[WARN] Failed to copy/remove PNG: {e}")
            except Exception:
                pass

    # Release/cleanup
    close_devices(dac, adc, enc, ldc_sensors)
    print("[INFO] Finished trajectory tracking", flush=True)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Cnt csv dq dqdes enc kpa ldc pid ppm qdes"
# End:
# End of file
