#!/usr/bin/env python
from __future__ import annotations

# Rewritten to use MyRobot hardware stack (apps/myrobot_lib) instead of affetto controller.
# Core algorithm is preserved: load reference trajectory -> generate qdes/dqdes over time ->
# real-time loop to track with PID -> log data -> repeat per reference file.
#/home/hosodalab2/Desktop/MyRobot/.venv-fix/bin/python -m apps.track_trajectory data/myrobot_model/trained_model.joblib -r data/recorded_trajectory/csv/reference_trajectory_3.csv -n 1 --interval-ms 33.33333333 -T 30

import argparse
import csv
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

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

    # helper to read encoder degrees (apply poll, ppr, invert, zero)
    def _read_enc_deg() -> float:
        if enc is None:
            return 0.0
        try:
            if hasattr(enc, "poll"):
                try:
                    enc.poll()
                except Exception:
                    pass
            if hasattr(enc, "degrees"):
                try:
                    val = enc.degrees(enc_ppr)  # type: ignore[arg-type]
                except TypeError:
                    val = enc.degrees()  # type: ignore[misc]
            else:
                val = 0.0
        except Exception:
            val = 0.0
        try:
            if enc_invert:
                val = -float(val)
            val = float(val) - float(enc_zero_deg)
        except Exception:
            val = float(val)
        return float(val)

    last_q_meas = _read_enc_deg()

    while True:
        now = time.perf_counter()
        if now < next_tick:
            # Poll encoder between frames to avoid missing edges (matches integrated_sensor_sin_python).
            # This helps capture fast quadrature transitions that would otherwise be lost when
            # sleeping for the whole frame interval. Polling interval ~150µs as in the integrated script.
            if enc is not None:
                poll_until = next_tick
                while time.perf_counter() < poll_until:
                    try:
                        if hasattr(enc, "poll"):
                            try:
                                enc.poll()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # Optional: print a raw-tick estimate for quick debugging. Limit prints by _debug_left
                    if verbose and _debug_left > 0:
                        try:
                            raw = None
                            # Try several common raw-tick attribute/method names used by various encoder impls
                            if hasattr(enc, "counts"):
                                raw = enc.counts if not callable(enc.counts) else enc.counts()
                            elif hasattr(enc, "count"):
                                raw = enc.count()
                            elif hasattr(enc, "ticks"):
                                raw = enc.ticks if not callable(enc.ticks) else enc.ticks()
                            elif hasattr(enc, "position"):
                                raw = enc.position if not callable(enc.position) else enc.position()
                            else:
                                # Fallback: read degrees and convert to approximate tick count
                                try:
                                    d = enc.degrees(enc_ppr)
                                except TypeError:
                                    d = enc.degrees()
                                raw = float(d) * float(enc_ppr) / 360.0
                            print(f"[ENC-POLL] t={time.perf_counter()-t0:6.3f}s raw_ticks={raw}", flush=True)
                        except Exception:
                            pass
                    time.sleep(0.00015)
            # Sleep remaining time (if any) to maintain cadence
            time.sleep(max(0.0, next_tick - now))
            now = time.perf_counter()
        t = now - t0
        if duration > 0 and t >= duration:
            break
        next_tick += loop_interval_ms / 1000.0

        # desired (degrees for logging; adapter unit conversion happens in qdes_vec_func)
        qdes = float(qdes_func(t)[0])
        dqdes = float(dqdes_func(t)[0])
        # measured encoder (degrees)
        q_meas = _read_enc_deg()
        # Verbose encoder confirmation (first few loops)
        if _debug_left > 0 and verbose:
            try:
                print(f"[ENC] t={t:6.3f}s q_meas={q_meas:.6f} deg (raw last_q_meas={last_q_meas:.6f})", flush=True)
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
                # ...existing code...
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
                # Assume model outputs raw control values in range 0..5 for ca/cb.
                # Scale to percent (0..100) by (raw / 5.0 * 100.0). Clip raw to 0..5 to avoid unexpected values.
                try:
                    raw_a = float(y_arr[0, 0])
                except Exception:
                    raw_a = 0.0
                try:
                    raw_b = float(y_arr[0, 1])
                except Exception:
                    raw_b = 0.0
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
                min_pct = float(getattr(controller, "min_pct", cfg.VALVE_MIN))
                max_pct = float(getattr(controller, "max_pct", cfg.VALVE_MAX))
            except Exception:
                min_pct = float(cfg.VALVE_MIN); max_pct = float(cfg.VALVE_MAX)

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
                        print(f"[DEBUG] raw_model_a={raw_a:.3f} raw_model_b={raw_b:.3f} -> a_pct_model={a_pct:.3f} b_pct_model={b_pct:.3f}", flush=True)
                    else:
                        print(f"[DEBUG] a_pct_model={a_pct:.3f} b_pct_model={b_pct:.3f}", flush=True)
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
    parser.add_argument("--min-valve", type=float, default=cfg.VALVE_MIN)
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


def main() -> None:
    args = parse()

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

    # Startup sequence: center valves -> capture encoder zero -> short wait
    enc_zero = args.zero_deg
    try:
        # Set initial valves to 60% and hold while capturing encoder zero (user requested behavior)
        if dac is not None:
            init_valve_pct = 60.0
            dac.set_channels(init_valve_pct, init_valve_pct)
            print(f"[INFO] Initial valves set to {init_valve_pct:.1f}% and will be held while capturing encoder zero", flush=True)
        time.sleep(1.0)
        if args.zero_at_start and enc is not None and enc_zero is None:
            try:
                # sample briefly similar to record_trajectory
                t0 = time.perf_counter()
                vals = []
                while time.perf_counter() - t0 < 0.08:
                    try:
                        enc.poll()
                        try:
                            a = enc.degrees(args.ppr)
                        except TypeError:
                            a = enc.degrees()
                        if args.encoder_invert:
                            a = -a
                        vals.append(float(a))
                    except Exception:
                        pass
                    time.sleep(0.002)
                if vals:
                    enc_zero = sum(vals) / len(vals)
                else:
                    try:
                        enc.poll()
                        try:
                            a = enc.degrees(args.ppr)
                        except TypeError:
                            a = enc.degrees()
                        if args.encoder_invert:
                            a = -a
                        enc_zero = float(a)
                    except Exception:
                        enc_zero = 0.0
                print(f"[INFO] Captured encoder zero: {float(enc_zero):.3f} deg", flush=True)
            except Exception:
                enc_zero = enc_zero if enc_zero is not None else 0.0
        if enc_zero is None:
            enc_zero = 0.0
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
        # Prepare logger header (ADC/LDC/ENC presence)
        header = make_header(has_adc=adc is not None, ldc_addrs=[getattr(s, 'addr', 0) for s in ldc_sensors], has_enc=enc is not None)

        # Determine starting index based on existing files
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

        base_idx = _next_idx(ref_dir)

        for j in range(args.n_repeat):
            # Reset controller state per run
            try:
                ctrl.reset()
            except Exception:
                pass
            # Prepare logger
            logger = DataLogger(str(ref_dir), args.output_prefix, header)
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
                header_text=header_text,
                loop_interval_ms=float(args.interval_ms),
                enc_ppr=int(args.ppr),
                enc_invert=bool(args.encoder_invert),
                enc_zero_deg=float(enc_zero or 0.0),
                target_joint=(None if args.target_joint is None else int(args.target_joint)),
                verbose=int(args.verbose),
            )
            print(f"[INFO] Motion file saved: {csv_path}")
            try:
                plot_csv(csv_path)
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
