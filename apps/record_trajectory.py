#!/usr/bin/env python
#source /home/hosodalab2/Desktop/MyRobot/.venv-fix/bin/activate

#uv run python -m apps.record_trajectory -T 30 -v

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import List, Optional
import pathlib

# Make top-level project (containing `apps/myrobot_lib`) importable when running from affetto-nn-ctrl/apps
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import MyRobot helper libraries (hardware + logging + config)
from apps.myrobot_lib.hardware import open_devices, close_devices  # type: ignore
from apps.myrobot_lib.logger import DataLogger, make_header, TerminalPrinter  # type: ignore
from apps.myrobot_lib import config as cfg  # type: ignore
from apps.myrobot_lib.plotter import plot_csv  # type: ignore

RUN = True


def _on_signal(signum, frame):  # noqa: ARG001
    global RUN
    RUN = False


for _sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(_sig, _on_signal)
    except Exception:
        pass


def _scan_next_index(dir_path: str, prefix: str) -> int:
    try:
        os.makedirs(dir_path, exist_ok=True)
        names = [n for n in os.listdir(dir_path) if n.startswith(prefix) and n.endswith('.csv')]
        return len(names) + 1
    except Exception:
        return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record kinesthetic (taught) motion trajectory for MyRobot")
    p.add_argument("-T", "--duration", type=float, default=cfg.DEFAULT_DURATION_S, help="Recording duration [s]")
    p.add_argument("-o", "--output", default=cfg.DEFAULT_OUTPUT_DIR, help="Output directory for CSV")
    p.add_argument("--output-prefix", default="reference_trajectory", help="Filename prefix for CSV")
    p.add_argument("--interval-ms", type=float, default=cfg.DEFAULT_LOOP_INTERVAL_MS, help="Loop interval in ms")
    p.add_argument("--mode", choices=["hold", "release"], default="hold", help="Valve mode during recording: hold=center, release=0%")
    p.add_argument("--center", type=float, default=60.0, help="Valve center percent when mode=hold")
    # Encoder options
    p.add_argument("--zero-at-start", dest="zero_at_start", action="store_true", default=True, help="Capture encoder zero at start")
    p.add_argument("--no-zero-at-start", dest="zero_at_start", action="store_false")
    p.add_argument("--zero-deg", type=float, default=None, help="Explicit encoder zero offset (deg)")
    p.add_argument("--encoder-invert", dest="encoder_invert", action="store_true", default=True, help="Invert encoder sign")
    p.add_argument("--no-encoder-invert", dest="encoder_invert", action="store_false")
    p.add_argument("--ppr", type=int, default=cfg.ENCODER_PPR, help="Encoder PPR per channel (if applicable)")
    # Verbose terminal printing
    p.add_argument("-v", "--verbose", action="count", default=0, help="Print CSV rows on terminal")
    return p.parse_args()


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


def _perform_rom_sweep(dac, enc, enc_zero: float, enc_invert: bool, enc_ppr: int, verbose: int = 0) -> tuple[float, float]:
    """Perform range-of-motion sweep: 60/60 -> 20/100 -> 100/20 -> 60/60"""
    if dac is None or enc is None:
        return 0.0, 0.0
    
    print("[INFO] Performing range-of-motion sweep: 60/60 -> 20/100 -> 100/20 -> 60/60", flush=True)
    
    # Sweep sequence
    ramps = [((60.0, 60.0), (20.0, 100.0)), ((20.0, 100.0), (100.0, 20.0)), ((100.0, 20.0), (60.0, 60.0))]
    overall_min = None
    overall_max = None
    
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
            
            # High-frequency polling during this step
            step_start = time.perf_counter()
            poll_until = step_start + step_dt
            step_min = None
            step_max = None
            
            while time.perf_counter() < poll_until:
                try:
                    v_rel = _read_angle(enc, enc_invert, enc_zero, enc_ppr)
                    step_min = v_rel if step_min is None else min(step_min, v_rel)
                    step_max = v_rel if step_max is None else max(step_max, v_rel)
                    overall_min = v_rel if overall_min is None else min(overall_min, v_rel)
                    overall_max = v_rel if overall_max is None else max(overall_max, v_rel)
                    
                    if verbose >= 2:  # Only very verbose mode
                        print(f"[SWEEP-POLL] a={a_pct:.1f}% b={b_pct:.1f}% enc_deg={v_rel:.3f}", flush=True)
                except Exception:
                    pass
                time.sleep(0.00015)  # ~150µs polling interval
            
            # Step summary
            if verbose >= 1:
                if step_min is not None and step_max is not None:
                    print(f"[SWEEP] ramp{ramp_idx+1} step{s}/{steps} a={a_pct:.1f}% b={b_pct:.1f}% range=[{step_min:.3f}, {step_max:.3f}]", flush=True)
            
            # Ensure step timing
            remaining = poll_until - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
    
    # Return to center
    try:
        dac.set_channels(60.0, 60.0)
    except Exception:
        pass
    
    if overall_min is not None and overall_max is not None:
        print(f"[INFO] Observed encoder range during sweep: min={overall_min:.3f} max={overall_max:.3f} deg", flush=True)
        return overall_min, overall_max
    else:
        print("[WARN] No encoder readings during sweep", flush=True)
        return 0.0, 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record kinesthetic (taught) motion trajectory for MyRobot")
    p.add_argument("-T", "--duration", type=float, default=cfg.DEFAULT_DURATION_S, help="Recording duration [s]")
    p.add_argument("-o", "--output", default=cfg.DEFAULT_OUTPUT_DIR, help="Output directory for CSV")
    p.add_argument("--output-prefix", default="reference_trajectory", help="Filename prefix for CSV")
    p.add_argument("--interval-ms", type=float, default=cfg.DEFAULT_LOOP_INTERVAL_MS, help="Loop interval in ms")
    p.add_argument("--mode", choices=["hold", "release"], default="hold", help="Valve mode during recording: hold=center, release=0%")
    p.add_argument("--center", type=float, default=60.0, help="Valve center percent when mode=hold")
    # Encoder options
    p.add_argument("--zero-at-start", dest="zero_at_start", action="store_true", default=True, help="Capture encoder zero at start")
    p.add_argument("--no-zero-at-start", dest="zero_at_start", action="store_false")
    p.add_argument("--zero-deg", type=float, default=None, help="Explicit encoder zero offset (deg)")
    p.add_argument("--encoder-invert", dest="encoder_invert", action="store_true", default=True, help="Invert encoder sign")
    p.add_argument("--no-encoder-invert", dest="encoder_invert", action="store_false")
    p.add_argument("--ppr", type=int, default=cfg.ENCODER_PPR, help="Encoder PPR per channel (if applicable)")
    # Verbose terminal printing
    p.add_argument("-v", "--verbose", action="count", default=0, help="Print CSV rows on terminal")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Open hardware
    try:
        dac, adc, enc, ldc_sensors = open_devices(i2c_bus=cfg.I2C_BUS, ldc_addrs=cfg.LDC_ADDRS)
        print("[INFO] DAC opened", flush=True)
    except Exception as e:
        print(f"[ERROR] {e}", flush=True)
        return 1
    if adc is not None:
        print("[INFO] ADC opened", flush=True)
    if enc is not None:
        print("[INFO] Encoder enabled", flush=True)
    if ldc_sensors:
        addrs = ','.join(f"0x{getattr(s, 'addr', 0):02X}" for s in ldc_sensors)
        print(f"[INFO] LDC sensors: {len(ldc_sensors)} ({addrs})", flush=True)

    # --- Startup sequence following collect_data_myrobot pattern ---
    center_val = float(args.center)
    try:
        # Apply center to both valves and settle before zero capture
        dac.set_channels(center_val, center_val)
        time.sleep(1.0)
        print(f"[INFO] Applied center valve {center_val:.1f}% and settled for encoder zero capture", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to apply center valve: {e}", flush=True)

    # Capture encoder zero using collect_data_myrobot method
    enc_zero = args.zero_deg
    try:
        if args.zero_at_start and enc is not None and enc_zero is None:
            enc_zero = _capture_zero(enc, args.encoder_invert, args.ppr)
            print(f"[INFO] Captured encoder zero: {enc_zero:.3f} deg", flush=True)
    except Exception:
        pass
    if enc_zero is None:
        enc_zero = 0.0

    # Wait before ROM sweep
    time.sleep(1.0)
    
    # Perform range-of-motion sweep before recording
    try:
        _perform_rom_sweep(dac, enc, enc_zero, args.encoder_invert, args.ppr, args.verbose)
        time.sleep(0.5)  # Small settle after sweep
    except Exception as e:
        print(f"[WARN] ROM sweep failed: {e}", flush=True)

    # Release valves to 0% for free movement during recording
    try:
        dac.set_channels(0.0, 0.0)
        a_fixed = 0.0
        b_fixed = 0.0
        print("[INFO] Valves released (0%). Will start recording immediately", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to release valves: {e}", flush=True)
        # fallback: still use center as fixed if release failed
        a_fixed = center_val
        b_fixed = center_val

    # Start recording immediately (no additional wait)

    # Prepare logger
    header = make_header(adc is not None, [getattr(s, 'addr', 0) for s in (ldc_sensors or [])], enc is not None)
    # Use fixed project locations for recorded trajectories
    csv_dir = os.path.join('data', 'recorded_trajectory', 'csv')
    graph_dir = os.path.join('data', 'recorded_trajectory', 'graph')
    logger = DataLogger(csv_dir, args.output_prefix, header)
    run_idx = _scan_next_index(csv_dir, args.output_prefix)
    csv_path = logger.open_file(run_index=run_idx)
    tprinter = TerminalPrinter(args.verbose)

    # Loop config
    interval_s = float(args.interval_ms) / 1000.0
    t0 = time.perf_counter()
    next_tick = t0
    last_tension = [None] * len(ldc_sensors) if ldc_sensors else []
    last_tension_time = 0.0

    try:
        while RUN:
            now = time.perf_counter()
            if now < next_tick:
                # Poll encoder for edge capture while waiting (collect_data_myrobot method)
                if enc is not None:
                    try:
                        enc.poll()
                    except Exception:
                        pass
                time.sleep(0.0002)
                continue
            t = now - t0
            if args.duration > 0 and t >= args.duration:
                break
            # Maintain cadence
            while now - next_tick > interval_s:
                next_tick += interval_s
            next_tick += interval_s

            # Apply fixed valve command
            try:
                dac.set_channels(a_fixed, b_fixed)
            except Exception as e:
                print(f"[ERROR] DAC write failed: {e}", flush=True)
                break

            # Read ADC
            adc_vals: Optional[tuple[int, float, float, int, float, float]] = None
            if adc is not None:
                try:
                    adc_vals = adc.read_pair()
                except Exception:
                    adc_vals = None

            # Read LDC tension occasionally
            if ldc_sensors and (t - last_tension_time) >= cfg.TENSION_POLL_INTERVAL:
                for i, s in enumerate(ldc_sensors):
                    try:
                        v = s.read_ch0_induct_uH()
                        if i < len(last_tension):
                            last_tension[i] = v
                    except Exception:
                        pass
                last_tension_time = t

            # Read encoder using collect_data_myrobot method
            enc_deg = None
            if enc is not None:
                try:
                    enc_deg = _read_angle(enc, args.encoder_invert, enc_zero, args.ppr)
                except Exception:
                    enc_deg = None

            # Build row
            elapsed_ms = int(t * 1000.0)
            row: List[str | float | int] = [elapsed_ms, f"{a_fixed:.1f}", f"{b_fixed:.1f}", "", ""]
            if adc_vals is not None:
                r0, v0, k0, r1, v1, k1 = adc_vals
                row += [r0, f"{v0:.3f}", f"{k0:.1f}", r1, f"{v1:.3f}", f"{k1:.1f}"]
            elif adc is not None:
                row += ["", "", "", "", "", ""]
            for v in (last_tension or []):
                row.append("" if v is None else f"{float(v):.5f}")
            if enc is not None:
                row.append("" if enc_deg is None else f"{float(enc_deg):.3f}")

            # Log and optionally print
            try:
                logger.write_row(row)
                tprinter.print_row(header, row)
            except Exception as e:
                print(f"[WARN] Failed to write row: {e}", flush=True)
    finally:
        # Safe stop: close valves and close hardware
        try:
            dac.set_channels(0.0, 0.0)
        except Exception:
            pass
        try:
            logger.close()
        except Exception:
            pass
        try:
            close_devices(dac, adc, enc, ldc_sensors)
        except Exception:
            pass
        print(f"[INFO] CSV saved: {csv_path}")
        # Attempt to plot CSV and move plot into graph directory
        try:
            plot_out = ''
            try:
                plot_out = plot_csv(csv_path)
            except Exception as e:
                print(f"[WARN] plot_csv failed: {e}", flush=True)
            if plot_out:
                try:
                    import shutil
                    os.makedirs(graph_dir, exist_ok=True)
                    dst = os.path.join(graph_dir, os.path.basename(plot_out))
                    shutil.move(plot_out, dst)
                    print(f"[INFO] Plot moved: {dst}", flush=True)
                except Exception as e:
                    print(f"[WARN] failed to move plot: {e}", flush=True)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
