#!/usr/bin/env python3
#uv run python apps/u_ramp_scan.py --u0 -100 --u1 100 --duration 12 --start-delay 1 --encoder-zero-on-start --encoder --encoder-invert
from __future__ import annotations
"""Linear open-loop differential command ramp to map encoder (and optional sensors) vs u_diff.
Now supports bidirectional sweep (e.g. -100 -> +100). Logs:
  ms, valve_a_pct, valve_b_pct, q_des(blank), pid_u(u_diff), [adc...], [tension...], enc_deg
Use --adc / --tension to enable extra sensors.
This variant runs min_pct->max_pct then back to min_pct by performing two u_ramp passes.

uv run python apps/range_of_motion.py --u0 -100 --u1 100 --duration 12 --start-delay 1 --encoder-zero-on-start --encoder --encoder-invert
↑(a,b)=(20,100)→(100,20)→(20,100)と動かすコード。これで-20~50度は確実に可動域として確保できる
"""

import argparse, os, time  # added time
import sys
import glob
from pathlib import Path
import csv

# plotting: import lazily to avoid startup/blocking during run
plt = None
np = None

from affetto_nn_ctrl.hw.integrated_session import IntegratedConfig, IntegratedSession

def build_parser():
    ap=argparse.ArgumentParser(description='Open-loop u_diff linear ramp (round-trip)')
    ap.add_argument('--u0', type=float, default=None, help='Optional start differential command (if omitted, derived from min/max)')
    ap.add_argument('--u1', type=float, default=None, help='Optional end differential command (if omitted, derived from min/max)')
    ap.add_argument('--duration', type=float, default=8.0)
    ap.add_argument('--interval-ms', type=int, default=50)
    ap.add_argument('--min-pct', type=float, default=20.0)
    ap.add_argument('--max-pct', type=float, default=100.0)
    ap.add_argument('--encoder', action='store_true')
    ap.add_argument('--encoder-invert', action='store_true')
    ap.add_argument('--encoder-zero-on-start', action='store_true', help='Capture initial encoder angle as zero before ramp (pre-actuation)')
    ap.add_argument('--adc', action='store_true', help='Enable ADC pressure channels')
    ap.add_argument('--adc-ch0', type=int, default=2)
    ap.add_argument('--adc-ch1', type=int, default=3)
    ap.add_argument('--tension', action='store_true', help='Enable LDC tension sensors scan')
    ap.add_argument('--csv-name', type=str, default=None, help='Custom CSV filename (stored under data/integrated_sensor)')
    ap.add_argument('--start-delay', type=float, default=0.0, help='Seconds to wait after opening session (after zero capture) before starting ramp')  # new
    ap.add_argument('--ramp-s', type=float, default=1.0, help='Seconds to perform smooth transitions between targets')
    return ap


def _run_once_with_cfg(cfg: IntegratedConfig, start_delay: float, ramp_s: float):
    sess = IntegratedSession(cfg)
    sess.open()
    # --- New: force center then capture encoder zero at center ---
    center = (cfg.min_pct + cfg.max_pct) / 2.0
    try:
        if sess.dac is not None:
            print(f"[INFO] Centering valves to {center:.1f}%/{center:.1f}% (A/B) for zero capture...", flush=True)
            sess.dac.set_channels(center, center)
    except Exception:
        pass
    hold_s = start_delay if start_delay > 0 else 0.6
    if hold_s > 0:
        print(f"[INFO] Holding center for {hold_s:.2f}s...", flush=True)
        t_end = time.time() + hold_s
        while time.time() < t_end:
            if cfg.encoder_enable and sess.enc is not None:
                try:
                    sess.enc.poll()
                except Exception:
                    pass
            time.sleep(0.02)
    if cfg.encoder_enable and cfg.encoder_zero_on_start and sess.enc is not None:
        try:
            for _ in range(5):
                sess.enc.poll(); time.sleep(0.02)
            val = sess.enc.degrees()
            if val is not None:
                if cfg.encoder_invert:
                    val = -val
                sess.enc_zero_offset = val
                print(f"[INFO] Encoder zero captured at center: {val:.3f} deg -> set to 0.000")
        except Exception:
            pass

    # Helper: smooth linear ramp between two valve setpoints
    def smooth_ramp(a_from: float, b_from: float, a_to: float, b_to: float, seconds: float):
        if sess.dac is None:
            return
        interval = max(1, cfg.interval_ms)
        steps = max(1, int(seconds * 1000.0 / interval))
        t0 = time.perf_counter()
        for i in range(1, steps + 1):
            frac = i / steps
            a = a_from + (a_to - a_from) * frac
            b = b_from + (b_to - b_from) * frac
            try:
                sess.dac.set_channels(a, b)
            except Exception:
                pass
            # Poll sensors for live log
            try:
                if sess.enc is not None:
                    sess.enc.poll()
                    cur = sess.enc.degrees()
                    if cur is not None:
                        if cfg.encoder_invert:
                            cur = -cur
                        if sess.enc_zero_offset is not None:
                            cur = cur - sess.enc_zero_offset
                        sess.last_enc_deg = cur
                sess._poll_between()
            except Exception:
                pass
            # print and write CSV entry using Session helpers
            elapsed = time.perf_counter() - t0
            try:
                sess._print_verbose(elapsed, a, b)
                sys.stdout.flush()
            except Exception:
                pass
            try:
                sess._write_csv(elapsed, a, b, sess.last_enc_deg)
            except Exception:
                pass
            time.sleep(interval / 1000.0)

    # perform the requested smooth transitions slowly
    print(f"[INFO] Smoothly ramping center -> min (first target) over {ramp_s:.2f}s", flush=True)
    first_a = cfg.min_pct
    first_b = cfg.max_pct
    # ramp from center to first target
    smooth_ramp(center, center, first_a, first_b, ramp_s)

    print("[INFO] Sequence: (60,60) -> (20,100) -> (100,20) -> (20,100)", flush=True)
    # Start the built-in u_ramp_roundtrip which will perform min->max->min
    for i, _ in enumerate(sess.run()):
        # flush periodically to ensure terminal shows activity
        if i % 1 == 0:
            try:
                sys.stdout.flush()
            except Exception:
                pass
        # tiny sleep to yield to I/O
        time.sleep(0)
        pass

    # --- after session finished: find CSV and plot if possible ---
    try:
        # import plotting libraries lazily (avoid blocking during run)
        try:
            import matplotlib.pyplot as plt_local
            import numpy as np_local
            plt = plt_local
            np = np_local
        except Exception:
            plt = None
            np = None
        # determine csv path: prefer cfg.csv_path, else pick latest file in data/integrated_sensor
        if cfg.csv_path:
            csvp = Path(cfg.csv_path)
        else:
            data_dir = Path('data/integrated_sensor')
            files = sorted(list(data_dir.glob('*.csv')), key=lambda p: p.stat().st_mtime) if data_dir.exists() else []
            csvp = files[-1] if files else None
        if csvp is None:
            print('[WARN] No CSV found to plot.', flush=True)
        else:
            print(f'[INFO] Plotting CSV: {csvp}', flush=True)
            if plt is None:
                print('[WARN] matplotlib not available; skipping plot.', flush=True)
            else:
                # Read CSV rows
                times = []
                a_vals = []
                b_vals = []
                enc_vals = []
                with open(csvp, 'r') as f:
                    rdr = csv.reader(f)
                    for row in rdr:
                        if not row:
                            continue
                        # skip non-numeric header lines
                        try:
                            ms = float(row[0])
                        except Exception:
                            continue
                        times.append(ms / 1000.0)
                        try:
                            a_vals.append(float(row[1]))
                        except Exception:
                            a_vals.append(float('nan'))
                        try:
                            b_vals.append(float(row[2]))
                        except Exception:
                            b_vals.append(float('nan'))
                        # encoder may be last column
                        try:
                            enc = float(row[-1])
                            enc_vals.append(enc)
                        except Exception:
                            # no encoder column
                            pass
                fig, ax1 = plt.subplots(figsize=(8, 4))
                if a_vals and b_vals:
                    ax1.plot(times, a_vals, label='A %', color='C0')
                    ax1.plot(times, b_vals, label='B %', color='C1')
                    ax1.set_ylabel('Valve %')
                    ax1.legend(loc='upper left')
                if enc_vals:
                    ax2 = ax1.twinx()
                    ax2.plot(times[:len(enc_vals)], enc_vals, label='Encoder deg', color='C2')
                    ax2.set_ylabel('Encoder (deg)')
                    # combine legends
                    lines1, labels1 = ax1.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                ax1.set_xlabel('Time (s)')
                # save
                out_png = csvp.with_suffix('.png')
                fig.tight_layout()
                fig.savefig(out_png, dpi=150)
                plt.close(fig)
                print(f'[INFO] Plot saved: {out_png}', flush=True)
    except Exception as e:
        print(f'[WARN] Plotting failed: {e}', flush=True)
    finally:
        # IntegratedSession exposes _cleanup() rather than a public close();
        # call _cleanup() if present, otherwise fall back to close().
        try:
            if hasattr(sess, '_cleanup'):
                sess._cleanup()
            elif hasattr(sess, 'close'):
                sess.close()
        except Exception:
            pass


def main():
    ap=build_parser(); args=ap.parse_args()
    # Derive differential command range from min/max if u0/u1 not provided
    span = float(args.max_pct) - float(args.min_pct)
    derived_u0 = -(span)
    derived_u1 = +(span)
    u0 = args.u0 if args.u0 is not None else derived_u0
    u1 = args.u1 if args.u1 is not None else derived_u1

    # Single continuous roundtrip: u0 -> u1 -> u0 within one session
    cfg = IntegratedConfig(min_pct=args.min_pct, max_pct=args.max_pct, interval_ms=args.interval_ms, total_s=args.duration,
                           mode='u_ramp_roundtrip', encoder_enable=args.encoder, encoder_invert=args.encoder_invert, encoder_zero_on_start=args.encoder_zero_on_start,
                           adc_enable=args.adc, adc_ch0=args.adc_ch0, adc_ch1=args.adc_ch1,
                           tension_enable=args.tension,
                           u_ramp_u0=u0, u_ramp_u1=u1, u_ramp_duration=args.duration,
                           csv_path=args.csv_name, verbose=True)
    print(f"[INFO] Running continuous roundtrip: {args.min_pct:.1f}% -> {args.max_pct:.1f}% -> {args.min_pct:.1f}% (u: {u0:.1f} -> {u1:.1f} -> {u0:.1f})")
    _run_once_with_cfg(cfg, args.start_delay, args.ramp_s)

    print('[INFO] Finished round-trip u_ramp scan.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())