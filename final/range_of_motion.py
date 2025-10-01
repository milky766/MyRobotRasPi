#!/usr/bin/env python3
#uv run python apps/u_ramp_scan.py --u0 -100 --u1 100 --duration 12 --start-delay 1 --encoder-zero-on-start --encoder --encoder-invert
from __future__ import annotations
"""Linear open-loop differential command ramp to map encoder (and optional sensors) vs u_diff.
Now supports bidirectional sweep (e.g. -100 -> +100). Logs:
  ms, valve_a_pct, valve_b_pct, q_des(blank), pid_u(u_diff), [adc...], [tension...], enc_deg
Use --adc / --tension to enable extra sensors.
This variant runs min_pct->max_pct then back to min_pct by performing two u_ramp passes.

uv run python apps/u_ramp_scan.py --u0 -100 --u1 100 --duration 12 --start-delay 1 --encoder-zero-on-start --encoder --encoder-invert
↑(a,b)=(20,100)→(100,20)→(20,100)と動かすコード。これで-20~50度は確実に可動域として確保できる
"""

import argparse, os, time  # added time
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
    return ap

def _run_once_with_cfg(cfg: IntegratedConfig, start_delay: float):
    sess = IntegratedSession(cfg)
    sess.open()
    if start_delay>0:
        print(f"[INFO] Start delay {start_delay:.2f}s before ramp...")
        t_end=time.time()+start_delay
        while time.time()<t_end:
            if cfg.encoder_enable and sess.enc is not None:
                try:
                    sess.enc.poll()
                except Exception:
                    pass
            time.sleep(0.02)
        if cfg.encoder_enable and sess.enc is not None and sess.enc_zero_offset is not None:
            try:
                sess.enc.poll()
                val = sess.enc.degrees()
                if val is not None:
                    if cfg.encoder_invert:
                        val = -val
                    if sess.enc_zero_offset is not None:
                        val = val - sess.enc_zero_offset
                    print(f"[INFO] Pre-actuation encoder (after delay, before first valve command): {val:.3f} deg")
            except Exception:
                pass
    for _ in sess.run():
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
    _run_once_with_cfg(cfg, args.start_delay)

    print('[INFO] Finished round-trip u_ramp scan.')
    return 0

if __name__=='__main__':
    raise SystemExit(main())