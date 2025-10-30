#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate angle-only plot (q, qdes(ESN), qdes(original) if provided) from a CSV file.
Usage: generate_angle_only_from_csv.py --csv <run_csv> [--reference <ref_csv>]
Saves: <csv_basename>_angle.png and .pdf next to mlp_esn graph folder convention.

/home/hosodalab2/Desktop/MyRobot/.venv/bin/python /home/hosodalab2/Desktop/MyRobot/tools/generate_angle_only_from_csv.py --csv /home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/Robot_ESN/data/mlp_esn/csv/20251026/mlp_esn_193555_2.csv --reference /home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/reference_trajectory_6.csv

"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', required=True)
    ap.add_argument('--reference', required=False, default='')
    args = ap.parse_args()
    src = Path(args.csv)
    # prepare graph folder mirroring mlp_esn convention
    parts = list(src.parts)
    # try to swap 'csv' to 'graph' if present
    if 'csv' in parts:
        new_parts = []
        swapped = False
        for p in parts:
            if not swapped and p == 'csv':
                new_parts.append('graph')
                swapped = True
            else:
                new_parts.append(p)
        # Build output filename by taking original stem and appending '_angle.png'
        orig = Path(*new_parts)
        out_png = orig.with_name(orig.stem + '_angle.png')
    else:
        out_png = src.with_name(src.stem + '_angle.png')
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix('.pdf')

    # read reference if provided
    ref_sec = []
    ref_enc = []
    if args.reference:
        try:
            with open(args.reference, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    try:
                        ms = float(r.get('ms', 0.0))
                        enc = r.get('enc_deg', r.get('q_des', r.get('enc', '')))
                        ref_sec.append(ms/1000.0)
                        ref_enc.append(float(enc))
                    except Exception:
                        continue
        except Exception:
            ref_sec = []
            ref_enc = []

    run_sec = []
    run_qdes = []
    run_q = []
    try:
        with open(src, 'r', encoding='utf-8') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                try:
                    run_sec.append(float(r.get('ms', 0.0))/1000.0)
                    qd = r.get('qdes', r.get('q_des', ''))
                    run_qdes.append(float(qd) if qd != '' else float('nan'))
                    qm = r.get('q', r.get('enc_deg', r.get('enc', '')))
                    run_q.append(float(qm) if qm != '' else float('nan'))
                except Exception:
                    run_sec.append(float('nan'))
                    run_qdes.append(float('nan'))
                    run_q.append(float('nan'))
    except Exception as e:
        print(f'[ERROR] Failed to read run CSV: {e}')
        return 2

    try:
        fig, ax = plt.subplots(figsize=(8, 3))
        h_ref = None
        h_esn = None
        h_q = None
        if ref_sec and ref_enc:
            h_ref, = ax.plot(ref_sec, ref_enc, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='qdes(original)')
        if run_sec and run_qdes:
            h_esn, = ax.plot(run_sec, run_qdes, color='green', linestyle='-', marker='o', markersize=3, linewidth=2, alpha=0.9, label='qdes(ESN)')
        if run_sec and run_q:
            h_q, = ax.plot(run_sec, run_q, color='red', linestyle='-', marker='o', markersize=4, linewidth=2, alpha=0.95, label='q')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Angle [deg]')
        handles = []
        labels = []
        if h_q is not None:
            handles.append(h_q); labels.append('q')
        if h_esn is not None:
            handles.append(h_esn); labels.append('qdes(ESN)')
        if h_ref is not None:
            handles.append(h_ref); labels.append('qdes(original)')
        # Add disturbance legend entry: hollow circle marker (no connecting line)
        import matplotlib.lines as mlines
        disturbance_line = mlines.Line2D([], [], color='black', marker='o', markerfacecolor='none', markeredgecolor='black', linestyle='None', markersize=6, label='disturbance')
        handles.append(disturbance_line)
        labels.append('disturbance')
        if handles:
            ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=9)
        plt.tight_layout()
        fig.savefig(out_png, dpi=150)
        try:
            fig.savefig(out_pdf)
        except Exception:
            pass
        plt.close(fig)
        print(f'[INFO] Angle-only plot saved: {out_png}')
        print(f'[INFO] Angle-only plot (PDF) saved: {out_pdf}')
    except Exception as e:
        print(f'[ERROR] Failed to generate angle plot: {e}')
        return 3
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
