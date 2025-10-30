#!/usr/bin/env python3
#/home/hosodalab2/Desktop/MyRobot/.venv/bin/python -u /home/hosodalab2/Desktop/MyRobot/tools/benchmark_tracking.py --ref-csv /home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/reference_trajectory_6.csv --esn-weights /home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/Robot_ESN/data/esn_optuna/traj6/20251026/015_weight.npy --esn-params /home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/Robot_ESN/data/esn_optuna/traj6/20251026/015_params.csv --duration 4 --python /home/hosodalab2/Desktop/MyRobot/.venv/bin/python
from __future__ import annotations
import argparse
import os
import sys
import subprocess
import shlex
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / 'MyRobot_RasPi_Desktop_Mix' / 'desktop' / 'yuma3' / 'Robot_ESN' / 'apps' / 'mlp_esn.py'
PYTHON_BIN = PROJECT_ROOT / '.venv' / 'bin' / 'python'
DEFAULT_PYTHONPATH = [
    str(PROJECT_ROOT),
    str(PROJECT_ROOT / 'MyRobot_RasPi_Desktop_Mix'),
    str(PROJECT_ROOT / 'MyRobot_RasPi_Desktop_Mix' / 'desktop' / 'yuma'),
]


def _latest_csv(dir_path: Path) -> Optional[Path]:
    try:
        files = sorted([p for p in dir_path.glob('*.csv')], key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    except Exception:
        return None


def _parse_csv_path_from_log(text: str) -> Optional[str]:
    for line in text.splitlines():
        if 'CSV saved:' in line:
            try:
                return line.split('CSV saved:', 1)[1].strip()
            except Exception:
                pass
    return None


def _load_reference(ref_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    # return (t_sec, q_ref_deg)
    df = pd.read_csv(ref_csv)
    if 'ms' not in df.columns:
        raise ValueError('reference CSV must contain ms column')
    # prefer enc_deg, else q_des
    if 'enc_deg' in df.columns:
        q = df['enc_deg'].to_numpy(dtype=float)
    elif 'q_des' in df.columns:
        q = df['q_des'].to_numpy(dtype=float)
    else:
        raise ValueError('reference CSV must contain enc_deg or q_des')
    t = df['ms'].to_numpy(dtype=float) / 1000.0
    return t, q


def _interp_ref(ref_t: np.ndarray, ref_q: np.ndarray, t: np.ndarray) -> np.ndarray:
    # simple 1D linear interpolation with edge hold
    return np.interp(t, ref_t, ref_q, left=ref_q[0], right=ref_q[-1])


def compute_metrics(run_csv: Path, ref_csv: Path) -> Dict[str, Any]:
    try:
        t_ref, q_ref = _load_reference(ref_csv)
        df = pd.read_csv(run_csv)
        if 'ms' not in df.columns:
            raise ValueError('run CSV missing ms column')
        # measured angle column name
        if 'q' in df.columns:
            q_meas = df['q'].to_numpy(dtype=float)
        elif 'enc_deg' in df.columns:
            q_meas = df['enc_deg'].to_numpy(dtype=float)
        else:
            raise ValueError('run CSV missing q (or enc_deg) column')
        t = df['ms'].to_numpy(dtype=float) / 1000.0
        # interpolate reference on run timeline
        q_ref_i = _interp_ref(t_ref, q_ref, t)
        err = q_ref_i - q_meas
        metrics = {
            'rmse': float(np.sqrt(np.nanmean(np.square(err)))),
            'mae': float(np.nanmean(np.abs(err))),
            'max_abs_err': float(np.nanmax(np.abs(err))),
            'p95_abs_err': float(np.nanpercentile(np.abs(err), 95)),
            'n': int(len(err)),
        }
        return metrics
    except Exception as e:
        return {'error': str(e)}


def run_mode(mode: str, base_env: Dict[str, str], ref_csv: Path, esn_weights: Path, esn_params: Optional[Path], duration: float, extra_args: List[str]) -> Tuple[Optional[Path], str]:
    args = [str(PYTHON_BIN), '-u', str(APP_PATH)]
    # common
    args += ['--duration', str(duration)]
    # provide reference path for overlay and CSV mode
    args += ['--trajectory-csv', str(ref_csv)]
    # ESN artifacts
    if esn_weights:
        args += ['--esn-weights', str(esn_weights)]
    if esn_params:
        args += ['--esn-params', str(esn_params)]
    # warmup steps (keep small)
    args += ['--esn-warmup-steps', '5']

    label = mode
    if mode == 'MLP+ESN':
        pass  # default
    elif mode == 'PID+ESN':
        args += ['--use-pid']
    elif mode == 'MLP+CSV':
        args += ['--use-csv-reference']
    elif mode == 'PID+CSV':
        args += ['--use-pid', '--use-csv-reference']
    else:
        raise ValueError(f'unknown mode: {mode}')

    # allow caller to inject extra args, e.g., PID gains
    if extra_args:
        args += extra_args

    cmd_str = ' '.join(shlex.quote(a) for a in args)
    env = os.environ.copy()
    env.update(base_env)

    print(f"\n[RUN] {label}: {cmd_str}")
    # Stream child output so cues appear in real-time
    logged_lines: List[str] = []
    try:
        proc = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end='')  # passthrough
            logged_lines.append(line)
        proc.wait()
        ret = proc.returncode
    except Exception as e:
        print(f"[ERROR] Failed to run child process: {e}")
        return None, 'spawn error'
    if ret != 0:
        return None, f'nonzero exit {ret}'

    # try parse csv path from captured stdout
    joined = ''.join(logged_lines)
    csv_path_str = _parse_csv_path_from_log(joined)
    if not csv_path_str:
        # fallback to latest csv under data/mlp_esn/csv/<today>
        date_str = datetime.now().strftime('%Y%m%d')
        csv_dir = PROJECT_ROOT / 'data' / 'mlp_esn' / 'csv' / date_str
        csv_path = _latest_csv(csv_dir)
    else:
        csv_path = Path(csv_path_str)

    return csv_path, ''


def main():
    ap = argparse.ArgumentParser(description='Benchmark 4 controller modes (tracking) vs reference CSV')
    ap.add_argument('--ref-csv', type=str, required=True, help='reference_trajectory_x.csv (ms, enc_deg)')
    ap.add_argument('--esn-weights', type=str, required=True, help='ESN weights .npy for ESN modes')
    ap.add_argument('--esn-params', type=str, default='', help='ESN params .csv for ESN modes')
    ap.add_argument('--duration', type=float, default=10.0)
    ap.add_argument('--kp', type=float, default=0.15)
    ap.add_argument('--ki', type=float, default=0.25)
    ap.add_argument('--kd', type=float, default=0.01)
    ap.add_argument('--esn-max-step', type=float, default=None, help='Pass --esn-max-step to mlp_esn (deg)')
    ap.add_argument('--out-dir', type=str, default='data/mlp_esn/benchmarks')
    # One-shot disturbance cue passthrough to mlp_esn
    ap.add_argument('--disturbance-cue-sec', type=float, default=None, help='If set, forward to mlp_esn to print a cue once at this time [s]')
    ap.add_argument('--disturbance-cue-text', type=str, default='DISTURB NOW', help='Cue text to forward to mlp_esn')
    # Do not reference PYTHON_BIN here because main() may rebind it; use None and assign below
    ap.add_argument('--python', type=str, default=None, help='Python executable to run mlp_esn (overrides bundled venv)')
    ap.add_argument('--dry', action='store_true', help='Only compute metrics for existing latest CSVs (no runs)')
    args = ap.parse_args()

    global PYTHON_BIN
    if args.python:
        PYTHON_BIN = Path(args.python)

    ref_csv = Path(args.ref_csv).resolve()
    esn_w = Path(args.esn_weights).resolve()
    esn_p = Path(args.esn_params).resolve() if args.esn_params else None

    # build env PYTHONPATH like user examples
    base_env = {
        'PYTHONPATH': ':'.join(DEFAULT_PYTHONPATH),
        # ensure plot overlays original reference
        'PLOT_REF_CSV': str(ref_csv),
        # include title with mode in plots
    }

    modes = ['PID+CSV', 'MLP+CSV', 'PID+ESN', 'MLP+ESN']
    results: List[Dict[str, Any]] = []
    csv_paths: Dict[str, Optional[Path]] = {}

    # prepare out dir
    date_str = datetime.now().strftime('%Y%m%d')
    out_root = PROJECT_ROOT / args.out_dir / date_str
    out_root.mkdir(parents=True, exist_ok=True)

    if args.dry:
        # pick latest 4 CSVs by name prefix if available
        csv_dir = PROJECT_ROOT / 'data' / 'mlp_esn' / 'csv' / date_str
        latest = sorted(csv_dir.glob('mlp_esn_*.csv'))[-4:]
        for m, p in zip(modes, latest):
            csv_paths[m] = p
    else:
        # run all modes
        pid_gains_extra = ['--kp', str(args.kp), '--ki', str(args.ki), '--kd', str(args.kd)]
        # common extras applied to all modes (e.g., esn-max-step, cue)
        common_extra: List[str] = []
        if args.esn_max_step is not None:
            common_extra += ['--esn-max-step', str(args.esn_max_step)]
        # disturbance cue passthrough
        if args.disturbance_cue_sec is not None:
            common_extra += ['--disturbance-cue-sec', str(args.disturbance_cue_sec)]
            if args.disturbance_cue_text:
                common_extra += ['--disturbance-cue-text', str(args.disturbance_cue_text)]
        for mode in modes:
            ex = list(common_extra) + (pid_gains_extra if 'PID' in mode else [])
            # annotate plot title
            base_env['PLOT_TITLE'] = f'{mode} tracking benchmark'
            csv_path, err = run_mode(mode, base_env, ref_csv, esn_w, esn_p, args.duration, ex)
            csv_paths[mode] = csv_path
            if err:
                print(f'[WARN] {mode}: {err}')


    # compute metrics
    for mode in modes:
        p = csv_paths.get(mode)
        if not p or not p.exists():
            results.append({'mode': mode, 'error': 'no csv'})
            continue
        m = compute_metrics(p, ref_csv)
        m.update({'mode': mode, 'csv': str(p)})
        results.append(m)

    # save summary
    summary_path = out_root / f'benchmark_summary_{datetime.now().strftime("%H%M%S")}.csv'
    cols = ['mode', 'rmse', 'mae', 'max_abs_err', 'p95_abs_err', 'n', 'csv']
    df = pd.DataFrame(results)
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(summary_path, index=False)
    print(f'[INFO] Summary saved: {summary_path}')

    # also write JSON
    json_path = summary_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[INFO] JSON saved: {json_path}')

    # Create combined angle-only overlay plot for all modes (if CSVs available)
    try:
        overlay_png = out_root / f'benchmark_overlay_{datetime.now().strftime("%H%M%S")}.png'
        overlay_pdf = overlay_png.with_suffix('.pdf')
        plt.figure(figsize=(10, 4))
        # plot reference first
        try:
            ref_t, ref_q = _load_reference(ref_csv)
            plt.plot(ref_t, ref_q, 'k--', label='reference')
        except Exception:
            pass
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        for i, mode in enumerate(modes):
            p = csv_paths.get(mode)
            if not p:
                continue
            p = Path(p)
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p)
                if 'ms' not in df.columns:
                    continue
                t = df['ms'].to_numpy(dtype=float) / 1000.0
                if 'q' in df.columns:
                    q = df['q'].to_numpy(dtype=float)
                elif 'enc_deg' in df.columns:
                    q = df['enc_deg'].to_numpy(dtype=float)
                elif 'q_des' in df.columns:
                    q = df['q_des'].to_numpy(dtype=float)
                else:
                    continue
                plt.plot(t, q, label=mode, color=colors[i % len(colors)], linewidth=1.4)
            except Exception:
                continue
        plt.xlabel('Time [s]')
        plt.ylabel('Angle [deg]')
        plt.title('Benchmark overlay: Angle [deg]')
        plt.legend(loc='upper left')
        # No grid per user request
        plt.tight_layout()
        plt.savefig(overlay_png, dpi=200)
        plt.savefig(overlay_pdf, dpi=200)
        print(f'[INFO] Combined overlay saved: {overlay_png}')
    except Exception as e:
        print(f'[WARN] Failed to create combined overlay: {e}')

    # Create 4-panel angle-only plots: each subplot shows measured q and original q_des;
    # for ESN modes also plot q_des (from ESN) if available in the run CSV ('qdes' column).
    try:
        panel_png = out_root / f'benchmark_angle_panels_{datetime.now().strftime("%H%M%S")}.png'
        panel_pdf = panel_png.with_suffix('.pdf')
        # Create vertical 4-panel layout as requested (PID+CSV, MLP+CSV, PID+ESN, MLP+ESN)
        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True, sharey=True)
        # axes is a 1-D array with length 4

        # load reference once
        try:
            ref_t, ref_q = _load_reference(ref_csv)
        except Exception:
            ref_t, ref_q = None, None

        panel_modes = ['PID+CSV', 'MLP+CSV', 'PID+ESN', 'MLP+ESN']
        for i, mode in enumerate(panel_modes):
            ax = axes[i]
            p = csv_paths.get(mode)
            if not p:
                ax.text(0.5, 0.5, 'no csv', transform=ax.transAxes, ha='center')
                ax.set_title(mode)
                continue
            p = Path(p)
            if not p.exists():
                ax.text(0.5, 0.5, 'missing file', transform=ax.transAxes, ha='center')
                ax.set_title(mode)
                continue
            try:
                df = pd.read_csv(p)
                if 'ms' not in df.columns:
                    ax.text(0.5, 0.5, 'no ms column', transform=ax.transAxes, ha='center')
                    ax.set_title(mode)
                    continue
                t = df['ms'].to_numpy(dtype=float) / 1000.0
                # measured angle
                if 'q' in df.columns:
                    q_meas = df['q'].to_numpy(dtype=float)
                elif 'enc_deg' in df.columns:
                    q_meas = df['enc_deg'].to_numpy(dtype=float)
                else:
                    q_meas = None

                # original reference interpolated to run timeline
                q_ref_i = None
                if ref_t is not None and ref_q is not None:
                    q_ref_i = _interp_ref(ref_t, ref_q, t)

                # ESN-produced q_des (present when mode uses ESN)
                q_esn = None
                if 'ESN' in mode and 'qdes' in df.columns:
                    q_esn = df['qdes'].to_numpy(dtype=float)

                # plot lines and capture handles
                h_q = h_esn = h_ref = None
                if q_meas is not None:
                    h_q, = ax.plot(t, q_meas, color='red', linestyle='-', marker='o', markersize=4, linewidth=2, alpha=0.95, label='q')
                if q_esn is not None:
                    h_esn, = ax.plot(t, q_esn, color='green', linestyle='-', marker='o', markersize=3, linewidth=2, alpha=0.9, label='qdes(ESN)')
                if q_ref_i is not None:
                    h_ref, = ax.plot(t, q_ref_i, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='qdes(original)')

                ax.set_title(mode)
                # match plotting style from mlp_esn.py: measured as black solid, reference dashed black,
                # ESN q_des dotted blue. Label axes (only bottom subplot gets x-label in a stacked layout).
                if i == len(panel_modes) - 1:
                    ax.set_xlabel('Time [s]')
                ax.set_ylabel('Angle [deg]')
                # No grid
                # Build legend in specific order and add a proxy 'disturbance' entry (legend only)
                handles: list = []
                labels: list = []
                if h_q is not None:
                    handles.append(h_q); labels.append('q')
                if h_esn is not None:
                    handles.append(h_esn); labels.append('qdes(ESN)')
                if h_ref is not None:
                    handles.append(h_ref); labels.append('qdes(original)')
                # disturbance proxy: hollow circle, no line
                disturbance_handle = Line2D([], [], linestyle='None', marker='o', markersize=6, markerfacecolor='none', markeredgecolor='black', label='disturbance')
                handles.append(disturbance_handle); labels.append('disturbance')
                ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=9)
            except Exception:
                ax.text(0.5, 0.5, 'plot error', transform=ax.transAxes, ha='center')
                ax.set_title(mode)

        # hide any unused axes (in case modes < 4)
        for j in range(len(modes), 4):
            axes[j].axis('off')

        plt.tight_layout()
        fig.savefig(panel_png, dpi=200)
        fig.savefig(panel_pdf, dpi=200)
        print(f'[INFO] 4-panel angle-only plot saved: {panel_png}')
    except Exception as e:
        print(f'[WARN] Failed to create 4-panel angle-only plot: {e}')


if __name__ == '__main__':
    main()
