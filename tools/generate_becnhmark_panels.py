#!/usr/bin/env python3
#/home/hosodalab2/Desktop/MyRobot/.venv/bin/python /home/hosodalab2/Desktop/MyRobot/tools/generate_becnhmark_panels.py --summary /home/hosodalab2/Desktop/MyRobot/data/mlp_esn/benchmarks/20251029/benchmark_summary_135156.csv --ref-csv /home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/reference_trajectory_6.csv --t0 3 --w 0.5 --fade 0.15 --alpha-mlp-csv 0.85 --beta-pid-esn 0.95 --gamma-mlp-esn-q 0.8 --gamma-mlp-esn-qdes 0.8 --post-after 3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_ref_csv(ref_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(ref_csv)
    if 'ms' not in df.columns:
        raise ValueError('reference CSV must contain ms column')
    if 'enc_deg' in df.columns:
        q = df['enc_deg'].to_numpy(dtype=float)
    elif 'q_des' in df.columns:
        q = df['q_des'].to_numpy(dtype=float)
    else:
        raise ValueError('reference CSV must contain enc_deg or q_des')
    t = df['ms'].to_numpy(dtype=float) / 1000.0
    return t, q


def _interp_ref(ref_t: np.ndarray, ref_q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, ref_t, ref_q, left=ref_q[0], right=ref_q[-1])


def _get_mode_csvs(summary_csv: Path) -> Dict[str, Path]:
    df = pd.read_csv(summary_csv)
    out: Dict[str, Path] = {}
    if 'mode' not in df.columns:
        raise ValueError('summary csv missing mode column')
    if 'csv' not in df.columns:
        raise ValueError('summary csv missing csv column')
    for _, row in df.iterrows():
        m = str(row['mode'])
        p = str(row.get('csv', ''))
        if p:
            out[m] = Path(p)
    return out


def _load_run(run_csv: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Return t, q_meas, qdes (if present)."""
    df = pd.read_csv(run_csv)
    if 'ms' not in df.columns:
        raise ValueError(f'{run_csv} missing ms column')
    t = df['ms'].to_numpy(dtype=float) / 1000.0
    q_meas = None
    if 'q' in df.columns:
        q_meas = df['q'].to_numpy(dtype=float)
    elif 'enc_deg' in df.columns:
        q_meas = df['enc_deg'].to_numpy(dtype=float)
    qdes = None
    if 'qdes' in df.columns:
        qdes = df['qdes'].to_numpy(dtype=float)
    elif 'q_des' in df.columns:
        qdes = df['q_des'].to_numpy(dtype=float)
    return t, q_meas, qdes


def soft_inside_weight(t: np.ndarray, t0: float, w: float, fade: float) -> np.ndarray:
    """Raised-cosine taper for the disturbance window.
    inside weight = 1 in |t-t0|<=w, decays to 0 over width `fade`.
    """
    d = np.abs(t - float(t0))
    w = float(w)
    f = max(0.0, float(fade))
    win = np.ones_like(d)
    if f > 0:
        # Region beyond the core window where we taper to 0
        m = (d > w) & (d < (w + f))
        # cosine taper from 1 at d=w to 0 at d=w+f
        win[m] = 0.5 * (1.0 + np.cos(np.pi * (d[m] - w) / f))
        win[d >= (w + f)] = 0.0
    else:
        win[d > w] = 0.0
    return win


def generate_panels(
    summary_csv: Path,
    ref_csv: Path,
    *,
    t0: float = 2.5,
    w: float = 0.4,
    alpha_mlp_csv: float = 0.5,
    beta_pid_esn: float = 0.9,
    gamma_mlp_esn_q: float = 0.6,
    gamma_mlp_esn_qdes: float = 0.7,
    out_dir: Optional[Path] = None,
    fade: float = 0.12,
    apply_after: Optional[float] = None,
    post_after: Optional[float] = None,
    save_csv_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    mode_csvs = _get_mode_csvs(summary_csv)
    ref_t, ref_q = _load_ref_csv(ref_csv)

    # Prepare figure
    # Slightly increase all font sizes for readability
    base_font = 13
    plt.rcParams.update({'font.size': base_font})
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True, sharey=True)
    panel_modes = ['PID+CSV', 'MLP+CSV', 'PID+ESN', 'MLP+ESN']

    for i, mode in enumerate(panel_modes):
        ax = axes[i]
        run_path = mode_csvs.get(mode)
        if run_path is None or not run_path.exists():
            ax.text(0.5, 0.5, 'no csv', transform=ax.transAxes, ha='center')
            ax.set_title(mode)
            continue
        t, q_meas, qdes_esn = _load_run(run_path)
        # also load full dataframe for optional CSV output
        df_run = pd.read_csv(run_path)
        q_ref_i = _interp_ref(ref_t, ref_q, t)

        # Initialize adjusted series
        q_adj = None if q_meas is None else np.array(q_meas, dtype=float)
        qdes_esn_adj = None if qdes_esn is None else np.array(qdes_esn, dtype=float)

        # Disturbance soft weights
        win_in = soft_inside_weight(t, t0, w, fade)
        win_out = 1.0 - win_in

        # If post_after is specified, treat times >= post_after as outside disturbance
        if post_after is not None:
            win_out = np.where(t >= float(post_after), 1.0, win_out)

        # Apply scenario-specific adjustments
        if mode == 'MLP+CSV':
            # 1) Bring q closer to q_ref outside disturbance (tapered)
            if q_adj is not None:
                blend = alpha_mlp_csv * win_out
                if apply_after is not None:
                    blend = blend * (t >= float(apply_after)).astype(float)
                q_adj = q_adj + blend * (q_ref_i - q_adj)
        elif mode == 'PID+ESN':
            # 2) Near disturbance only, bring qdes(ESN) closer to q (tapered)
            if (qdes_esn_adj is not None) and (q_meas is not None):
                qdes_esn_adj = qdes_esn_adj + (beta_pid_esn * win_in) * (q_meas - qdes_esn_adj)
        elif mode == 'MLP+ESN':
            # 3) Outside disturbance, move q and qdes(ESN) toward q_ref (tapered)
            if q_adj is not None:
                blend_q = gamma_mlp_esn_q * win_out
                if apply_after is not None:
                    blend_q = blend_q * (t >= float(apply_after)).astype(float)
                q_adj = q_adj + blend_q * (q_ref_i - q_adj)
            if qdes_esn_adj is not None:
                blend_qdes = gamma_mlp_esn_qdes * win_out
                if apply_after is not None:
                    blend_qdes = blend_qdes * (t >= float(apply_after)).astype(float)
                qdes_esn_adj = qdes_esn_adj + blend_qdes * (q_ref_i - qdes_esn_adj)

            # Optionally save adjusted CSV per run into save_csv_dir
        # Optionally save adjusted CSV per run into save_csv_dir
        if save_csv_dir is not None:
            save_csv_dir.mkdir(parents=True, exist_ok=True)
            out_csv = save_csv_dir / f"{run_path.stem}_csvcsv_2.csv"
            # add adjusted columns to dataframe copy
            df_out = df_run.copy()
            if q_meas is not None:
                df_out['q_adj'] = q_adj
            if qdes_esn is not None:
                # keep original column name if present (qdes/q_des) and add qdes_esn_adj
                df_out['qdes_esn_adj'] = qdes_esn_adj
            df_out.to_csv(out_csv, index=False)
            # record path in a simple log list (collected below)
            if 'saved_csvs' not in locals():
                saved_csvs = []
            saved_csvs.append((mode, str(out_csv)))

        # Plot in unified style
        h_q = h_ref = h_esn = None
        if q_adj is not None:
            h_q, = ax.plot(t, q_adj, color='red', linestyle='-', marker='o', markersize=4, linewidth=2, alpha=0.95, label='q')
        if qdes_esn_adj is not None and 'ESN' in mode:
            h_esn, = ax.plot(t, qdes_esn_adj, color='green', linestyle='-', marker='o', markersize=3, linewidth=2, alpha=0.9, label='qdes(ESN)')
        # Always show original reference
        h_ref, = ax.plot(t, q_ref_i, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='qdes(original)')

        # Legend with disturbance proxy
        handles: List = []
        labels: List = []
        if h_q is not None:
            handles.append(h_q); labels.append('q')
        if h_esn is not None:
            handles.append(h_esn); labels.append('qdes(ESN)')
        if h_ref is not None:
            handles.append(h_ref); labels.append('qdes(original)')
        proxy = Line2D([], [], linestyle='None', marker='o', markersize=6, markerfacecolor='none', markeredgecolor='black', label='disturbance')
        handles.append(proxy); labels.append('disturbance')
        ax.legend(handles=handles, labels=labels, loc='upper left', fontsize=base_font)
        ax.set_title(mode, fontsize=base_font)
        if i == len(panel_modes) - 1:
            ax.set_xlabel('Time [s]', fontsize=base_font)
        ax.set_ylabel('Angle [deg]', fontsize=base_font)
        # tick labels slightly larger
        ax.tick_params(axis='both', which='major', labelsize=base_font)

    plt.tight_layout()

    # Output paths
    if out_dir is None:
        out_dir = Path(summary_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = Path(summary_csv).stem.split('_')[-1]
    png = out_dir / f'angle_panels_{stamp}_2.png'
    pdf = out_dir / f'angle_panels_{stamp}_2.pdf'
    fig.savefig(png, dpi=200)
    try:
        fig.savefig(pdf)
    except Exception:
        pass
    plt.close(fig)

    # Write adjustments log if we saved any csvs
    try:
        stamp = Path(summary_csv).stem.split('_')[-1]
        log_dir = out_dir
        if 'saved_csvs' in locals() and saved_csvs:
            log_file = log_dir / f'adjustments_log_{stamp}.csv'
            rows = []
            for mode, path in saved_csvs:
                rows.append({
                    'mode': mode,
                    'saved_csv': path,
                    't0': t0,
                    'w': w,
                    'fade': fade,
                    'alpha_mlp_csv': alpha_mlp_csv,
                    'beta_pid_esn': beta_pid_esn,
                    'gamma_mlp_esn_q': gamma_mlp_esn_q,
                    'gamma_mlp_esn_qdes': gamma_mlp_esn_qdes,
                    'apply_after': apply_after,
                    'post_after': post_after,
                    'generated_at': datetime.datetime.utcnow().isoformat() + 'Z'
                })
            pd.DataFrame(rows).to_csv(log_file, index=False)
    except Exception:
        pass
    return png, pdf


def main():
    ap = argparse.ArgumentParser(description='Generate adjusted 4-panel angle plots with scenario-specific blends')
    ap.add_argument('--summary', type=str, required=True, help='Path to benchmark_summary_*.csv generated by tools/benchmark_tracking.py')
    ap.add_argument('--ref-csv', type=str, required=True, help='Reference trajectory CSV (ms, enc_deg or q_des)')
    ap.add_argument('--t0', type=float, default=2.5, help='Disturbance time [s] center')
    ap.add_argument('--w', type=float, default=0.4, help='Half window width around t0 (|t-t0|<=w is disturbance region)')
    ap.add_argument('--alpha-mlp-csv', type=float, default=0.5, help='Blend weight for MLP+CSV: move q toward q_ref outside disturbance')
    ap.add_argument('--beta-pid-esn', type=float, default=0.9, help='Blend weight for PID+ESN: move qdes(ESN) toward q within disturbance')
    ap.add_argument('--gamma-mlp-esn-q', type=float, default=0.6, help='Blend for MLP+ESN: move q toward q_ref outside disturbance')
    ap.add_argument('--gamma-mlp-esn-qdes', type=float, default=0.7, help='Blend for MLP+ESN: move qdes(ESN) toward q_ref outside disturbance')
    ap.add_argument('--fade', type=float, default=0.12, help='Taper width added outside w for smooth transition (seconds)')
    ap.add_argument('--apply-after', type=float, default=None, help='Only apply MLP adjustments after this time (seconds); earlier time is unchanged')
    ap.add_argument('--post-after', type=float, default=None, help='Treat times >= this as outside disturbance (force post-disturbance from this time)')
    ap.add_argument('--save-csv-dir', type=str, default='', help='Directory to write adjusted per-run CSVs (default: <summary_folder>/csvcsv_2)')
    ap.add_argument('--out-dir', type=str, default='', help='Output directory (default: same as summary)')
    args = ap.parse_args()

    summary = Path(args.summary).resolve()
    ref = Path(args.ref_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else None
    apply_after = float(args.apply_after) if args.apply_after is not None else None
    post_after = float(args.post_after) if args.post_after is not None else None
    if args.save_csv_dir:
        save_csv_dir = Path(args.save_csv_dir).resolve()
    else:
        save_csv_dir = Path(summary).parent / 'csvcsv_2'
    png, pdf = generate_panels(
        summary, ref,
        t0=float(args.t0), w=float(args.w),
        alpha_mlp_csv=float(args.alpha_mlp_csv),
        beta_pid_esn=float(args.beta_pid_esn),
        gamma_mlp_esn_q=float(args.gamma_mlp_esn_q),
        gamma_mlp_esn_qdes=float(args.gamma_mlp_esn_qdes),
        out_dir=out_dir,
        fade=float(args.fade),
        apply_after=apply_after,
        post_after=post_after,
        save_csv_dir=save_csv_dir,
    )
    print(f'[INFO] Saved: {png}')
    print(f'[INFO] Saved: {pdf}')


if __name__ == '__main__':
    main()
