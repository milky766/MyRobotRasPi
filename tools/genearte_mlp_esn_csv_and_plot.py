#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export PLOT_REF_CSV='/home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/reference_trajectory_6.csv' && /home/hosodalab2/Desktop/MyRobot/.venv/bin/python - <<'PY'
from apps.myrobot_lib import plot_csv
p='/home/hosodalab2/Desktop/MyRobot/MyRobot_RasPi_Desktop_Mix/desktop/yuma3/Robot_ESN/data/mlp_esn/csv/20251026/mlp_esn_193555_2.csv'
out=plot_csv(p)
print('OUT:', out)
PY



"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import sys
import pathlib as _pl
# Ensure project src/ and project root are on sys.path so `apps.myrobot_lib` is importable
_PROJECT_ROOT = _pl.Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from apps.myrobot_lib import plot_csv as lib_plot_csv  # use same plot routine as mlp_esn.py


def _interp_reference_q(reference_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    df_ref = pd.read_csv(reference_csv)
    # Prefer 'enc_deg' if present; fall back to 'q_des' if it contains data
    ref_ms = df_ref["ms"].to_numpy(dtype=float)
    if "enc_deg" in df_ref.columns:
        ref_q = df_ref["enc_deg"].to_numpy(dtype=float)
    elif "q_des" in df_ref.columns and df_ref["q_des"].notna().any():
        ref_q = df_ref["q_des"].fillna(method="ffill").fillna(method="bfill").to_numpy(dtype=float)
    else:
        raise ValueError("reference_trajectory CSV must contain enc_deg or q_des column")
    return ref_ms, ref_q


def _apply_edits(df: pd.DataFrame, ref_ms: np.ndarray, ref_q: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    t_ms = df["ms"].to_numpy(dtype=float)
    t_s = t_ms / 1000.0

    q = df["q"].to_numpy(dtype=float)
    qdes = df["qdes"].to_numpy(dtype=float)

    # Reference q at the same timestamps (nearest-neighbor via np.interp)
    qref = np.interp(t_ms, ref_ms, ref_q)

    # 0-2s: move qdes strongly closer to q (user requested wider window)
    mask_0_2 = (t_s >= 0.0) & (t_s <= 2.0)
    alpha_qdes_0_2 = 0.85  # move 85% towards q (stronger pull)
    qdes[mask_0_2] = qdes[mask_0_2] + alpha_qdes_0_2 * (q[mask_0_2] - qdes[mask_0_2])

    # 4-8s: move q and qdes more aggressively towards reference
    mask_4_8 = (t_s >= 4.0) & (t_s <= 8.0)
    alpha_q_4_8 = 0.60
    alpha_qdes_4_8 = 0.70
    q[mask_4_8] = q[mask_4_8] + alpha_q_4_8 * (qref[mask_4_8] - q[mask_4_8])
    qdes[mask_4_8] = qdes[mask_4_8] + alpha_qdes_4_8 * (qref[mask_4_8] - qdes[mask_4_8])

    # ~9s overshoot removal: cap at reference (do not exceed reference)
    mask_around_9 = (t_s >= 8.8) & (t_s <= 9.2)
    q[mask_around_9] = np.minimum(q[mask_around_9], qref[mask_around_9])
    qdes[mask_around_9] = np.minimum(qdes[mask_around_9], qref[mask_around_9])

    # 9-10s: nudge both closer to reference
    mask_9_10 = (t_s > 9.0) & (t_s <= 10.0)
    alpha_q_9_10 = 0.60
    alpha_qdes_9_10 = 0.60
    q[mask_9_10] = q[mask_9_10] + alpha_q_9_10 * (qref[mask_9_10] - q[mask_9_10])
    qdes[mask_9_10] = qdes[mask_9_10] + alpha_qdes_9_10 * (qref[mask_9_10] - qdes[mask_9_10])

    # Write back
    df["q"] = q
    df["qdes"] = qdes

    # Add a helper column for plotting if not present
    if "time_s" not in df.columns:
        df.insert(1, "time_s", t_s)

    # Keep a computed column with reference for plotting
    if "qref" not in df.columns:
        df["qref"] = qref
    return df


def _plot(df: pd.DataFrame, out_png: Path):
    t_s = df["time_s"].to_numpy()

    fig = plt.figure(figsize=(8, 10))
    gs = fig.add_gridspec(5, 1, hspace=0.45)

    # 1) Angle
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t_s, df["q"], color="red", label="q")
    ax1.plot(t_s, df["qdes"], color="green", label="qdes(ESN)")
    ax1.plot(t_s, df["qref"], color="gray", linestyle="--", label="qdes(original)")
    ax1.set_ylabel("Angle [deg]")
    ax1.legend(loc="upper left", fontsize=8)

    # 2) Angle velocity (keep original data)
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(t_s, df["dq"], "o-", color="red", markersize=3, label="dq")
    ax2.set_ylabel("Angle Velocity [deg/s]")
    ax2.legend(loc="upper left", fontsize=8)

    # 3) Valve command
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.plot(t_s, df["ca"], "o-", label="ca")
    ax3.plot(t_s, df["cb"], "o-", label="cb")
    ax3.set_ylabel("Valve Command")
    ax3.legend(loc="upper left", fontsize=8)

    # 4) Pressure
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.plot(t_s, df["pa"], "o-", label="pa")
    ax4.plot(t_s, df["pb"], "o-", label="pb")
    ax4.set_ylabel("Pressure [kPa]")
    ax4.legend(loc="upper left", fontsize=8)

    # 5) Tension
    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax5.plot(t_s, df["Ta"], "o-", label="Ta")
    ax5.plot(t_s, df["Tb"], "o-", label="Tb")
    ax5.set_ylabel("Tension [V]")
    ax5.set_xlabel("Time [s]")
    ax5.legend(loc="upper left", fontsize=8)

    fig.suptitle("MLP+ESN run (edited top panel)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to mlp_esn_*.csv to edit")
    ap.add_argument("--reference", required=True, help="Path to reference_trajectory_*.csv")
    args = ap.parse_args()

    src_csv = Path(args.csv).resolve()
    ref_csv = Path(args.reference).resolve()

    # Load
    df = pd.read_csv(src_csv)
    ref_ms, ref_q = _interp_reference_q(ref_csv)

    # Edit
    df2 = _apply_edits(df, ref_ms, ref_q)

    # Save CSV next to original
    dst_csv = src_csv.with_name(src_csv.stem + "_2.csv")
    df2.to_csv(dst_csv, index=False)

    # Graph destination (mirror mlp_esn.py behavior)
    date_str = datetime.fromtimestamp(src_csv.stat().st_mtime).strftime("%Y%m%d")
    graph_root = Path("data/mlp_esn/graph") / date_str
    graph_root.mkdir(parents=True, exist_ok=True)

    # Set environment variable PLOT_REF_CSV so lib_plot_csv will overlay qdes(original)
    import os
    os.environ["PLOT_REF_CSV"] = str(ref_csv)

    # Use the same plot generator used by mlp_esn.py so styles/colors match
    try:
        svg_or_png = lib_plot_csv(str(dst_csv))
    except Exception as e:
        print("[WARN] lib_plot_csv failed, falling back to internal plot:", e)
        out_png = graph_root / (src_csv.stem + "_2.png")
        _plot(df2, out_png)
        print("Edited CSV:", dst_csv)
        print("Figure:", out_png)
        return

    # If lib_plot_csv returned a path, move or copy into our graph folder with _2 suffix
    out_png = graph_root / (src_csv.stem + "_2.png")
    if svg_or_png:
        src_path = Path(svg_or_png)
        if src_path.exists():
            # If it is already a PNG, copy. If SVG, try to convert via cairosvg if available, else copy SVG.
            if src_path.suffix.lower() == ".png":
                shutil.copy(src_path, out_png)
            elif src_path.suffix.lower() == ".svg":
                try:
                    import cairosvg

                    cairosvg.svg2png(url=str(src_path), write_to=str(out_png))
                except Exception:
                    # fallback: copy svg next to png name (user can convert manually)
                    svgtarget = out_png.with_suffix(".svg")
                    shutil.copy(src_path, svgtarget)
                    print(f"[WARN] cairosvg not available; saved SVG at {svgtarget}")
        else:
            print(f"[WARN] plot generator returned path but file missing: {svg_or_png}")

    print("Edited CSV:", dst_csv)
    print("Figure:", out_png)


if __name__ == "__main__":
    main()
