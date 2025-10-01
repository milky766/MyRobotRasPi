#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np


def _iter_tracked_csv(csv_dir: Path, *, prefix: str | None = None, contains: str | None = None) -> list[Path]:
    files = []
    for p in csv_dir.glob("tracked_trajectory_*.csv"):
        # Optional filters
        if prefix is not None and not p.name.startswith(prefix):
            continue
        if contains is not None and contains not in p.name:
            continue
        try:
            stem = p.stem
            idx = int(stem.rsplit("_", 1)[-1])
        except Exception:
            idx = -1
        files.append((idx, p))
    # 降順（新しい番号が大きい想定）
    files.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in files]


def _safe_float(x: str | float | int | None) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _load_columns(path: Path, q_key: str = "q_des", enc_key: str = "enc_deg") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_ms: list[float] = []
    q_des: list[float] = []
    enc: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = _safe_float(row.get("ms"))
            q = _safe_float(row.get(q_key))
            e = _safe_float(row.get(enc_key))
            if t is None or q is None or e is None:
                continue
            t_ms.append(t)
            q_des.append(q)
            enc.append(e)
    if len(t_ms) == 0:
        return np.array([]), np.array([]), np.array([])
    t_arr = np.asarray(t_ms, dtype=float)
    q_arr = np.asarray(q_des, dtype=float)
    e_arr = np.asarray(enc, dtype=float)
    return t_arr, q_arr, e_arr


def _metrics(q: np.ndarray, e: np.ndarray) -> dict[str, float]:
    if q.size == 0 or e.size == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "bias": np.nan,
            "max_abs": np.nan,
            "p95_abs": np.nan,
            "r2": np.nan,
            "nrmse_std": np.nan,
            "nrmse_range": np.nan,
        }
    diff = q - e
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    max_abs = float(np.max(np.abs(diff)))
    p95_abs = float(np.percentile(np.abs(diff), 95))
    # R^2: 1 - SSE/SST（SSTが極小のときはNaN）
    var = float(np.var(q))
    if var > 1e-12:
        r2 = float(1.0 - np.sum(diff**2) / (len(q) * var))
    else:
        r2 = float("nan")
    # NRMSE: 標準偏差基準とレンジ基準の2種
    std_q = float(np.std(q))
    ptp_q = float(np.ptp(q))  # max - min
    nrmse_std = float(rmse / std_q) if std_q > 1e-12 else float("nan")
    nrmse_range = float(rmse / ptp_q) if ptp_q > 1e-12 else float("nan")
    return {
        "n": int(q.size),
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "max_abs": max_abs,
        "p95_abs": p95_abs,
        "r2": r2,
        "nrmse_std": nrmse_std,
        "nrmse_range": nrmse_range,
    }


def summarize(csv_paths: Iterable[Path]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for p in csv_paths:
        t, q, e = _load_columns(p)
        m = _metrics(q, e)
        dur_ms = float(t[-1] - t[0]) if t.size > 1 else 0.0
        rows.append(
            {
                "file": p.name,
                "n": m["n"],
                "duration_ms": round(dur_ms, 1),
                "mae_deg": round(m["mae"], 4) if isinstance(m["mae"], float) else m["mae"],
                "rmse_deg": round(m["rmse"], 4) if isinstance(m["rmse"], float) else m["rmse"],
                "bias_deg": round(m["bias"], 4) if isinstance(m["bias"], float) else m["bias"],
                "max_abs_deg": round(m["max_abs"], 4) if isinstance(m["max_abs"], float) else m["max_abs"],
                "p95_abs_deg": round(m["p95_abs"], 4) if isinstance(m["p95_abs"], float) else m["p95_abs"],
                "r2": round(m["r2"], 6) if isinstance(m["r2"], float) else m["r2"],
                "nrmse_std": round(m["nrmse_std"], 6) if isinstance(m["nrmse_std"], float) else m["nrmse_std"],
                "nrmse_range": round(m["nrmse_range"], 6) if isinstance(m["nrmse_range"], float) else m["nrmse_range"],
            }
        )
    return rows


def save_summary(rows: list[dict[str, float | int | str]], out_dir: Path, base_name: str = "summary") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 連番ファイル名
    base = base_name
    existing = [p for p in out_dir.glob(f"{base}_*.csv")]
    max_idx = 0
    for p in existing:
        try:
            idx = int(p.stem.rsplit("_", 1)[-1])
            max_idx = max(max_idx, idx)
        except Exception:
            pass
    out = out_dir / f"{base}_{max_idx+1}.csv"
    if len(rows) == 0:
        out.write_text("", encoding="utf-8")
        return out
    # カラム順
    cols = [
        "file",
        "n",
        "duration_ms",
        "mae_deg",
        "rmse_deg",
        "bias_deg",
        "max_abs_deg",
        "p95_abs_deg",
        "r2",
        "nrmse_std",
        "nrmse_range",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate tracking metrics (q_des vs enc_deg).")
    parser.add_argument("--csv-dir", default="/home/hosodalab2/Desktop/MyRobot/data/tracked_trajectory/csv")
    parser.add_argument("--take", type=int, default=10, help="最新番号からN件を集計")
    parser.add_argument("--out-dir", default="/home/hosodalab2/Desktop/MyRobot/data/tracked_trajectory/summary")
    parser.add_argument("--prefix", type=str, default=None, help="ファイル名の先頭一致でフィルタ (例: tracked_trajectory_reference_trajectory_16_) ")
    parser.add_argument("--contains", type=str, default=None, help="ファイル名に含まれる文字列でフィルタ")
    parser.add_argument("--sort-by", type=str, default="rmse", choices=[
        "rmse", "mae", "bias", "max_abs", "p95_abs", "r2", "nrmse_std", "nrmse_range"
    ], help="ランキングに使う指標（デフォルトrmse、低いほど良いと仮定）")
    parser.add_argument("--topk", type=int, default=0, help="上位K件を抽出（0なら全件）")
    parser.add_argument("--copy-to", type=str, default=None, help="上位K件のCSV/PNGをコピーする先のディレクトリ")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    paths = _iter_tracked_csv(csv_dir, prefix=args.prefix, contains=args.contains)
    if args.take > 0:
        paths = paths[: args.take]

    rows = summarize(paths)
    out_dir = Path(args.out_dir)
    out = save_summary(rows, out_dir, base_name="summary")

    # 画面にもコンパクトに出す
    print("\nPer-run metrics (latest first):")
    for r in rows:
        print(
            f"{r['file']:>28s} | rmse={r['rmse_deg']!s:>7} deg | mae={r['mae_deg']!s:>7} deg | bias={r['bias_deg']!s:>7} deg | p95={r['p95_abs_deg']!s:>7} deg | R2={r['r2']!s:>6} | nrmse(std)={r['nrmse_std']!s:>6}"
        )
    # 平均/標準偏差
    try:
        arr_rmse = np.array([x["rmse_deg"] for x in rows], dtype=float)
        arr_mae = np.array([x["mae_deg"] for x in rows], dtype=float)
        arr_r2 = np.array([x["r2"] for x in rows], dtype=float)
        arr_nrmse = np.array([x["nrmse_std"] for x in rows], dtype=float)
        print(
            f"\nSummary: RMSE mean={np.mean(arr_rmse):.4f} (std={np.std(arr_rmse):.4f}), "
            f"MAE mean={np.mean(arr_mae):.4f} (std={np.std(arr_mae):.4f}), "
            f"R2 mean={np.nanmean(arr_r2):.4f}, NRMSE(std) mean={np.nanmean(arr_nrmse):.4f}"
        )
    except Exception:
        pass
    print(f"Saved: {out}")

    # 上位K件の抽出と保存/コピー
    if args.topk and args.topk > 0 and len(rows) > 0:
        # 並び替え（rmseやmaeなどは小さいほど良い、r2は大きいほど良い）
        key = args.sort_by
        reverse = True if key == "r2" else False
        def _val(r):
            try:
                return float(r[f"{key}_deg"]) if key in ("rmse", "mae", "bias", "max_abs", "p95_abs") else float(r[key])
            except Exception:
                return float("inf") if not reverse else float("-inf")
        ranked = sorted(rows, key=_val, reverse=reverse)
        topk_rows = ranked[: args.topk]
        top_path = save_summary(topk_rows, out_dir, base_name=f"summary_top{args.topk}_{key}")
        print(f"Saved top{args.topk} ({key}) to: {top_path}")

        # 必要ならファイルをコピー
        if args.copy_to:
            dest = Path(args.copy_to)
            dest.mkdir(parents=True, exist_ok=True)
            csv_dir = Path(args.csv_dir)
            # graphディレクトリはcsvの兄弟フォルダと仮定
            graph_dir = csv_dir.parent / "graph"
            for r in topk_rows:
                fname = str(r["file"])  # file名
                src_csv = csv_dir / fname
                try:
                    shutil.copy2(src_csv, dest / src_csv.name)
                except Exception:
                    pass
                try:
                    png = graph_dir / (src_csv.stem + ".png")
                    if png.exists():
                        shutil.copy2(png, dest / png.name)
                except Exception:
                    pass
            print(f"Copied top{args.topk} files to: {dest}")


if __name__ == "__main__":
    main()
