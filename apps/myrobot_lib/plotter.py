from __future__ import annotations
from typing import Dict, List, Any
import os

def plot_csv(csv_path: str) -> str:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except Exception:
        print('[WARN] matplotlib not available; skipping plot', flush=True)
        return ''
    import csv
    import numpy as np

    rows = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    except Exception as e:
        print(f'[WARN] failed to read CSV for plotting: {e}', flush=True)
        return ''
    if not rows:
        print('[WARN] CSV empty; nothing to plot', flush=True)
        return ''

    def safe_float(s: Any) -> float:
        try:
            if s is None:
                return float('nan')
            # handle empty strings
            if isinstance(s, str) and s.strip() == '':
                return float('nan')
            return float(s)
        except Exception:
            return float('nan')

    ms = [int(r.get('ms', 0) or 0) for r in rows]
    # convert ms to seconds for plotting
    sec = [m / 1000.0 for m in ms]

    # support both old and new CSV column names (e.g. q_des vs qdes, valve_a_pct vs ca)
    q_des = [safe_float(r.get('qdes', r.get('q_des', ''))) for r in rows]
    dq_des = [safe_float(r.get('dqdes', r.get('dq_des', ''))) for r in rows]
    # Try known dq keys first; will compute from encoder if missing
    dq_meas = [safe_float(r.get('dq', r.get('dq_meas', ''))) for r in rows]
    # Accept encoder values under 'q', 'enc_deg' or 'enc' (some CSV writers use 'q')
    enc = [safe_float(r.get('q', r.get('enc_deg', r.get('enc', '')))) for r in rows]

    # Valve command: prefer raw DAC 'ca'/'cb', but commonly CSVs have 'valve_a_pct'/'valve_b_pct' (percent)
    # Also accept 'a_pct'/'b_pct' or 'valve_a'/'valve_b' variants.
    def pick_valve_series(key_candidates: list[str]) -> list[float]:
        for k in key_candidates:
            if k in rows[0]:
                return [safe_float(r.get(k, '')) for r in rows]
        # fallback: return NaNs
        return [float('nan')] * len(rows)

    a = pick_valve_series(['ca', 'valve_a_pct', 'a_pct', 'valve_a', 'caper'])
    b = pick_valve_series(['cb', 'valve_b_pct', 'b_pct', 'valve_b', 'cbper'])
    # percent -> volt 変換（5V = 100%）
    try:
        if any(k in rows[0] for k in ('valve_a_pct','a_pct','valve_a','caper')) and not ('ca' in rows[0]):
            a = [ (x * 0.05) if np.isfinite(x) else float('nan') for x in a ]
        if any(k in rows[0] for k in ('valve_b_pct','b_pct','valve_b','cbper')) and not ('cb' in rows[0]):
            b = [ (x * 0.05) if np.isfinite(x) else float('nan') for x in b ]
    except Exception:
        pass

    # ADC/pressure support (keep original keys if present)
    has_adc0 = ('adc0_kpa' in rows[0]) or ('pa' in rows[0])
    p0 = [safe_float(r.get('adc0_kpa', r.get('pa', ''))) for r in rows] if has_adc0 else None
    p1 = [safe_float(r.get('adc1_kpa', r.get('pb', ''))) for r in rows] if ('adc1_kpa' in rows[0] or 'pb' in rows[0]) else None

    # Tension/LDC support: accept 'Ta'/'Tb', 'ta'/'tb', or 'tension_0xXX' keys
    tension_keys: list[str] = []
    for k in rows[0].keys():
        lk = k.lower()
        if lk in ('ta', 'tb', 'ta_str', 'tb_str'):
            tension_keys.append(k)
        elif lk.startswith('tension_0x') or lk.startswith('tension'):
            tension_keys.append(k)
    tensions: dict[str, list[float]] = {}
    if tension_keys:
        for k in tension_keys:
            tensions[k] = [safe_float(r.get(k, '')) for r in rows]

    # If dq_meas is not present (all NaNs), try to compute from encoder values
    try:
        if all(np.isnan(x) for x in dq_meas):
            computed = []
            for i in range(len(enc)):
                if i == 0:
                    computed.append(float('nan'))
                    continue
                dt = sec[i] - sec[i - 1]
                if dt == 0 or np.isnan(enc[i]) or np.isnan(enc[i - 1]):
                    computed.append(float('nan'))
                else:
                    # simple finite difference (deg/s)
                    computed.append((enc[i] - enc[i - 1]) / dt)
            dq_meas = computed
    except Exception:
        # ignore computation errors and leave dq_meas as-is
        pass

    # label mapping requested by user (accept both old/new keys)
    # Use superscript plus/minus in legend labels via matplotlib mathtext
    label_map = {
        'q_des': 'qdes', 'qdes': 'qdes',
        'enc_deg': 'q', 'enc': 'q',
        # control/valve labels: c⁺ / c⁻
        'valve_a_pct': 'c$^{+}$', 'ca': 'c$^{+}$',
        'valve_b_pct': 'c$^{-}$', 'cb': 'c$^{-}$',
        # pressures: p⁺ / p⁻
        'adc0_kpa': 'p$^{+}$', 'pa': 'p$^{+}$',
        'adc1_kpa': 'p$^{-}$', 'pb': 'p$^{-}$',
        # tensions: T⁺ / T⁻
        'tension_0x2B': 'T$^{-}$', 'tension_0x2A': 'T$^{+}$',
        'Ta': 'T$^{+}$', 'Tb': 'T$^{-}$',
    }

    n_subplots = 5
    fig = plt.figure(figsize=(10, 12))

    # (ユーザー要望) グラフ上部のタイトルは表示しない
    _ = os.environ.get('PLOT_TITLE', None)

    # Angle plot
    ax1 = fig.add_subplot(n_subplots, 1, 1)
    # plot encoder first so legend shows 'q' before 'qdes'
    # make markers larger and lines thicker so measured angle is visible
    # user requested: q -> 赤, qdes -> 緑
    ax1.plot(sec, enc, '-o', color='red', linewidth=1.5, markersize=4, label='q')
    ax1.plot(sec, q_des, '-o', color='tab:green', linewidth=1.2, markersize=3, alpha=0.95, label='qdes')

    # expand y-limits to include both series (ignore NaNs)
    try:
        import numpy as _np
        combined = _np.array([v for v in (enc + q_des) if not _np.isnan(v)])
        if combined.size > 0:
            lo = float(_np.nanmin(combined))
            hi = float(_np.nanmax(combined))
            rng = max(1.0, (hi - lo) * 0.1)
            ax1.set_ylim(lo - rng, hi + rng)
    except Exception:
        pass
    ax1.legend(loc='upper right')
    ax1.set_title('Angle [deg]')
    # make left y-axis have 5 major ticks
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


    # Angle velocity (deg/s) plot — replace pid_u plot
    ax2 = fig.add_subplot(n_subplots, 1, 2)
    # plot measured and desired angular velocity with distinct colors
    # user requested: dq -> 赤, dqdes -> 緑
    ax2.plot(sec, dq_meas, '-o', color='red', label='dq')
    if not all(np.isnan(x) for x in dq_des):
        ax2.plot(sec, dq_des, '-', color='tab:green', alpha=0.8, label='dqdes')
    ax2.legend(loc='upper right')
    ax2.set_title('Angle Velocity [deg/s]')
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # Valve command plot
    ax3 = fig.add_subplot(n_subplots, 1, 3)
    # user requested: c+ -> 黄色, c- -> 青（凡例は c^+, c^-）
    ax3.plot(sec, a, '-o', color='gold', label=r'c$^{+}$')
    ax3.plot(sec, b, '-o', color='tab:blue', label=r'c$^{-}$')
    ax3.legend(loc='upper right')
    ax3.set_title('Valve Command [V]')
    try:
        # 指示により縦軸を 1V〜5V に設定
        ax3.set_ylim(1.0, 5.0)
    except Exception:
        pass
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # Pressure plot
    ax4 = fig.add_subplot(n_subplots, 1, 4)
    # user requested: (pa,pb) = (黄色, 青) ＋ 凡例は p^+, p^-
    if p0 is not None:
        ax4.plot(sec, p0, '-o', color='gold', label=r'p$^{+}$')
    if p1 is not None:
        ax4.plot(sec, p1, '-o', color='tab:blue', label=r'p$^{-}$')
    if p0 is None and p1 is None:
        ax4.text(0.5, 0.5, 'No pressure (ADC) data', ha='center', va='center')
    ax4.legend(loc='upper right')
    ax4.set_title('Pressure [kPa]')
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    # Tension sensors
    ax5 = fig.add_subplot(n_subplots, 1, 5)
    if tensions:
        # swap colors for Ta/Tb as requested
        # user requested: Ta -> 黄色, Tb -> 青
        color_map = {
            'tension_0x2A': 'gold',   # Ta
            'tension_0x2a': 'gold',
            'tension_0x2B': 'tab:blue', # Tb
            'tension_0x2b': 'tab:blue',
            'Ta': 'gold',
            'Tb': 'tab:blue',
            'ta': 'gold',
            'tb': 'tab:blue',
        }
        for k, vals in tensions.items():
            kl = k.lower()
            if kl in ('ta',) or kl.endswith('_0x2a'):
                lab = r'T$^{+}$'
            elif kl in ('tb',) or kl.endswith('_0x2b'):
                lab = r'T$^{-}$'
            else:
                lab = label_map.get(k, k)
            col = color_map.get(k, None)
            if col:
                ax5.plot(sec, vals, '-o', label=lab, color=col)
            else:
                ax5.plot(sec, vals, '-o', label=lab)
        # make Ta/Tb legend vertical (one column) and invert order (上下を入れ替え)
        handles, labels = ax5.get_legend_handles_labels()
        if handles and labels:
            # If both Ta and Tb are present, ensure Ta appears above Tb in the legend.
            # Preserve other labels' relative order but place Ta then Tb at the top.
            try:
                lbls = list(labels)
                hds = list(handles)
                if 'Ta' in lbls and 'Tb' in lbls:
                    ta_idx = lbls.index('Ta')
                    tb_idx = lbls.index('Tb')
                    # Collect other (handle,label) pairs in original order excluding Ta/Tb
                    others = [(h, l) for (h, l) in zip(hds, lbls) if l not in ('Ta', 'Tb')]
                    new_order = []
                    new_order.append((hds[ta_idx], lbls[ta_idx]))
                    new_order.append((hds[tb_idx], lbls[tb_idx]))
                    new_order.extend(others)
                    new_handles, new_labels = zip(*new_order) if new_order else ([], [])
                    ax5.legend(new_handles, new_labels, loc='upper right', ncol=1, fontsize='small')
                else:
                    # Fallback: reverse order as before
                    ax5.legend(handles[::-1], labels[::-1], loc='upper right', ncol=1, fontsize='small')
            except Exception:
                # Best-effort fallback to previous behavior on any error
                ax5.legend(handles[::-1], labels[::-1], loc='upper right', ncol=1, fontsize='small')
        else:
            ax5.legend(loc='upper right', ncol=1, fontsize='small')
    else:
        ax5.text(0.5, 0.5, 'No tension (LDC) data', ha='center', va='center')
    ax5.set_xlabel('Time [s]')
    # display units in micro-Henry as requested
    ax5.set_title('Tension [μH]')
    ax5.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    plt.tight_layout()
    # Allow user to select output format via environment variable PLOT_FORMAT (png or pdf)
    import os as _os
    # Default to PNG; allow overriding via PLOT_FORMAT environment variable
    fmt = _os.environ.get('PLOT_FORMAT', 'png').lower()
    ext = 'pdf' if fmt == 'pdf' else 'png'
    out = csv_path.rsplit('.', 1)[0] + f'.{ext}'
    # graph へ直接保存（.../data/tracked_trajectory/csv -> .../graph）
    try:
        import pathlib as _pl
        p = _pl.Path(csv_path)
        parts = [s for s in p.parts]
        if 'data' in parts and 'tracked_trajectory' in parts and 'csv' in parts:
            new_parts = []
            swapped = False
            for s in parts:
                if not swapped and s == 'csv':
                    new_parts.append('graph')
                    swapped = True
                else:
                    new_parts.append(s)
            out_path = _pl.Path(*new_parts).with_suffix(f'.{ext}')
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out = str(out_path)
    except Exception:
        pass
    # Save main figure (best-effort)
    try:
        if fmt == 'pdf':
            plt.savefig(out, format='pdf')
        else:
            plt.savefig(out)
        print(f'[INFO] Plot saved: {out}', flush=True)
    except Exception as e:
        print(f'[WARN] failed to save main plot: {e}', flush=True)
        out = ''

    # If we found any tension channels, produce a dedicated tension plot as well
    try:
        # Do not produce a separate tension-only figure per user request.
        # Keep tension panel in the main figure but skip creating a dedicated plot.
        if False and tensions:
            import matplotlib.pyplot as _plt
            tfig, tax = _plt.subplots(figsize=(6, 3))
            # color mapping: prefer Ta -> gold, Tb -> tab:blue when present
            for k, series in tensions.items():
                key_low = k.lower()
                if key_low == 'ta' or key_low.endswith('_0x2a'):
                    c = 'gold'
                    lab = 'Ta'
                elif key_low == 'tb' or key_low.endswith('_0x2b'):
                    c = 'tab:blue'
                    lab = 'Tb'
                else:
                    c = None
                    lab = k
                tax.plot(sec, series, label=lab, color=c)
            tax.set_xlabel('Time [s]')
            tax.set_ylabel('Tension (uH)')
            tax.legend(loc='upper right')
            tax.grid(True, alpha=0.3)
            tension_ext = ext
            tension_png = csv_path.rsplit('.', 1)[0] + f'_tension.{tension_ext}'
            tfig.tight_layout()
            try:
                if fmt == 'pdf':
                    tfig.savefig(tension_png, format='pdf', dpi=150)
                else:
                    tfig.savefig(tension_png, dpi=150)
                print(f'[INFO] Tension plot saved: {tension_png}', flush=True)
            except Exception as e:
                print(f'[WARN] failed to save tension plot: {e}', flush=True)
            try:
                _plt.close(tfig)
            except Exception:
                pass
    except Exception:
        # best-effort: do not fail main plotting on tension plotting errors
        pass

    return out
