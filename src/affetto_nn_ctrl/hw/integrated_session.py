from __future__ import annotations

import os
import csv
import time
import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np  # added for trajectory mode

from .dac8564 import Dac8564
from .max11632 import Max11632
from .ldc1614 import LDC1614
from .encoder import EncoderSimple
from .scheduler import FrameScheduler
from .modes import compute_valves, RandomWalk2Ch

# --- new control imports ---
try:
    from affetto_nn_ctrl.control.random_trajectory import RandomTrajectory  # type: ignore
    from affetto_nn_ctrl.control.trajectory_presets import PRESETS  # type: ignore
    from affetto_nn_ctrl.control.pid import PID  # type: ignore
except Exception:  # noqa: BLE001
    RandomTrajectory = None  # type: ignore
    PRESETS = {}  # type: ignore
    PID = None  # type: ignore

try:
    from smbus2 import SMBus  # type: ignore
except Exception:  # noqa: BLE001
    SMBus = None  # type: ignore

@dataclass
class IntegratedConfig:
    # Updated defaults unified with valve_sine_test (was 40-80)
    min_pct: float = 20.0
    max_pct: float = 100.0
    cycle_s: float = 10.0
    total_s: float = 60.0  # 0 => infinite
    interval_ms: int = 50
    mode: str = "sine"  # or "random" or "trajectory" or "random_angle" or "pid_step" or "u_ramp"
    rand_delta: float = 2.0
    seed: int | None = None
    antagonistic: bool = False
    adc_enable: bool = False
    adc_ch0: int = 0
    adc_ch1: int = 1
    adc_print_interval: float = 0.2
    encoder_enable: bool = False
    encoder_print_interval: float = 0.1
    tension_enable: bool = False
    tension_print_interval: float = 0.2
    csv_path: str | None = None
    verbose: bool = False
    # --- trajectory mode specific ---
    trajectory_preset: str | None = None  # key in PRESETS
    pid_kp: float = 2.0
    pid_ki: float = 0.0
    pid_kd: float = 0.0
    # --- pid step test specific ---
    step_q0: float = 0.0          # initial setpoint (encoder units)
    step_q1: float = 50.0         # post-step setpoint
    step_t_change: float = 2.0    # time at which setpoint changes (s)
    encoder_invert: bool = True  # new: invert encoder sign if True
    encoder_zero_on_start: bool = False  # new: capture initial encoder as zero reference
    # --- u_ramp (open-loop diff command) ---
    u_ramp_u0: float = 0.0
    u_ramp_u1: float = 100.0
    u_ramp_duration: float = 8.0

# Constants moved from script
SPI_BUS = 0
SPI_DEV = 0
GPIO_CS_DAC = 19
GPIO_CS_ADC = 24
I2C_BUS = 1
LDC_ADDRS = [0x1A,0x1B,0x2B,0x2A,0x24,0x14,0x15]
ENC_CHIP = '/dev/gpiochip4'
ENC1A = 14; ENC1B = 4

class IntegratedSession:
    def __init__(self, cfg: IntegratedConfig):
        self.cfg = cfg
        self.center = (cfg.min_pct + cfg.max_pct) / 2.0
        self.amp = (cfg.max_pct - cfg.min_pct) / 2.0
        # Devices
        self.dac: Dac8564 | None = None
        self.adc: Max11632 | None = None
        self.enc: EncoderSimple | None = None
        self.ldc_bus = None
        self.ldc_sensors: list[LDC1614] = []
        self.detected_ldc_addrs: list[int] = []
        self.last_tension_values: list[float | None] = []
        # Caches
        self.last_adc_print = 0.0
        self.last_enc_print = 0.0
        self.last_adc_data = None
        self.last_enc_deg: float | None = None
        self.last_tension_print = 0.0
        # CSV
        self.csv_f = None
        self.csv_writer = None
        self.csv_path: str | None = None
        # Random walk
        self.random_walk: RandomWalk2Ch | None = None
        # Trajectory mode
        self.traj: RandomTrajectory | None = None  # type: ignore
        self.pid: PID | None = None  # type: ignore
        self.last_q_des: float | None = None
        self.last_pid_u: float | None = None
        # random_angle internal state
        self.ra_current_target: float | None = None
        self.ra_prev_target: float | None = None
        self.ra_seg_start_t: float = 0.0
        self.ra_seg_duration: float = 1.0
        # encoder zero offset
        self.enc_zero_offset: float | None = None

    # ---- Setup ----
    def open(self) -> None:
        cfg = self.cfg
        # DAC
        self.dac = Dac8564(SPI_BUS, SPI_DEV, GPIO_CS_DAC)
        self.dac.open()
        # ADC
        if cfg.adc_enable:
            try:
                self.adc = Max11632(SPI_BUS, SPI_DEV, GPIO_CS_ADC, cfg.adc_ch0, cfg.adc_ch1)
                self.adc.open()
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] ADC init failed: {e}")
                self.adc = None
        # Encoder
        if cfg.encoder_enable:
            try:
                self.enc = EncoderSimple(ENC_CHIP, ENC1A, ENC1B)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Encoder init failed: {e}")
                self.enc = None
        # If requested, capture encoder zero BEFORE any valve command loop starts
        if cfg.encoder_enable and cfg.encoder_zero_on_start and self.enc is not None:
            try:
                # Do NOT change DAC outputs here — leave valves in their current state and only sample encoder
                # poll longer for stable reading (~0.6s total)
                for _ in range(15):
                    self.enc.poll()
                    time.sleep(0.04)
                raw = self.enc.degrees()
                if raw is not None:
                    if cfg.encoder_invert:
                        raw = -raw
                    self.enc_zero_offset = raw
                    print(f"[INFO] Encoder zero captured (pre-actuation, valves untouched): {raw:.3f} deg -> set to 0.000 (wait~0.6s)")
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] Encoder zero capture failed: {e}")
        # LDC tension
        if cfg.tension_enable and SMBus is not None:
            try:
                self.ldc_bus = SMBus(I2C_BUS)
                for addr in LDC_ADDRS:
                    sensor = LDC1614(self.ldc_bus, addr)
                    if sensor.init():
                        self.ldc_sensors.append(sensor)
                        self.detected_ldc_addrs.append(addr)
                        print(f"[INFO] LDC1614 detected 0x{addr:02X}")
                self.last_tension_values = [None] * len(self.ldc_sensors)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] LDC init failed: {e}")
        # Random walk
        if cfg.mode == "random":
            if cfg.seed is not None:
                random.seed(cfg.seed)
            self.random_walk = RandomWalk2Ch(cfg.min_pct, cfg.max_pct, cfg.rand_delta, self.center, cfg.antagonistic)
        # Trajectory
        if cfg.mode == 'trajectory':
            if RandomTrajectory is None:
                print('[ERROR] trajectory mode requested but control modules unavailable.')
            else:
                if cfg.trajectory_preset is None:
                    print('[ERROR] trajectory_preset must be set for trajectory mode')
                elif cfg.trajectory_preset not in PRESETS:
                    print(f"[ERROR] unknown trajectory_preset: {cfg.trajectory_preset}")
                else:
                    preset = PRESETS[cfg.trajectory_preset]
                    # Single DOF assumption for now
                    t0 = time.perf_counter()
                    q0 = np.array([0.0])
                    try:
                        self.traj = RandomTrajectory([0], t0, q0,
                                                     update_t_range=preset.duration_range,
                                                     update_q_range=preset.amplitude_range,
                                                     update_q_limit=(0.0, 100.0),
                                                     update_profile=preset.profile,
                                                     seed=cfg.seed,
                                                     async_update=not preset.sync)
                        if PID is not None:
                            # 双方向用の最大差動幅 (安全マージンでamp*2)
                            umax = (cfg.max_pct - cfg.min_pct)
                            self.pid = PID(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd, u_min=-umax, u_max=umax)
                    except Exception as e:  # noqa: BLE001
                        print(f"[ERROR] failed to init trajectory: {e}")
        # random_angle init
        if cfg.mode == 'random_angle':
            if PID is not None:
                umax = (cfg.max_pct - cfg.min_pct)
                self.pid = PID(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd, u_min=-umax, u_max=umax)
            if cfg.seed is not None:
                random.seed(cfg.seed)
            self._ra_new_segment(0.0)
        # pid_step init
        if cfg.mode == 'pid_step':
            if PID is not None:
                umax = (cfg.max_pct - cfg.min_pct)
                self.pid = PID(cfg.pid_kp, cfg.pid_ki, cfg.pid_kd, u_min=-umax, u_max=umax)
        # u_ramp mode (open-loop diff command) #
        # CSV
        self._init_csv()

    def _ra_new_segment(self, t: float):
        # choose new target in [0,100], random duration 0.5-2.0s
        self.ra_prev_target = self.ra_current_target if self.ra_current_target is not None else 0.0
        self.ra_current_target = random.uniform(0.0, 100.0)
        self.ra_seg_start_t = t
        self.ra_seg_duration = random.uniform(0.5, 2.0)

    def _init_csv(self) -> None:
        cfg = self.cfg
        base_dir = os.path.join('data','integrated_sensor')
        os.makedirs(base_dir, exist_ok=True)
        if cfg.csv_path is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            cfg.csv_path = f'integrated_log_{ts}.csv'
        self.csv_path = cfg.csv_path if os.path.isabs(cfg.csv_path) else os.path.join(base_dir, cfg.csv_path)
        try:
            self.csv_f = open(self.csv_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_f)
            header = ['ms','valve_a_pct','valve_b_pct']
            # trajectory & random_angle & pid_step & u_ramp extras
            if self.cfg.mode in ('trajectory','random_angle','pid_step','u_ramp'):
                header += ['q_des','pid_u']
            if cfg.adc_enable and self.adc is not None:
                header += ['adc0_raw','adc0_volt','adc0_kpa','adc1_raw','adc1_volt','adc1_kpa']
            if cfg.tension_enable:
                for addr in self.detected_ldc_addrs:
                    header.append(f'tension_0x{addr:02X}')
            if cfg.encoder_enable and self.enc is not None:
                header.append('enc_deg')
            self.csv_writer.writerow(header)
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] CSV init failed: {e}")
            self.csv_writer = None

    # ---- Loop ----
    def run(self):  # generator
        cfg = self.cfg
        scheduler = FrameScheduler(cfg.interval_ms / 1000.0)
        scheduler.start()
        start_msg = f"Start session: {cfg.mode} {cfg.min_pct:.1f}-{cfg.max_pct:.1f}% cycle={cfg.cycle_s:.2f}s total={'∞' if cfg.total_s==0 else f'{cfg.total_s:.1f}s'}"
        print(start_msg)
        try:
            while True:
                t, frame_start = scheduler.wait(poll_cb=self._poll_between)
                if cfg.total_s > 0 and t > cfg.total_s:
                    break
                # Read encoder early for PID
                current_enc_deg = None
                if self.enc is not None:
                    try:
                        self.enc.poll()
                    except Exception:  # noqa: BLE001
                        pass
                    current_enc_deg = self.enc.degrees()
                    if self.cfg.encoder_invert and current_enc_deg is not None:
                        current_enc_deg = -current_enc_deg
                    # removed in-loop zero capture; zero offset set in open() if requested
                    if self.enc_zero_offset is not None and current_enc_deg is not None:
                        current_enc_deg = current_enc_deg - self.enc_zero_offset
                    if frame_start - self.last_enc_print >= cfg.encoder_print_interval:
                        self.last_enc_deg = current_enc_deg
                        self.last_enc_print = frame_start
                a_pct, b_pct = self._compute_valves(t, current_enc_deg)
                if a_pct < cfg.min_pct: a_pct = cfg.min_pct
                if a_pct > cfg.max_pct: a_pct = cfg.max_pct
                if b_pct < cfg.min_pct: b_pct = cfg.min_pct
                if b_pct > cfg.max_pct: b_pct = cfg.max_pct
                if self.dac is not None:
                    self.dac.set_channels(a_pct, b_pct)
                if self.adc is not None and (frame_start - self.last_adc_print) >= cfg.adc_print_interval:
                    try:
                        self.last_adc_data = self.adc.read_pair()
                    except Exception:  # noqa: BLE001
                        self.last_adc_data = None
                    self.last_adc_print = frame_start
                if cfg.tension_enable and self.ldc_sensors and (t - self.last_tension_print) >= cfg.tension_print_interval:
                    self.last_tension_print = t
                self._write_csv(t, a_pct, b_pct, current_enc_deg)
                if cfg.verbose:
                    self._print_verbose(t, a_pct, b_pct)
                yield t
        finally:
            self._cleanup()

    def _compute_valves(self, t: float, enc_deg: float | None) -> tuple[float, float]:
        cfg = self.cfg
        if cfg.mode == 'sine':
            return compute_valves(t, cfg.cycle_s, self.center, self.amp)
        if cfg.mode == 'trajectory' and self.traj is not None and self.pid is not None:
            q_des_all = self.traj.qdes(t)
            q_des = float(q_des_all[0])
            self.last_q_des = q_des
            angle = enc_deg if enc_deg is not None else 0.0
            u = self.pid.step(q_des, angle, cfg.interval_ms/1000.0)
            self.last_pid_u = u
            a_pct = self.center + u/2.0
            b_pct = self.center - u/2.0
            return a_pct, b_pct
        if cfg.mode == 'random_angle' and self.pid is not None:
            # piecewise linear interpolation between previous and current target
            if self.ra_current_target is None:
                self._ra_new_segment(t)
            seg_u = (t - self.ra_seg_start_t) / self.ra_seg_duration
            if seg_u >= 1.0:
                self._ra_new_segment(t)
                seg_u = 0.0
            q0 = self.ra_prev_target if self.ra_prev_target is not None else 0.0
            q1 = self.ra_current_target if self.ra_current_target is not None else 0.0
            q_des = q0 + (q1 - q0) * max(0.0, min(1.0, seg_u))
            self.last_q_des = q_des
            angle = enc_deg if enc_deg is not None else 0.0
            u = self.pid.step(q_des, angle, cfg.interval_ms/1000.0)
            self.last_pid_u = u
            a_pct = self.center + u/2.0
            b_pct = self.center - u/2.0
            return a_pct, b_pct
        if cfg.mode == 'pid_step' and self.pid is not None:
            q_des = cfg.step_q0 if t < cfg.step_t_change else cfg.step_q1
            self.last_q_des = q_des
            angle = enc_deg if enc_deg is not None else 0.0
            u = self.pid.step(q_des, angle, cfg.interval_ms/1000.0)
            self.last_pid_u = u
            a_pct = self.center + u/2.0
            b_pct = self.center - u/2.0
            return a_pct, b_pct
        if cfg.mode == 'u_ramp' or cfg.mode == 'u_ramp_roundtrip':
            # Open-loop linear ramp of differential command.
            # 'u_ramp' behaves as before (single leg). 'u_ramp_roundtrip' performs
            # u0 -> u1 over the first half of cfg.u_ramp_duration and then u1 -> u0
            # over the second half, producing a continuous back-and-forth sweep.
            dur = max(1e-6, cfg.u_ramp_duration)
            if cfg.mode == 'u_ramp_roundtrip':
                half = dur / 2.0
                if t <= half:
                    alpha = t / half if half > 0 else 1.0
                    u = cfg.u_ramp_u0 + (cfg.u_ramp_u1 - cfg.u_ramp_u0) * alpha
                else:
                    alpha = (t - half) / half if half > 0 else 1.0
                    u = cfg.u_ramp_u1 + (cfg.u_ramp_u0 - cfg.u_ramp_u1) * min(alpha, 1.0)
            else:
                alpha = t / dur
                if alpha > 1.0:
                    alpha = 1.0
                u = cfg.u_ramp_u0 + (cfg.u_ramp_u1 - cfg.u_ramp_u0) * alpha
            self.last_pid_u = u  # reuse column for convenience
            a_pct = self.center + u/2.0
            b_pct = self.center - u/2.0
            return a_pct, b_pct
        if self.random_walk is not None:
            return self.random_walk.step()
        return self.center, self.center

    def _poll_between(self) -> None:
        if self.enc is not None:
            try:
                self.enc.poll()
            except Exception:  # noqa: BLE001
                pass
        if self.ldc_sensors:
            for i, s in enumerate(self.ldc_sensors):
                try:
                    v = s.read_ch0_induct_uH()
                    if v is not None and i < len(self.last_tension_values):
                        self.last_tension_values[i] = v
                except Exception:  # noqa: BLE001
                    pass

    def _write_csv(self, t: float, a_pct: float, b_pct: float, current_enc_deg: float | None) -> None:
        if self.csv_writer is None:
            return
        cfg = self.cfg
        row: list[object] = [int(round(t * 1000)), f'{a_pct:.1f}', f'{b_pct:.1f}']
        if cfg.mode in ('trajectory','random_angle','pid_step'):
            row.append('' if self.last_q_des is None else f'{self.last_q_des:.3f}')
            row.append('' if self.last_pid_u is None else f'{self.last_pid_u:.3f}')
        elif cfg.mode == 'u_ramp':
            # open-loop: no q_des, reuse pid_u column for differential command
            row.append('')  # q_des blank
            row.append('' if self.last_pid_u is None else f'{self.last_pid_u:.3f}')
        if cfg.adc_enable and self.adc is not None:
            if self.last_adc_data is not None:
                r0,v0,k0,r1,v1,k1 = self.last_adc_data
                row += [r0, f'{v0:.3f}', f'{k0:.1f}', r1, f'{v1:.3f}', f'{k1:.1f}']
            else:
                row += ['', '', '', '', '', '']
        if cfg.tension_enable:
            for idx, addr in enumerate(self.detected_ldc_addrs):  # noqa: ARG001
                v = self.last_tension_values[idx] if idx < len(self.last_tension_values) else None
                row.append('' if v is None else f'{v:.5f}')
        if cfg.encoder_enable and current_enc_deg is not None:
            row.append(f'{current_enc_deg:.3f}')
        try:
            self.csv_writer.writerow(row)
            if self.csv_f:
                self.csv_f.flush()
        except Exception:  # noqa: BLE001
            pass

    def _print_verbose(self, t: float, a_pct: float, b_pct: float) -> None:
        out = [f'{int(round(t*1000)):8d}ms', f'A={a_pct:5.1f}%', f'B={b_pct:5.1f}%']
        if self.cfg.mode in ('trajectory','random_angle','pid_step'):
            if self.last_q_des is not None:
                out.append(f'q_des={self.last_q_des:6.2f}')
            if self.last_pid_u is not None:
                out.append(f'u={self.last_pid_u:6.2f}')
        elif self.cfg.mode == 'u_ramp':
            if self.last_pid_u is not None:
                out.append(f'u={self.last_pid_u:6.2f}')
        if self.last_adc_data is not None:
            r0,v0,k0,r1,v1,k1 = self.last_adc_data
            out.append(f'P0={k0:6.1f}kPa'); out.append(f'P1={k1:6.1f}kPa')
        if self.cfg.tension_enable:
            disp = []
            for v in [val for val in self.last_tension_values if val is not None][:2]:
                disp.append(f'T={v:7.3f}uH')
            out += disp
        if self.last_enc_deg is not None:
            out.append(f'Enc={self.last_enc_deg:7.2f}deg')
        print(' '.join(out))

    def _cleanup(self) -> None:
        print('\nStopping session... closing valves.')
        try:
            if self.dac is not None:
                self.dac.set_channels(0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        if self.adc is not None:
            try:
                self.adc.close()
            except Exception:  # noqa: BLE001
                pass
        if self.enc is not None:
            try:
                self.enc.close()
            except Exception:  # noqa: BLE001
                pass
        if self.ldc_bus is not None:
            try:
                self.ldc_bus.close()
            except Exception:  # noqa: BLE001
                pass
        if self.dac is not None:
            try:
                self.dac.close()
            except Exception:  # noqa: BLE001
                pass
        if self.csv_f is not None:
            try:
                self.csv_f.close()
            except Exception:  # noqa: BLE001
                pass
        if self.csv_path:
            print(f'[INFO] CSV saved: {self.csv_path}')
        print('Done.')

__all__ = ["IntegratedConfig", "IntegratedSession"]
