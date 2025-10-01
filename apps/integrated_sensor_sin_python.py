#!/usr/bin/env python3　range of motionを測る機能も兼ねている
"""Python equivalent of integrated_sensor_sin.cpp (DAC direct control)
env PYTHONPATH=/home/hosodalab2/Desktop/MyRobot/src uv run python -u apps/integrated_sensor_sin_python.py --encoder-invert --center-hold-s 1.0
"""
import time
import math
import os
import signal
from datetime import datetime
import argparse
import sys, pathlib
from pathlib import Path
import csv

# Remove IntegratedSession usage; use local DAC like valve_sine_test
try:
    import spidev  # type: ignore
except Exception:  # noqa: BLE001
    spidev = None  # type: ignore
try:
    import gpiod  # type: ignore
    from gpiod.line import Direction, Value  # type: ignore
except Exception:  # noqa: BLE001
    gpiod = None  # type: ignore
    Direction = Value = None  # type: ignore
# New: encoder & LDC imports
# Ensure local 'src' is on sys.path so we can import affetto_nn_ctrl.* when running from workspace
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))
try:
    from smbus2 import SMBus  # type: ignore
except Exception:  # noqa: BLE001
    SMBus = None  # type: ignore
try:
    from affetto_nn_ctrl.hw.encoder import EncoderSimple  # type: ignore
    from affetto_nn_ctrl.hw.ldc1614 import LDC1614  # type: ignore
except Exception:  # noqa: BLE001
    EncoderSimple = None  # type: ignore
    LDC1614 = None  # type: ignore

# Sine curve control parameters
DUTY_CENTER = 60.0
DUTY_AMPLITUDE = 40.0
CYCLE_SEC = 10.0
# Run exactly two cycles by default
TOTAL_SEC = CYCLE_SEC * 2
LOOP_INTERVAL_MS = 50

# SPI/DAC constants (match valve_sine_test)
SPI_BUS = 0
SPI_DEV = 0           # /dev/spidev0.0
SPI_MAX_HZ = 1_000_000
SPI_MODE_DAC = 0b01   # MODE1
GPIO_CS_DAC = 19      # manual CS GPIO

# New: ADC (MAX11632) constants for pressure sensor
SPI_MODE_ADC = 0b00
GPIO_CS_ADC = 24
ADC_CH0 = 0
ADC_CH1 = 1
ADC_PRINT_INTERVAL = 0.2
# Pressure conversion params
P_MIN = 0.003  # MPa
P_MAX = 0.6    # MPa
V_MIN = 1.0    # V
V_MAX = 5.0    # V

# New: LDC and Encoder constants
I2C_BUS = 1
LDC_ADDRS = [0x1A, 0x1B, 0x2B, 0x2A, 0x24, 0x14, 0x15]
TENSION_PRINT_INTERVAL = 0.2
ENCODER_PRINT_INTERVAL = 0.1
ENC_CHIP = '/dev/gpiochip4'
ENC1A = 14; ENC1B = 4

RUN = True

def _on_signal(signum, frame):  # noqa: ARG001
    global RUN
    RUN = False

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, _on_signal)

class CsLine:
    """Minimal CS line using libgpiod v2 (same as valve_sine_test)."""
    def __init__(self, pin: int):
        if gpiod is None:
            raise RuntimeError('python3-libgpiod is required')
        self.pin = pin
        self.chip = gpiod.Chip('/dev/gpiochip0')
        ls = gpiod.LineSettings()
        if Direction is not None:
            ls.direction = Direction.OUTPUT
        if Value is not None:
            ls.output_value = Value.ACTIVE
        self.req = self.chip.request_lines(consumer='dac_cs', config={pin: ls})
        try:
            # default inactive (high for active-low CS)
            self.set(1)
        except Exception:
            pass
    def set(self, val: int):
        if Value is not None:
            self.req.set_values({self.pin: (Value.ACTIVE if val else Value.INACTIVE)})
        else:  # pragma: no cover
            self.req.set_values({self.pin: val})  # type: ignore[arg-type]
    def close(self):
        try:
            self.set(1)
        except Exception:
            pass
        try:
            self.req.release()
        except Exception:
            pass
        try:
            self.chip.close()
        except Exception:
            pass

class Dac8564:
    """Direct DAC8564 access with manual CS toggling per transfer."""
    def __init__(self, bus: int, dev: int, cs_gpio: int):
        if spidev is None:
            raise RuntimeError('spidev is required')
        self.spi = spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        self.cs = CsLine(cs_gpio)
        self.opened = False
    def open(self):
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            self.spi.max_speed_hz = SPI_MAX_HZ
            self.spi.mode = SPI_MODE_DAC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_and_init()
    def close(self):  # pragma: no cover - hardware cleanup
        try:
            self.set_channels(0.0, 0.0)
        except Exception:
            pass
        if self.opened:
            try:
                self.spi.close()
            except Exception:
                pass
        try:
            self.cs.close()
        except Exception:
            pass
    def _xfer(self, data):
        if not self.opened:
            raise RuntimeError('SPI not opened yet')
        self.spi.mode = SPI_MODE_DAC
        self.cs.set(0)
        self.spi.xfer2(list(data))
        self.cs.set(1)
    @staticmethod
    def _pct_to_code(pct: float) -> int:
        if pct < 0.0: pct = 0.0
        if pct > 100.0: pct = 100.0
        return int(pct * 65535.0 / 100.0 + 0.5)
    @staticmethod
    def _cmd(ch: int) -> int:
        # Use 0x10 (write input register). Board LDAC is tied low so output updates.
        return 0x10 | (ch << 1)
    def set_channels(self, a_pct: float, b_pct: float):
        a_code = self._pct_to_code(a_pct)
        b_code = self._pct_to_code(b_pct)
        a = [self._cmd(0), (a_code >> 8) & 0xFF, a_code & 0xFF]
        b = [self._cmd(1), (b_code >> 8) & 0xFF, b_code & 0xFF]
        self._xfer(a)
        self._xfer(b)
    def _reset_and_init(self):
        self._xfer([0x28,0x00,0x01])
        time.sleep(0.001)
        self._xfer([0x38,0x00,0x01])

def calculate_valve_a_opening(time_sec: float) -> float:
    sin_value = math.sin(2.0 * math.pi * time_sec / CYCLE_SEC - math.pi / 2.0)
    opening = DUTY_CENTER + DUTY_AMPLITUDE * sin_value
    return max(0.0, min(100.0, opening))

def calculate_valve_b_opening(time_sec: float) -> float:
    sin_value = math.sin(2.0 * math.pi * time_sec / CYCLE_SEC + math.pi / 2.0)
    opening = DUTY_CENTER + DUTY_AMPLITUDE * sin_value
    return max(0.0, min(100.0, opening))

def create_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

# --- New: MAX11632 ADC for pressure ---
class Max11632:
    def __init__(self, bus: int, dev: int, cs_gpio: int, ch0: int, ch1: int):
        import spidev as _spidev  # local import to reuse module
        self.spi = _spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        self.cs = CsLine(cs_gpio)
        self.opened = False
        self.ch0 = ch0
        self.ch1 = ch1
    def open(self):
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            self.spi.max_speed_hz = SPI_MAX_HZ
            self.spi.mode = SPI_MODE_ADC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_init()
    def close(self):  # pragma: no cover
        if self.opened:
            try:
                self.spi.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.cs.close()
        except Exception:  # noqa: BLE001
            pass
    def _xfer(self, tx, rxlen=0):
        if not self.opened:
            raise RuntimeError('ADC SPI not opened')
        self.spi.mode = SPI_MODE_ADC
        self.cs.set(0)
        if rxlen:
            rx = self.spi.xfer2(list(tx) + [0x00]*rxlen)
            self.cs.set(1)
            return rx[len(tx):]
        else:
            self.spi.xfer2(list(tx))
            self.cs.set(1)
            return []
    def _reset_init(self):
        # Reset (0x00) then setup (0x64)
        self._xfer([0x00])
        time.sleep(0.001)
        self._xfer([0x64])
        time.sleep(0.005)
    def read_raw(self, ch: int) -> int:
        cmd = 0x80 | ((ch & 0x0F) << 3) | (0b11 << 1)
        self._xfer([cmd])
        time.sleep(0.0001)  # 100us conversion
        # Read exactly 16 clocks like the C++ version (no leading extra byte)
        d = self._xfer([], rxlen=2)
        raw = ((d[0] & 0x0F) << 8) | d[1]
        return raw
    @staticmethod
    def raw_to_voltage(raw: int) -> float:
        return (raw / 4095.0) * 5.0
    @staticmethod
    def voltage_to_kpa(v: float) -> float:
        mpa = (v - V_MIN) * (P_MAX - P_MIN) / (V_MAX - V_MIN) + P_MIN
        return mpa * 1000.0
    def read_pair(self):
        r0 = self.read_raw(self.ch0)
        r1 = self.read_raw(self.ch1)
        v0 = self.raw_to_voltage(r0)
        v1 = self.raw_to_voltage(r1)
        k0 = self.voltage_to_kpa(v0)
        k1 = self.voltage_to_kpa(v1)
        return (r0, v0, k0, r1, v1, k1)

# Simple CSV setup extended with tension/encoder columns

def _init_csv(path_hint: str | None, detected_ldc_addrs: list[int], has_enc: bool, include_adc: bool):
    base_dir = os.path.join('data','integrated_sensor')
    os.makedirs(base_dir, exist_ok=True)
    ts = create_timestamp()
    filename = path_hint or f'integrated_log_{ts}_sin_wave_python.csv'
    csv_path = filename if os.path.isabs(filename) else os.path.join(base_dir, filename)
    try:
        f = open(csv_path, 'w', buffering=1)
        header = ['ms','valve_a_pct','valve_b_pct']
        if include_adc:
            header += ['adc0_raw','adc0_volt','adc0_kpa','adc1_raw','adc1_volt','adc1_kpa']
        for addr in detected_ldc_addrs:
            header.append(f'tension_0x{addr:02X}')
        if has_enc:
            header.append('enc_deg')
        f.write(','.join(header) + '\n')
        return f, csv_path
    except Exception:
        return None, None

def main():
    # CLI: encoder invert and center-hold seconds for zero capture
    ap = argparse.ArgumentParser(description='Direct DAC sine with optional encoder invert and zero capture at center (60/60)')
    ap.add_argument('--encoder-invert', action='store_true', help='Invert encoder sign')
    ap.add_argument('--center-hold-s', type=float, default=0.6, help='Seconds to hold at 60/60 before starting; used to capture encoder zero')
    args = ap.parse_args()

    print('Python sine wave control starting (direct DAC)...')
    dac = Dac8564(SPI_BUS, SPI_DEV, GPIO_CS_DAC)
    try:
        dac.open()
    except Exception as e:  # noqa: BLE001
        print(f'[ERROR] DAC init failed: {e}')
        return 1

    # New: init ADC (pressure)
    adc = None
    last_adc_data = None
    last_adc_print = 0.0
    try:
        adc = Max11632(SPI_BUS, SPI_DEV, GPIO_CS_ADC, ADC_CH0, ADC_CH1)
        adc.open()
        print(f'[INFO] ADC enabled ch{ADC_CH0}, ch{ADC_CH1}')
    except Exception as e:  # noqa: BLE001
        print(f'[WARN] ADC init failed: {e}')
        adc = None

    # New: init encoder
    enc = None
    last_enc_deg = None
    last_enc_print = 0.0
    enc_zero_offset = None
    if EncoderSimple is not None:
        try:
            enc = EncoderSimple(ENC_CHIP, ENC1A, ENC1B)  # type: ignore[call-arg]
            print(f'[INFO] Encoder enabled A={ENC1A} B={ENC1B}')
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] Encoder init failed: {e}')
            enc = None

    # Center valves and capture encoder zero at center (pre-actuation for sine)
    try:
        dac.set_channels(DUTY_CENTER, DUTY_CENTER)
        print(f'[INFO] Centering valves to {DUTY_CENTER:.1f}%/{DUTY_CENTER:.1f}% for zero capture...')
        t_end = time.perf_counter() + max(0.0, float(args.center_hold_s))
        # poll encoder during hold for stable reading
        while time.perf_counter() < t_end:
            if enc is not None:
                try:
                    enc.poll()
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.02)
        if enc is not None:
            try:
                # take multiple samples for robustness
                vals = []
                for _ in range(5):
                    try:
                        enc.poll()
                    except Exception:
                        pass
                    v = enc.degrees()  # type: ignore[union-attr]
                    if v is not None:
                        vals.append(v)
                    time.sleep(0.02)
                if vals:
                    raw = sum(vals) / len(vals)
                    if args.encoder_invert:
                        raw = -raw
                    enc_zero_offset = raw
                    print(f'[INFO] Encoder zero captured at center: {raw:.3f} deg -> set to 0.000')
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass

    # New: init tension sensors (LDC1614)
    ldc_bus = None
    ldc_sensors = []
    detected_ldc_addrs: list[int] = []
    last_tension_values: list[float | None] = []
    last_tension_print = 0.0
    if SMBus is not None and LDC1614 is not None:
        try:
            ldc_bus = SMBus(I2C_BUS)
            for addr in LDC_ADDRS:
                try:
                    sensor = LDC1614(ldc_bus, addr)  # type: ignore[call-arg]
                    if sensor.init():
                        ldc_sensors.append(sensor)
                        detected_ldc_addrs.append(addr)
                        last_tension_values.append(None)
                        print(f'[INFO] LDC1614 detected 0x{addr:02X}')
                except Exception:  # noqa: BLE001
                    pass
            if not ldc_sensors:
                print('[WARN] No LDC1614 detected')
        except Exception as e:  # noqa: BLE001
            print(f'[WARN] LDC init failed: {e}')
            ldc_bus = None

    csv_f, csv_path = _init_csv(f'integrated_log_{create_timestamp()}_sin_wave_python.csv', detected_ldc_addrs, enc is not None, adc is not None)

    t0 = time.perf_counter()
    interval_s = LOOP_INTERVAL_MS / 1000.0
    next_tick = t0
    try:
        while RUN:
            now = time.perf_counter()
            if now < next_tick:
                # poll encoder between frames for better edge capture
                if enc is not None:
                    try:
                        enc.poll()
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(0.00015)
                continue
            t = now - t0
            if TOTAL_SEC > 0 and t > TOTAL_SEC:
                break
            # keep stable cadence
            while now - next_tick > interval_s:
                next_tick += interval_s
            next_tick += interval_s
            a_pct = calculate_valve_a_opening(t)
            b_pct = calculate_valve_b_opening(t)
            dac.set_channels(a_pct, b_pct)
            # ADC periodic read
            if adc is not None and (now - last_adc_print) >= ADC_PRINT_INTERVAL:
                try:
                    last_adc_data = adc.read_pair()
                except Exception:  # noqa: BLE001
                    last_adc_data = None
                last_adc_print = now
            # update encoder sampled value at interval
            current_enc_deg = None
            if enc is not None:
                try:
                    enc.poll()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    current_enc_deg = enc.degrees()  # type: ignore[union-attr]
                except Exception:
                    current_enc_deg = None
                # apply invert and zero offset if available
                if current_enc_deg is not None:
                    if args.encoder_invert:
                        current_enc_deg = -current_enc_deg
                    if enc_zero_offset is not None:
                        current_enc_deg = current_enc_deg - enc_zero_offset
                if now - last_enc_print >= ENCODER_PRINT_INTERVAL:
                    last_enc_deg = current_enc_deg
                    last_enc_print = now
            # read tension sensors periodically
            if ldc_sensors and (t - last_tension_print) >= TENSION_PRINT_INTERVAL:
                for i, s in enumerate(ldc_sensors):
                    try:
                        v = s.read_ch0_induct_uH()
                        if i < len(last_tension_values):
                            last_tension_values[i] = v
                    except Exception:  # noqa: BLE001
                        pass
                last_tension_print = t
            # output
            elapsed_ms = int(t * 1000)
            out = [f"{elapsed_ms:8d}ms", f"A={a_pct:5.1f}%", f"B={b_pct:5.1f}%"]
            if last_adc_data is not None:
                _, v0, k0, _, v1, k1 = last_adc_data
                out.append(f'P0={k0:6.1f}kPa'); out.append(f'P1={k1:6.1f}kPa')
            # show up to two tension values
            for v in [val for val in last_tension_values if val is not None][:2]:
                out.append(f'T={v:7.3f}uH')
            if last_enc_deg is not None:
                out.append(f'Enc={last_enc_deg:7.2f}deg')
            print(' '.join(out))
            # CSV
            if csv_f is not None:
                try:
                    row = [str(elapsed_ms), f'{a_pct:.1f}', f'{b_pct:.1f}']
                    if adc is not None:
                        if last_adc_data is not None:
                            r0, v0, k0, r1, v1, k1 = last_adc_data
                            row += [str(r0), f'{v0:.3f}', f'{k0:.1f}', str(r1), f'{v1:.3f}', f'{k1:.1f}']
                        else:
                            row += ['', '', '', '', '', '']
                    for v in last_tension_values:
                        row.append('' if v is None else f'{v:.5f}')
                    if enc is not None and last_enc_deg is not None:
                        row.append(f'{last_enc_deg:.3f}')
                    elif enc is not None:
                        row.append('')
                    csv_f.write(','.join(row) + '\n')
                except Exception:
                    pass
    finally:
        print('\nStopping... closing valves.')
        try:
            dac.set_channels(0.0, 0.0)
        except Exception:
            pass
        try:
            dac.close()
        except Exception:
            pass
        if adc is not None:
            try:
                adc.close()
            except Exception:
                pass
        if enc is not None:
            try:
                enc.close()  # type: ignore[union-attr]
            except Exception:
                pass
        if ldc_bus is not None:
            try:
                ldc_bus.close()
            except Exception:
                pass
        if csv_f is not None:
            try:
                csv_f.close()
            except Exception:
                pass
            if csv_path:
                print(f'[INFO] CSV saved: {csv_path}')
                # Try to plot the CSV lazily (matplotlib may not be installed on uv env)
                try:
                    try:
                        import matplotlib
                        matplotlib.use('Agg')
                        import matplotlib.pyplot as plt_local
                        import numpy as np_local
                        plt = plt_local
                        np = np_local
                    except Exception as e:
                        plt = None
                        np = None
                    if plt is not None:
                        # read CSV and plot valve % and encoder
                        times = []
                        a_vals = []
                        b_vals = []
                        enc_vals = []
                        try:
                            with open(csv_path, 'r') as f:
                                rdr = csv.reader(f)
                                for row in rdr:
                                    if not row:
                                        continue
                                    try:
                                        ms = float(row[0])
                                    except Exception:
                                        continue
                                    times.append(ms/1000.0)
                                    try:
                                        a_vals.append(float(row[1]))
                                    except Exception:
                                        a_vals.append(np.nan)
                                    try:
                                        b_vals.append(float(row[2]))
                                    except Exception:
                                        b_vals.append(np.nan)
                                    try:
                                        enc_vals.append(float(row[-1]))
                                    except Exception:
                                        pass
                            fig, ax1 = plt.subplots(figsize=(8,4))
                            if a_vals and b_vals:
                                ax1.plot(times, a_vals, label='A %', color='C0')
                                ax1.plot(times, b_vals, label='B %', color='C1')
                                ax1.set_ylabel('Valve %')
                                ax1.legend(loc='upper left')
                            if enc_vals:
                                ax2 = ax1.twinx()
                                ax2.plot(times[:len(enc_vals)], enc_vals, label='Encoder deg', color='C2')
                                ax2.set_ylabel('Encoder (deg)')
                                lines1, labels1 = ax1.get_legend_handles_labels()
                                lines2, labels2 = ax2.get_legend_handles_labels()
                                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                            ax1.set_xlabel('Time (s)')
                            out_png = Path(csv_path).with_suffix('.png')
                            fig.tight_layout()
                            fig.savefig(out_png, dpi=150)
                            plt.close(fig)
                            print(f'[INFO] Plot saved: {out_png}')
                        except Exception as e:
                            print(f'[WARN] Plotting failed: {e}')
                except Exception:
                    pass
        print('Done.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
