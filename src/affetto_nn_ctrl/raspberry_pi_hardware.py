"""Raspberry Pi hardware abstraction (skeleton, simulation only).

Goal: Provide affctrllib-compatible triple (comm, ctrl, state) interfaces so that
existing control_utility functions can operate without modification.

This initial version:
- Does NOT access real hardware (SPI/I2C/GPIO not invoked)
- Provides zero (or constant) sensor values
- Accepts command arrays and stores the last sent values
- Simulates timing via timestamps only (no background thread yet)

Next steps (future incremental additions):
1. Implement DAC (DAC8564) write over spidev when available
2. Implement ADC (MAX11632) read over smbus2 with EOC handling
3. Implement LDC1614 initialization & read
4. Implement quadrature encoder counting (gpiod events) & dq estimation
5. Add filtering / scaling using config calibration

Usage pattern mirrors affctrllib objects:
    comm.send_commands(ca, cb)
    ca, cb = ctrl.update(t, q, dq, pa, pb, qdes, dqdes)
    rq, rdq, rpa, rpb = state.get_raw_states()

"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional
from pathlib import Path
import numpy as np
import threading
import time
try:
    import tomllib  # Python 3.11+
except Exception:  # noqa: BLE001
    import tomli as tomllib  # type: ignore

try:
    import spidev  # type: ignore
except Exception:  # noqa: BLE001
    spidev = None  # type: ignore
try:
    import smbus2  # type: ignore
except Exception:  # noqa: BLE001
    smbus2 = None  # type: ignore
try:
    import gpiod  # type: ignore
except Exception:  # noqa: BLE001
    gpiod = None  # type: ignore
# Try importing higher-level hw helpers if present
try:
    from affetto_nn_ctrl.hw.encoder import EncoderSimple  # type: ignore
except Exception:  # noqa: BLE001
    EncoderSimple = None  # type: ignore
try:
    from affetto_nn_ctrl.hw.ldc1614 import LDC1614  # type: ignore
except Exception:  # noqa: BLE001
    LDC1614 = None  # type: ignore

# ---------------------------------------------------------------------------
# Communication (command dispatch)
# ---------------------------------------------------------------------------
class RPiComm:
    """Mimic AffComm minimal interface."""

    def __init__(self, dof: int) -> None:
        self.dof = dof
        self._last_ca = np.zeros(dof, dtype=float)
        self._last_cb = np.zeros(dof, dtype=float)

    def create_command_socket(self) -> None:  # compatibility no-op
        pass

    def send_commands(self, ca: np.ndarray, cb: np.ndarray) -> None:
        # Clamp 0..1 for safety (simulation assumption)
        self._last_ca = np.clip(ca, 0.0, 1.0)
        self._last_cb = np.clip(cb, 0.0, 1.0)

    @property
    def last_commands(self) -> tuple[np.ndarray, np.ndarray]:
        return self._last_ca, self._last_cb


# ---------------------------------------------------------------------------
# Position Controller (placeholder)
# ---------------------------------------------------------------------------
class RPiPosCtrl:
    """Placeholder controller replicating interface of AffPosCtrl.

    Current behavior: pass-through of desired pressures generated from q-des.
    Later will implement mapping (e.g., simple P controller or learned policy).
    """

    def __init__(self, dof: int, freq: float | None = None) -> None:
        self.dof = dof
        self.freq = 50.0 if freq is None else freq  # Hz
        self._inactive = np.zeros(dof, dtype=bool)

    # Mirror expected method signature
    def update(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray,
        pa: np.ndarray,
        pb: np.ndarray,
        qdes: np.ndarray,
        dqdes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Simple placeholder: command A proportional to (qdes - q) normalized; B zero
        err = qdes - q
        ca = 0.5 + 0.01 * err  # crude scaling bringing value nominally into 0..1
        cb = np.zeros(self.dof, dtype=float)
        ca = np.clip(ca, 0.0, 1.0)
        # Inactive joints forced to zero
        ca[self._inactive] = 0.0
        cb[self._inactive] = 0.0
        return ca, cb

    def set_inactive_joints(self, indices: list[int], *, pressure: float = 0.0) -> None:
        for i in indices:
            if 0 <= i < self.dof:
                self._inactive[i] = True
        # (pressure parameter ignored in placeholder)


# ---------------------------------------------------------------------------
# State Thread (sensor acquisition simulation)
# ---------------------------------------------------------------------------
class RPiStateThread:
    """Simulated state acquisition.

    Provides getters similar to AffStateThread.
    Raw vs filtered are identical here.
    """

    def __init__(self, dof: int, freq: float | None = None, *, butterworth: bool = True) -> None:  # noqa: ARG002
        self.dof = dof
        self.freq = 50.0 if freq is None else freq
        self._started = False
        self._t0 = perf_counter()
        # Internal state arrays
        self.rq = np.full(dof, 50.0)  # raw joint position (deg or arbitrary)
        self.rdq = np.zeros(dof)
        self.rpa = np.zeros(dof)  # raw pressure A
        self.rpb = np.zeros(dof)  # raw pressure B (unused)
        # Filtered (same for now)
        self.q = self.rq.copy()
        self.dq = self.rdq.copy()
        self.pa = self.rpa.copy()
        self.pb = self.rpb.copy()
        self._last_update_t = self._t0

    # Compatibility methods
    def prepare(self) -> None:  # placeholder (e.g., allocate buffers)
        pass

    def start(self) -> None:
        self._started = True
        self._t0 = perf_counter()
        self._last_update_t = self._t0

    # Update simulation (could be called externally before reads)
    def _simulate_update(self) -> None:
        now = perf_counter()
        dt = now - self._last_update_t
        if dt <= 0:
            return
        # Simple integrator for rq using rdq (still zeros)
        self._last_update_t = now
        # Filtered copies
        self.q = self.rq.copy()
        self.dq = self.rdq.copy()
        self.pa = self.rpa.copy()
        self.pb = self.rpb.copy()

    # API expected by control_utility
    def get_raw_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._simulate_update()
        return self.rq.copy(), self.rdq.copy(), self.rpa.copy(), self.rpb.copy()

    def get_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._simulate_update()
        return self.q.copy(), self.dq.copy(), self.pa.copy(), self.pb.copy()


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------
@dataclass
class RPiControllerBundle:
    comm: RPiComm
    ctrl: RPiPosCtrl
    state: RPiStateThread

    @property
    def triple(self) -> tuple[RPiComm, RPiPosCtrl, RPiStateThread]:
        return self.comm, self.ctrl, self.state


def create_rpi_controller(
    dof: int,
    sensor_freq: Optional[float] = None,
    control_freq: Optional[float] = None,
) -> tuple[RPiComm, RPiPosCtrl, RPiStateThread]:
    """Create simulated Raspberry Pi controller triple.

    Returns
    -------
    (comm, ctrl, state): tuple implementing minimal AffComm/AffPosCtrl/AffStateThread interface.
    """
    comm = RPiComm(dof)
    state = RPiStateThread(dof, freq=sensor_freq)
    ctrl = RPiPosCtrl(dof, freq=control_freq)
    state.prepare()
    state.start()
    return comm, ctrl, state


# ---------------------------------------------------------------------------
# Real hardware support (partial)
# ---------------------------------------------------------------------------
class _DAC8564:
    """Minimal DAC8564 writer (single device assumption).

    NOTE: chip index in mapping is currently ignored (TODO: multi-chip support).
    """
    def __init__(self, bus: int, device: int, mode: int, max_speed_hz: int) -> None:
        if spidev is None:
            raise RuntimeError("spidev not available")
        self._spi = spidev.SpiDev()
        self._spi.open(bus, device)
        self._spi.mode = mode
        self._spi.max_speed_hz = max_speed_hz

    def write_channel(self, channel: int, value_norm: float) -> None:
        v = int(max(0.0, min(1.0, value_norm)) * 0xFFFF)  # 16-bit
        cmd = 0x3  # write & update
        addr = channel & 0x0F
        b0 = (cmd << 4) | addr
        b1 = (v >> 8) & 0xFF
        b2 = v & 0xFF
        self._spi.xfer2([b0, b1, b2])

    def close(self) -> None:  # pragma: no cover - trivial
        try:
            self._spi.close()
        except Exception:  # noqa: BLE001
            pass


class _MAX11632:
    """Minimal MAX11632 single-ended polling reader."""
    def __init__(self, bus: int, address: int, v_ref: float) -> None:
        if smbus2 is None:
            raise RuntimeError("smbus2 not available")
        self._bus = smbus2.SMBus(bus)
        self.address = address
        self.v_ref = v_ref
        # Basic setup: internal ref, unipolar, single-ended; see datasheet.
        # Setup byte: 1 0 SEL2 SEL1 SEL0 CLK REF PD1 PD0
        # Use internal ref, internal clock, normal mode.
        setup = 0b10011000  # example safe default
        self._bus.write_byte(self.address, setup)

    def read_channel(self, ch: int) -> int:
        # Config byte: 0 1 SCAN2 SCAN1 SCAN0 CS3 CS2 CS1 CS0 (datasheet simplified)
        # Simplify: single channel conversion request each time.
        config = 0b01100000 | (ch & 0x0F)  # single channel
        self._bus.write_byte(self.address, config)
        # Wait conversion (~ few tens of us); use 0.5ms conservative.
        time.sleep(0.0005)
        data = self._bus.read_word_data(self.address, 0)  # 16 bits; upper 12 valid
        # read_word_data may swap endian; adjust by bus function (depends). Simplify:
        raw = ((data & 0xFF) << 8) | (data >> 8)
        raw >>= 4  # align 12-bit
        return raw & 0x0FFF

    def read_channels(self, channels: list[int]) -> list[int]:
        return [self.read_channel(ch) for ch in channels]

    def to_voltage(self, raw: int) -> float:
        return self.v_ref * raw / 4095.0

    def close(self) -> None:  # pragma: no cover
        try:
            self._bus.close()
        except Exception:  # noqa: BLE001
            pass


class RealRPiStateThread(RPiStateThread):
    """State thread with real DAC/ADC interaction (encoders/LDC pending)."""
    def __init__(self, config: dict, *, freq: float | None = None) -> None:
        robot = config.get("robot", {})
        dof = int(robot.get("dof", 6))
        super().__init__(dof, freq=freq)
        hw = config.get("hardware", {})
        dac_cfg = hw.get("dac", {})
        adc_cfg = hw.get("adc", {})
        self._dac_map = dac_cfg.get("map", [])
        self._normalize = dac_cfg.get("normalize", True)
        self._dac = None
        self._adc = None
        # Optional hardware interfaces
        self._encoders: list[object] = []
        self._ldc_sensors: list[object] = []
        self._last_enc_vals: list[float | None] = [None] * self.dof
        self._last_enc_t: float | None = None
        if _HW_AVAILABLE:
            try:
                # Prefer manual-CS DAC if configuration provides cs_gpio
                cs_gpio = dac_cfg.get("cs_gpio", None)
                self._dac = _DAC8564_CS(
                    dac_cfg.get("spi_bus", 0),
                    dac_cfg.get("spi_device", 0),
                    dac_cfg.get("spi_mode", 1),
                    dac_cfg.get("spi_max_speed_hz", 1_000_000),
                    cs_gpio,
                )
            except Exception as e:  # noqa: BLE001
                print("[WARN] DAC init failed:", e)
            try:
                adc_cs = adc_cfg.get("cs_gpio", None)
                self._adc = _MAX11632_SPI(adc_cfg.get("spi_bus", 0), adc_cfg.get("spi_device", 0), adc_cs, adc_cfg.get("v_ref", 5.0))
            except Exception as e:  # noqa: BLE001
                print("[WARN] ADC init failed:", e)
            # Encoder initialization if configuration present
            try:
                enc_cfg = hw.get("encoders", [])
                if EncoderSimple is not None and isinstance(enc_cfg, list) and enc_cfg:
                    for econf in enc_cfg[: self.dof]:
                        chip = econf.get("chip", None)
                        a = econf.get("a", None)
                        b = econf.get("b", None)
                        if chip is not None and a is not None and b is not None:
                            try:
                                enc = EncoderSimple(chip, a, b)  # type: ignore[call-arg]
                                self._encoders.append(enc)
                            except Exception:
                                pass
            except Exception:
                pass
            # LDC sensors initialization
            try:
                ldc_cfg = hw.get("ldc", {})
                i2c_bus = ldc_cfg.get("i2c_bus", 1)
                addrs = ldc_cfg.get("addrs", [])
                if LDC1614 is not None and addrs:
                    try:
                        import smbus2 as _smbus
                        bus = _smbus.SMBus(i2c_bus)
                        for addr in addrs:
                            try:
                                s = LDC1614(bus, addr)  # type: ignore[call-arg]
                                if s.init():
                                    self._ldc_sensors.append(s)
                            except Exception:
                                pass
                    except Exception:
                        pass
        self._adc_use = adc_cfg.get("use_channels", list(range(dof)))
        # Pressure scaling (raw voltage to MPa)
        self._vmin = adc_cfg.get("pressure_v_min", 0.0)
        self._vmax = adc_cfg.get("pressure_v_max", 5.0)
        self._pmin = adc_cfg.get("pressure_p_min", 0.0)
        self._pmax = adc_cfg.get("pressure_p_max", 1.0)
        # Thread control
        self._run = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:  # override
        self._run = True
        # open devices lazily
        try:
            if hasattr(self, "_dac") and self._dac is not None:
                if hasattr(self._dac, "open"):
                    self._dac.open()
        except Exception:
            pass
        try:
            if hasattr(self, "_adc") and self._adc is not None:
                if hasattr(self._adc, "open"):
                    self._adc.open()
        except Exception:
            pass
        # start encoder state timestamps
        self._last_enc_t = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:  # compatibility helper
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout)
        # Close devices
        if getattr(self, "_dac", None) is not None:
            try:
                self._dac.close()
            except Exception:
                pass
        if getattr(self, "_adc", None) is not None:
            try:
                self._adc.close()
            except Exception:
                pass
        # close ldc/enc
        for e in getattr(self, "_encoders", []):
            try:
                e.close()  # type: ignore[union-attr]
            except Exception:
                pass
        try:
            if getattr(self, "_ldc_sensors", None):
                for s in self._ldc_sensors:
                    try:
                        s.close()
                    except Exception:
                        pass
        except Exception:
            pass

    def _sample(self) -> None:
        with self._lock:
            # Read ADC pressures
            if self._adc is not None:
                try:
                    raws = [self._adc.read_raw(ch) for ch in self._adc_use]
                    voltages = [self._adc.to_voltage(r) for r in raws]
                except Exception:
                    raws = [0 for _ in self._adc_use]
                    voltages = [0.0 for _ in self._adc_use]
                pressures = []
                for v in voltages:
                    if self._vmax > self._vmin:
                        ratio = (v - self._vmin) / (self._vmax - self._vmin)
                        pressures.append(self._pmin + ratio * (self._pmax - self._pmin))
                    else:
                        pressures.append(0.0)
                # Store to rpa (truncate/pad to dof)
                for i in range(self.dof):
                    self.rpa[i] = pressures[i] if i < len(pressures) else 0.0
                self.pa = self.rpa.copy()
            # Read encoders if available and update q/dq
            now = time.perf_counter()
            if self._encoders:
                for i, enc in enumerate(self._encoders[: self.dof]):
                    try:
                        deg = enc.degrees()  # type: ignore[union-attr]
                        last = self._last_enc_vals[i]
                        self.rq[i] = deg
                        self.q[i] = deg
                        if last is None or self._last_enc_t is None:
                            self.dq[i] = 0.0
                        else:
                            dt = max(1e-6, now - self._last_enc_t)
                            self.dq[i] = (deg - last) / dt
                        self._last_enc_vals[i] = deg
                    except Exception:
                        pass
                self._last_enc_t = now
            # Read LDC sensors (store but do not map to pressure array)
            if self._ldc_sensors:
                for idx, s in enumerate(self._ldc_sensors):
                    try:
                        # use a representative read method if available
                        v = None
                        if hasattr(s, 'read_ch0_induct_uH'):
                            v = s.read_ch0_induct_uH()  # type: ignore[union-attr]
                        elif hasattr(s, 'read'):
                            v = s.read()
                        # store to rpa tail if room (best-effort)
                        if v is not None and idx + len(self._adc_use) < self.dof:
                            self.rpa[len(self._adc_use) + idx] = float(v)
                    except Exception:
                        pass
                # reflect in pa
                self.pa = self.rpa.copy()
            # Keep pb copy
            self.pb = self.rpb.copy()
            # Ensure filtered copies sync
            self.q = self.rq.copy()
            self.dq = self.rdq.copy()


class RealRPiComm(RPiComm):
    def __init__(self, dof: int, dac: _DAC8564 | None, dac_map: list[dict], normalize: bool = True) -> None:
        super().__init__(dof)
        self._dac_hw = dac
        self._map = dac_map
        self._normalize = normalize

    def send_commands(self, ca: np.ndarray, cb: np.ndarray) -> None:  # noqa: D401
        super().send_commands(ca, cb)
        if self._dac_hw is None:
            return
        # Write only active mapping entries length == dof
        for logical_index, entry in enumerate(self._map[: self.dof]):
            try:
                ch = entry.get("ch", logical_index)
            except AttributeError:
                ch = logical_index
            val = float(ca[logical_index])
            self._dac_hw.write_channel(ch, val if self._normalize else val / 65535.0)


def create_rpi_controller_from_config(
    config_path: str | Path,
    sensor_freq: float | None = None,
    control_freq: float | None = None,
    *,

    real: bool = False,
) -> tuple[RPiComm, RPiPosCtrl, RPiStateThread]:
    """Create controller using config file (real or simulated)."""
    with Path(config_path).open("rb") as f:
        cfg = tomllib.load(f)
    robot = cfg.get("robot", {})
    dof = int(robot.get("dof", 6))
    if real:
        state: RPiStateThread = RealRPiStateThread(cfg, freq=sensor_freq)
        hw = cfg.get("hardware", {})
        dac_cfg = hw.get("dac", {})
        dac_map = dac_cfg.get("map", [])
        normalize = dac_cfg.get("normalize", True)
        dac_hw = None
        if isinstance(state, RealRPiStateThread):
            dac_hw = state._dac  # type: ignore[attr-defined]
        comm: RPiComm = RealRPiComm(dof, dac_hw, dac_map, normalize)
        ctrl = RPiPosCtrl(dof, freq=control_freq)
        state.prepare()
        state.start()
        return comm, ctrl, state
    # simulated fallback
    return create_rpi_controller(dof, sensor_freq=sensor_freq, control_freq=control_freq)

__all__ = [
    "RPiComm",
    "RPiPosCtrl",
    "RPiStateThread",
    "RPiControllerBundle",
    "create_rpi_controller",
    "create_rpi_controller_from_config",
    "RealRPiStateThread",
    "RealRPiComm",
]
