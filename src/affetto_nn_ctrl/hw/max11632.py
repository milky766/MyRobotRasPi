from __future__ import annotations

import time
try:
    import spidev  # type: ignore
except Exception:  # noqa: BLE001
    spidev = None  # type: ignore
from .gpio import CsLine

SPI_MAX_HZ = 1_000_000
SPI_MODE_ADC = 0b00

class Max11632:
    """MAX11632 SPI ADC (manual CS), single-ended immediate conversion polling.

    NOTE: Current implementation mirrors previous monolithic script. Additional
    features (EOC line, burst scanning) can be added later.
    """
    def __init__(self, bus: int, dev: int, cs_gpio: int, ch0: int, ch1: int):
        if spidev is None:
            raise RuntimeError("spidev not available")
        self.spi = spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        # CS is active-low on our hardware
        self.cs = CsLine(cs_gpio, active_high=False)
        self.opened = False
        self.ch0 = ch0
        self.ch1 = ch1

    def open(self) -> None:
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            # Ensure kernel CS is disabled to avoid CE0 toggling
            try:
                self.spi.no_cs = True
            except Exception:
                pass
            self.spi.max_speed_hz = SPI_MAX_HZ
            self.spi.mode = SPI_MODE_ADC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_init()

    def close(self) -> None:  # pragma: no cover - hw cleanup
        if self.opened:
            try:
                self.spi.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.cs.close()
        except Exception:  # noqa: BLE001
            pass

    def _xfer(self, tx, rxlen: int = 0):
        if not self.opened:
            raise RuntimeError("ADC SPI not opened")
        self.spi.mode = SPI_MODE_ADC
        # Assert CS, transfer, deassert CS (single cycle)
        try:
            self.cs.set_active()
        except Exception:
            try:
                self.cs.set(0)
            except Exception:
                pass
        time.sleep(0.00001)
        if rxlen:
            rx = self.spi.xfer2(list(tx) + [0x00] * rxlen)
            try:
                self.cs.set_inactive()
            except Exception:
                try:
                    self.cs.set(1)
                except Exception:
                    pass
            time.sleep(0.00001)
            return rx[len(tx):]
        self.spi.xfer2(list(tx))
        try:
            self.cs.set_inactive()
        except Exception:
            try:
                self.cs.set(1)
            except Exception:
                pass
        time.sleep(0.00001)
        return []

    def _reset_init(self) -> None:
        self._xfer([0x00])  # reset
        time.sleep(0.001)
        self._xfer([0x64])  # setup example
        time.sleep(0.005)

    def read_raw(self, ch: int) -> int:
        cmd = 0x80 | ((ch & 0x0F) << 3) | (0b11 << 1)
        # Start conversion, then read exactly 16 clocks under same CS (no leading extra byte)
        self._xfer([cmd])
        time.sleep(0.0001)  # 100 us conversion
        d = self._xfer([], rxlen=2)
        raw = ((d[0] & 0x0F) << 8) | d[1]
        return raw

    @staticmethod
    def raw_to_voltage(raw: int) -> float:
        return (raw / 4095.0) * 5.0

    @staticmethod
    def voltage_to_kpa(v: float, *, v_min: float = 1.0, v_max: float = 5.0, p_min_mpa: float = 0.003, p_max_mpa: float = 0.6) -> float:
        mpa = (v - v_min) * (p_max_mpa - p_min_mpa) / (v_max - v_min) + p_min_mpa
        return mpa * 1000.0

    def read_pair(self):
        r0 = self.read_raw(self.ch0)
        r1 = self.read_raw(self.ch1)
        v0 = self.raw_to_voltage(r0)
        v1 = self.raw_to_voltage(r1)
        k0 = self.voltage_to_kpa(v0)
        k1 = self.voltage_to_kpa(v1)
        return (r0, v0, k0, r1, v1, k1)

__all__ = ["Max11632"]
