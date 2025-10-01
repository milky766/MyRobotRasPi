from __future__ import annotations

import time
try:
    import spidev  # type: ignore
except Exception:  # noqa: BLE001
    spidev = None  # type: ignore
from .gpio import CsLine

SPI_MAX_HZ = 1_000_000
SPI_MODE_DAC = 0b01

class Dac8564:
    """Minimal DAC8564 interface with manual CS (shared SPI bus)."""
    def __init__(self, bus: int, dev: int, cs_gpio: int):
        if spidev is None:
            raise RuntimeError("spidev not available")
        self.spi = spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        # DAC CS is active-low on our hardware
        self.cs = CsLine(cs_gpio, active_high=False)
        self.opened = False

    def open(self) -> None:
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            # disable kernel-controlled CS so we control CS via GPIO manually
            try:
                self.spi.no_cs = True
            except Exception:
                pass
            self.spi.max_speed_hz = SPI_MAX_HZ
            self.spi.mode = SPI_MODE_DAC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_and_init()

    def close(self) -> None:  # pragma: no cover - hardware cleanup
        try:
            self.set_channels(0.0, 0.0)
        except Exception:  # noqa: BLE001
            pass
        if self.opened:
            try:
                self.spi.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.cs.close()
        except Exception:  # noqa: BLE001
            pass

    def _xfer(self, data) -> None:
        # Debug log to /tmp/hw_debug.log
        try:
            with open("/tmp/hw_debug.log", "a") as f:
                f.write(f"[Dac8564] xfer: {data}\n")
        except Exception:
            pass
        if not self.opened:
            raise RuntimeError("SPI not opened")
        self.spi.mode = SPI_MODE_DAC
        # Assert CS (active) then transfer then deassert CS (inactive)
        try:
            self.cs.set_active()
        except Exception:
            try:
                self.cs.set(0)
            except Exception:
                pass
        # tiny delay to ensure device samples MOSI after CS assertion
        time.sleep(0.00001)
        self.spi.xfer2(list(data))
        # deactivate CS (single cycle) and allow settle
        try:
            self.cs.set_inactive()
        except Exception:
            try:
                self.cs.set(1)
            except Exception:
                pass
        # allow CS to settle
        time.sleep(0.00001)
    @staticmethod
    def _pct_to_code(pct: float) -> int:
        if pct < 0.0:
            pct = 0.0
        if pct > 100.0:
            pct = 100.0
        return int(pct * 65535.0 / 100.0 + 0.5)

    @staticmethod
    def _cmd(ch: int) -> int:
        # Match the working script: 0x10 = write input register for DAC n.
        # Our board ties LDAC low, so output updates immediately after write.
        return 0x10 | (ch << 1)

    def set_channels(self, a_pct: float, b_pct: float) -> None:
        a_code = self._pct_to_code(a_pct)
        b_code = self._pct_to_code(b_pct)
        # Map logical A/B to physical DAC channels 0 and 1 (match C++ implementation)
        a = [self._cmd(0), (a_code >> 8) & 0xFF, a_code & 0xFF]
        b = [self._cmd(1), (b_code >> 8) & 0xFF, b_code & 0xFF]
        self._xfer(a)
        self._xfer(b)

    def _reset_and_init(self) -> None:
        # Software reset then power-up (same as working python script)
        self._xfer([0x28, 0x00, 0x01])
        time.sleep(0.001)
        self._xfer([0x38, 0x00, 0x01])
        # Do NOT disable internal reference here (board relies on it)

__all__ = ["Dac8564"]
