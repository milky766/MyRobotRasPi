"""LDC1614 driver implementation moved from app script."""
import time
import math
from apps.myrobot_lib import config as cfg

class LDC1614:
    def __init__(self, bus, addr: int):
        self.bus = bus
        self.addr = addr
        self.ok = False
    def _w16(self, reg: int, val: int):
        hi = (val >> 8) & 0xFF; lo = val & 0xFF
        self.bus.write_i2c_block_data(self.addr, reg, [hi, lo])
        time.sleep(0.010)
    def _r16(self, reg: int) -> int | None:
        try:
            self.bus.write_byte(self.addr, reg)
            data = self.bus.read_i2c_block_data(self.addr, reg, 2)
            return (data[0] << 8) | data[1]
        except Exception:
            return None
    def init(self) -> bool:
        try:
            self._w16(0x08,0x4E20); self._w16(0x09,0x4E20); self._w16(0x0A,0x4E20); self._w16(0x0B,0x4E20)
            self._w16(0x0C,0x0000); self._w16(0x0D,0x0000); self._w16(0x0E,0x0000); self._w16(0x0F,0x0000)
            self._w16(0x10,0x1D4C); self._w16(0x11,0x1D4C); self._w16(0x12,0x1D4C); self._w16(0x13,0x1D4C)
            self._w16(0x14,0x1001); self._w16(0x15,0x1001); self._w16(0x16,0x1001); self._w16(0x17,0x1001)
            self._w16(0x1E,0x8C40); self._w16(0x1F,0x8C40); self._w16(0x20,0x8C40); self._w16(0x21,0x8C40)
            self._w16(0x1B,0xC20C); self._w16(0x1A,0x1E01)
            self.ok = True
        except Exception:
            self.ok = False
        return self.ok
    def read_ch0_induct_uH(self) -> float | None:
        if not self.ok:
            return None
        hi = self._r16(0x00); lo = self._r16(0x01)
        if hi is None or lo is None:
            return None
        raw28 = ((hi & 0x0FFF) << 16) | lo
        freq = (raw28 / (1 << 28)) * cfg.LDC_FREF_HZ
        if freq <= 0:
            return None
        L = 1.0 / ((2.0 * math.pi * freq) ** 2 * cfg.LDC_CPAR_F)
        return L * 1e6  # uH

__all__ = ['LDC1614']
