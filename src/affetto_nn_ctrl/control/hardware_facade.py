from __future__ import annotations
from typing import List
from affetto_nn_ctrl.hw.dac8564 import Dac8564
from affetto_nn_ctrl.hw.max11632 import Max11632
from affetto_nn_ctrl.hw.encoder import EncoderSimple
from affetto_nn_ctrl.hw.ldc1614 import LDC1614

class HardwareFacade:
    def __init__(self, dac: Dac8564, adc: Max11632 | None, enc: EncoderSimple | None, ldc_sensors: List[LDC1614]):
        self.dac = dac
        self.adc = adc
        self.enc = enc
        self.ldc_sensors = ldc_sensors
    def set_valves(self, a: float, b: float):
        self.dac.set_channels(a, b)
    def read_angle(self) -> float | None:
        if self.enc is None:
            return None
        try:
            self.enc.poll(); return self.enc.degrees()
        except Exception:  # noqa: BLE001
            return None
    def read_pressures(self):
        if self.adc is None:
            return None
        try:
            r0,v0,k0,r1,v1,k1 = self.adc.read_pair(); return (k0, k1)
        except Exception:  # noqa: BLE001
            return None
    def read_tensions(self) -> List[float | None]:
        vals: List[float | None] = []
        for s in self.ldc_sensors:
            try: vals.append(s.read_ch0_induct_uH())
            except Exception: vals.append(None)  # noqa: BLE001
        return vals
__all__ = ['HardwareFacade']
