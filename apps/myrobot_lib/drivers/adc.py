"""MAX11632 driver implementation moved from app script."""
import time
try:
    import spidev  # type: ignore
except Exception:
    spidev = None

SPI_MODE_ADC = 0b00

class CsLine:
    def __init__(self, pin: int):
        try:
            import gpiod
            from gpiod.line import Direction, Value
        except Exception:
            gpiod = None
            Direction = Value = None
        if gpiod is None:
            raise RuntimeError('python3-libgpiod is required')
        self.pin = pin
        self.chip = gpiod.Chip('/dev/gpiochip0')
        ls = gpiod.LineSettings()
        if Direction is not None:
            ls.direction = Direction.OUTPUT
        if Value is not None:
            ls.output_value = Value.ACTIVE
        self.req = self.chip.request_lines(consumer='adc_cs', config={pin: ls})
        try:
            self.set(1)
        except Exception:
            pass
    def set(self, val: int):
        try:
            from gpiod.line import Value
            self.req.set_values({self.pin: (Value.ACTIVE if val else Value.INACTIVE)})
        except Exception:
            try:
                self.req.set_values({self.pin: val})
            except Exception:
                pass
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

class Max11632:
    def __init__(self, bus: int, dev: int, cs_gpio: int, ch0: int, ch1: int):
        if spidev is None:
            raise RuntimeError('spidev is required')
        self.spi = spidev.SpiDev()
        self.bus = bus
        self.dev = dev
        self.cs = CsLine(cs_gpio)
        self.opened = False
        self.ch0 = ch0
        self.ch1 = ch1
    def open(self):
        if not self.opened:
            self.spi.open(self.bus, self.dev)
            self.spi.max_speed_hz = 1_000_000
            self.spi.mode = SPI_MODE_ADC
            self.spi.bits_per_word = 8
            self.opened = True
            self._reset_init()
    def close(self):
        if self.opened:
            try:
                self.spi.close()
            except Exception:
                pass
        try:
            self.cs.close()
        except Exception:
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
        self._xfer([0x00])
        time.sleep(0.001)
        self._xfer([0x64])
        time.sleep(0.005)
    def read_raw(self, ch: int) -> int:
        cmd = 0x80 | ((ch & 0x0F) << 3) | (0b11 << 1)
        self._xfer([cmd])
        time.sleep(0.0001)  # 100us conversion
        d = self._xfer([], rxlen=2)
        raw = ((d[0] & 0x0F) << 8) | d[1]
        return raw
    @staticmethod
    def raw_to_voltage(raw: int) -> float:
        return (raw / 4095.0) * 5.0
    @staticmethod
    def voltage_to_kpa(v: float, *, v_min: float = 1.0, v_max: float = 5.0, p_min_mpa: float = 0.003, p_max_mpa: float = 0.6) -> float:
        """
        Convert ADC voltage to kPa using the same linear mapping as in pid_tune/integrated scripts:
        defaults map 1.0–5.0 V -> 0.003–0.6 MPa (i.e. 3–600 kPa).
        An optional per-sensor calibration (scale, offset) from `apps.myrobot_lib.config.ADC_CALIBRATION`
        is applied to the resulting kPa value.
        """
        try:
            from apps.myrobot_lib import config as cfg
            calib = getattr(cfg, 'ADC_CALIBRATION', None)
            if isinstance(calib, dict):
                scale = float(calib.get('scale', 1.0))
                offset = float(calib.get('offset', 0.0))
            else:
                scale = 1.0
                offset = 0.0
        except Exception:
            scale = 1.0
            offset = 0.0
        # linear map voltage -> MPa
        mpa = (v - v_min) * (p_max_mpa - p_min_mpa) / (v_max - v_min) + p_min_mpa
        kpa = mpa * 1000.0
        return kpa * scale + offset
    def read_pair(self):
        r0 = self.read_raw(self.ch0)
        r1 = self.read_raw(self.ch1)
        v0 = self.raw_to_voltage(r0)
        v1 = self.raw_to_voltage(r1)
        k0 = self.voltage_to_kpa(v0)
        k1 = self.voltage_to_kpa(v1)
        return (r0, v0, k0, r1, v1, k1)

__all__ = ['Max11632']
