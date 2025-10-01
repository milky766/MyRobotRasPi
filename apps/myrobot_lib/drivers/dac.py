"""Simple DAC8564 driver reverted to original import-at-module-time behavior."""
import time
try:
    import spidev  # type: ignore
except Exception:  # pragma: no cover
    spidev = None  # type: ignore

try:
    import gpiod  # type: ignore
    # Newer libgpiod exposes line helpers in gpiod.line
    try:
        from gpiod.line import Direction, Value  # type: ignore
    except Exception:  # pragma: no cover
        Direction = None  # type: ignore
        Value = None  # type: ignore
except Exception:  # pragma: no cover
    gpiod = None  # type: ignore
    Direction = None  # type: ignore
    Value = None  # type: ignore

# SPI / DAC constants
SPI_MAX_HZ = 1_000_000
SPI_MODE_DAC = 0b01  # MODE1 for DAC8564

# Safety cap for valve percent
VALVE_MAX_PCT = 100.0

class CsLine:
    """Minimal CS line using libgpiod v2-style request_lines when available.

    Falls back to older line.request / request_output APIs if needed.
    """
    def __init__(self, pin: int):
        if gpiod is None:
            raise RuntimeError('python3-libgpiod is required')
        self.pin = pin
        # Prefer opening chip0 as in pid_tune.py
        self.chip = gpiod.Chip('/dev/gpiochip0')
        # Try new request_lines + LineSettings API
        try:
            ls = gpiod.LineSettings()
            if Direction is not None:
                ls.direction = Direction.OUTPUT
            if Value is not None:
                ls.output_value = Value.ACTIVE
            # request_lines expects a mapping pin->LineSettings in some bindings
            try:
                # Some libgpiod bindings accept config dict
                self.req = self.chip.request_lines(consumer='dac_cs', config={pin: ls})
            except Exception:
                # Older/newer variations might accept (consumer, {pin: ls}) or different signature
                # Try a simple form and fall back to old-style API below
                try:
                    self.req = self.chip.request_lines({pin: ls})
                except Exception:
                    raise
            try:
                # Set inactive (high)
                self.set(1)
            except Exception:
                pass
            return
        except Exception:
            # Fall back to old-style line.request APIs
            pass

        try:
            line = self.chip.get_line(pin)
            try:
                # preferred old-style request API
                line.request(consumer='dac_cs', type=gpiod.LINE_REQ_DIR_OUT, default_vals=[1])
            except Exception:
                # fallback to request_output helper
                try:
                    line.request_output(consumer='dac_cs', default_vals=[1])
                except Exception:
                    raise
            self.req = line
        except Exception as e:
            raise RuntimeError('DAC CS init failed') from e
        try:
            self.set(1)
        except Exception:
            pass

    def set(self, val: int):
        # Use Value constants if available (new API)
        try:
            if Value is not None and hasattr(self.req, 'set_values'):
                self.req.set_values({self.pin: (Value.ACTIVE if val else Value.INACTIVE)})
                return
        except Exception:
            pass
        # Old-style API: set_value or set
        try:
            if hasattr(self.req, 'set_value'):
                self.req.set_value(int(bool(val)))
            elif hasattr(self.req, 'set'):
                self.req.set(int(bool(val)))
        except Exception:
            pass

    def close(self):
        try:
            self.set(1)
        except Exception:
            pass
        try:
            if hasattr(self.req, 'release'):
                self.req.release()
        except Exception:
            pass
        try:
            self.chip.close()
        except Exception:
            pass


class Dac8564:
    """Direct DAC8564 access with manual CS per transfer (aligned with pid_tune.py)."""
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
            try:
                self.spi.bits_per_word = 8
            except Exception:
                pass
            self.opened = True
            self._reset_and_init()

    def close(self):  # pragma: no cover
        if self.opened:
            try:
                self.spi.close()
            except Exception:
                pass
        try:
            if self.cs is not None:
                self.cs.close()
        except Exception:
            pass

    def _xfer(self, data: list[int]):
        if not self.opened:
            raise RuntimeError('SPI not opened yet')
        # ensure correct spi mode
        try:
            self.spi.mode = SPI_MODE_DAC
        except Exception:
            pass
        if self.cs is None:
            raise RuntimeError('CS line not initialized')
        self.cs.set(0)
        try:
            self.spi.xfer2(list(data))
        finally:
            # always release CS
            self.cs.set(1)

    @staticmethod
    def _pct_to_code(pct: float) -> int:
        if pct < 0.0:
            pct = 0.0
        if pct > VALVE_MAX_PCT:
            pct = VALVE_MAX_PCT
        return int(pct * 65535.0 / 100.0 + 0.5)

    @staticmethod
    def _cmd(ch: int) -> int:
        # 0x10 (write input) with LDAC low -> immediate update
        return 0x10 | (ch << 1)

    def set_channels(self, a_pct: float, b_pct: float):
        a_code = self._pct_to_code(a_pct)
        b_code = self._pct_to_code(b_pct)
        a = [self._cmd(0), (a_code >> 8) & 0xFF, a_code & 0xFF]
        b = [self._cmd(1), (b_code >> 8) & 0xFF, b_code & 0xFF]
        self._xfer(a)
        self._xfer(b)

    def _reset_and_init(self):
        # Reset and power config (matching pid_tune.py)
        self._xfer([0x28, 0x00, 0x01])
        time.sleep(0.001)
        self._xfer([0x38, 0x00, 0x01])


__all__ = ['Dac8564']
