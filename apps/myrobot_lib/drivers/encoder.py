"""Encoder driver implementation moved from app script."""
try:
    import gpiod
    from gpiod.line import Direction, Value
except Exception:
    gpiod = None
    Direction = Value = None

class EncoderSimple:
    def __init__(self, chip_path: str, a_pin: int, b_pin: int):
        if gpiod is None:
            raise RuntimeError('gpiod required for EncoderSimple')
        self.chip_path = chip_path
        self.a_pin = a_pin
        self.b_pin = b_pin
        self.req = None
        self.last_a = None
        self.last_b = None
        self.count = 0
        self._open()
    def _open(self):
        if gpiod is None:
            return
        self.chip = gpiod.Chip(self.chip_path)
        ls_in = gpiod.LineSettings()
        if Direction is not None:
            ls_in.direction = Direction.INPUT
        self.req = self.chip.request_lines(consumer='enc', config={self.a_pin: ls_in, self.b_pin: ls_in})
        self._sample_initial()
    def _decode_val(self, v):
        if hasattr(v, 'value'):
            return 1 if v.value == 1 else 0
        s = str(v)
        if s.endswith('ACTIVE') and not s.endswith('INACTIVE'):
            return 1
        try:
            return 1 if int(v) == 1 else 0
        except Exception:
            return 0
    def _sample_initial(self):
        vals = self.req.get_values([self.a_pin, self.b_pin])
        self.last_a = self._decode_val(vals[0])
        self.last_b = self._decode_val(vals[1])
    def poll(self):
        if gpiod is None or self.req is None:
            return
        vals = self.req.get_values([self.a_pin, self.b_pin])
        a = self._decode_val(vals[0])
        b = self._decode_val(vals[1])
        if a != self.last_a:
            if b != a:
                self.count += 1
            else:
                self.count -= 1
        self.last_a = a; self.last_b = b
    def degrees(self, ppr: int = 2048) -> float:
        return self.count * 360.0 / (2.0 * ppr)
    def close(self):
        try:
            if self.req: self.req.release()
        except Exception:
            pass
        try:
            if hasattr(self, 'chip'): self.chip.close()
        except Exception:
            pass

__all__ = ['EncoderSimple']
