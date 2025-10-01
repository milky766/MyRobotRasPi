from __future__ import annotations

# Import gpiod robustly (support both v1 and v2 Python bindings)
try:
    import gpiod  # type: ignore
    HAS_GPIOD = True
except Exception:  # noqa: BLE001
    gpiod = None  # type: ignore
    HAS_GPIOD = False

# Try to import v2 Direction API if present (optional)
try:
    from gpiod.line import Direction as GpioDirection  # type: ignore
    HAS_V2 = True
except Exception:  # noqa: BLE001
    GpioDirection = None  # type: ignore
    HAS_V2 = False

class EncoderSimple:
    """Minimal polled quadrature encoder.

    Poll in a tight loop between frames for best edge capture.
    Works with libgpiod v2 (preferred) and falls back to v1.
    """
    def __init__(self, chip_path: str, a_pin: int, b_pin: int):
        if not HAS_GPIOD or gpiod is None:
            raise RuntimeError("gpiod not available")
        self.chip_path = chip_path
        self.a_pin = a_pin
        self.b_pin = b_pin
        self.req = None  # v2 request_lines handle
        self.lines = None  # v1 get_lines handle
        self.last_a = None
        self.last_b = None
        self.count = 0
        self._is_v2 = False
        self._open()

    def _open(self) -> None:
        # Open chip by path or name
        self.chip = gpiod.Chip(self.chip_path)
        if HAS_V2 and hasattr(gpiod, "LineSettings") and hasattr(self.chip, "request_lines"):
            # libgpiod v2 path
            ls_in = gpiod.LineSettings()
            # When Direction API is present, use it
            if GpioDirection is not None:
                ls_in.direction = GpioDirection.INPUT  # type: ignore[attr-defined]
            else:
                # Some bindings accept string names; ignore if not supported
                try:
                    ls_in.direction = "input"  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.req = self.chip.request_lines(consumer="enc", config={self.a_pin: ls_in, self.b_pin: ls_in})
            self._is_v2 = True
        else:
            # libgpiod v1 fallback
            self.lines = self.chip.get_lines([self.a_pin, self.b_pin])
            # Request inputs
            self.lines.request(consumer="enc", type=gpiod.LINE_REQ_DIR_IN)
            self._is_v2 = False
        self._sample_initial()

    def _decode_val(self, v):  # noqa: ANN001
        # Support v2 Value types, ints, or other representations
        if hasattr(v, "value"):
            try:
                return 1 if v.value == 1 else 0
            except Exception:  # noqa: BLE001
                pass
        s = str(v)
        if s.endswith("ACTIVE") and not s.endswith("INACTIVE"):
            return 1
        try:
            return 1 if int(v) == 1 else 0
        except Exception:  # noqa: BLE001
            return 0

    def _read_ab(self):
        if self._is_v2 and self.req is not None:
            vals = self.req.get_values([self.a_pin, self.b_pin])
            return self._decode_val(vals[0]), self._decode_val(vals[1])
        elif self.lines is not None:
            vals = self.lines.get_values()
            # v1 returns list aligned with requested order
            return self._decode_val(vals[0]), self._decode_val(vals[1])
        else:  # pragma: no cover
            raise RuntimeError("encoder lines/request not initialized")

    def _sample_initial(self) -> None:
        a, b = self._read_ab()
        self.last_a = a
        self.last_b = b

    def poll(self) -> None:
        a, b = self._read_ab()
        if a != self.last_a:
            if b != a:
                self.count += 1
            else:
                self.count -= 1
        self.last_a = a
        self.last_b = b

    def degrees(self, ppr: int = 2048) -> float:
        return self.count * 360.0 / (2.0 * ppr)

    def close(self) -> None:  # pragma: no cover
        try:
            if self.req is not None and hasattr(self.req, "release"):
                self.req.release()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.lines is not None and hasattr(self.lines, "release"):
                self.lines.release()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            if hasattr(self, "chip"):
                self.chip.close()
        except Exception:  # noqa: BLE001
            pass

__all__ = ["EncoderSimple"]
