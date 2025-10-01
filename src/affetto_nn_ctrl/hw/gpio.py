from __future__ import annotations

import time
# Robust gpiod import supporting both v2 and v1 bindings
try:
    import gpiod  # type: ignore
except Exception:  # noqa: BLE001
    gpiod = None  # type: ignore

# Try optional v2 helpers; if unavailable keep as None
try:  # noqa: SIM105
    from gpiod.line import Direction, Value  # type: ignore
except Exception:  # noqa: BLE001
    Direction = None  # type: ignore
    Value = None  # type: ignore

class CsLine:
    """Chip-select abstraction supporting libgpiod v2 and v1 APIs.

    Active state drives the electrical level that asserts the device CS.
    When active_high=False, the line is configured as active-low so that
    Value.ACTIVE outputs a low level (on v2). For v1, we emulate via 0/1.
    """
    def __init__(self, pin: int, chip: str = "/dev/gpiochip0", *, active_high: bool = True):
        if gpiod is None:
            raise RuntimeError("libgpiod not available")
        self.pin = pin
        self._active_high = active_high
        self._req = None
        self._line = None
        # Open chip
        self._chip = gpiod.Chip(chip)
        # Try v2 request_lines path first
        try:
            ls = gpiod.LineSettings()
            # direction
            if Direction is not None:
                ls.direction = Direction.OUTPUT
            else:  # best-effort if bindings differ
                try:
                    ls.direction = "output"  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Configure active_low mapping if API provides it
            try:
                ls.active_low = (not active_high)  # type: ignore[attr-defined]
            except Exception:
                pass
            # Default to inactive state
            try:
                if Value is not None:
                    ls.output_value = Value.INACTIVE
            except Exception:
                pass
            # Request line(s)
            self._req = self._chip.request_lines(consumer="cs", config={pin: ls})
            # Ensure inactive
            self.set_inactive()
            self._is_v2 = True
            return
        except Exception:
            # Fall through to v1 API
            self._is_v2 = False

        # v1 fallback: get single line and request as output high (inactive by default)
        try:
            # Try several ways to obtain a Line object depending on binding version
            line = None
            if hasattr(self._chip, 'get_line'):
                line = self._chip.get_line(pin)
            elif hasattr(self._chip, 'get_lines'):
                lines = self._chip.get_lines([pin])
                line = lines[0] if lines else None
            else:
                # some bindings expose a lines() method
                try:
                    lines = self._chip.lines()
                    line = lines[pin]
                except Exception:
                    line = None
            if line is None:
                raise RuntimeError('failed to obtain line from chip')
            try:
                line.request(consumer="cs", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[1 if active_high else 0])
            except Exception:
                # Older helpers
                line.request_output(consumer="cs", default_vals=[1 if active_high else 0])
            self._line = line
            # Ensure inactive (logical 0 means inactive when active_high=True)
            self.set_inactive()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("libgpiod v1 request failed") from e

    def set_active(self) -> None:
        try:
            with open("/tmp/hw_debug.log", "a") as f:
                f.write(f"[CsLine] set_active pin={self.pin}\n")
        except Exception:
            pass
        if self._is_v2 and self._req is not None:
            try:
                if Value is not None:
                    self._req.set_values({self.pin: Value.ACTIVE})
                    return
            except Exception:
                pass
            # Fallback toggle with literal 1
            self._req.set_values({self.pin: 1})  # type: ignore[arg-type]
        elif self._line is not None:
            # For active-low wiring (active_high=False), active means drive 0
            val = 1 if self._active_high else 0
            try:
                if hasattr(self._line, 'set_value'):
                    self._line.set_value(val)
                else:
                    self._line.set(val)  # type: ignore[attr-defined]
            except Exception:
                pass

    def set_inactive(self) -> None:
        try:
            with open("/tmp/hw_debug.log", "a") as f:
                f.write(f"[CsLine] set_inactive pin={self.pin}\n")
        except Exception:
            pass
        if self._is_v2 and self._req is not None:
            try:
                if Value is not None:
                    self._req.set_values({self.pin: Value.INACTIVE})
                    return
            except Exception:
                pass
            self._req.set_values({self.pin: 0})  # type: ignore[arg-type]
        elif self._line is not None:
            # For active-low wiring, inactive means drive 1
            val = 0 if self._active_high else 1
            try:
                if hasattr(self._line, 'set_value'):
                    self._line.set_value(val)
                else:
                    self._line.set(val)  # type: ignore[attr-defined]
            except Exception:
                pass

    def pulse(self, dt: float = 1e-6) -> None:
        self.set_active(); time.sleep(dt); self.set_inactive()

    def set(self, val: int) -> None:
        if val:
            self.set_active()
        else:
            self.set_inactive()

    def close(self) -> None:
        try:
            self.set_inactive()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._req is not None and hasattr(self._req, 'release'):
                self._req.release()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._line is not None and hasattr(self._line, 'release'):
                self._line.release()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
        try:
            self._chip.close()
        except Exception:  # noqa: BLE001
            pass

__all__ = ["CsLine"]
