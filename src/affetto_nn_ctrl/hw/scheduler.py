from __future__ import annotations

import time
from dataclasses import dataclass

@dataclass
class FrameScheduler:
    interval_s: float
    t0: float | None = None
    next_tick: float | None = None

    def start(self) -> None:
        self.t0 = time.perf_counter()
        self.next_tick = self.t0

    def wait(self, *, poll_cb=None, poll_sleep: float = 0.00015):  # noqa: ANN001
        if self.next_tick is None:
            raise RuntimeError("scheduler not started")
        while True:
            now = time.perf_counter()
            if now >= self.next_tick:
                frame_start = now
                break
            if poll_cb is not None:
                try:
                    poll_cb()
                except Exception:  # noqa: BLE001
                    pass
            if poll_sleep > 0:
                time.sleep(poll_sleep)
        while frame_start - self.next_tick > self.interval_s:
            self.next_tick += self.interval_s
        self.next_tick += self.interval_s
        t = frame_start - (self.t0 or frame_start)
        return t, frame_start

__all__ = ["FrameScheduler"]
