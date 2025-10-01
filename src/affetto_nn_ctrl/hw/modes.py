from __future__ import annotations

import math
import random
from dataclasses import dataclass

# Sine pair (phase shifted)

def compute_valves(t: float, cycle: float, center: float, amp: float) -> tuple[float, float]:
    s_a = math.sin(2.0 * math.pi * t / cycle - math.pi / 2.0)
    s_b = math.sin(2.0 * math.pi * t / cycle + math.pi / 2.0)
    return center + amp * s_a, center + amp * s_b

@dataclass
class RandomWalk2Ch:
    min_pct: float
    max_pct: float
    delta: float
    center: float
    antagonistic: bool = False

    def __post_init__(self):
        self.a = self.center
        self.b = self.center

    def _step_one(self, val: float) -> float:
        val += random.uniform(-self.delta, self.delta)
        if val > self.max_pct:
            val = self.max_pct - (val - self.max_pct)
        if val < self.min_pct:
            val = self.min_pct + (self.min_pct - val)
        if val > self.max_pct:
            val = self.max_pct
        if val < self.min_pct:
            val = self.min_pct
        return val

    def step(self) -> tuple[float, float]:
        self.a = self._step_one(self.a)
        if self.antagonistic:
            self.b = self.center - (self.a - self.center)
            if self.b < self.min_pct:
                self.b = self.min_pct
            if self.b > self.max_pct:
                self.b = self.max_pct
        else:
            self.b = self._step_one(self.b)
        return self.a, self.b

__all__ = ["compute_valves", "RandomWalk2Ch"]
