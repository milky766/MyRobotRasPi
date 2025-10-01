from __future__ import annotations
from dataclasses import dataclass

@dataclass
class PID:
    Kp: float
    Ki: float
    Kd: float
    u_min: float = -100.0  # 双方向化: デフォルトで対称範囲
    u_max: float = 100.0
    def __post_init__(self):
        self._sum = 0.0
        self._prev_e = None
    def reset(self):
        self._sum = 0.0
        self._prev_e = None
    def step(self, target: float, meas: float, dt: float) -> float:
        e = target - meas
        de = 0.0
        if self._prev_e is not None and dt > 0:
            de = (e - self._prev_e) / dt
        # 積分候補
        sum_new = self._sum + e * dt
        # 生出力計算（候補）
        raw = self.Kp * e + self.Ki * sum_new + self.Kd * de
        # 飽和適用
        u = raw
        if u < self.u_min: u = self.u_min
        if u > self.u_max: u = self.u_max
        # アンチワインドアップ: 飽和し かつ その方向にさらに押し込む誤差なら積分更新しない
        if not ((u == self.u_max and raw >= self.u_max and e > 0) or (u == self.u_min and raw <= self.u_min and e < 0)):
            self._sum = sum_new  # 許容される場合のみ確定
        self._prev_e = e
        return u
__all__ = ['PID']
