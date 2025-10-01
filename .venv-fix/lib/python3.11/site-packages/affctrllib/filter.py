from __future__ import annotations

from collections import deque
from typing import Generic, TypeVar

import numpy as np
from scipy.signal import butter

T = TypeVar("T", float, np.ndarray)

DEFAULT_FILTER_N_POINTS = 5


class Filter(Generic[T]):
    _n_points: int
    _x_buffer: deque[T | float]
    _y_prev: T | float

    def __init__(self, n_points: int | None = None) -> None:
        self._n_points = DEFAULT_FILTER_N_POINTS
        if n_points is not None:
            self.n_points = n_points

        self._x_buffer = deque()
        for _ in range(self.n_points):
            self._x_buffer.append(0.0)
        self._y_prev = 0.0

    def set_n_points(self, n_points: int) -> None:
        self._n_points = n_points

    @property
    def n_points(self) -> int:
        return self._n_points

    @n_points.setter
    def n_points(self, n_points: int) -> None:
        self.set_n_points(n_points)

    def update(self, x: T) -> T:
        self._x_buffer.append(x)
        self._y_prev = self._y_prev + x - self._x_buffer.popleft()
        return self._y_prev / self.n_points


class LiveFilter(Generic[T]):
    def process(self, x: T) -> T:
        # do not process NaN
        if np.any(np.isnan(x)):
            return x

        return self._process(x)

    def __call__(self, x: T) -> T:
        return self.process(x)

    def _process(self, x: T) -> T:
        _ = x
        msg = "Derived class must implement _process"
        raise NotImplementedError(msg)


class LiveLFilter(LiveFilter[T]):
    """Live implementation of digital filter using difference equations.

    The following is almost equivalent to calling scipy.lfilter(b, a, xs):
    >>> lfilter = LiveLFilter(b, a)
    >>> [lfilter(x) for x in xs]
    """

    _xs: deque[T | float]
    _ys: deque[T | float]

    def __init__(self, b: np.ndarray, a: np.ndarray) -> None:
        """Initialize live filter based on difference equation.

        Args:
            b (array-like): numerator coefficients obtained from scipy
                filter design.
            a (array-like): denominator coefficients obtained from scipy
                filter design.
        """
        self.b = b
        self.a = a
        self._xs = deque(maxlen=len(b))
        self._ys = deque(maxlen=len(a) - 1)

    def _process(self, x: T) -> T:
        """Filter incoming data with standard difference equations."""
        if len(self._xs) == 0:
            for _ in range(len(self.b)):
                if isinstance(x, np.ndarray):
                    self._xs.append(np.zeros(len(x)))
                else:
                    self._xs.append(0.0)
        if len(self._ys) == 0:
            for _ in range(len(self.a) - 1):
                if isinstance(x, np.ndarray):
                    self._ys.append(np.zeros(len(x)))
                else:
                    self._ys.append(0.0)

        self._xs.appendleft(x)
        y = np.dot(self.b, np.asarray(self._xs)) - np.dot(self.a[1:], np.asarray(self._ys))
        y = y / self.a[0]
        self._ys.appendleft(y)

        return y


class Butterworth(Generic[T]):
    N: int  # the order of the filter
    Wn: float  # the critical frequency
    fs: float  # the sampling frequency

    def __init__(self, cutoff: float, fs: float, order: int = 5) -> None:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        self.N = order
        self.Wn = normal_cutoff
        self.fs = fs
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        self.lfilter = LiveLFilter(b, a)

    def update(self, x: T) -> T:
        return self.lfilter.process(x)  # type: ignore[return-value,arg-type]


# Local Variables:
# jinx-local-words: "Args LiveLFilter NaN arg lfilter scipy xs"
# End:
