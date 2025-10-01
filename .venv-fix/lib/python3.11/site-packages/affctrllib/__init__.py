"""Affetto Control Library.

This is a small library to control NEDO Affetto in Python.
"""

from __future__ import annotations

from importlib import metadata

__version__ = metadata.version("affctrllib")
__all__ = ["__version__"]

from affctrllib.affcomm import (
    AffComm,
    convert_array_to_bytes,
    convert_array_to_string,
    split_received_msg,
    unzip_array,
    unzip_array_as_ndarray,
    zip_arrays,
    zip_arrays_as_ndarray,
)
from affctrllib.affctrl import AffCtrl, AffCtrlThread
from affctrllib.affmock import AffMock
from affctrllib.affposctrl import AffPosCtrl, AffPosCtrlThread
from affctrllib.affstate import AffState, AffStateThread
from affctrllib.filter import Filter
from affctrllib.logger import Logger
from affctrllib.ptp import PTP
from affctrllib.timer import Timer

__all__ = [
    "AffComm",
    "AffCtrl",
    "AffCtrl",
    "AffCtrlThread",
    "AffCtrlThread",
    "AffMock",
    "AffPosCtrl",
    "AffPosCtrlThread",
    "AffState",
    "AffStateThread",
    "Filter",
    "Logger",
    "PTP",
    "Timer",
    "convert_array_to_bytes",
    "convert_array_to_string",
    "split_received_msg",
    "unzip_array",
    "unzip_array_as_ndarray",
    "zip_arrays",
    "zip_arrays_as_ndarray",
]
