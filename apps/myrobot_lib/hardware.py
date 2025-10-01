from __future__ import annotations
from typing import Any, List, Tuple
import pathlib
import time

# Use driver shims located in the same package so application-level module
# `apps.integrated_sensor_sin_python` is not required at import time.
from apps.myrobot_lib import dac as _dac_shim
from apps.myrobot_lib import adc as _adc_shim
from apps.myrobot_lib import encoder as _encoder_shim
from apps.myrobot_lib import ldc as _ldc_shim
from apps.myrobot_lib import config as cfg

Dac8564 = getattr(_dac_shim, 'Dac8564', None)
Max11632 = getattr(_adc_shim, 'Max11632', None)
EncoderSimple = getattr(_encoder_shim, 'EncoderSimple', None)
LDC1614 = getattr(_ldc_shim, 'LDC1614', None)


def open_devices(i2c_bus: int = cfg.I2C_BUS, ldc_addrs: list[int] | None = None) -> Tuple[Any, Any, Any, List[Any]]:
    """Open and return (dac, adc, encoder, ldc_sensors).
    Raises RuntimeError if DAC driver not available / cannot be opened.
    """
    dac = None
    adc = None
    enc = None
    ldc_sensors: List[Any] = []

    if Dac8564 is None:
        raise RuntimeError('Dac8564 driver/class not available')
    try:
        dac_obj = Dac8564(0, 0, 19)
        dac_obj.open()
        dac = dac_obj
    except Exception as e:
        raise RuntimeError(f'DAC open failed: {e}')

    if Max11632 is not None:
        try:
            adc_obj = Max11632(0, 0, 24, 0, 1)
            adc_obj.open()
            adc = adc_obj
        except Exception:
            adc = None

    if EncoderSimple is not None:
        try:
            # Use configurable chip and pin assignments from config
            enc_chip = getattr(cfg, 'ENC_CHIP', '/dev/gpiochip4')
            enc_a = getattr(cfg, 'ENC_PIN_A', 14)
            enc_b = getattr(cfg, 'ENC_PIN_B', 4)
            enc_obj = EncoderSimple(enc_chip, enc_a, enc_b)
            enc = enc_obj
        except Exception:
            enc = None

    if LDC1614 is not None and (ldc_addrs or cfg.LDC_ADDRS):
        try:
            from smbus2 import SMBus  # type: ignore
            bus = SMBus(i2c_bus)
            for addr in (ldc_addrs or cfg.LDC_ADDRS):
                try:
                    s = LDC1614(bus, addr)
                    if s.init():
                        ldc_sensors.append(s)
                except Exception:
                    pass
        except Exception:
            ldc_sensors = []

    return dac, adc, enc, ldc_sensors


def close_devices(dac: Any, adc: Any, enc: Any, ldc_sensors: list[Any] | None) -> None:
    try:
        if dac is not None:
            try:
                dac.set_channels(0.0, 0.0)
            except Exception:
                pass
            try:
                dac.close()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if adc is not None:
            try:
                adc.close()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if enc is not None:
            try:
                enc.close()
            except Exception:
                pass
    except Exception:
        pass

    try:
        if ldc_sensors:
            for s in ldc_sensors:
                try:
                    s.close()
                except Exception:
                    pass
    except Exception:
        pass
