"""ADC driver shim for MyRobot"""
try:
    from apps.myrobot_lib.drivers.adc import Max11632  # type: ignore
except Exception:
    try:
        from apps.integrated_sensor_sin_python import Max11632  # type: ignore
    except Exception:
        Max11632 = None

__all__ = ['Max11632']
