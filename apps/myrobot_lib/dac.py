"""DAC driver shim for MyRobot
Attempts to import the real driver from the application module
`apps.integrated_sensor_sin_python` when available. If not present sets
`Dac8564 = None` so higher-level code can run in dry/sim mode.
"""
try:
    from apps.myrobot_lib.drivers.dac import Dac8564  # type: ignore
except Exception:
    try:
        from apps.integrated_sensor_sin_python import Dac8564  # type: ignore
    except Exception:
        Dac8564 = None

__all__ = ['Dac8564']
