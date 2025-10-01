"""LDC sensor shim for MyRobot"""
try:
    from apps.myrobot_lib.drivers.ldc import LDC1614  # type: ignore
except Exception:
    try:
        from apps.integrated_sensor_sin_python import LDC1614  # type: ignore
    except Exception:
        LDC1614 = None

__all__ = ['LDC1614']
