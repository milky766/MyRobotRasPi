"""Encoder shim for MyRobot — prefer affetto_nn_ctrl implementation when present."""
# Try affetto implementation first (robust v1/v2 gpiod handling)
try:
    from affetto_nn_ctrl.hw.encoder import EncoderSimple  # type: ignore
except Exception:
    try:
        from apps.myrobot_lib.drivers.encoder import EncoderSimple  # type: ignore
    except Exception:
        try:
            from apps.integrated_sensor_sin_python import EncoderSimple  # type: ignore
        except Exception:
            EncoderSimple = None

__all__ = ['EncoderSimple']