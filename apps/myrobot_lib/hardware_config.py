# Hardware-specific configuration for MyRobot
# Put board and sensor addresses, valve bounds and other hardware-dependent
# defaults here so they can be changed independently of experiment defaults.

# I2C / LDC
I2C_BUS = 1
LDC_ADDRS = [0x1A, 0x1B, 0x2B, 0x2A, 0x24, 0x14, 0x15]

# Valve bounds and mapping defaults (percent)
VALVE_MIN = 20.0
VALVE_MAX = 100.0
VALVE_CENTER = 50.0
VALVE_SPAN = 40.0

# Encoder defaults
ENCODER_PPR = 2048

# Tension polling / sensor timings
TENSION_POLL_INTERVAL = 0.2

__all__ = [
    'I2C_BUS', 'LDC_ADDRS',
    'VALVE_MIN', 'VALVE_MAX', 'VALVE_CENTER', 'VALVE_SPAN',
    'ENCODER_PPR', 'TENSION_POLL_INTERVAL'
]
