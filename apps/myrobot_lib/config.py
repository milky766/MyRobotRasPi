# Shared MyRobot configuration constants
# Put hardware and experiment defaults here for reuse across scripts

# I2C / LDC
I2C_BUS = 1
LDC_ADDRS = [0x1A, 0x1B, 0x2B, 0x2A, 0x24, 0x14, 0x15]

# LDC hardware reference constants used by LDC1614 driver to convert
# raw frequency ticks into inductance. These match values used in
# affetto_nn_ctrl.hw.ldc1614 and apps/valve_sine_test.
LDC_FREF_HZ = 40_000_000.0
LDC_CPAR_F = 330e-12

# Joint / encoder usable motion range (degrees) discovered from range_of_motion
JOINT_MIN = -20.0
JOINT_MAX = 60.0

# Valve bounds and mapping defaults (percent)
VALVE_MIN = 20.0
VALVE_MAX = 100.0
VALVE_CENTER = 50.0
VALVE_SPAN = 40.0

# Experiment / run defaults
DEFAULT_LOOP_INTERVAL_MS = 33.333333
DEFAULT_DURATION_S = 10.0
DEFAULT_OUTPUT_DIR = 'data/myrobot'
DEFAULT_OUTPUT_PREFIX = 'myrobot_motion'
DEFAULT_N_REPEAT = 3
DEFAULT_SEED = 42

# Controller defaults (PID)
DEFAULT_KP = 0.1
DEFAULT_KI = 0.04
DEFAULT_KD = 0.01
INCREMENTAL_VALVE_DEFAULT = True
INCR_GAIN_DEFAULT = 1.0

# Operational safety joint limits (can be stricter than JOINT_MIN/MAX)
SAFETY_Q_MIN = JOINT_MIN
SAFETY_Q_MAX = JOINT_MAX

# GPIO / hardware pin assignments (fill with your platform's BCM numbers)
# Set to None for not-applicable platforms
EMERGENCY_STOP_GPIO = None
VALVE_ENABLE_GPIO = None
# Low-level SPI / CS pins for DAC/ADC
SPI_BUS = 0
SPI_DEV = 0
SPI_MAX_HZ = 1_000_000
GPIO_CS_DAC = 19
GPIO_CS_ADC = 24
# Encoder hardware defaults
ENC_CHIP = '/dev/gpiochip4'
ENC_PIN_A = 14
ENC_PIN_B = 4

ENCODER_A_GPIO = None
ENCODER_B_GPIO = None

# Encoder defaults / calibration
ENCODER_ZERO_DEFAULT = 0.0  # deg offset applied if provided
ENCODER_PPR = 2048  # pulses per revolution (default)

# Misc defaults
DEFAULT_Q_LIMIT = (0.0, 100.0)

# Sensor polling and channel mapping
TENSION_POLL_INTERVAL = 0.2
ADC_CHANNEL_MAP = {'pressure0': 0, 'pressure1': 1}
DAC_CHANNEL_MAP = {'valve_a': 0, 'valve_b': 1}

# Plotting/logging defaults
CSV_TIMESTAMP_FMT = '%Y%m%d_%H%M%S'
PLOT_DPI = 150
PLOT_FIGSIZE = (8, 6)
PLOT_BACKEND = 'Agg'

# Calibration maps (optional, key by sensor id/address)
LDC_SCALE_FACTORS = {addr: 1.0 for addr in LDC_ADDRS}
ADC_CALIBRATION = {'scale': 1.0, 'offset': 0.0}
PRESSURE_SENSOR_OFFSETS = {'pressure0': 0.0, 'pressure1': 0.0}

# Update __all__ to export the new symbols
__all__ = [
    'I2C_BUS', 'LDC_ADDRS',
    'JOINT_MIN', 'JOINT_MAX',
    'VALVE_MIN', 'VALVE_MAX', 'VALVE_CENTER', 'VALVE_SPAN',
    'ENCODER_PPR', 'DEFAULT_Q_LIMIT',
    'DEFAULT_LOOP_INTERVAL_MS', 'DEFAULT_DURATION_S', 'DEFAULT_OUTPUT_DIR', 'DEFAULT_OUTPUT_PREFIX', 'DEFAULT_N_REPEAT', 'DEFAULT_SEED',
    'DEFAULT_KP', 'DEFAULT_KI', 'DEFAULT_KD', 'INCREMENTAL_VALVE_DEFAULT', 'INCR_GAIN_DEFAULT',
    'SAFETY_Q_MIN', 'SAFETY_Q_MAX',
    'EMERGENCY_STOP_GPIO', 'VALVE_ENABLE_GPIO', 'ENCODER_A_GPIO', 'ENCODER_B_GPIO',
    'ENC_CHIP', 'ENC_PIN_A', 'ENC_PIN_B',
    'SPI_BUS', 'SPI_DEV', 'SPI_MAX_HZ', 'GPIO_CS_DAC', 'GPIO_CS_ADC',
    'ENCODER_ZERO_DEFAULT', 'TENSION_POLL_INTERVAL', 'ADC_CHANNEL_MAP', 'DAC_CHANNEL_MAP',
    'CSV_TIMESTAMP_FMT', 'PLOT_DPI', 'PLOT_FIGSIZE', 'PLOT_BACKEND',
    'LDC_SCALE_FACTORS', 'ADC_CALIBRATION', 'PRESSURE_SENSOR_OFFSETS',
    # exported LDC constants
    'LDC_FREF_HZ', 'LDC_CPAR_F'
]
