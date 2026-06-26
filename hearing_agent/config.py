# Configuration settings for the Hearing Test Calibration Agent

# Standard audiometry frequencies (Hz) used for hearing test calibration
AUDIOMETRY_FREQUENCIES = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]


# Database priority order. Higher in list means more authoritative.
DATABASE_PRIORITY = ['oratory1990', 'rtings']

# Safety limits for correction factors (dB)
# Prevents clipping, audio distortion, or excessively loud/quiet volume levels.
MAX_CORRECTION_DB = 15.0
MIN_CORRECTION_DB = -15.0

# Threshold for vetting database discrepancies (dB)
# If standard deviation of response values across databases is higher than this,
# the user will be alerted about the discrepancy.
VETTING_DISCREPANCY_THRESHOLD_DB = 3.0

# Keywords to detect bone conduction headphones (case-insensitive search)
BONE_CONDUCTION_KEYWORDS = [
    'shokz',
    'aftershokz',
    'bone conduction',
    'bone-conduction',
    'openmove',
    'openrun',
    'opencomm',
    'aeropex'
]
