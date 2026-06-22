"""
config.py — All tunable parameters for the SoLEXS/HEL1OS flare-detection pipeline.
Units are noted in parentheses for every physical quantity.
"""
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # BAH/
GOES_PRIMARY_MD   = ROOT / "GOES(PRIMARY).md"
GOES_SECONDARY_MD = ROOT / "GOES(SECONDARY).md"

SOLEXS_DATA_ROOT = ROOT / "SoLEXS_data"
HEL1OS_DATA_ROOT = ROOT / "HeL1OS_data" / "2026" / "06"

OUTPUTS = Path(__file__).resolve().parent / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ── Date range ────────────────────────────────────────────────────────────────
DATE_START = "2026-06-17"
DATE_STOP  = "2026-06-19"   # inclusive

# ── GOES-18 SXR flare detection (Step 1) ─────────────────────────────────────
GOES_BG_WINDOW_MIN    = 60      # trailing window for background (minutes)
GOES_STD_WINDOW_MIN   = 30      # trailing window for rolling σ (minutes)
K_SIGMA               = 3.0     # detection threshold multiplier
K_SIGMA_RETRY         = 2.0     # fallback if <5 events found
MIN_SUSTAINED_MIN     = 3       # consecutive minutes above threshold for start
DECAY_SIGMA           = 1.0     # stop criterion: back to bg + DECAY_SIGMA*σ
DECAY_TIMEOUT_MIN     = 30      # hard-stop after peak if no decay (minutes)
MERGE_GAP_MIN         = 5       # merge two events if starts within N minutes

# GOES flux → class boundaries (W/m²)
CLASS_THRESHOLDS = {
    "X": 1e-4,
    "M": 1e-5,
    "C": 1e-6,
    "B": 1e-7,
    "A": 0.0,
}

# ── SoLEXS ingestion (Step 2) ─────────────────────────────────────────────────
SOLEXS_BG_WINDOW_MIN  = 30      # trailing background window (minutes)
SOLEXS_GAP_THRESHOLD_S = 5     # gap definition: missing >N consecutive seconds

# ── HEL1OS ingestion (Step 3) ─────────────────────────────────────────────────
# Nominal HK ranges. Updated from actual data inspection:
# CdTe operates cryo-cooled at ~-40 to -35 degC; CZT at room temp ~17-22 degC.
# Temperature quality check uses a +-5 degC margin around observed range.
# HV is checked dynamically (+-20% of observed median) in hel1os_ingest.py.
HEL1OS_TEMP_MIN  = -45.0        # detector temperature lower bound (degC)
HEL1OS_TEMP_MAX  =  30.0        # detector temperature upper bound (degC)
HEL1OS_HV_MIN    = 400.0        # high-voltage lower bound (V)
HEL1OS_HV_MAX    = 1200.0       # high-voltage upper bound (V)

# ── SoLEXS classical detector (Step 5) ───────────────────────────────────────
SG_WINDOW        = 15           # Savitzky-Golay window (samples)
SG_POLY          = 3            # Savitzky-Golay polynomial order
DET_BG_WINDOW_MIN = 10          # rolling σ window for detection (minutes)
DET_K_SIGMA      = 3.0          # start threshold
DET_DECAY_SIGMA  = 1.0          # stop threshold
DET_PEAK_WINDOW_MIN = 30        # max time from start to peak (minutes)
DET_STOP_CUTOFF_MIN = 60        # hard stop after peak (minutes)
MIN_EVENT_DURATION_S = 30       # reject events shorter than this (seconds)

# ── Evaluation (Step 6) ───────────────────────────────────────────────────────
OVERLAP_FRACTION  = 0.50        # TP if overlap ≥ 50% of GT event duration
