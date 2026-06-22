# -*- coding: utf-8 -*-
"""
hel1os_detector.py - Phase C.2: HEL1OS CdTe Classical Flare Detector.

Same algorithm as detector.py (SoLEXS) applied to HEL1OS CdTe1 broadband
(1.8-90 keV) count-rate light curve.

CdTe1 characteristics:
  - Typical quiescent rate: ~0.2-0.5 cts/s
  - Solar X-ray flares show clear enhancement (up to ~10x background)
  - 1-second cadence, MJD timestamps rounded to nearest second

Detection algorithm (identical to SoLEXS detector):
  1. Load GOOD-quality HEL1OS CdTe1 data
  2. Savitzky-Golay smooth (window=15s, poly=3)
  3. Rolling sigma (10-min trailing window on GOOD data only)
  4. Start: smoothed > background + 3*sigma AND slope > 0, sustained >= 5s
  5. Peak: local maximum within 30 min of start
  6. Stop: flux drops to background + 1*sigma, or 60-min hard cutoff
  7. Reject events shorter than 30 seconds total
  8. Classify using same calibration (with CdTe-specific offset documented)

Confidence SNR:
  confidence_snr = (peak_ctr - background_at_peak) / sigma_at_peak

Units
-----
count_rate         : counts/s
confidence_snr     : dimensionless
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from config import (
    OUTPUTS,
    SG_WINDOW,
    SG_POLY,
    DET_BG_WINDOW_MIN,
    DET_K_SIGMA,
    DET_DECAY_SIGMA,
    DET_PEAK_WINDOW_MIN,
    DET_STOP_CUTOFF_MIN,
    MIN_EVENT_DURATION_S,
    CLASS_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_from_calibration_cdte(peak_cr: float, a: float, b: float,
                                     offset: float = 0.0) -> str:
    """
    Map HEL1OS CdTe count rate to approximate GOES class via calibration.

    CdTe responds to higher-energy photons than SoLEXS SDD2; its sensitivity
    is different so we apply a documented offset (offset=0 by default = use
    same calibration as SoLEXS, which is conservative).

    Parameters
    ----------
    peak_cr : float  -- HEL1OS CdTe peak count rate (counts/s)
    a       : float  -- log-log slope from calibration
    b       : float  -- log-log intercept from calibration
    offset  : float  -- additive offset in log10(CR) space to account for
                        detector sensitivity difference (default 0)

    Returns
    -------
    str -- GOES letter class ('A', 'B', 'C', 'M', or 'X')
    """
    if peak_cr <= 0:
        return "A"
    # Apply offset: log10(flux) = (log10(cr) - offset - b) / a
    log_flux = (np.log10(peak_cr) - offset - b) / a
    flux     = 10 ** log_flux

    for cls in ["X", "M", "C", "B"]:
        if flux >= CLASS_THRESHOLDS[cls]:
            return cls
    return "A"


# ---------------------------------------------------------------------------
# Phase C.2 -- HEL1OS CdTe Classical Detector
# ---------------------------------------------------------------------------

def detect_hel1os_flares(
    hel1os_path: Path   = OUTPUTS / "hel1os_clean.parquet",
    coeffs_path: Path   = OUTPUTS / "calibration_coeffs.json",
    output_path: Path   = OUTPUTS / "hel1os_detections.parquet",
    detector_col: str   = "ctr_cdte1",
    bg_col: str         = "background_cdte1",
) -> pd.DataFrame:
    """
    Detect flares in HEL1OS CdTe1 broadband light curve.

    Uses exactly the same Savitzky-Golay + rolling-threshold state-machine as
    the SoLEXS detector (detector.py).

    Parameters
    ----------
    hel1os_path  : Path -- hel1os_clean.parquet from hel1os_ingest.py
    coeffs_path  : Path -- calibration_coeffs.json from calibration.py
    output_path  : Path -- where to save hel1os_detections.parquet
    detector_col : str  -- column to run detection on (default: ctr_cdte1)
    bg_col       : str  -- pre-computed background column

    Returns
    -------
    pd.DataFrame with schema:
        event_id | start | peak_time | stop |
        peak_count_rate (counts/s) | goes_equivalent_class | confidence_snr | instrument
    """
    print("\n" + "=" * 70)
    print("  PHASE C.2 -- HEL1OS CdTe Classical Flare Detector")
    print("=" * 70)

    # Load data
    hel1os = pd.read_parquet(hel1os_path)
    hel1os = hel1os.set_index("timestamp").sort_index()

    # Load calibration
    if not coeffs_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {coeffs_path}")
    with open(coeffs_path) as f:
        coeffs = json.load(f)
    a_cal = coeffs["a"]
    b_cal = coeffs["b"]
    print(f"\n  Using calibration: log10(CR) = {a_cal:.4f}*log10(flux) + {b_cal:.4f}")
    print(f"  Detector channel  : {detector_col}  (HEL1OS CdTe1, 1.8-90 keV)")

    good_mask   = hel1os["quality_flag"] == "GOOD"
    counts_raw  = hel1os[detector_col].where(good_mask)
    bg_series   = hel1os[bg_col].values

    n_good = good_mask.sum()
    n_tot  = len(hel1os)
    print(f"\n  GOOD samples: {n_good:,} / {n_tot:,} ({n_good/n_tot*100:.1f}%)")

    # Fill short gaps for smoothing
    counts_interp = counts_raw.interpolate(method="linear", limit=10)

    # Savitzky-Golay smoothing
    raw_array  = counts_interp.values.copy()
    nan_mask   = np.isnan(raw_array)
    raw_filled = np.where(nan_mask, 0.0, raw_array)

    smoothed = savgol_filter(raw_filled, window_length=SG_WINDOW, polyorder=SG_POLY)
    smoothed[nan_mask] = np.nan

    times = hel1os.index

    # Rolling sigma on GOOD data (10-min window, 1-s cadence)
    sigma_window = DET_BG_WINDOW_MIN * 60
    counts_s = pd.Series(smoothed, index=times)
    sigma = (
        counts_s
        .where(good_mask.values)
        .rolling(sigma_window, min_periods=30)
        .std()
    )
    sigma = sigma.bfill().fillna(
        float(np.nanstd(smoothed[~np.isnan(smoothed)])) * 0.1
    )
    sigma_arr = sigma.values

    # Global background fallback
    global_bg = float(np.nanpercentile(smoothed[~np.isnan(smoothed)], 10))
    bg_arr      = np.where(np.isnan(bg_series),     global_bg, bg_series)
    thr_on_arr  = bg_arr + DET_K_SIGMA * sigma_arr
    thr_off_arr = bg_arr + DET_DECAY_SIGMA * sigma_arr
    slope_arr   = np.diff(smoothed, prepend=np.nan)

    n = len(times)
    events = []

    IDLE, COUNTING, IN_FLARE = 0, 1, 2
    state     = IDLE
    count     = 0
    start_idx = None

    i = 0
    while i < n:
        v = smoothed[i]
        if np.isnan(v):
            state = IDLE; count = 0; start_idx = None
            i += 1; continue

        s = slope_arr[i] if not np.isnan(slope_arr[i]) else 0.0

        if state == IDLE:
            if v > thr_on_arr[i] and s > 0:
                state     = COUNTING
                count     = 1
                start_idx = i

        elif state == COUNTING:
            if v > thr_on_arr[i]:
                count += 1
                if count >= 5:
                    state = IN_FLARE
            else:
                state = IDLE; count = 0; start_idx = None

        if state == IN_FLARE:
            peak_end = min(i + DET_PEAK_WINDOW_MIN * 60, n)
            seg_vals = smoothed[i:peak_end]
            valid    = ~np.isnan(seg_vals)
            if not valid.any():
                state = IDLE; count = 0; start_idx = None
                i += 1; continue

            peak_off  = int(np.nanargmax(seg_vals))
            peak_idx  = i + peak_off
            peak_cr   = float(smoothed[peak_idx])
            peak_t    = times[peak_idx]
            start_t   = times[start_idx]

            timeout_idx = min(peak_idx + DET_STOP_CUTOFF_MIN * 60, n - 1)
            stop_idx    = timeout_idx
            for j in range(peak_idx + 1, timeout_idx + 1):
                sv = smoothed[j]
                if not np.isnan(sv) and sv <= thr_off_arr[j]:
                    stop_idx = j
                    break

            total_duration = (times[stop_idx] - start_t).total_seconds()

            if total_duration >= MIN_EVENT_DURATION_S and peak_cr > 0:
                bg_at_peak    = bg_arr[peak_idx]
                sigma_at_peak = sigma_arr[peak_idx]
                snr           = (peak_cr - bg_at_peak) / max(sigma_at_peak, 1e-10)
                goes_class    = _classify_from_calibration_cdte(peak_cr, a_cal, b_cal)

                events.append({
                    "start":                 start_t,
                    "peak_time":             peak_t,
                    "stop":                  times[stop_idx],
                    "peak_count_rate":       peak_cr,
                    "goes_equivalent_class": goes_class,
                    "confidence_snr":        float(snr),
                    "instrument":            "HEL1OS-CdTe1",
                })

            i         = stop_idx + 1
            state     = IDLE; count = 0; start_idx = None
            continue

        i += 1

    if events:
        df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
        df.insert(0, "event_id", [f"HX{i+1:03d}" for i in range(len(df))])
    else:
        df = pd.DataFrame(columns=[
            "event_id", "start", "peak_time", "stop",
            "peak_count_rate", "goes_equivalent_class", "confidence_snr", "instrument"
        ])

    print(f"\n  Detected {len(df)} HEL1OS-CdTe flare events")
    if len(df) > 0:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 140)
        print(df.to_string(index=False))

    df.to_parquet(output_path, index=False)
    print(f"\n  Saved -> {output_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = detect_hel1os_flares()
    print("\n  All done -- hel1os_detector.py complete.")
