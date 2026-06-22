# -*- coding: utf-8 -*-
"""
detector.py - Step 5: SoLEXS Classical Flare Detector.

Applies a Savitzky-Golay smoothed threshold detector to SoLEXS count-rate
light curve, classifies detections using the Step 4 calibration, and outputs
a flare detection catalog.

Detection algorithm
-------------------
1. Load GOOD-quality SoLEXS data; mask BAD.
2. Savitzky-Golay smooth (window=15 samples, poly_order=3).
3. Compute rolling std (sigma) over 10-min trailing window.
4. IDLE -> COUNTING: smoothed flux > background + 3*sigma AND slope > 0.
5. COUNTING -> IN_FLARE: sustained for >= 5 consecutive seconds.
6. Peak: local maximum within 30 min of start.
7. Stop: flux drops to background + 1*sigma, or 60-min hard cutoff.
8. Reject events shorter than 30 seconds total duration.
9. Classify: apply calibration to map peak count rate -> equivalent GOES class.

Units
-----
count_rate         : counts/s
confidence_snr     : (peak - background) / sigma  (dimensionless)
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

def _classify_from_calibration(peak_cr: float, a: float, b: float) -> str:
    """
    Classify SoLEXS count rate to GOES letter class via calibration.

    Parameters
    ----------
    peak_cr : float  -- SoLEXS peak count rate (counts/s)
    a       : float  -- log-log slope
    b       : float  -- log-log intercept

    Returns
    -------
    str -- GOES letter class ('A', 'B', 'C', 'M', or 'X')
    """
    if peak_cr <= 0:
        return "A"
    log_flux = (np.log10(peak_cr) - b) / a
    flux     = 10 ** log_flux

    for cls in ["X", "M", "C", "B"]:
        if flux >= CLASS_THRESHOLDS[cls]:
            return cls
    return "A"


# ---------------------------------------------------------------------------
# Step 5 -- SoLEXS Classical Detector
# ---------------------------------------------------------------------------

def detect_solexs_flares(
    solexs_path: Path    = OUTPUTS / "solexs_clean.parquet",
    coeffs_path: Path    = OUTPUTS / "calibration_coeffs.json",
    output_path: Path    = OUTPUTS / "solexs_detections.parquet",
) -> pd.DataFrame:
    """
    Detect flares in SoLEXS light curve using classical threshold algorithm.

    Parameters
    ----------
    solexs_path  : Path -- solexs_clean.parquet from solexs_ingest.py
    coeffs_path  : Path -- calibration_coeffs.json from calibration.py
    output_path  : Path -- where to save solexs_detections.parquet

    Returns
    -------
    pd.DataFrame with schema:
        event_id | start | peak_time | stop |
        peak_count_rate (counts/s) | goes_equivalent_class | confidence_snr
    """
    print("\n" + "=" * 70)
    print("  STEP 5 -- SoLEXS Classical Flare Detector")
    print("=" * 70)

    # --- Load data ---
    solexs = pd.read_parquet(solexs_path)
    solexs = solexs.set_index("timestamp").sort_index()

    # Load calibration coefficients
    if not coeffs_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {coeffs_path}")
    with open(coeffs_path) as f:
        coeffs = json.load(f)
    a_cal = coeffs["a"]
    b_cal = coeffs["b"]
    print(f"\n  Calibration: log10(CR) = {a_cal:.4f}*log10(flux) + {b_cal:.4f}")
    print(f"  Method: {coeffs['method']}")

    # --- Use GOOD data only ---
    good_mask  = solexs["quality_flag"] == "GOOD"
    counts_raw = solexs["counts"].where(good_mask)

    n_good = good_mask.sum()
    print(f"\n  GOOD samples      : {n_good:,} / {len(solexs):,} ({n_good/len(solexs)*100:.1f}%)")

    # --- Fill short gaps for smoothing (interpolate up to 10s) ---
    counts_interp = counts_raw.interpolate(method="linear", limit=10)

    # --- Savitzky-Golay smoothing ---
    raw_array = counts_interp.values.copy()
    # Only smooth where we have data; set NaN to 0 for smoothing then restore
    nan_mask = np.isnan(raw_array)
    raw_filled = np.where(nan_mask, 0.0, raw_array)

    smoothed = savgol_filter(raw_filled, window_length=SG_WINDOW, polyorder=SG_POLY)
    smoothed[nan_mask] = np.nan

    times    = solexs.index
    bg_series = solexs["background_counts"].values  # pre-computed in ingestion

    # --- Rolling sigma (10-min trailing window, 1-second cadence) ---
    sigma_window = DET_BG_WINDOW_MIN * 60
    counts_s = pd.Series(smoothed, index=times)
    # Only compute std on GOOD data
    sigma = (
        counts_s
        .where(good_mask.values)
        .rolling(sigma_window, min_periods=30)
        .std()
    )
    sigma = sigma.bfill().fillna(smoothed[~np.isnan(smoothed)].std() * 0.1)
    sigma_arr = sigma.values

    # Detection thresholds
    bg_arr     = bg_series
    thr_on_arr = bg_arr + DET_K_SIGMA * sigma_arr
    thr_off_arr= bg_arr + DET_DECAY_SIGMA * sigma_arr

    # Handle NaN background (early in series)
    global_bg = float(np.nanpercentile(smoothed[~np.isnan(smoothed)], 10))
    bg_arr     = np.where(np.isnan(bg_arr),     global_bg, bg_arr)
    thr_on_arr = np.where(np.isnan(thr_on_arr), global_bg, thr_on_arr)
    thr_off_arr= np.where(np.isnan(thr_off_arr),global_bg, thr_off_arr)

    # Slope (finite difference of smoothed, counts/s/s)
    slope_arr = np.diff(smoothed, prepend=np.nan)

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
            state = IDLE
            count = 0
            start_idx = None
            i += 1
            continue

        s = slope_arr[i] if not np.isnan(slope_arr[i]) else 0.0

        if state == IDLE:
            if v > thr_on_arr[i] and s > 0:
                state     = COUNTING
                count     = 1
                start_idx = i

        elif state == COUNTING:
            if v > thr_on_arr[i]:
                count += 1
                if count >= 5:   # 5-second minimum sustained
                    state = IN_FLARE
            else:
                state     = IDLE
                count     = 0
                start_idx = None

        if state == IN_FLARE:
            # Find peak within next DET_PEAK_WINDOW_MIN minutes
            peak_end = min(i + DET_PEAK_WINDOW_MIN * 60, n)
            seg_vals = smoothed[i:peak_end]
            valid    = ~np.isnan(seg_vals)
            if not valid.any():
                state = IDLE; count = 0; start_idx = None; i += 1; continue

            peak_off  = int(np.nanargmax(seg_vals))
            peak_idx  = i + peak_off
            peak_cr   = float(smoothed[peak_idx])
            peak_t    = times[peak_idx]
            start_t   = times[start_idx]

            # Event duration so far
            duration_s = (peak_t - start_t).total_seconds()

            # Find stop
            timeout_idx = min(peak_idx + DET_STOP_CUTOFF_MIN * 60, n - 1)
            stop_idx    = timeout_idx
            for j in range(peak_idx + 1, timeout_idx + 1):
                sv = smoothed[j]
                if not np.isnan(sv) and sv <= thr_off_arr[j]:
                    stop_idx = j
                    break

            total_duration = (times[stop_idx] - start_t).total_seconds()

            if total_duration >= MIN_EVENT_DURATION_S and peak_cr > 0:
                # SNR at peak
                bg_at_peak    = bg_arr[peak_idx]
                sigma_at_peak = sigma_arr[peak_idx]
                snr = (peak_cr - bg_at_peak) / max(sigma_at_peak, 1e-10)

                goes_class = _classify_from_calibration(peak_cr, a_cal, b_cal)

                events.append({
                    "start":               start_t,
                    "peak_time":           peak_t,
                    "stop":                times[stop_idx],
                    "peak_count_rate":     peak_cr,
                    "goes_equivalent_class": goes_class,
                    "confidence_snr":      float(snr),
                })

            i         = stop_idx + 1
            state     = IDLE
            count     = 0
            start_idx = None
            continue

        i += 1

    # Build output DataFrame
    if events:
        df = pd.DataFrame(events).sort_values("start").reset_index(drop=True)
        df.insert(0, "event_id", [f"SX{i+1:03d}" for i in range(len(df))])
    else:
        df = pd.DataFrame(columns=[
            "event_id","start","peak_time","stop",
            "peak_count_rate","goes_equivalent_class","confidence_snr"
        ])

    print(f"\n  Detected {len(df)} flare event(s):")
    if len(df) > 0:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 130)
        print(df.to_string(index=False))

    df.to_parquet(output_path, index=False)
    print(f"\n  Saved -> {output_path}")

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = detect_solexs_flares()
    print("\n  All done -- detector.py complete.")
