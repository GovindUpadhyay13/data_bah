# -*- coding: utf-8 -*-
"""
goes_parser.py - Step 0 + Step 1 of the SoLEXS flare-detection pipeline.

Step 0: Parse GOES-18 and GOES-19 X-ray flux JSON from markdown wrappers.
Step 1: Detect solar flare events from GOES-18 SXR (0.1-0.8 nm) light curve.

Units
-----
flux       : W/m^2
time_tag   : UTC ISO-8601 strings -> pandas datetime64[ns, UTC]
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Force stdout to UTF-8 on Windows to avoid cp1252 encode errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from config import (
    GOES_PRIMARY_MD,
    GOES_SECONDARY_MD,
    OUTPUTS,
    DATE_START,
    DATE_STOP,
    GOES_BG_WINDOW_MIN,
    GOES_STD_WINDOW_MIN,
    K_SIGMA,
    K_SIGMA_RETRY,
    MIN_SUSTAINED_MIN,
    DECAY_SIGMA,
    DECAY_TIMEOUT_MIN,
    MERGE_GAP_MIN,
    CLASS_THRESHOLDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_flux(flux_wm2: float) -> str:
    """Map peak flux (W/m^2) to GOES letter class (A/B/C/M/X)."""
    if flux_wm2 >= CLASS_THRESHOLDS["X"]:
        return "X"
    elif flux_wm2 >= CLASS_THRESHOLDS["M"]:
        return "M"
    elif flux_wm2 >= CLASS_THRESHOLDS["C"]:
        return "C"
    elif flux_wm2 >= CLASS_THRESHOLDS["B"]:
        return "B"
    else:
        return "A"


def _parse_goes_md(md_path: Path, satellite_label: str) -> pd.DataFrame:
    """
    Parse a GOES X-ray flux markdown file into a tidy DataFrame.

    The file has a 4-line header (Source URL + blank + --- + blank) followed
    by a single line containing the JSON array with markdown-escaped brackets
    and underscores.  We unescape and parse it here.

    Parameters
    ----------
    md_path : Path
        Path to the markdown file (e.g. GOES(PRIMARY).md).
    satellite_label : str
        Short label to attach ('g18' or 'g19').

    Returns
    -------
    pd.DataFrame with columns:
        timestamp, satellite, flux (W/m^2), observed_flux (W/m^2),
        electron_correction (W/m^2), electron_flag, energy, label
    """
    if not md_path.exists():
        raise FileNotFoundError(f"GOES markdown file not found: {md_path}")

    raw = md_path.read_text(encoding="utf-8")

    # Find the line containing the JSON array (starts with \[ or [)
    lines = raw.splitlines()
    json_line = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("\\[") or stripped.startswith("[{"):
            json_line = stripped
            break

    if json_line is None:
        raise ValueError(f"Could not locate JSON array in {md_path}")

    # Unescape markdown escapes: \[ -> [, \] -> ], \_ -> _, \- -> -
    json_str = json_line
    json_str = json_str.replace("\\[", "[").replace("\\]", "]")
    json_str = re.sub(r"\\_(\w)", r"_\1", json_str)
    json_str = json_str.replace("\\_", "_")
    # Fix escaped negative exponents: e\-08 -> e-08
    json_str = re.sub(r"([eE])\\(-\d+)", r"\1\2", json_str)

    records = json.loads(json_str)
    df = pd.DataFrame(records)

    # Rename the misspelled column if present (electron_contaminaton -> electron_contamination)
    if "electron_contaminaton" in df.columns:
        df.rename(columns={"electron_contaminaton": "electron_contamination"}, inplace=True)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["time_tag"], utc=True)
    df["electron_flag"] = df["electron_contamination"].astype(bool)
    df["label"] = satellite_label

    return df[["timestamp", "satellite", "flux", "observed_flux",
               "electron_correction", "electron_flag", "energy", "label"]]


# ---------------------------------------------------------------------------
# Step 0 -- Parse GOES data
# ---------------------------------------------------------------------------

def parse_goes(
    primary_md: Path = GOES_PRIMARY_MD,
    secondary_md: Path = GOES_SECONDARY_MD,
    date_start: str = DATE_START,
    date_stop: str = DATE_STOP,
) -> pd.DataFrame:
    """
    Parse GOES-18 (primary) and GOES-19 (secondary) X-ray flux data and merge
    into a single wide-format DataFrame.

    Parameters
    ----------
    primary_md : Path
        Path to GOES(PRIMARY).md (GOES-18).
    secondary_md : Path
        Path to GOES(SECONDARY).md (GOES-19).
    date_start : str
        First date to keep, inclusive (YYYY-MM-DD).
    date_stop : str
        Last date to keep, inclusive (YYYY-MM-DD).

    Returns
    -------
    pd.DataFrame with columns:
        timestamp | sxr_g18 (W/m^2) | sxr_g19 (W/m^2) |
        hxr_g18 (W/m^2) | hxr_g19 (W/m^2) |
        electron_flag_g18 | electron_flag_g19
    """
    print("\n" + "=" * 70)
    print("  STEP 0 -- Parsing GOES X-ray flux data")
    print("=" * 70)

    # --- Load both satellites ---
    print(f"\n  Loading primary   (GOES-18): {primary_md}")
    df18 = _parse_goes_md(primary_md, "g18")
    print(f"  Loaded {len(df18):,} raw records (GOES-18)")

    print(f"\n  Loading secondary (GOES-19): {secondary_md}")
    df19 = _parse_goes_md(secondary_md, "g19")
    print(f"  Loaded {len(df19):,} raw records (GOES-19)")

    # --- Date filter ---
    t_start = pd.Timestamp(date_start, tz="UTC")
    t_stop  = pd.Timestamp(date_stop,  tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    def _filter_date(df):
        return df[(df["timestamp"] >= t_start) & (df["timestamp"] <= t_stop)].copy()

    df18 = _filter_date(df18)
    df19 = _filter_date(df19)
    print(f"\n  After date filter ({date_start} to {date_stop}):")
    print(f"    GOES-18 records : {len(df18):,}")
    print(f"    GOES-19 records : {len(df19):,}")

    # --- Split by energy band ---
    def _band(df, energy, flux_col):
        lbl = df["label"].iloc[0]
        sub = df[df["energy"] == energy][["timestamp", "flux", "electron_flag"]].copy()
        sub = sub.rename(columns={
            "flux": flux_col,
            "electron_flag": f"electron_flag_{lbl}",
        })
        return sub.set_index("timestamp")

    sxr18 = _band(df18, "0.1-0.8nm",  "sxr_g18")
    hxr18 = _band(df18, "0.05-0.4nm", "hxr_g18")
    sxr19 = _band(df19, "0.1-0.8nm",  "sxr_g19")
    hxr19 = _band(df19, "0.05-0.4nm", "hxr_g19")

    # --- Merge on timestamp ---
    merged = (
        sxr18
        .join(hxr18[["hxr_g18"]], how="outer")
        .join(sxr19, how="outer")
        .join(hxr19[["hxr_g19"]], how="outer")
    )

    merged = merged.reset_index().rename(columns={"index": "timestamp"})
    # The index name may already be 'timestamp' if set_index was called
    if "timestamp" not in merged.columns:
        merged = merged.reset_index()
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    # Ensure correct column order and presence
    out_cols = ["timestamp", "sxr_g18", "sxr_g19", "hxr_g18", "hxr_g19",
                "electron_flag_g18", "electron_flag_g19"]
    for c in out_cols:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[out_cols]

    # --- Summary statistics ---
    print(f"\n  Merged output: {len(merged):,} rows")
    for col in ["sxr_g18", "sxr_g19", "hxr_g18", "hxr_g19"]:
        s = merged[col].dropna()
        if len(s) > 0:
            print(f"    {col:12s}: min={s.min():.4e}  max={s.max():.4e}  "
                  f"mean={s.mean():.4e}  W/m^2")

    eflag18 = merged["electron_flag_g18"].sum()
    eflag19 = merged["electron_flag_g19"].sum()
    print(f"\n    electron_flag_g18 set: {int(eflag18):,} rows")
    print(f"    electron_flag_g19 set: {int(eflag19):,} rows")

    # --- Save ---
    out_path = OUTPUTS / "goes_parsed.parquet"
    merged.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    return merged


# ---------------------------------------------------------------------------
# Step 1 -- Derive ground-truth flare catalog from GOES-18 SXR
# ---------------------------------------------------------------------------

def detect_flares(
    goes_df: pd.DataFrame,
    k_sigma: float = K_SIGMA,
) -> pd.DataFrame:
    """
    Detect solar flares from GOES-18 SXR (0.1-0.8 nm) light curve using a
    classical background-subtraction + threshold algorithm.

    Parameters
    ----------
    goes_df : pd.DataFrame
        Output from parse_goes(); must contain 'timestamp' and 'sxr_g18' (W/m^2).
    k_sigma : float
        Detection threshold in units of rolling sigma above background.

    Returns
    -------
    pd.DataFrame with schema:
        event_id | start | peak_time | stop | peak_flux_W_m2 (W/m^2) | goes_class
    """
    print("\n" + "=" * 70)
    print("  STEP 1 -- Deriving ground-truth flare catalog from GOES-18 SXR")
    print("=" * 70)

    ts = goes_df.set_index("timestamp")["sxr_g18"].dropna().sort_index()
    ts = ts[~ts.index.duplicated(keep="first")]

    # Resample to 1-min uniform grid (fill gaps by linear interpolation, limit 10 min)
    ts_1min = ts.resample("1min").mean().interpolate(method="time", limit=10)
    print(f"\n  Time series length (1-min grid): {len(ts_1min):,} samples")
    print(f"  Coverage: {ts_1min.index[0]}  ->  {ts_1min.index[-1]}")

    # Rolling background: 10th percentile over trailing 60-min window (W/m^2)
    bg  = ts_1min.rolling(f"{GOES_BG_WINDOW_MIN}min",  min_periods=10).quantile(0.10)
    std = ts_1min.rolling(f"{GOES_STD_WINDOW_MIN}min", min_periods=5).std()

    # Fill leading NaNs in std with a global fallback
    global_std = float(ts_1min.std()) * 0.05
    std = std.bfill().fillna(global_std)
    bg  = bg.bfill().fillna(float(ts_1min.quantile(0.10)))

    # Run state-machine detector
    events = _state_machine_detect(ts_1min, bg, std, k_sigma)
    print(f"\n  Events found at k_sigma={k_sigma}: {len(events)}")

    if len(events) < 5:
        print(f"  WARNING: Only {len(events)} event(s) found with k_sigma={k_sigma}.")
        print(f"    Retrying with k_sigma={K_SIGMA_RETRY} ...")
        events = _state_machine_detect(ts_1min, bg, std, K_SIGMA_RETRY)
        print(f"  Events found at k_sigma={K_SIGMA_RETRY}: {len(events)}")

    # Merge events whose starts are within MERGE_GAP_MIN of each other
    events = _merge_events(events, MERGE_GAP_MIN)
    print(f"  Events after merging close detections: {len(events)}")

    # Build catalog DataFrame
    if not events:
        catalog = pd.DataFrame(columns=[
            "event_id", "start", "peak_time", "stop", "peak_flux_W_m2", "goes_class"
        ])
    else:
        catalog = pd.DataFrame(events)
        catalog["goes_class"] = catalog["peak_flux_W_m2"].apply(_classify_flux)
        catalog = catalog.sort_values("start").reset_index(drop=True)
        catalog.insert(0, "event_id", [f"GT{i+1:03d}" for i in range(len(catalog))])

    print(f"\n  Detected {len(catalog)} flare event(s):\n")
    if len(catalog) > 0:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 120)
        print(catalog.to_string(index=False))
    else:
        print("  (none)")

    out_path = OUTPUTS / "ground_truth_catalog.parquet"
    catalog.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    return catalog


def _state_machine_detect(
    ts: pd.Series,
    bg: pd.Series,
    std: pd.Series,
    k: float,
) -> list:
    """
    State-machine flare detector on a 1-min uniform time series.

    Parameters
    ----------
    ts  : pd.Series  -- SXR flux (W/m^2), DatetimeIndex (UTC, 1-min cadence)
    bg  : pd.Series  -- rolling 10th-pct background (W/m^2)
    std : pd.Series  -- rolling standard deviation (W/m^2)
    k   : float      -- detection threshold multiplier

    Returns
    -------
    list of dicts: {start, peak_time, stop, peak_flux_W_m2}
    """
    vals    = ts.values
    times   = ts.index
    thr_on  = (bg + k * std).values
    thr_off = (bg + DECAY_SIGMA * std).values
    slope   = np.diff(vals, prepend=np.nan)

    n = len(times)
    events = []

    IDLE, COUNTING, IN_FLARE = 0, 1, 2
    state = IDLE
    count = 0
    start_idx = None

    i = 0
    while i < n:
        v = vals[i]
        s = slope[i] if not np.isnan(slope[i]) else 0.0

        if state == IDLE:
            if v > thr_on[i] and s > 0:
                state     = COUNTING
                count     = 1
                start_idx = i

        elif state == COUNTING:
            if v > thr_on[i]:
                count += 1
                if count >= MIN_SUSTAINED_MIN:
                    state = IN_FLARE
                    # start_idx stays at where counting began
            else:
                state = IDLE
                count = 0
                start_idx = None

        if state == IN_FLARE:
            # Find peak within next DECAY_TIMEOUT_MIN samples
            peak_end = min(i + DECAY_TIMEOUT_MIN, n)
            peak_off = int(np.nanargmax(vals[i:peak_end]))
            peak_idx = i + peak_off
            peak_flux = float(vals[peak_idx])
            peak_t    = times[peak_idx]
            start_t   = times[start_idx]

            # Find stop: flux drops to thr_off or timeout
            timeout_idx = min(peak_idx + DECAY_TIMEOUT_MIN, n - 1)
            stop_idx    = timeout_idx
            for j in range(peak_idx + 1, timeout_idx + 1):
                if not np.isnan(vals[j]) and vals[j] <= thr_off[j]:
                    stop_idx = j
                    break

            events.append({
                "start":          start_t,
                "peak_time":      peak_t,
                "stop":           times[stop_idx],
                "peak_flux_W_m2": peak_flux,
            })

            i         = stop_idx + 1
            state     = IDLE
            count     = 0
            start_idx = None
            continue

        i += 1

    return events


def _merge_events(events: list, gap_min: int) -> list:
    """
    Merge events whose start times are within gap_min minutes.
    Keeps the higher-peak version; extends stop to the maximum.

    Parameters
    ----------
    events  : list of dicts {start, peak_time, stop, peak_flux_W_m2}
    gap_min : int -- merge window (minutes)

    Returns
    -------
    Merged list of event dicts.
    """
    if not events:
        return events

    events = sorted(events, key=lambda e: e["start"])
    merged = [dict(events[0])]

    for ev in events[1:]:
        last = merged[-1]
        gap  = (ev["start"] - last["start"]).total_seconds() / 60.0
        if gap <= gap_min:
            if ev["peak_flux_W_m2"] > last["peak_flux_W_m2"]:
                merged[-1] = {
                    "start":          last["start"],
                    "peak_time":      ev["peak_time"],
                    "stop":           max(last["stop"], ev["stop"]),
                    "peak_flux_W_m2": ev["peak_flux_W_m2"],
                }
            else:
                merged[-1]["stop"] = max(last["stop"], ev["stop"])
        else:
            merged.append(dict(ev))

    return merged


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    goes_df = parse_goes()
    catalog = detect_flares(goes_df)
    print("\n  All done -- goes_parser.py complete.")
