# -*- coding: utf-8 -*-
"""
solexs_ingest.py - Step 2: SoLEXS Light Curve Ingestion.

Reads AL1 SoLEXS L1 data for Jun 17-19 2026:
  - Light curve (.lc FITS)  -> TIME (Unix s) and COUNTS (counts/s)
  - Good Time Intervals (.gti FITS) -> START/STOP (Unix s)
  - Spectrum (.pi FITS)  -> loaded but not processed

Key findings from file inspection:
  - TIME column: Unix seconds (epoch 1970-01-01, MJDREFI=40587)
  - COUNTS: total counts per second (single broadband SDD2 channel)
  - GTI: same Unix second epoch
  - Data is 1-second cadence (TIMEDEL=1)
  - Only SDD2 has LC data; SDD1 only has GTI (used for combined flagging)

Units
-----
TIME      : Unix seconds (s) since 1970-01-01T00:00:00 UTC
COUNTS    : counts/s
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

from config import (
    SOLEXS_DATA_ROOT,
    OUTPUTS,
    DATE_START,
    DATE_STOP,
    SOLEXS_BG_WINDOW_MIN,
    SOLEXS_GAP_THRESHOLD_S,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unix_to_utc(unix_s: np.ndarray) -> pd.DatetimeIndex:
    """Convert Unix seconds (float64) to UTC DatetimeIndex (ns precision)."""
    return pd.to_datetime(unix_s, unit="s", utc=True)


def _find_solexs_files(day: str) -> dict:
    """
    Locate SoLEXS FITS files for a given observation date.

    Parameters
    ----------
    day : str -- observation date in YYYYMMDD format (e.g. '20260617')

    Returns
    -------
    dict with keys: 'lc', 'gti', 'pi' (pathlib.Path objects or None)
    """
    date_dir = SOLEXS_DATA_ROOT / f"AL1_SLX_L1_{day}_v1.0" / f"AL1_SLX_L1_{day}_v1.0"
    sdd2_dir = date_dir / "SDD2"

    files = {}
    for ext in ["lc", "gti", "pi"]:
        # Pattern: AL1_SOLEXS_{day}_SDD2_L1.{ext} inside a same-named dir
        candidate_dir  = sdd2_dir / f"AL1_SOLEXS_{day}_SDD2_L1.{ext}"
        candidate_file = candidate_dir / f"AL1_SOLEXS_{day}_SDD2_L1.{ext}"
        if candidate_file.exists():
            files[ext] = candidate_file
        else:
            files[ext] = None
            print(f"    [WARN] {ext.upper()} file not found at expected path: {candidate_file}")

    return files


def _load_lc_day(lc_path: Path, gti_path: Path) -> pd.DataFrame:
    """
    Load a single-day SoLEXS light curve and apply GTI mask.

    Parameters
    ----------
    lc_path  : Path -- .lc FITS file
    gti_path : Path -- .gti FITS file (may be None; all data kept as GOOD)

    Returns
    -------
    pd.DataFrame with columns:
        timestamp (UTC, 1-s) | counts (counts/s) | quality_flag (GOOD/BAD)
    """
    if not lc_path.exists():
        raise FileNotFoundError(f"SoLEXS LC file not found: {lc_path}")

    # Load LC
    with fits.open(lc_path) as hdul:
        hdr = hdul[0].header
        print(f"\n  -- LC file: {lc_path.name}")
        print(f"     MISSION  : {hdr.get('MISSION','N/A')}")
        print(f"     INSTRUME : {hdr.get('INSTRUME','N/A')}")
        print(f"     DATE-OBS : {hdul[1].header.get('DATE-OBS','N/A')}")
        print(f"     TIMEDEL  : {hdul[1].header.get('TIMEDEL','N/A')} s")
        print(f"     MJDREFI  : {hdul[1].header.get('MJDREFI','N/A')}")
        print(f"     TIMESYS  : {hdul[1].header.get('TIMESYS','N/A')}")
        print(f"     Columns  : {[c.name for c in hdul[1].columns]}")
        print(f"     Rows     : {len(hdul[1].data)}")

        rate_hdu = hdul["RATE"]
        time_col   = rate_hdu.data["TIME"]     # Unix seconds
        counts_col = rate_hdu.data["COUNTS"]   # counts/s

    timestamps = _unix_to_utc(time_col)
    df = pd.DataFrame({
        "timestamp": timestamps,
        "counts":    counts_col.astype(np.float64),
    })

    # Load GTI
    gti_intervals = []
    if gti_path is not None and gti_path.exists():
        with fits.open(gti_path) as hdul:
            gti_data = hdul["GTI"].data
            for row in gti_data:
                gti_intervals.append((float(row["START"]), float(row["STOP"])))
        print(f"     GTI segments: {len(gti_intervals)}")
    else:
        print("     GTI file not found -> all data flagged GOOD")

    # Apply GTI mask
    if gti_intervals:
        t_unix = time_col
        good_mask = np.zeros(len(t_unix), dtype=bool)
        for start_s, stop_s in gti_intervals:
            good_mask |= (t_unix >= start_s) & (t_unix <= stop_s)
        df["quality_flag"] = np.where(good_mask, "GOOD", "BAD")
    else:
        df["quality_flag"] = "GOOD"

    return df


# ---------------------------------------------------------------------------
# Step 2 -- SoLEXS Ingestion
# ---------------------------------------------------------------------------

def ingest_solexs(
    date_start: str = DATE_START,
    date_stop:  str = DATE_STOP,
) -> pd.DataFrame:
    """
    Load and process SoLEXS light curves for Jun 17-19 2026.

    Steps
    -----
    1. For each day, find and load .lc and .gti FITS files.
    2. Confirm .pi opens successfully.
    3. Convert timestamps to UTC datetime64[ns].
    4. Apply GTI mask (BAD outside valid intervals).
    5. Concatenate all three days into one DataFrame.
    6. Resample to 1-second UTC grid.
    7. Compute rolling background (10th percentile, 30-min trailing window).

    Parameters
    ----------
    date_start : str -- first date (YYYY-MM-DD), inclusive
    date_stop  : str -- last date (YYYY-MM-DD), inclusive

    Returns
    -------
    pd.DataFrame with columns:
        timestamp | counts (counts/s) | quality_flag | background_counts (counts/s)
    """
    print("\n" + "=" * 70)
    print("  STEP 2 -- SoLEXS Ingestion")
    print("=" * 70)

    # Build list of YYYYMMDD day strings
    dates = pd.date_range(date_start, date_stop, freq="D")
    day_strings = [d.strftime("%Y%m%d") for d in dates]

    all_dfs = []
    for day in day_strings:
        print(f"\n  Processing day: {day}")
        files = _find_solexs_files(day)

        if files["lc"] is None:
            print(f"  [WARN] No LC file for {day} -- skipping day")
            continue

        # Confirm PI opens
        if files["pi"] is not None:
            try:
                with fits.open(files["pi"]) as hdul:
                    print(f"     PI file OK ({len(hdul)} HDUs, {sum(len(h.data) if hasattr(h,'data') and h.data is not None else 0 for h in hdul)} rows)")
            except Exception as e:
                print(f"     [WARN] PI file could not be opened: {e}")
        else:
            print("     PI file: not found")

        day_df = _load_lc_day(files["lc"], files["gti"])
        all_dfs.append(day_df)
        good_pct = (day_df["quality_flag"] == "GOOD").mean() * 100
        print(f"     Day rows: {len(day_df):,}  |  GOOD: {good_pct:.1f}%")

    if not all_dfs:
        raise RuntimeError("No SoLEXS data loaded for any day.")

    # Concatenate all days
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop duplicate timestamps (keep first)
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    # Resample to 1-second UTC grid across full range
    t_start = pd.Timestamp(date_start, tz="UTC")
    t_stop  = pd.Timestamp(date_stop,  tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df_indexed = df.set_index("timestamp")

    # Build full 1-second grid
    full_index = pd.date_range(t_start, t_stop, freq="1s")
    df_resampled = df_indexed.reindex(full_index)
    df_resampled.index.name = "timestamp"

    # Forward-fill quality flag for newly created rows (mark as BAD if data was missing)
    df_resampled["quality_flag"] = df_resampled["quality_flag"].fillna("BAD")

    df_resampled = df_resampled.reset_index()

    # Rolling background: 10th percentile over trailing 30-min window (GOOD data only)
    good_mask = df_resampled["quality_flag"] == "GOOD"
    counts_clean = df_resampled["counts"].where(good_mask)

    # Window in seconds: SOLEXS_BG_WINDOW_MIN minutes * 60 s/min
    bg_window_s = SOLEXS_BG_WINDOW_MIN * 60
    bg = (
        counts_clean
        .rolling(bg_window_s, min_periods=60)
        .quantile(0.10)
    )
    df_resampled["background_counts"] = bg

    # Gap detection (missing >5 consecutive seconds)
    missing = df_resampled["counts"].isna()
    # Count runs of consecutive missing values
    gap_run = (missing != missing.shift()).cumsum()
    gap_lengths = missing.groupby(gap_run).transform("sum")
    n_gaps = (missing & (gap_lengths > SOLEXS_GAP_THRESHOLD_S)).sum()

    # Summary statistics
    total_rows = len(df_resampled)
    bad_pct    = (~good_mask).mean() * 100
    good_df    = df_resampled[good_mask]
    print(f"\n  ---- SoLEXS Summary ----")
    print(f"  Total rows     : {total_rows:,}")
    print(f"  % BAD quality  : {bad_pct:.1f}%")
    print(f"  Data gaps (>{SOLEXS_GAP_THRESHOLD_S}s consecutive missing): {int(n_gaps):,}")
    if len(good_df) > 0:
        c = good_df["counts"].dropna()
        print(f"  counts (GOOD)  : min={c.min():.2f}  max={c.max():.2f}  "
              f"mean={c.mean():.2f}  counts/s")

    # Save
    out_cols = ["timestamp", "counts", "quality_flag", "background_counts"]
    out = df_resampled[out_cols]
    out_path = OUTPUTS / "solexs_clean.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = ingest_solexs()
    print("\n  All done -- solexs_ingest.py complete.")
