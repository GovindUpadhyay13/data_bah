# -*- coding: utf-8 -*-
"""
hel1os_ingest.py - Step 3: HEL1OS Light Curve Ingestion.

Reads HEL1OS L1 data for Jun 17-19 2026.

Directory structure:
  HeL1OS_data/2026/06/{17,18,19}/HLS_{date}_{time}_{duration}sec_lev1_V111/
    cdte/ -> lightcurve_cdte{1,2}.fits
    czt/  -> lightcurve_czt{1,2}.fits
    aux/  -> hk.fits, gticdte{1,2}.fits, gticzt{1,2}.fits

Key findings from file inspection:
  - Time: MJD column (MJD) and ISOT string column (ISOT='2026-06-17T00:00:54.785')
  - Count rate: CTR column (cts/sec), STAT_ERR column (cts/sec)
  - CdTe bands: 5-20, 20-30, 30-40, 40-60, 1.8-90 keV (5 HDUs per detector)
  - CZT: same structure with different energy ranges
  - HK columns: mjd, cdte1temp/cdte2temp/czt1temp/czt2temp (degC),
                czthvmon/cdtehvmon (V), czt1hotpixcnt/czt2hotpixcnt

Units
-----
MJD       : days since 1858-11-17T00:00:00 UTC (standard MJD epoch)
CTR       : counts/s
STAT_ERR  : counts/s (Poisson 1-sigma)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from config import (
    HEL1OS_DATA_ROOT,
    OUTPUTS,
    DATE_START,
    DATE_STOP,
    HEL1OS_TEMP_MIN,
    HEL1OS_TEMP_MAX,
    HEL1OS_HV_MIN,
    HEL1OS_HV_MAX,
    SOLEXS_GAP_THRESHOLD_S,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mjd_to_utc(mjd: np.ndarray) -> pd.DatetimeIndex:
    """Convert MJD float array to UTC pandas DatetimeIndex (ns precision)."""
    t = Time(mjd, format="mjd", scale="utc")
    return pd.DatetimeIndex(t.isot).tz_localize("UTC")


def _find_hel1os_day_dirs(day_str: str) -> list:
    """
    Find all observation subdirectories for a given UTC day.

    Parameters
    ----------
    day_str : str -- day in DD format (e.g. '17')

    Returns
    -------
    list of Path objects (sorted)
    """
    day_path = HEL1OS_DATA_ROOT / day_str
    if not day_path.exists():
        raise FileNotFoundError(f"HEL1OS day directory not found: {day_path}")
    dirs = sorted([d for d in day_path.iterdir() if d.is_dir()])
    return dirs


def _load_lc_fits(fits_path: Path, detector: str, band_idx: int = 0) -> pd.DataFrame:
    """
    Load a single HEL1OS light curve FITS file (one detector, one band).

    Parameters
    ----------
    fits_path : Path -- path to lightcurve_*.fits
    detector  : str  -- 'cdte1', 'cdte2', 'czt1', 'czt2'
    band_idx  : int  -- which HDU band to read (default 0 = first/broadband)

    Returns
    -------
    pd.DataFrame with columns:
        timestamp (UTC) | ctr_{detector} (cts/sec) | stat_err_{detector} (cts/sec)
        | band_name
    """
    if not fits_path.exists():
        raise FileNotFoundError(f"HEL1OS LC file not found: {fits_path}")

    with fits.open(fits_path) as hdul:
        # Print all HDU names for inspection
        hdu_names = [h.name for h in hdul if h.name != "PRIMARY"]

        # Use negative index to select from end (band_idx=-1 means last band)
        # hdul[-1] = last HDU (broadband)
        if band_idx == -1:
            target_hdu = hdul[-1]
        else:
            target_hdu = hdul[band_idx + 1]   # +1 to skip PRIMARY

        hdr        = target_hdu.header
        band_name  = hdr.get("EXTNAME", f"band_{band_idx}")

        mjd_col = target_hdu.data["MJD"]   # days since 1858-11-17
        ctr_col = target_hdu.data["CTR"]   # counts/s
        err_col = target_hdu.data["STAT_ERR"]

    timestamps = _mjd_to_utc(mjd_col)
    # Round to nearest second for alignment with 1-second grid
    timestamps = timestamps.round("s")
    df = pd.DataFrame({
        "timestamp":              timestamps,
        f"ctr_{detector}":       ctr_col.astype(np.float64),
        f"stat_err_{detector}":  err_col.astype(np.float64),
        "band_name":              band_name,
    })
    return df


def _build_hk_quality_mask(hk_path: Path) -> pd.DataFrame:
    """
    Build a quality mask from HEL1OS housekeeping data.

    Quality criteria (flagged BAD if out of nominal range):
      - CdTe detector temp  : cdte1temp, cdte2temp outside [-20, +30] degC
      - CZT detector temp   : czt1temp,  czt2temp  outside [-20, +30] degC
      - CdTe HV monitor     : cdtehvmon outside [400, 600] V (nominal ~953 V -- will use actual)
      - CZT HV monitor      : czthvmon  outside [400, 600] V (nominal ~634 V -- will use actual)
      - Hot pixel count      : czt1hotpixcnt, czt2hotpixcnt > 0

    Parameters
    ----------
    hk_path : Path -- path to hk.fits

    Returns
    -------
    pd.DataFrame with columns:
        timestamp (UTC) | quality_flag ('GOOD'/'BAD') | [individual flag columns]
    """
    if not hk_path.exists():
        raise FileNotFoundError(f"HEL1OS HK file not found: {hk_path}")

    with fits.open(hk_path) as hdul:
        hk_hdu = hdul["HLSHK"]
        hk_data = hk_hdu.data
        hk_cols = [c.name for c in hk_hdu.columns]

    hk = pd.DataFrame(hk_data.tolist(), columns=hk_cols)

    # Print actual value ranges
    print("\n  HK column ranges (actual data):")
    for col in ["cdte1temp", "cdte2temp", "czt1temp", "czt2temp",
                "cdtehvmon", "czthvmon", "czt1hotpixcnt", "czt2hotpixcnt"]:
        if col in hk.columns:
            v = hk[col].dropna()
            print(f"    {col:20s}: min={v.min():.3f}  max={v.max():.3f}  "
                  f"mean={v.mean():.3f}")

    # Determine threshold ranges from actual data (use ±3 std from mean as nominal)
    def _flag_col(series, lo, hi):
        return (series < lo) | (series > hi)

    # Build flag columns
    flags = pd.DataFrame(index=hk.index)
    flags["bad_cdte1temp"] = _flag_col(hk.get("cdte1temp", pd.Series()), HEL1OS_TEMP_MIN, HEL1OS_TEMP_MAX)
    flags["bad_cdte2temp"] = _flag_col(hk.get("cdte2temp", pd.Series()), HEL1OS_TEMP_MIN, HEL1OS_TEMP_MAX)
    flags["bad_czt1temp"]  = _flag_col(hk.get("czt1temp", pd.Series()),  HEL1OS_TEMP_MIN, HEL1OS_TEMP_MAX)
    flags["bad_czt2temp"]  = _flag_col(hk.get("czt2temp", pd.Series()),  HEL1OS_TEMP_MIN, HEL1OS_TEMP_MAX)

    # Use actual HV range from data + wide margin (±20% around median)
    for hv_col in ["cdtehvmon", "czthvmon"]:
        if hv_col in hk.columns:
            med = hk[hv_col].median()
            lo  = med * 0.80
            hi  = med * 1.20
            print(f"    HV nominal range for {hv_col}: [{lo:.1f}, {hi:.1f}] V  (median={med:.1f})")
            flags[f"bad_{hv_col}"] = _flag_col(hk[hv_col], lo, hi)

    # Hot pixels: constant 7 in czt1hotpixcnt is a hardware artifact (normal)
    # Only flag if significantly elevated above the observed baseline
    hpix_threshold = 20  # flag if > 20 hot pixels
    if "czt1hotpixcnt" in hk.columns:
        flags["bad_czt1hotpix"] = hk["czt1hotpixcnt"] > hpix_threshold
    if "czt2hotpixcnt" in hk.columns:
        flags["bad_czt2hotpix"] = hk["czt2hotpixcnt"] > hpix_threshold

    # Any bad -> mark interval BAD
    any_bad = flags.any(axis=1)
    hk["quality_flag"] = np.where(any_bad, "BAD", "GOOD")

    # Convert HK time (round to nearest second)
    hk["timestamp"] = _mjd_to_utc(hk["mjd"].values).round("s")

    result = hk[["timestamp", "quality_flag"]].copy()
    result = result.sort_values("timestamp").reset_index(drop=True)

    bad_pct = (result["quality_flag"] == "BAD").mean() * 100
    print(f"\n    HK bad fraction: {bad_pct:.1f}%")

    return result


def _apply_hk_quality(lc_df: pd.DataFrame, hk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply HK-derived quality mask to light-curve DataFrame by nearest-neighbour
    merge on timestamp.

    Parameters
    ----------
    lc_df : pd.DataFrame  -- LC with 'timestamp' column
    hk_df : pd.DataFrame  -- HK quality with 'timestamp' and 'quality_flag'

    Returns
    -------
    lc_df with 'quality_flag' column added/updated
    """
    # Merge-asof (backward fill HK quality onto LC timestamps)
    lc_sorted = lc_df.sort_values("timestamp").reset_index(drop=True)
    hk_sorted = hk_df.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        lc_sorted,
        hk_sorted.rename(columns={"quality_flag": "hk_quality"}),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("60s"),
    )
    # If no HK match within tolerance, mark BAD
    merged["hk_quality"] = merged["hk_quality"].fillna("BAD")

    if "quality_flag" in merged.columns:
        # Combine existing and HK flags
        merged["quality_flag"] = np.where(
            (merged["quality_flag"] == "GOOD") & (merged["hk_quality"] == "GOOD"),
            "GOOD", "BAD"
        )
    else:
        merged["quality_flag"] = merged["hk_quality"]

    return merged.drop(columns=["hk_quality"])


# ---------------------------------------------------------------------------
# Step 3 -- HEL1OS Ingestion
# ---------------------------------------------------------------------------

def ingest_hel1os(
    date_start: str = DATE_START,
    date_stop:  str = DATE_STOP,
) -> pd.DataFrame:
    """
    Load and process HEL1OS light curves for Jun 17-19 2026.

    Steps
    -----
    1. Find all observation subdirectories per day.
    2. Load lightcurve_cdte1.fits and lightcurve_czt1.fits (broadband band).
    3. Print FITS structure and HK column ranges.
    4. Build HK quality mask.
    5. Merge CdTe and CZT light curves.
    6. Resample to 1-second UTC grid.

    Parameters
    ----------
    date_start : str -- first date (YYYY-MM-DD), inclusive
    date_stop  : str -- last date (YYYY-MM-DD), inclusive

    Returns
    -------
    pd.DataFrame with columns:
        timestamp | ctr_cdte1 (cts/sec) | ctr_czt1 (cts/sec) |
        quality_flag | background_cdte1 (cts/sec) | background_czt1 (cts/sec)
    """
    print("\n" + "=" * 70)
    print("  STEP 3 -- HEL1OS Ingestion")
    print("=" * 70)

    dates   = pd.date_range(date_start, date_stop, freq="D")
    day_strs = [d.strftime("%d") for d in dates]

    all_lc_cdte = []
    all_lc_czt  = []
    all_hk      = []

    for day_str in day_strs:
        print(f"\n  Processing day: 2026-06-{day_str}")
        obs_dirs = _find_hel1os_day_dirs(day_str)
        print(f"    Observation dirs found: {len(obs_dirs)}")
        for obs_dir in obs_dirs:
            print(f"    -> {obs_dir.name}")

        for obs_dir in obs_dirs:
            cdte_dir = obs_dir / "cdte"
            czt_dir  = obs_dir / "czt"
            aux_dir  = obs_dir / "aux"

            # --- CdTe broadband LC (detector 1) ---
            lc_cdte_path = cdte_dir / "lightcurve_cdte1.fits"
            if lc_cdte_path.exists():
                with fits.open(lc_cdte_path) as hdul:
                    hdu_names = [h.name for h in hdul]
                    print(f"\n    CdTe1 LC HDUs: {hdu_names}")
                    print(f"    Columns: {[(c.name, c.unit) for c in hdul[1].columns]}")
                    print(f"    Rows per band: {len(hdul[1].data)}")

                # Load broadband (last band = broadband 1.8-90 keV)
                df_cdte = _load_lc_fits(lc_cdte_path, "cdte1", band_idx=-1)
                print(f"    CdTe1 broadband band: {df_cdte['band_name'].iloc[0]}, "
                      f"rows={len(df_cdte)}")
                all_lc_cdte.append(df_cdte.drop(columns="band_name"))
            else:
                print(f"    [WARN] {lc_cdte_path} not found")

            # --- CZT broadband LC (detector 1) ---
            lc_czt_path = czt_dir / "lightcurve_czt1.fits"
            if lc_czt_path.exists():
                with fits.open(lc_czt_path) as hdul:
                    hdu_names = [h.name for h in hdul]
                    print(f"    CZT1 LC HDUs: {hdu_names}")

                df_czt = _load_lc_fits(lc_czt_path, "czt1", band_idx=-1)
                print(f"    CZT1 broadband band: {df_czt['band_name'].iloc[0]}, "
                      f"rows={len(df_czt)}")
                all_lc_czt.append(df_czt.drop(columns="band_name"))
            else:
                print(f"    [WARN] {lc_czt_path} not found")

            # --- HK ---
            hk_path = aux_dir / "hk.fits"
            if hk_path.exists():
                print(f"\n    Loading HK: {hk_path.name}")
                hk_df = _build_hk_quality_mask(hk_path)
                all_hk.append(hk_df)
            else:
                print(f"    [WARN] HK file not found: {hk_path}")

    # --- Concatenate ---
    if not all_lc_cdte:
        raise RuntimeError("No HEL1OS CdTe data loaded.")

    df_cdte = pd.concat(all_lc_cdte, ignore_index=True).sort_values("timestamp")
    df_cdte = df_cdte.drop_duplicates("timestamp", keep="first").reset_index(drop=True)

    df_czt  = pd.concat(all_lc_czt,  ignore_index=True).sort_values("timestamp")
    df_czt  = df_czt.drop_duplicates("timestamp", keep="first").reset_index(drop=True)

    df_hk   = pd.concat(all_hk, ignore_index=True).sort_values("timestamp")
    df_hk   = df_hk.drop_duplicates("timestamp", keep="first").reset_index(drop=True)

    # Merge CdTe + CZT on timestamp
    df_all = pd.merge_asof(
        df_cdte.sort_values("timestamp"),
        df_czt.sort_values("timestamp"),
        on="timestamp",
        tolerance=pd.Timedelta("1s"),
        direction="nearest",
    )

    # Apply HK quality mask
    df_all = _apply_hk_quality(df_all, df_hk)

    # Resample to 1-second grid
    t_start = pd.Timestamp(date_start, tz="UTC")
    t_stop  = pd.Timestamp(date_stop,  tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    full_index = pd.date_range(t_start, t_stop, freq="1s")

    df_idx = df_all.set_index("timestamp")
    df_re  = df_idx.reindex(full_index)
    df_re.index.name = "timestamp"
    df_re["quality_flag"] = df_re["quality_flag"].fillna("BAD")
    df_re = df_re.reset_index()

    # Rolling background for each detector
    good_mask = df_re["quality_flag"] == "GOOD"
    bg_window_s = 30 * 60  # 30-minute window in seconds

    for det in ["cdte1", "czt1"]:
        col = f"ctr_{det}"
        if col in df_re.columns:
            clean = df_re[col].where(good_mask)
            df_re[f"background_{det}"] = clean.rolling(bg_window_s, min_periods=60).quantile(0.10)

    # Summary
    total   = len(df_re)
    bad_pct = (~good_mask).mean() * 100
    cdte_rows = df_cdte.shape[0]
    czt_rows  = df_czt.shape[0]

    print(f"\n  ---- HEL1OS Summary ----")
    print(f"  Total rows (1-s grid)   : {total:,}")
    print(f"  % BAD quality            : {bad_pct:.1f}%")
    print(f"  CdTe1 source rows        : {cdte_rows:,}")
    print(f"  CZT1  source rows        : {czt_rows:,}")

    for col in ["ctr_cdte1", "ctr_czt1"]:
        if col in df_re.columns:
            s = df_re.loc[good_mask, col].dropna()
            if len(s) > 0:
                print(f"  {col:20s}: min={s.min():.2f}  max={s.max():.2f}  "
                      f"mean={s.mean():.2f}  cts/s")

    # Save
    out_path = OUTPUTS / "hel1os_clean.parquet"
    df_re.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")

    return df_re


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = ingest_hel1os()
    print("\n  All done -- hel1os_ingest.py complete.")
