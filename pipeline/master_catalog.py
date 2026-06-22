# -*- coding: utf-8 -*-
"""
master_catalog.py - Phase C.3: Multi-Instrument Master Catalog Fusion.

Merges SoLEXS and HEL1OS detections into a single master catalog by
temporal overlap. Assigns confidence scores:

  confidence = 1  ->  single-instrument detection (SoLEXS or HEL1OS only)
  confidence = 2  ->  both instruments agree (>= 50% temporal overlap)

The merged event interval is the UNION of the two instrument intervals.
The classification uses the SoLEXS equivalent class when both agree;
if SoLEXS is missing, uses HEL1OS class.

Output
------
master_catalog.parquet with schema:
    event_id        : MC{NNN}
    start           : UTC timestamp (earliest of matched pair)
    peak_time       : UTC timestamp (from primary = SoLEXS if available)
    stop            : UTC timestamp (latest of matched pair)
    peak_count_rate : peak CR from primary instrument (counts/s)
    goes_equivalent_class : 'A','B','C','M','X'
    confidence      : 1 or 2
    instruments     : comma-separated list of instruments confirming event
    sx_event_id     : SoLEXS event ID (or None)
    hx_event_id     : HEL1OS event ID (or None)
    sx_snr          : SoLEXS confidence_snr (NaN if no SoLEXS match)
    hx_snr          : HEL1OS confidence_snr (NaN if no HEL1OS match)

Units
-----
start, peak_time, stop : UTC datetime64[ns, UTC]
peak_count_rate        : counts/s
sx_snr, hx_snr         : dimensionless
"""

from pathlib import Path

import numpy as np
import pandas as pd

from config import OUTPUTS, OVERLAP_FRACTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlap_fraction_ab(a_start, a_stop, b_start, b_stop) -> float:
    """
    Fraction of interval B that overlaps with interval A.

    Parameters
    ----------
    a_start, a_stop : timestamps (interval A)
    b_start, b_stop : timestamps (interval B = reference for fraction)

    Returns
    -------
    float in [0, 1]
    """
    b_dur = (b_stop - b_start).total_seconds()
    if b_dur <= 0:
        return 0.0
    ol_start = max(a_start, b_start)
    ol_stop  = min(a_stop,  b_stop)
    ol_s     = max(0.0, (ol_stop - ol_start).total_seconds())
    return ol_s / b_dur


def _symmetric_overlap(s1, e1, s2, e2) -> float:
    """
    Symmetric overlap fraction: max(overlap/dur_A, overlap/dur_B).

    Parameters
    ----------
    s1, e1 : start/stop of interval 1
    s2, e2 : start/stop of interval 2

    Returns
    -------
    float in [0, 1]
    """
    ol_start = max(s1, s2)
    ol_stop  = min(e1, e2)
    ol_s     = max(0.0, (ol_stop - ol_start).total_seconds())
    dur1     = max((e1 - s1).total_seconds(), 1.0)
    dur2     = max((e2 - s2).total_seconds(), 1.0)
    return max(ol_s / dur1, ol_s / dur2)


# ---------------------------------------------------------------------------
# Phase C.3 -- Master Catalog Fusion
# ---------------------------------------------------------------------------

def build_master_catalog(
    sx_path:  Path = OUTPUTS / "solexs_detections.parquet",
    hx_path:  Path = OUTPUTS / "hel1os_detections.parquet",
    gt_path:  Path = OUTPUTS / "ground_truth_catalog.parquet",
    out_path: Path = OUTPUTS / "master_catalog.parquet",
    overlap_threshold: float = OVERLAP_FRACTION,
) -> pd.DataFrame:
    """
    Fuse SoLEXS and HEL1OS detection catalogs into a master catalog.

    Matching algorithm
    ------------------
    1. For each SoLEXS event, find all HEL1OS events with symmetric temporal
       overlap >= overlap_threshold (default 50%).
    2. Greedy best-overlap pairing (each SoLEXS event matched to at most one
       HEL1OS event, each HEL1OS event matched to at most one SoLEXS event).
    3. Matched pairs -> confidence=2; unmatched singles -> confidence=1.
    4. Merged interval = [min(sx_start, hx_start), max(sx_stop, hx_stop)].

    Parameters
    ----------
    sx_path           : Path -- SoLEXS detections parquet
    hx_path           : Path -- HEL1OS detections parquet
    gt_path           : Path -- ground truth catalog (for diagnostic only)
    out_path          : Path -- output master_catalog.parquet
    overlap_threshold : float -- minimum symmetric overlap to count as match

    Returns
    -------
    pd.DataFrame -- master catalog
    """
    print("\n" + "=" * 70)
    print("  PHASE C.3 -- Master Catalog Fusion")
    print("=" * 70)

    sx = pd.read_parquet(sx_path)
    hx = pd.read_parquet(hx_path)
    gt = pd.read_parquet(gt_path)

    print(f"\n  SoLEXS detections  : {len(sx)}")
    print(f"  HEL1OS detections  : {len(hx)}")
    print(f"  Overlap threshold  : {overlap_threshold:.0%}")

    # Convert timestamps
    for df in [sx, hx, gt]:
        for col in ["start", "stop", "peak_time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

    # ---------------------------------------------------------------------------
    # Step 1: Build overlap score matrix (SoLEXS x HEL1OS)
    # ---------------------------------------------------------------------------
    overlap_scores = np.zeros((len(sx), len(hx)), dtype=np.float32)
    for i, sx_row in sx.iterrows():
        for j, hx_row in hx.iterrows():
            ov = _symmetric_overlap(
                sx_row.start, sx_row.stop,
                hx_row.start, hx_row.stop,
            )
            overlap_scores[i, j] = ov

    # ---------------------------------------------------------------------------
    # Step 2: Greedy best-overlap matching
    # ---------------------------------------------------------------------------
    sx_matched = {}   # sx_idx -> hx_idx
    hx_used    = set()

    # Sort pairs by overlap descending
    pairs = [
        (overlap_scores[i, j], i, j)
        for i in range(len(sx))
        for j in range(len(hx))
        if overlap_scores[i, j] >= overlap_threshold
    ]
    pairs.sort(reverse=True)

    for ov_score, i, j in pairs:
        if i in sx_matched or j in hx_used:
            continue
        sx_matched[i] = j
        hx_used.add(j)

    n_matched = len(sx_matched)
    print(f"\n  Matched (confidence=2) : {n_matched}")
    print(f"  SoLEXS-only (conf=1)  : {len(sx) - n_matched}")
    print(f"  HEL1OS-only (conf=1)  : {len(hx) - len(hx_used)}")

    # ---------------------------------------------------------------------------
    # Step 3: Build master catalog rows
    # ---------------------------------------------------------------------------
    records = []

    # Confidence=2 (both instruments)
    for sx_idx, hx_idx in sx_matched.items():
        sx_row = sx.iloc[sx_idx]
        hx_row = hx.iloc[hx_idx]

        merged_start = min(sx_row.start, hx_row.start)
        merged_stop  = max(sx_row.stop,  hx_row.stop)
        # Primary = SoLEXS for class / peak (it has direct SXR calibration)
        records.append({
            "start":                 merged_start,
            "peak_time":             sx_row.peak_time,
            "stop":                  merged_stop,
            "peak_count_rate":       sx_row.peak_count_rate,
            "goes_equivalent_class": sx_row.goes_equivalent_class,
            "confidence":            2,
            "instruments":           "SoLEXS,HEL1OS-CdTe1",
            "sx_event_id":           sx_row.event_id,
            "hx_event_id":           hx_row.event_id,
            "sx_snr":                sx_row.confidence_snr,
            "hx_snr":                hx_row.confidence_snr,
        })

    # SoLEXS-only (confidence=1)
    for sx_idx in range(len(sx)):
        if sx_idx in sx_matched:
            continue
        sx_row = sx.iloc[sx_idx]
        records.append({
            "start":                 sx_row.start,
            "peak_time":             sx_row.peak_time,
            "stop":                  sx_row.stop,
            "peak_count_rate":       sx_row.peak_count_rate,
            "goes_equivalent_class": sx_row.goes_equivalent_class,
            "confidence":            1,
            "instruments":           "SoLEXS",
            "sx_event_id":           sx_row.event_id,
            "hx_event_id":           None,
            "sx_snr":                sx_row.confidence_snr,
            "hx_snr":                np.nan,
        })

    # HEL1OS-only (confidence=1)
    for hx_idx in range(len(hx)):
        if hx_idx in hx_used:
            continue
        hx_row = hx.iloc[hx_idx]
        records.append({
            "start":                 hx_row.start,
            "peak_time":             hx_row.peak_time,
            "stop":                  hx_row.stop,
            "peak_count_rate":       hx_row.peak_count_rate,
            "goes_equivalent_class": hx_row.goes_equivalent_class,
            "confidence":            1,
            "instruments":           "HEL1OS-CdTe1",
            "sx_event_id":           None,
            "hx_event_id":           hx_row.event_id,
            "sx_snr":                np.nan,
            "hx_snr":                hx_row.confidence_snr,
        })

    df = pd.DataFrame(records).sort_values("start").reset_index(drop=True)
    df.insert(0, "event_id", [f"MC{i+1:03d}" for i in range(len(df))])

    # ---------------------------------------------------------------------------
    # Step 4: Summary statistics
    # ---------------------------------------------------------------------------
    conf_counts = df["confidence"].value_counts().sort_index()
    print(f"\n  Master catalog total events: {len(df)}")
    for conf, cnt in conf_counts.items():
        label = "both instruments" if conf == 2 else "single instrument"
        print(f"    Confidence={conf} ({label}): {cnt}")

    class_counts = df["goes_equivalent_class"].value_counts()
    print(f"\n  Class distribution:")
    for cls, cnt in sorted(class_counts.items()):
        print(f"    {cls}: {cnt}")

    df.to_parquet(out_path, index=False)
    print(f"\n  Saved -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = build_master_catalog()
    print("\n  All done -- master_catalog.py complete.")
