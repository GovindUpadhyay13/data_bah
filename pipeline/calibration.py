# -*- coding: utf-8 -*-
"""
calibration.py - Step 4: Flux-to-Class Calibration.

Pairs each GOES-derived ground-truth flare with its SoLEXS peak count rate,
fits a log-log linear regression, and saves calibration coefficients.

Calibration model
-----------------
    log10(solexs_peak_counts)  =  a * log10(goes_peak_flux)  +  b
where:
    solexs_peak_counts  : peak count rate during event (counts/s)
    goes_peak_flux      : peak GOES-18 SXR flux (W/m^2)

If fewer than 3 matched events are available, a default mapping based on
approximate literature values is used (documented in calibration_coeffs.json).

Units
-----
goes_flux      : W/m^2
count_rate     : counts/s
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Windows
import matplotlib.pyplot as plt

from config import OUTPUTS

# ---------------------------------------------------------------------------
# Defaults (literature-based approximate calibration for SoLEXS-class instrument)
# ---------------------------------------------------------------------------
DEFAULT_SLOPE     = 1.0    # log-log slope (linear relationship assumed)
DEFAULT_INTERCEPT = 10.0   # log-log intercept (rough order-of-magnitude)
DEFAULT_R2        = None   # marks this as a fallback

GOOD_FRACTION_MIN = 0.50   # minimum fraction of GOOD samples to use peak


# ---------------------------------------------------------------------------
# Step 4 -- Calibration
# ---------------------------------------------------------------------------

def calibrate(
    catalog_path: Path    = OUTPUTS / "ground_truth_catalog.parquet",
    solexs_path: Path     = OUTPUTS / "solexs_clean.parquet",
    coeffs_out: Path      = OUTPUTS / "calibration_coeffs.json",
    plot_out: Path        = OUTPUTS / "calibration_curve.png",
) -> dict:
    """
    Match GOES-18 ground-truth flares to SoLEXS peak count rates and
    fit a log-log linear calibration.

    Parameters
    ----------
    catalog_path : Path -- ground_truth_catalog.parquet from goes_parser.py
    solexs_path  : Path -- solexs_clean.parquet from solexs_ingest.py
    coeffs_out   : Path -- output JSON with a, b, r2 coefficients
    plot_out     : Path -- output PNG calibration scatter plot

    Returns
    -------
    dict with keys: a (slope), b (intercept), r2, method ('regression'|'default')
    """
    print("\n" + "=" * 70)
    print("  STEP 4 -- Flux-to-Class Calibration")
    print("=" * 70)

    # Load data
    catalog = pd.read_parquet(catalog_path)
    solexs  = pd.read_parquet(solexs_path)
    solexs  = solexs.set_index("timestamp")

    print(f"\n  Ground-truth events  : {len(catalog)}")
    print(f"  SoLEXS total rows    : {len(solexs):,}")

    pairs = []

    for _, evt in catalog.iterrows():
        start = pd.Timestamp(evt["start"])
        stop  = pd.Timestamp(evt["stop"])

        # Extract SoLEXS data during event
        mask = (solexs.index >= start) & (solexs.index <= stop)
        seg  = solexs.loc[mask]

        if len(seg) == 0:
            print(f"    {evt['event_id']}: no SoLEXS data in interval [{start}, {stop}]")
            continue

        good_frac = (seg["quality_flag"] == "GOOD").mean()
        if good_frac < GOOD_FRACTION_MIN:
            print(f"    {evt['event_id']}: GOOD fraction={good_frac:.2f} < {GOOD_FRACTION_MIN} -- skipped")
            continue

        # Peak count rate (GOOD samples only)
        good_counts = seg.loc[seg["quality_flag"] == "GOOD", "counts"]
        peak_cr     = float(good_counts.max())

        if np.isnan(peak_cr) or peak_cr <= 0:
            print(f"    {evt['event_id']}: invalid peak counts={peak_cr} -- skipped")
            continue

        pairs.append({
            "event_id":          evt["event_id"],
            "goes_class":        evt["goes_class"],
            "goes_peak_flux":    evt["peak_flux_W_m2"],
            "solexs_peak_cr":    peak_cr,
        })
        print(f"    {evt['event_id']} ({evt['goes_class']}): "
              f"GOES={evt['peak_flux_W_m2']:.3e} W/m^2  "
              f"SoLEXS peak={peak_cr:.1f} cts/s  GOOD={good_frac:.0%}")

    pairs_df = pd.DataFrame(pairs)
    print(f"\n  Matched pairs: {len(pairs_df)}")

    if len(pairs_df) >= 3:
        # Log-log linear regression
        log_flux = np.log10(pairs_df["goes_peak_flux"].values)
        log_cr   = np.log10(pairs_df["solexs_peak_cr"].values)

        result = stats.linregress(log_flux, log_cr)
        a   = float(result.slope)
        b   = float(result.intercept)
        r2  = float(result.rvalue ** 2)
        method = "regression"

        print(f"\n  Calibration fit:")
        print(f"    log10(CR) = {a:.4f} * log10(flux) + {b:.4f}")
        print(f"    R^2       = {r2:.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            pairs_df["goes_peak_flux"],
            pairs_df["solexs_peak_cr"],
            c=pairs_df["goes_class"].map({"A":0,"B":1,"C":2,"M":3,"X":4}),
            cmap="plasma", s=80, zorder=5, label="Matched events",
        )
        # Fit line
        flux_range = np.logspace(
            np.log10(pairs_df["goes_peak_flux"].min()) - 0.5,
            np.log10(pairs_df["goes_peak_flux"].max()) + 0.5, 100
        )
        cr_fit = 10 ** (a * np.log10(flux_range) + b)
        ax.plot(flux_range, cr_fit, "r--", lw=2, label=f"Fit: log10(CR)={a:.2f}*log10(F)+{b:.2f}, R2={r2:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("GOES-18 SXR Peak Flux (W/m^2)")
        ax.set_ylabel("SoLEXS Peak Count Rate (counts/s)")
        ax.set_title("SoLEXS Flux-Count Rate Calibration")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add class markers
        for cls, lo, hi in [("B",1e-7,1e-6),("C",1e-6,1e-5),("M",1e-5,1e-4),("X",1e-4,1e-3)]:
            ax.axvspan(lo, hi, alpha=0.05, label=cls)

        plt.tight_layout()
        plt.savefig(plot_out, dpi=150)
        plt.close()
        print(f"  Saved plot -> {plot_out}")

    else:
        # Default fallback
        print(f"\n  WARNING: Only {len(pairs_df)} matched pairs -- using default calibration.")
        print("  ASSUMPTION: Literature approximate for SoLEXS-class SDD detector.")
        print("  Default: log10(CR) = 1.0 * log10(flux) + 10.0")
        print("  This assumes ~1 count/s per 1e-10 W/m^2 (rough order-of-magnitude).")
        a       = DEFAULT_SLOPE
        b       = DEFAULT_INTERCEPT
        r2      = None
        method  = "default"

    coeffs = {
        "a":          a,
        "b":          b,
        "r2":         r2,
        "method":     method,
        "n_pairs":    len(pairs_df),
        "description": (
            "log10(solexs_peak_counts) = a * log10(goes_peak_flux_wm2) + b"
        ),
    }

    with open(coeffs_out, "w") as f:
        json.dump(coeffs, f, indent=2)
    print(f"\n  Saved coefficients -> {coeffs_out}")
    print(f"  Coefficients: a={a:.4f}, b={b:.4f}, R2={r2}")

    return coeffs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = calibrate()
    print("\n  All done -- calibration.py complete.")
