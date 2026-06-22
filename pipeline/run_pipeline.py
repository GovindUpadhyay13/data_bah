# -*- coding: utf-8 -*-
"""
run_pipeline.py - Full SoLEXS/HEL1OS Flare Detection Pipeline Orchestrator.

Runs all steps in order:
  Step 0  -- Parse GOES JSON data
  Step 1  -- Detect ground-truth flare catalog from GOES-18 SXR
  Step 2  -- Ingest SoLEXS light curves
  Step 3  -- Ingest HEL1OS light curves
  Step 4  -- Flux-count rate calibration
  Step 5  -- SoLEXS classical flare detector
  Step 6  -- Evaluation vs. ground truth

Usage
-----
  python run_pipeline.py            # run all steps
  python run_pipeline.py --from 4  # resume from step 4 (requires prior outputs)
"""

import sys
import argparse
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Banner helper
# ---------------------------------------------------------------------------

def banner(msg: str):
    """Print a phase banner."""
    line = "#" * (len(msg) + 4)
    print(f"\n{line}")
    print(f"# {msg} #")
    print(f"{line}\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SoLEXS/HEL1OS Flare Pipeline")
    parser.add_argument("--from", dest="from_step", type=int, default=0,
                        help="Start from this step (0-6); requires prior outputs.")
    args = parser.parse_args()
    start_step = args.from_step

    banner(f"ADITYA-L1 SOLAR FLARE DETECTION PIPELINE  (starting at Step {start_step})")

    goes_df  = None
    catalog  = None
    solexs   = None
    hel1os   = None
    coeffs   = None
    det_df   = None

    # -------------------------------------------------------------------------
    # Step 0 + 1 -- GOES parsing and ground-truth catalog
    # -------------------------------------------------------------------------
    if start_step <= 1:
        banner("Phase A: GOES Data -- Steps 0 + 1")
        try:
            from goes_parser import parse_goes, detect_flares
            goes_df = parse_goes()
            catalog = detect_flares(goes_df)
        except Exception as e:
            print(f"ERROR in Steps 0/1: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        from config import OUTPUTS
        import pandas as pd
        goes_df = pd.read_parquet(OUTPUTS / "goes_parsed.parquet")
        catalog = pd.read_parquet(OUTPUTS / "ground_truth_catalog.parquet")
        print(f"Loaded goes_parsed.parquet ({len(goes_df):,} rows)")
        print(f"Loaded ground_truth_catalog.parquet ({len(catalog)} events)")

    # -------------------------------------------------------------------------
    # Step 2 -- SoLEXS ingestion
    # -------------------------------------------------------------------------
    if start_step <= 2:
        banner("Phase B: SoLEXS Ingestion -- Step 2")
        try:
            from solexs_ingest import ingest_solexs
            solexs = ingest_solexs()
        except Exception as e:
            print(f"ERROR in Step 2: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        from config import OUTPUTS
        import pandas as pd
        solexs = pd.read_parquet(OUTPUTS / "solexs_clean.parquet")
        print(f"Loaded solexs_clean.parquet ({len(solexs):,} rows)")

    # -------------------------------------------------------------------------
    # Step 3 -- HEL1OS ingestion
    # -------------------------------------------------------------------------
    if start_step <= 3:
        banner("Phase C: HEL1OS Ingestion -- Step 3")
        try:
            from hel1os_ingest import ingest_hel1os
            hel1os = ingest_hel1os()
        except Exception as e:
            print(f"ERROR in Step 3: {e}")
            traceback.print_exc()
            print("(Continuing with SoLEXS-only analysis)")

    # -------------------------------------------------------------------------
    # Step 4 -- Calibration
    # -------------------------------------------------------------------------
    if start_step <= 4:
        banner("Phase D: Calibration -- Step 4")
        try:
            from calibration import calibrate
            coeffs = calibrate()
        except Exception as e:
            print(f"ERROR in Step 4: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 5 -- SoLEXS detector
    # -------------------------------------------------------------------------
    if start_step <= 5:
        banner("Phase E: SoLEXS Detector -- Step 5")
        try:
            from detector import detect_solexs_flares
            det_df = detect_solexs_flares()
        except Exception as e:
            print(f"ERROR in Step 5: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 6 -- Evaluation
    # -------------------------------------------------------------------------
    if start_step <= 6:
        banner("Phase F: Evaluation -- Step 6")
        try:
            from evaluation import evaluate
            match_df = evaluate()
        except Exception as e:
            print(f"ERROR in Step 6: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 7 -- HEL1OS Detector
    # -------------------------------------------------------------------------
    if start_step <= 7:
        banner("Phase G: HEL1OS Detector -- Step 7")
        try:
            from hel1os_detector import detect_hel1os_flares
            detect_hel1os_flares()
        except Exception as e:
            print(f"ERROR in Step 7: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 8 -- Master Catalog Fusion
    # -------------------------------------------------------------------------
    if start_step <= 8:
        banner("Phase H: Master Catalog Fusion -- Step 8")
        try:
            from master_catalog import build_master_catalog
            build_master_catalog()
        except Exception as e:
            print(f"ERROR in Step 8: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 9 -- Master Catalog Evaluation
    # -------------------------------------------------------------------------
    if start_step <= 9:
        banner("Phase I: Master Catalog Evaluation -- Step 9")
        try:
            from evaluation import evaluate
            from config import OUTPUTS
            evaluate(det_path=OUTPUTS / "master_catalog.parquet")
        except Exception as e:
            print(f"ERROR in Step 9: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 10 -- Flare Forecasting
    # -------------------------------------------------------------------------
    if start_step <= 10:
        banner("Phase J: Flare Forecasting -- Step 10")
        try:
            from forecasting import build_features_and_train
            build_features_and_train()
        except Exception as e:
            print(f"ERROR in Step 10: {e}")
            traceback.print_exc()
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 11 -- Dashboard Generation
    # -------------------------------------------------------------------------
    if start_step <= 11:
        banner("Phase K: Dashboard Generation -- Step 11")
        try:
            from dashboard import build_dashboard
            build_dashboard()
        except Exception as e:
            print(f"ERROR in Step 11: {e}")
            traceback.print_exc()
            sys.exit(1)

    banner("PIPELINE COMPLETE")
    print("Output files in: pipeline/outputs/")
    from config import OUTPUTS
    for f in sorted(OUTPUTS.iterdir()):
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name:45s}  {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
