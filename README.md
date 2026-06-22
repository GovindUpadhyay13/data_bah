# Aditya-L1 Solar Flare Detection & Forecasting Pipeline

An end-to-end space weather data processing and machine learning pipeline for solar flare detection and forecasting. This project merges observations from the Aditya-L1 spacecraft (using the SoLEXS and HEL1OS instruments) with GOES-18 and GOES-19 satellite data, trains a forecasting model, and compiles an interactive weather dashboard.

---

## 1. Directory Structure

```
BAH/
├── README.md                  # Project overview and execution guide (this file)
├── DATA_EXTRACTION.md         # Detailed telemetry extraction guide (no code)
└── pipeline/                  # Pipeline source scripts
    ├── config.py              # Configuration thresholds and parameters
    ├── goes_parser.py         # GOES parser & Ground Truth generator
    ├── solexs_ingest.py       # SoLEXS FITS ingestion & GTI filter
    ├── hel1os_ingest.py       # HEL1OS FITS ingestion & HK health validator
    ├── calibration.py         # Flux-to-count rate log-log linear scaling
    ├── detector.py            # SoLEXS classical threshold detector
    ├── hel1os_detector.py     # HEL1OS CdTe classical threshold detector
    ├── master_catalog.py      # Multi-instrument master catalog fusion
    ├── evaluation.py          # Catalog evaluation module (TSS, TPR, lead times)
    ├── forecasting.py         # Machine learning forecasting model (LightGBM)
    ├── dashboard.py           # Base64 embedded HTML dashboard generator
    ├── run_pipeline.py        # Pipeline orchestrator
    └── outputs/               # Clean Parquet tables, metrics, and plots (ignored by Git)
```

---

## 2. Pipeline Phase Details

### Phase A: GOES parsing & Ground Truth Catalog
* Strips markdown formatting, repairs escaped scientific formats, and resamples GOES data to a 1-minute cadence.
* Extracts **55 ground-truth flare events** from GOES-18 SXR flux using a rolling background subtraction and threshold triggers (3σ).

### Phase B: SoLEXS Ingestion & GTI Filtering
* Ingests SoLEXS SDD2 light curves and filters timestamps against Good Time Intervals (GTIs).
* Reindexes counts onto a 1-second grid covering the three-day target range (June 17–19, 2026).

### Phase C: HEL1OS Ingestion & Housekeeping Validation
* Ingests HEL1OS CdTe and CZT light curves.
* Runs health validation using housekeeping auxiliary data (sensor operating temperatures, high voltage bias stability, and hot pixel counts).
* Rounds timestamps to align with the 1-second grid.

### Phase D: Cross-Instrument Calibration
* Pairs ground-truth GOES peak fluxes with SoLEXS peak counts and computes a log-log scaling coefficients model:
  $$\log_{10}(\text{SoLEXS Peak Count Rate}) = 1.4676 \cdot \log_{10}(\text{GOES SXR Flux}) + 10.4049$$

### Phase E: Classical Flare Detectors
* Runs state-machine threshold detectors on SoLEXS SXR counts and HEL1OS CdTe1 counts using rolling background standard deviations.
* Classifies detected flares using the calibration coefficients.

### Phase F: Master Catalog Fusion & Evaluation
* Fuses detections from SoLEXS and HEL1OS based on a $\ge 50\%$ symmetric temporal overlap.
* Assigns confidence scores (1 for single-instrument detection, 2 for multi-instrument consensus).
* Fusing instruments increases the **True Positive Rate (Recall) to 80%** (up from 63.64% with SoLEXS-only).

### Phase G: Flare Forecasting Model (ML)
* Computes trailing 30-minute rolling features: SoLEXS SXR slope, HEL1OS/SoLEXS hardness ratios (mean/max), and time elapsed since the last catalog flare.
* Trains a **LightGBM binary classifier** to predict whether a flare starts in the lookahead interval $(T, T + 15\text{ minutes}]$.
* Achieves a **TSS of 0.1229** on the 12-hour held-out test set, successfully predicting **7 out of 9 flares** with a **median lead time of 10.0 minutes**.

### Phase H: Interactive HTML Dashboard
* Compiles base64 embedded matplotlib figures (light curves, correlation matrix, ROC/PR curves) and a master event table into a single standalone HTML report: `pipeline/outputs/dashboard.html`.

---

## 3. How to Run the Pipeline

### Prerequisites
Install the required packages:
```bash
pip install numpy pandas scipy scikit-learn lightgbm seaborn matplotlib astropy
```

### End-to-End Orchestration
To execute the entire data extraction, detection, calibration, model training, and dashboard generation sequence, run:
```bash
cd pipeline
python run_pipeline.py
```

### Resume from Steps
To resume the pipeline from a specific step (e.g. step 7 - HEL1OS detector):
```bash
python run_pipeline.py --from 7
```
This is useful if you have already generated earlier clean ingestion Parquet files.
