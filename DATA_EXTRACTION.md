# Aditya-L1 and GOES Data Ingestion & Extraction Guide

This document provides detailed instructions to ingest, clean, validate, and merge solar observation data from scratch using the raw data directories and instrument manuals. This guide focuses strictly on data preparation and contains no source code.

---

## 1. Directory & Raw Data File Mapping

When extracting data from scratch, locate the files in the following directory layout:

* **GOES Satellite Data:**
  * Primary: `GOES(PRIMARY).md` (GOES-18 1-minute flux observations)
  * Secondary: `GOES(SECONDARY).md` (GOES-19 1-minute flux observations)
* **Aditya-L1 SoLEXS Data:**
  * Located under: `SoLEXS_data/AL1_SLX_L1_{date}_v1.0/SDD2/`
  * Light Curve: `AL1_SOLEXS_{date}_SDD2_L1.lc` (FITS format)
  * Good Time Intervals (GTI): `AL1_SOLEXS_{date}_SDD2_L1.gti` (FITS format)
* **Aditya-L1 HEL1OS Data:**
  * Located under: `HeL1OS_data/2026/06/{day}/HLS_{date}_*sec_lev1_V111/`
  * CdTe Sensor Light Curve: `cdte/lightcurve_cdte1.fits` (FITS format)
  * CZT Sensor Light Curve: `czt/lightcurve_czt1.fits` (FITS format)
  * Housekeeping Telemetry: `aux/hk.fits` (FITS format)

---

## 2. GOES Satellite Data Extraction

### A. Raw Format & Syntax Corrections

The raw GOES files are markdown files wrapping a large JSON text payload. The JSON array contains 1-minute cadence flux observations. Before parsing the JSON:

1. Strip the first 4 lines of markdown headers.
2. Search and replace markdown translation escape backslashes:
   - Replace `\[` and `\]` with standard brackets `[` and `]`.
   - Replace `\_` with standard underscores `_`.
   - Replace scientific notation escapes like `e\-08` or `e\-07` with `e-08` or `e-07`.
3. Correct the typo in the field name `electron_contaminaton` by renaming it to `electron_contamination`.

### B. Filtering & Column Extraction

1. Filter the dataset to retain only records where `electron_contamination` is `False`.
2. Extract the two distinct energy bands:
   - **Soft X-Ray (SXR):** Column `flux_01_08` ($0.1 - 0.8\text{ nm}$ band). This is the canonical band used to determine standard GOES flare classes (A, B, C, M, and X).
   - **Hard X-Ray (HXR) Proxy:** Column `flux_005_04` ($0.05 - 0.4\text{ nm}$ band).
3. Align the timestamps of both satellites to a uniform 1-minute grid using the GOES-18 SXR flux as the primary ground-truth benchmark.

---

## 3. SoLEXS Data Ingestion & Filtering

### A. Time Coordinate Reconstruction

SoLEXS FITS binary tables contain a `TIME` column representing elapsed seconds. To convert these to UTC:

1. Read the FITS header parameter `MJDREFI = 40587` (Modified Julian Date reference integer). This parameter specifies that the reference time coordinate corresponds directly to the Unix Epoch (January 1, 1970).
2. Compute the UTC datetime coordinate for each record:
   $$
   \text{UTC Datetime} = \text{Unix Epoch (1970-01-01)} + \text{TIME (seconds)}
   $$
3. Project these coordinates onto the Coordinated Universal Time (UTC) reference frame.

### B. Quality Filtering via Good Time Intervals (GTI)

1. Open the `.gti` FITS file and extract the binary table containing columns `START` and `STOP` (also expressed in elapsed seconds since the Unix Epoch).
2. For each record in the `.lc` (light curve) table, compare its timestamp against the GTI segments.
3. Mark a record's quality as **GOOD** if it falls within any interval:
   $$
   \text{START}_i \le \text{Timestamp} \le \text{STOP}_i
   $$

   Otherwise, flag the record as **BAD** (representing night-side passes or satellite telemetry dropouts).
4. Extract the `COUNTS` column (representing the soft X-ray count rate in counts per second).

---

## 4. HEL1OS Ingestion & Quality Validation

### A. Sensor Band Ingestion & Time Conversion

HEL1OS includes two different sensor arrays (CdTe for lower-energy ranges, CZT for higher-energy ranges).

1. Open the CdTe FITS binary table and extract the Modified Julian Date (`MJD`) column and the broadband counts column (`ctr_cdte1`).
2. Open the CZT FITS binary table and extract the Modified Julian Date (`MJD`) column and the broadband counts column (`ctr_czt1`).
3. Convert the MJD floating-point values to Coordinated Universal Time (UTC) datetimes:
   $$
   JD = MJD + 2400000.5
   $$

   Then convert the Julian Date to UTC.
4. Because HEL1OS timestamps contain sub-second offsets, round all UTC datetimes to the nearest integer second to allow clean merges on a standard 1-second grid.

### B. Housekeeping Telemetry Quality Checks

Open the auxiliary `hk.fits` file and align its parameters with your light curve timestamps. Flag any records as **BAD** if they fail any of the following manual parameters:

- **Operating Temperature Constraints:**
  - CdTe sensor arrays are cryo-cooled. Verify that the temperature of the CdTe sensor stays between $-45^\circ\text{C}$ and $-30^\circ\text{C}$.
  - CZT sensor arrays operate near ambient temperature. Verify that the CZT temperature stays between $15^\circ\text{C}$ and $25^\circ\text{C}$.
- **High Voltage Power Telemetry:** Ensure that the sensor bias voltage remains stable by rejecting records where the high voltage deviates by more than $\pm 20\%$ from its running median.
- **Hot Pixel Counts:** Monitor the active hot pixel counter (`czt1hotpixcnt`). Values exceeding 20 signify detector anomalies; flag these times as bad quality.

---

## 5. File Format Utilization Summary (.fits, .lc, .gti, .hk)

When processing solar data from Aditya-L1, several distinct file types are utilized. Here is why and how they are used:

### A. Why FITS Format is Used

* **Standardization:** FITS (Flexible Image Transport System) is the standard file format in astronomy and space physics for storing multidimensional datasets, tables, and associated metadata.
* **Self-Documentation:** FITS files contain detailed headers that store critical metadata parameters (such as the reference epoch, coordinate frames, and sampling cadence) alongside the actual binary data.

### B. In What Ways Specific Files are Utilized

1. **Light Curve Files (`.lc` FITS tables):**
   * **Why they are used:** These files store the raw photon counts detected by the sensors.
   * **How they are used:** Used to extract the primary time-series signal (`COUNTS` and `TIME` columns). The header metadata (like `MJDREFI` and `TIMEDEL`) is read to reconstruct absolute UTC timestamps from the relative elapsed time coordinates.
2. **Good Time Interval Files (`.gti` FITS tables):**
   * **Why they are used:** Telemetry data contains night-side orbits, sensor saturation, and communication outages.
   * **How they are used:** Used to create quality masks. We check if a light curve timestamp falls between the start and stop of a GTI window to mark the reading as **GOOD** or **BAD**.
3. **Housekeeping Telemetry Files (`hk.fits` FITS tables):**
   * **Why they are used:** Active space instruments suffer from thermal fluctuations, voltage drifts, and hot pixels that introduce artificial noise into counts.
   * **How they are used:** Used to validate engineering parameters (high voltage stability, cryo-cooling thresholds, and active hot pixel counts) to discard sensor noise.
4. **Markdown-Wrapped JSON Files (`.md` GOES tables):**
   * **Why they are used:** To align the raw counts with independent satellite instruments.
   * **How they are used:** Used to parse ground-truth SXR flux to train ML models and calibrate the instrument response coefficients.

---

## 6. Master Time Alignment & Combination

To build the combined solar dataset from scratch:

1. **Construct a Continuous Reference Grid:** Create a master datetime series at a 1-second cadence covering the entire target period (from `2026-06-17 00:00:00 UTC` to `2026-06-19 23:59:59 UTC`), yielding exactly 259,200 rows.
2. **Standardize and Project:**
   - Map the SoLEXS counts onto this 1-second grid, filling short gaps (up to 10 seconds) with linear interpolation.
   - Map the HEL1OS CdTe and CZT counts onto the same 1-second grid.
3. **Data Combination:** Perform an inner join on the timestamps to compile a single master dataframe containing SoLEXS SXR, HEL1OS CdTe, and HEL1OS CZT count rates.
4. **Resample to Analysis Cadence:** To compare with the GOES satellite data, resample the 1-second unified dataset to a 1-minute cadence by calculating the mean count rates for each minute interval, excluding all intervals flagged as bad quality.
