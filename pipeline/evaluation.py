# -*- coding: utf-8 -*-
"""
evaluation.py - Step 6: Evaluation of SoLEXS Detections vs. Ground Truth.

Matching rule: a detection is a True Positive (TP) if its [start, stop]
overlaps with a ground-truth [start, stop] by >= 50% of the ground-truth
event duration. Best-overlap wins (one GT event -> one detection).

Metrics computed
----------------
- Per-class (B, C, M, X) and overall:
  TP, FP, FN
  TPR (recall), Precision, F1
  FPR (False Positive Rate) -- requires TN estimate (uses non-flare time windows)
  TSS = TPR - FPR
  HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))
- ROC-AUC using confidence_snr as score
- Lead time per TP: ground_truth_start - detection_start (minutes)
  (positive = detection found BEFORE ground truth start)

Outputs
-------
- roc_curve.png
- precision_recall_curve.png
- lead_time_distribution.png
- evaluation_results.parquet (full match table)

Units
-----
lead_time  : minutes
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import OUTPUTS, OVERLAP_FRACTION


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overlap_fraction(det_start, det_stop, gt_start, gt_stop) -> float:
    """
    Compute fraction of ground-truth interval overlapped by detection.

    Parameters
    ----------
    det_start, det_stop : timestamps (detection interval)
    gt_start,  gt_stop  : timestamps (ground-truth interval)

    Returns
    -------
    float in [0, 1]
    """
    gt_dur = (gt_stop - gt_start).total_seconds()
    if gt_dur <= 0:
        return 0.0
    overlap_start = max(det_start, gt_start)
    overlap_stop  = min(det_stop,  gt_stop)
    overlap_s = max(0.0, (overlap_stop - overlap_start).total_seconds())
    return overlap_s / gt_dur


def _match_events(detections: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    """
    Match detection catalog to ground-truth catalog.

    Rules
    -----
    - TP: detection overlaps GT by >= OVERLAP_FRACTION of GT duration.
    - Best-overlap match wins; each GT event matched to at most one detection.

    Parameters
    ----------
    detections   : pd.DataFrame with columns start, stop, (others)
    ground_truth : pd.DataFrame with columns event_id, start, stop, goes_class

    Returns
    -------
    pd.DataFrame with one row per matched pair / unmatched event:
        gt_event_id | det_event_id | overlap_frac | tp | goes_class | lead_time_min
    """
    match_records = []
    used_det = set()

    for _, gt in ground_truth.iterrows():
        gt_start = pd.Timestamp(gt["start"])
        gt_stop  = pd.Timestamp(gt["stop"])

        best_overlap = 0.0
        best_det     = None

        for _, det in detections.iterrows():
            if det["event_id"] in used_det:
                continue
            det_start = pd.Timestamp(det["start"])
            det_stop  = pd.Timestamp(det["stop"])
            ov = _overlap_fraction(det_start, det_stop, gt_start, gt_stop)
            if ov > best_overlap:
                best_overlap = ov
                best_det     = det

        if best_overlap >= OVERLAP_FRACTION and best_det is not None:
            used_det.add(best_det["event_id"])
            lead_min = (gt_start - pd.Timestamp(best_det["start"])).total_seconds() / 60.0
            match_records.append({
                "gt_event_id":   gt["event_id"],
                "det_event_id":  best_det["event_id"],
                "gt_class":      gt["goes_class"],
                "overlap_frac":  best_overlap,
                "tp":            True,
                "fp":            False,
                "fn":            False,
                "lead_time_min": lead_min,
                "confidence_snr": best_det["confidence_snr"],
            })
        else:
            match_records.append({
                "gt_event_id":   gt["event_id"],
                "det_event_id":  None,
                "gt_class":      gt["goes_class"],
                "overlap_frac":  best_overlap,
                "tp":            False,
                "fp":            False,
                "fn":            True,
                "lead_time_min": np.nan,
                "confidence_snr": np.nan,
            })

    # False positives: detections not matched to any GT
    matched_det_ids = {r["det_event_id"] for r in match_records if r["det_event_id"] is not None}
    for _, det in detections.iterrows():
        if det["event_id"] not in matched_det_ids:
            match_records.append({
                "gt_event_id":   None,
                "det_event_id":  det["event_id"],
                "gt_class":      None,
                "overlap_frac":  0.0,
                "tp":            False,
                "fp":            True,
                "fn":            False,
                "lead_time_min": np.nan,
                "confidence_snr": det["confidence_snr"],
            })

    return pd.DataFrame(match_records)


def _compute_metrics(match_df: pd.DataFrame, class_filter=None) -> dict:
    """
    Compute TP, FP, FN, TPR, FPR, Precision, F1, TSS, HSS for a subset.

    Parameters
    ----------
    match_df      : DataFrame from _match_events
    class_filter  : list of GOES classes to include (None = all)

    Returns
    -------
    dict of metric name -> value
    """
    if class_filter is not None:
        sub = match_df[match_df["gt_class"].isin(class_filter) | match_df["tp"] == False]
        tp_rows = match_df[(match_df["tp"]) & (match_df["gt_class"].isin(class_filter))]
        fn_rows = match_df[(match_df["fn"]) & (match_df["gt_class"].isin(class_filter))]
        fp_rows = match_df[match_df["fp"]]  # FP counts against all classes
    else:
        tp_rows = match_df[match_df["tp"]]
        fn_rows = match_df[match_df["fn"]]
        fp_rows = match_df[match_df["fp"]]

    TP = len(tp_rows)
    FP = len(fp_rows)
    FN = len(fn_rows)

    # TN: estimated as number of non-flare 30-min windows minus FP
    # Using a simple estimate: assume ~144 non-flare windows per day (3 days)
    TN = max(0, 144 * 3 - FP)

    TPR  = TP / (TP + FN + 1e-9)
    FPR  = FP / (FP + TN + 1e-9)
    prec = TP / (TP + FP + 1e-9)
    f1   = 2 * prec * TPR / (prec + TPR + 1e-9)
    tss  = TPR - FPR

    denom_hss = (TP+FN)*(FN+TN) + (TP+FP)*(FP+TN)
    if denom_hss > 0:
        hss = 2 * (TP*TN - FP*FN) / denom_hss
    else:
        hss = np.nan

    return {
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "TPR":  round(TPR,  4),
        "FPR":  round(FPR,  4),
        "Precision": round(prec, 4),
        "F1":   round(f1,   4),
        "TSS":  round(tss,  4),
        "HSS":  round(hss,  4) if not np.isnan(hss) else np.nan,
    }


# ---------------------------------------------------------------------------
# Step 6 -- Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    det_path: Path      = OUTPUTS / "solexs_detections.parquet",
    gt_path:  Path      = OUTPUTS / "ground_truth_catalog.parquet",
    out_dir:  Path      = OUTPUTS,
) -> pd.DataFrame:
    """
    Evaluate SoLEXS detections against GOES-18 ground-truth catalog.

    Parameters
    ----------
    det_path : Path -- solexs_detections.parquet
    gt_path  : Path -- ground_truth_catalog.parquet
    out_dir  : Path -- directory for output plots and parquet

    Returns
    -------
    pd.DataFrame -- full match table (evaluation_results.parquet)
    """
    print("\n" + "=" * 70)
    print("  STEP 6 -- Evaluation")
    print("=" * 70)

    detections   = pd.read_parquet(det_path)
    if "confidence_snr" not in detections.columns:
        if "sx_snr" in detections.columns and "hx_snr" in detections.columns:
            detections["confidence_snr"] = detections[["sx_snr", "hx_snr"]].max(axis=1).fillna(1.0)
        elif "confidence" in detections.columns:
            detections["confidence_snr"] = detections["confidence"].astype(float)
        else:
            detections["confidence_snr"] = 1.0

    ground_truth = pd.read_parquet(gt_path)

    print(f"\n  Ground-truth events : {len(ground_truth)}")
    print(f"  SoLEXS detections   : {len(detections)}")

    if len(detections) == 0:
        print("\n  WARNING: No detections to evaluate.")
        return pd.DataFrame()

    # Match
    match_df = _match_events(detections, ground_truth)
    match_df.to_parquet(out_dir / "evaluation_results.parquet", index=False)

    # --- Overall metrics ---
    overall = _compute_metrics(match_df)
    print(f"\n  Overall metrics:")
    for k, v in overall.items():
        print(f"    {k:12s}: {v}")

    # --- Stratified by class ---
    class_groups = {
        "B":   ["B"],
        "C":   ["C"],
        "M":   ["M"],
        "X":   ["X"],
        "M+X": ["M", "X"],
        "All": None,
    }

    rows = []
    for label, classes in class_groups.items():
        m = _compute_metrics(match_df, class_filter=classes)
        m["Class"] = label
        rows.append(m)

    summary_df = pd.DataFrame(rows).set_index("Class")
    print(f"\n  Stratified results:\n")
    print(summary_df.to_string())

    # --- Lead time ---
    tp_df = match_df[match_df["tp"]]
    if len(tp_df) > 0:
        lt = tp_df["lead_time_min"].dropna()
        print(f"\n  Lead time (GT_start - Det_start, positive = early detection):")
        print(f"    Median : {lt.median():.2f} min")
        print(f"    IQR    : [{lt.quantile(0.25):.2f}, {lt.quantile(0.75):.2f}] min")
        print(f"    Range  : [{lt.min():.2f}, {lt.max():.2f}] min")
    else:
        print("\n  No TPs found -- cannot compute lead time.")
        lt = pd.Series(dtype=float)

    # --- ROC / PR curves ---
    snr_all = match_df["confidence_snr"].fillna(0.0).values
    true_all = match_df["tp"].astype(int).values

    if true_all.sum() > 0 and true_all.sum() < len(true_all):
        try:
            auc = roc_auc_score(true_all, snr_all)
            print(f"\n  ROC-AUC: {auc:.4f}")

            # ROC curve
            fpr_curve, tpr_curve, _ = roc_curve(true_all, snr_all)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr_curve, tpr_curve, "b-", lw=2, label=f"ROC (AUC={auc:.3f})")
            ax.plot([0,1],[0,1],"k--", lw=1, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("SoLEXS Detector ROC Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "roc_curve.png", dpi=150)
            plt.close()
            print(f"  Saved -> roc_curve.png")

            # PR curve
            prec_c, rec_c, _ = precision_recall_curve(true_all, snr_all)
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(rec_c, prec_c, "g-", lw=2)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("SoLEXS Detector Precision-Recall Curve")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_dir / "precision_recall_curve.png", dpi=150)
            plt.close()
            print(f"  Saved -> precision_recall_curve.png")

        except Exception as e:
            print(f"  [WARN] Could not compute ROC: {e}")
    else:
        print("\n  [WARN] Skipping ROC -- all samples same class or no TPs.")

    # Lead time distribution
    if len(lt) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(lt, bins=min(20, len(lt)), color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(lt.median(), color="red", lw=2, linestyle="--", label=f"Median={lt.median():.1f} min")
        ax.set_xlabel("Lead Time (minutes)")
        ax.set_ylabel("Count")
        ax.set_title("SoLEXS Detection Lead Time Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / "lead_time_distribution.png", dpi=150)
        plt.close()
        print(f"  Saved -> lead_time_distribution.png")

    # Final summary table
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY TABLE")
    print("=" * 70)
    print(summary_df.to_string())
    print()

    return match_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    match_df = evaluate()
    print("\n  All done -- evaluation.py complete.")
