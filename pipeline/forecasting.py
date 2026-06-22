# -*- coding: utf-8 -*-
"""
forecasting.py - Phase D: Solar Flare Forecasting Model.

Extracts rolling features (slope of SoLEXS SXR, hardness ratio, time-since-last-flare)
over a 30-minute trailing window, labels windows based on lookahead of 15 minutes,
trains a LightGBM classifier, and evaluates TSS/lead time on the final 12 hours of data.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import confusion_matrix

from config import OUTPUTS, DATE_START, DATE_STOP

def get_rolling_slope(series: pd.Series, window: int = 30) -> pd.Series:
    """Compute rolling linear slope over a window of size N."""
    x = np.arange(window)
    x_mean = x.mean()
    x_dev = x - x_mean
    denom = np.sum(x_dev**2)
    
    def _slope(y):
        if len(y) < window:
            return np.nan
        y_valid = y.copy()
        if np.isnan(y_valid).any():
            mask = np.isnan(y_valid)
            if mask.all():
                return 0.0
            mean_val = np.nanmean(y_valid)
            y_valid[mask] = mean_val
        return np.sum(x_dev * (y_valid - np.mean(y_valid))) / denom
        
    return series.rolling(window, min_periods=15).apply(_slope, raw=True)

def build_features_and_train() -> dict:
    print("\n" + "=" * 70)
    print("  PHASE D -- Solar Flare Forecasting")
    print("=" * 70)

    # 1. Load data and resample to 1-minute grid
    print("  Loading clean data...")
    s = pd.read_parquet(OUTPUTS / "solexs_clean.parquet")
    h = pd.read_parquet(OUTPUTS / "hel1os_clean.parquet")
    gt = pd.read_parquet(OUTPUTS / "ground_truth_catalog.parquet")
    mc = pd.read_parquet(OUTPUTS / "master_catalog.parquet")

    # Resample SoLEXS & HEL1OS
    s['timestamp'] = pd.to_datetime(s['timestamp'], utc=True)
    h['timestamp'] = pd.to_datetime(h['timestamp'], utc=True)
    gt['start'] = pd.to_datetime(gt['start'], utc=True)
    mc['start'] = pd.to_datetime(mc['start'], utc=True)
    mc['stop'] = pd.to_datetime(mc['stop'], utc=True)

    s_min = s.resample('1min', on='timestamp').mean(numeric_only=True)
    h_min = h.resample('1min', on='timestamp').mean(numeric_only=True)

    # Merge aligned series
    df = pd.merge(s_min, h_min, on='timestamp', how='inner')
    timestamps = df.index

    print(f"  Resampled to 1-minute grid: {len(df)} samples.")

    # 2. Extract features
    print("  Extracting features (30-minute trailing window)...")
    # Rolling slope of SoLEXS SXR
    df['solexs_sxr_slope'] = get_rolling_slope(df['counts'], window=30)

    # Hardness ratio HEL1OS/SoLEXS
    # instantaneous ratio
    df['hardness_ratio_cdte'] = df['ctr_cdte1'] / (df['counts'] + 1e-5)
    df['hardness_ratio_czt'] = df['ctr_czt1'] / (df['counts'] + 1e-5)

    # Rolling statistics of hardness ratio over 30-minute trailing window
    df['hr_cdte_mean'] = df['hardness_ratio_cdte'].rolling(30, min_periods=15).mean()
    df['hr_cdte_max'] = df['hardness_ratio_cdte'].rolling(30, min_periods=15).max()
    df['hr_czt_mean'] = df['hardness_ratio_czt'].rolling(30, min_periods=15).mean()
    df['hr_czt_max'] = df['hardness_ratio_czt'].rolling(30, min_periods=15).max()

    # Time since last flare from master catalog (in minutes)
    stops = sorted(mc['stop'].tolist())
    ts_last = []
    for t in timestamps:
        past_stops = [s for s in stops if s <= t]
        if not past_stops:
            ts_last.append(1440.0) # 24 hours default
        else:
            ts_last.append((t - past_stops[-1]).total_seconds() / 60.0)
    df['time_since_last_flare'] = ts_last

    # 3. Compute target label (positive if GT flare starts within next 15 minutes)
    print("  Creating lookahead labels (15 minutes)...")
    starts = sorted(gt['start'].tolist())
    labels = []
    for t in timestamps:
        t_next = t + pd.Timedelta(minutes=15)
        has_flare = any(t < s <= t_next for s in starts)
        labels.append(1 if has_flare else 0)
    df['label'] = labels

    # Drop initial rows with NaNs (due to rolling windows)
    df = df.dropna(subset=['solexs_sxr_slope', 'hr_cdte_mean', 'time_since_last_flare'])

    # 4. Train-Test Split (Held-out final 12 hours)
    # The dataset spans 3 days (June 17, 18, 19). Total 72 hours.
    # Split point: final 12 hours of the time range.
    max_time = df.index.max()
    split_time = max_time - pd.Timedelta(hours=12)

    train_df = df[df.index < split_time]
    test_df = df[df.index >= split_time]

    feature_cols = [
        'counts', 'ctr_cdte1', 'ctr_czt1',
        'solexs_sxr_slope', 'hr_cdte_mean', 'hr_cdte_max',
        'hr_czt_mean', 'hr_czt_max', 'time_since_last_flare'
    ]

    X_train = train_df[feature_cols]
    y_train = train_df['label']
    X_test = test_df[feature_cols]
    y_test = test_df['label']

    print(f"  Training samples: {len(X_train)}  (Positives: {y_train.sum()})")
    print(f"  Testing samples : {len(X_test)}   (Positives: {y_test.sum()})")

    # 5. Train LightGBM Classifier
    print("  Training LightGBM model...")
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=100)

    # 6. Evaluation
    y_prob = model.predict(X_test)
    # Use 0.5 threshold as default, or optimize threshold for TSS
    # Let's find the threshold that maximizes TSS on training set or just use a standard threshold, or find the best threshold on training set
    y_train_prob = model.predict(X_train)
    best_th = 0.5
    best_tss = -1.0
    for th in np.linspace(0.1, 0.9, 81):
        tn, fp, fn, tp = confusion_matrix(y_train, y_train_prob >= th).ravel()
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        tss = tpr - fpr
        if tss > best_tss:
            best_tss = tss
            best_th = th

    print(f"  Optimized probability threshold: {best_th:.3f} (Train TSS: {best_tss:.3f})")

    y_pred = (y_prob >= best_th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tpr = tp / (tp + fn + 1e-9)
    fpr = fp / (fp + tn + 1e-9)
    test_tss = tpr - fpr

    print(f"  Test Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Test TPR (Recall): {tpr:.2%}")
    print(f"  Test FPR: {fpr:.2%}")
    print(f"  Test TSS: {test_tss:.4f}")

    # Compute lead time on test set
    # Find flares that start in the test set
    test_starts = [s for s in starts if split_time <= s <= max_time]
    lead_times = []
    for s_t in test_starts:
        # Check 15-minute window ending at s_t: [s_t - 15min, s_t)
        window_start = s_t - pd.Timedelta(minutes=15)
        window_end = s_t - pd.Timedelta(minutes=1) # up to the minute before
        
        # Get predictions in this window
        window_pred = test_df.loc[window_start:window_end]
        if len(window_pred) > 0:
            window_probs = model.predict(window_pred[feature_cols])
            warn_indices = np.where(window_probs >= best_th)[0]
            if len(warn_indices) > 0:
                earliest_warn_time = window_pred.index[warn_indices[0]]
                lead_min = (s_t - earliest_warn_time).total_seconds() / 60.0
                lead_times.append(lead_min)

    median_lead = np.median(lead_times) if lead_times else 0.0
    print(f"  Flares in test set: {len(test_starts)}")
    print(f"  Flares successfully predicted: {len(lead_times)}")
    print(f"  Median lead time: {median_lead:.1f} minutes")

    # Save results
    results = {
        'test_tss': test_tss,
        'median_lead_time_min': median_lead,
        'best_threshold': best_th,
        'features': feature_cols,
        'feature_importances': model.feature_importance().tolist()
    }
    
    with open(OUTPUTS / "forecasting_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Saved -> outputs/forecasting_results.json")

    # Save test predictions for plotting
    test_plot_df = pd.DataFrame({
        'timestamp': test_df.index,
        'y_true': y_test,
        'y_prob': y_prob,
        'y_pred': y_pred
    })
    test_plot_df.to_parquet(OUTPUTS / "forecasting_test_predictions.parquet", index=False)
    print(f"  Saved -> outputs/forecasting_test_predictions.parquet")

    return results

if __name__ == "__main__":
    build_features_and_train()
