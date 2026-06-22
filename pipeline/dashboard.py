# -*- coding: utf-8 -*-
"""
dashboard.py - Phase E: Interactive HTML Dashboard.

Generates base64-encoded matplotlib plots and embeds them in a single, premium
HTML file outputs/dashboard.html, showing light curves, master catalog detections,
feature correlations, ROC/PR curves, and a table of events.
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from config import OUTPUTS

def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to a base64 png string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_str

def build_dashboard():
    print("\n" + "=" * 70)
    print("  PHASE E -- Dashboard Generation")
    print("=" * 70)

    # Load datasets
    print("  Loading results for dashboard...")
    s = pd.read_parquet(OUTPUTS / "solexs_clean.parquet")
    h = pd.read_parquet(OUTPUTS / "hel1os_clean.parquet")
    mc = pd.read_parquet(OUTPUTS / "master_catalog.parquet")
    gt = pd.read_parquet(OUTPUTS / "ground_truth_catalog.parquet")
    
    # Resample light curves to 1-minute to make plotting fast and clean
    s['timestamp'] = pd.to_datetime(s['timestamp'], utc=True)
    h['timestamp'] = pd.to_datetime(h['timestamp'], utc=True)
    mc['start'] = pd.to_datetime(mc['start'], utc=True)
    mc['stop'] = pd.to_datetime(mc['stop'], utc=True)
    gt['start'] = pd.to_datetime(gt['start'], utc=True)

    s_min = s.resample('1min', on='timestamp').mean(numeric_only=True)
    h_min = h.resample('1min', on='timestamp').mean(numeric_only=True)
    df_plot = pd.merge(s_min, h_min, on='timestamp', how='inner')

    # Load forecasting results
    forecast_results = {}
    forecast_path = OUTPUTS / "forecasting_results.json"
    if forecast_path.exists():
        with open(forecast_path) as f:
            forecast_results = json.load(f)

    # 1. Generate Plot 1: Overlaid Light Curves
    print("  Generating Plot 1: Light Curves Overlaid...")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot SoLEXS counts on left y-axis (SXR)
    color = '#1f77b4'
    ax1.set_xlabel('Time (UTC)', color='#333333', fontsize=12)
    ax1.set_ylabel('SoLEXS SXR Rate (counts/s)', color=color, fontsize=12)
    line1 = ax1.plot(df_plot.index, df_plot['counts'], color=color, alpha=0.8, label='SoLEXS SXR')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot HEL1OS CdTe on right y-axis (HXR proxy)
    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('HEL1OS CdTe1/CZT1 Rate (counts/s)', color=color, fontsize=12)
    line2 = ax2.plot(df_plot.index, df_plot['ctr_cdte1'], color=color, alpha=0.7, label='HEL1OS CdTe1')
    line3 = ax2.plot(df_plot.index, df_plot['ctr_czt1'], color='#2ca02c', alpha=0.5, label='HEL1OS CZT1')
    ax2.tick_params(axis='y', labelcolor=color)

    # Shade master catalog events
    first_shade = True
    for _, row in mc.iterrows():
        label = "Master Catalog Flare" if first_shade else None
        ax1.axvspan(row['start'], row['stop'], color='red', alpha=0.15, label=label)
        first_shade = False

    # Title & Legends
    plt.title('SoLEXS & HEL1OS Light Curves (Jun 17-19, 2026)', fontsize=14, fontweight='bold', pad=15)
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    # Add axvspan label manually
    if not mc.empty:
        lines.append(plt.Rectangle((0,0),1,1,fc="red",alpha=0.15))
        labels.append("Master Catalog Flare")
    ax1.legend(lines, labels, loc='upper left')
    
    ax1.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    lc_plot_base64 = fig_to_base64(fig)

    # 2. Generate Plot 2: Correlation Heatmap
    print("  Generating Plot 2: Correlation Heatmap...")
    # Re-extract features at 1-min grid to compute correlation matrix
    # (Simplified correlation of key features to label)
    from forecasting import get_rolling_slope
    df_plot['solexs_sxr_slope'] = get_rolling_slope(df_plot['counts'], window=30)
    df_plot['hr_cdte_mean'] = (df_plot['ctr_cdte1'] / (df_plot['counts'] + 1e-5)).rolling(30, min_periods=15).mean()
    df_plot['hr_czt_mean'] = (df_plot['ctr_czt1'] / (df_plot['counts'] + 1e-5)).rolling(30, min_periods=15).mean()
    
    # Time since last flare
    stops = sorted(mc['stop'].tolist())
    ts_last = []
    for t in df_plot.index:
        past_stops = [s for s in stops if s <= t]
        ts_last.append(1440.0 if not past_stops else (t - past_stops[-1]).total_seconds() / 60.0)
    df_plot['time_since_last_flare'] = ts_last
    
    # Target
    starts = sorted(gt['start'].tolist())
    df_plot['target_label'] = [1 if any(t < s <= t + pd.Timedelta(minutes=15) for s in starts) else 0 for t in df_plot.index]

    corr_cols = ['counts', 'ctr_cdte1', 'ctr_czt1', 'solexs_sxr_slope', 'hr_cdte_mean', 'hr_czt_mean', 'time_since_last_flare', 'target_label']
    corr_matrix = df_plot[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar=True, square=True, annot_kws={"size": 10})
    plt.title('Feature Correlation Matrix', fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    corr_plot_base64 = fig_to_base64(fig)

    # 3. Generate Plot 3: Model ROC/PR curves
    print("  Generating Plot 3: ROC/PR Curves...")
    roc_pr_plot_base64 = ""
    pred_path = OUTPUTS / "forecasting_test_predictions.parquet"
    if pred_path.exists():
        preds = pd.read_parquet(pred_path)
        y_true = preds['y_true'].values
        y_prob = preds['y_prob'].values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('Receiver Operating Characteristic')
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)
        
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax2.plot(recall, precision, color='blue', lw=2, label='PR curve')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        roc_pr_plot_base64 = fig_to_base64(fig)

    # Generate HTML report
    print("  Writing HTML dashboard...")
    
    # Event list HTML table
    table_rows = []
    for _, row in mc.sort_values('start', ascending=False).head(50).iterrows():
        bg_cls = "badge-primary" if row['confidence'] == 2 else "badge-secondary"
        conf_label = "Confirmed (Both)" if row['confidence'] == 2 else "Single-Instrument"
        table_rows.append(f"""
        <tr>
            <td><strong>{row['event_id']}</strong></td>
            <td>{row['start'].strftime('%Y-%m-%d %H:%M:%S')}</td>
            <td>{row['stop'].strftime('%Y-%m-%d %H:%M:%S')}</td>
            <td>{row['goes_equivalent_class']}</td>
            <td><span class="badge {bg_cls}">{conf_label}</span></td>
            <td>{row['peak_count_rate']:.1f}</td>
            <td>{row['instruments']}</td>
        </tr>
        """)
    table_content = "\n".join(table_rows)

    tss_val = f"{forecast_results.get('test_tss', 0.0):.4f}" if forecast_results else "N/A"
    lead_time_val = f"{forecast_results.get('median_lead_time_min', 0.0):.1f} min" if forecast_results else "N/A"
    th_val = f"{forecast_results.get('best_threshold', 0.0):.3f}" if forecast_results else "N/A"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aditya-L1 Solar Flare Detection & Forecasting Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --border-color: #334155;
        }}
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.5;
            padding: 2rem;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        header {{
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
        }}
        header h1 {{
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(to right, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        header p {{
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 1.1rem;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }}
        .card-title {{
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.5rem;
        }}
        .card-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-main);
        }}
        .card-value.highlight {{
            color: var(--accent);
        }}
        .card-value.success {{
            color: var(--success);
        }}
        .card-value.warning {{
            color: var(--warning);
        }}
        .chart-box {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin-bottom: 2rem;
        }}
        .chart-box h2 {{
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            border-left: 4px solid var(--accent);
            padding-left: 0.75rem;
        }}
        .chart-img {{
            width: 100%;
            border-radius: 0.5rem;
            background-color: #ffffff;
            padding: 0.5rem;
        }}
        .charts-row {{
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        @media (max-width: 1024px) {{
            .charts-row {{
                grid-template-columns: 1fr;
            }}
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
            font-size: 0.9rem;
        }}
        th, td {{
            text-align: left;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border-color);
        }}
        th {{
            background-color: rgba(255, 255, 255, 0.05);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}
        tr:hover td {{
            background-color: rgba(255, 255, 255, 0.02);
        }}
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 0.25rem;
        }}
        .badge-primary {{
            background-color: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        .badge-secondary {{
            background-color: rgba(148, 163, 184, 0.2);
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.3);
        }}
        .table-container {{
            max-height: 500px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Aditya-L1 Solar Flare Space Weather Panel</h1>
            <p>India Runs Data and AI Challenge Submission — Multi-Instrument Fusion & Forecasting</p>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-title">Master Catalog Events</div>
                <div class="card-value highlight">{len(mc)}</div>
            </div>
            <div class="card">
                <div class="card-title">Forecast TSS (Test Set)</div>
                <div class="card-value success">{tss_val}</div>
            </div>
            <div class="card">
                <div class="card-title">Median Lead Time</div>
                <div class="card-value warning">{lead_time_val}</div>
            </div>
            <div class="card">
                <div class="card-title">Optimized Probability Threshold</div>
                <div class="card-value">{th_val}</div>
            </div>
        </div>

        <div class="chart-box">
            <h2>SoLEXS & HEL1OS Integrated Light Curves (3-Day Sequence)</h2>
            <img class="chart-img" src="data:image/png;base64,{lc_plot_base64}" alt="Overlaid Light Curves">
        </div>

        <div class="charts-row">
            <div class="chart-box">
                <h2>Predictive Model Performance (ROC & Precision-Recall Curves)</h2>
                {"<img class='chart-img' src='data:image/png;base64," + roc_pr_plot_base64 + "' alt='ROC and PR Curves'>" if roc_pr_plot_base64 else "<p>Model predictions not found.</p>"}
            </div>
            <div class="chart-box">
                <h2>Feature Correlations</h2>
                <img class="chart-img" src="data:image/png;base64,{corr_plot_base64}" alt="Feature Correlation Matrix">
            </div>
        </div>

        <div class="chart-box">
            <h2>Detected Flare Catalog (Master Catalog - Head 50 Events)</h2>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Event ID</th>
                            <th>Start Time (UTC)</th>
                            <th>Stop Time (UTC)</th>
                            <th>Equivalent Class</th>
                            <th>Confidence</th>
                            <th>Peak CR (cts/s)</th>
                            <th>Instruments</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_content}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
</body>
</html>
"""
    
    out_html = OUTPUTS / "dashboard.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  Saved Dashboard HTML -> {out_html}")

if __name__ == "__main__":
    build_dashboard()
