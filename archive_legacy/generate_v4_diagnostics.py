import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

OUT = Path("outputs/amazon_cds_v3")

def generate_diagnostics():
    md_lines = []
    md_lines.append("# V4 PIPELINE DIAGNOSTICS\n")
    
    # 1. Sanity / Leakage Table
    md_lines.append("## 1. Sanity Check & Leakage Assertion\n")
    df = pd.read_csv(OUT / "phase2" / "customer_augmented_v3.csv")
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    
    # Check for review leakage
    revs = pd.read_csv(OUT / "phase0" / "reviews.csv")
    revs["purchase_date"] = pd.to_datetime(revs["purchase_date"])
    snap_date = pd.Timestamp("2018-10-02")
    leakage_count = (revs["purchase_date"] >= snap_date).sum()
    
    try:
        with open(OUT / "phase3" / "split_info.json") as f:
            split_info = json.load(f)
        t_max = split_info["train_max_date"]
        v_min = split_info["val_min_date"]
        v_max = split_info["val_max_date"]
        t_min = split_info["test_min_date"]
        n_train = split_info["train_snapshots"]
        n_val = split_info["val_snapshots"]
        n_test = split_info["test_snapshots"]
    except Exception:
        t_max = v_min = v_max = t_min = "N/A"
        n_train = n_val = n_test = 0

    md_lines.append("| Metric | Value |")
    md_lines.append("|--------|-------|")
    md_lines.append(f"| Total customers | {len(df):,} |")
    md_lines.append(f"| Total reviews | {len(revs):,} |")
    md_lines.append(f"| Train snapshots | {n_train:,} |")
    md_lines.append(f"| Val snapshots | {n_val:,} |")
    md_lines.append(f"| Test snapshots | {n_test:,} |")
    md_lines.append(f"| train_max_date | {t_max} |")
    md_lines.append(f"| val_min_date | {v_min} |")
    md_lines.append(f"| val_max_date | {v_max} |")
    md_lines.append(f"| test_min_date | {t_min} |")
    md_lines.append(f"| train_max_date < val_min_date | {t_max < v_min if t_max != 'N/A' else 'N/A'} |")
    md_lines.append(f"| val_max_date < test_min_date | {v_max < t_min if v_max != 'N/A' else 'N/A'} |")
    md_lines.append(f"| review_time < cutoff violation | {leakage_count} |")
    md_lines.append(f"| Censoring rate | {(1 - df['E'].mean())*100:.1f}% |\n")

    # 2. Feature Diagnostics
    md_lines.append("## 2. Feature Diagnostics (Churn vs Non-Churn)\n")
    churned = df[df["E"] == 1]
    active = df[df["E"] == 0]
    
    md_lines.append("| Feature | Churned mean | Non-churned mean | Difference |")
    md_lines.append("|---------|-------------:|-----------------:|-----------:|")
    feats = ["latest_fail_severity", "sentiment_min", "sentiment_volatility", "mixed_flag", "loyal_x_failure", "negative_review_rate", "sentiment_missing_flag_latest", "loyal_x_missing_sentiment", "loyal_x_fail_flag", "negative_sentiment_shock", "loyal_x_sentiment_shock", "failure_after_positive_history", "loyal_x_failure_after_positive_history"]
    for f in feats:
        if f in df.columns:
            c_mean = churned[f].mean()
            a_mean = active[f].mean()
            diff = c_mean - a_mean
            md_lines.append(f"| `{f}` | {c_mean:.4f} | {a_mean:.4f} | {diff:+.4f} |")
            
    md_lines.append("\nPrevalence:")
    for f in ["loyal_customer_flag", "mixed_flag"]:
        if f in df.columns:
            md_lines.append(f"- % {f} = 1: {(df[f] > 0).mean()*100:.1f}%")
    for f in ["latest_fail_severity", "loyal_x_failure", "negative_sentiment_shock", "loyal_x_sentiment_shock"]:
        if f in df.columns:
            md_lines.append(f"- % {f} > 0: {(df[f] > 0).mean()*100:.1f}%")
    for f in ["failure_after_positive_history", "loyal_x_failure_after_positive_history"]:
        if f in df.columns:
            md_lines.append(f"- % {f} = 1: {(df[f] > 0).mean()*100:.1f}%")
            
    md_lines.append("\n*Note: `loyal_x_mixed` was removed from diagnostics and the feature set due to near-zero variance (no valid loyal+mixed latest-review cases).*")
            
    # 3. Ablation A-F Table
    md_lines.append("\n## 3. Ablation A-F/G\n")
    try:
        with open(OUT / "phase3" / "phase3_results_v4.json") as f:
            p3 = json.load(f)["results"]
        
        md_lines.append("| Model | Feature set | C-index OOS | Brier_Raw@1095d | Brier_Cal@1095d | MACG_Raw@1095d | MACG_Cal@1095d | IBS |")
        md_lines.append("|-------|-------------|------------:|----------------:|----------------:|---------------:|---------------:|----:|")
        for model, r in p3.items():
            features = len(r['features'])
            c_index = r['c_oos']
            ibs = r.get('ibs', 0.0)
            md_lines.append(f"| {model} | {features} feats | {c_index:.4f} | {r.get('Brier_Raw@1095d', '')} | {r.get('Brier_Cal@1095d', '')} | {r.get('MACG_Raw@1095d', '')} | {r.get('MACG_Cal@1095d', '')} | {ibs:.4f} |")
    except Exception as e:
        md_lines.append(f"Could not load ablation table: {e}")

    # 3.5 CoxPH
    md_lines.append("\n## 3.5 CoxPH Hazard Ratios for Interpretation\n")
    try:
        cox_df = pd.read_csv(OUT / "phase3" / "coxph_summary.csv", index_col=0)
        md_lines.append("| Feature | Hazard Ratio (exp(coef)) | p-value | 95% CI Lower | 95% CI Upper |")
        md_lines.append("|---------|-------------------------:|--------:|-------------:|-------------:|")
        for idx, row in cox_df.iterrows():
            if "loyal" in idx:
                md_lines.append(f"| `{idx}` | {row['exp(coef)']:.4f} | {row['p']:.4f} | {row['exp(coef) lower 95%']:.4f} | {row['exp(coef) upper 95%']:.4f} |")
    except Exception as e:
        md_lines.append(f"Could not load CoxPH summary: {e}")

    # 4. Calibration Table
    md_lines.append("\n## 5. Calibration Tables\n")
    try:
        dec_df = pd.read_csv(OUT / "phase5" / "phase5_decisions.csv")
        
        # Raw Calibration
        dec_df["risk_decile_raw"] = pd.qcut(dec_df["churn_prob_raw"], 10, labels=False, duplicates='drop') + 1
        md_lines.append("### Raw Calibration (Uncalibrated)")
        md_lines.append("| Risk Decile | Avg Pred Churn Prob | Actual Churn Rate | Gap |")
        md_lines.append("|-------------|--------------------:|------------------:|----:|")
        for d in sorted(dec_df["risk_decile_raw"].unique()):
            sub = dec_df[dec_df["risk_decile_raw"] == d]
            pred = sub["churn_prob_raw"].mean()
            act = sub["E"].mean()
            md_lines.append(f"| {d} | {pred:.4f} | {act:.4f} | {pred - act:+.4f} |")
            
        # Calibrated Calibration
        dec_df["risk_decile_calib"] = pd.qcut(dec_df["churn_prob_calib"], 10, labels=False, duplicates='drop') + 1
        md_lines.append("\n### Isotonic Calibration")
        md_lines.append("| Risk Decile | Avg Pred Churn Prob | Actual Churn Rate | Gap |")
        md_lines.append("|-------------|--------------------:|------------------:|----:|")
        for d in sorted(dec_df["risk_decile_calib"].unique()):
            sub = dec_df[dec_df["risk_decile_calib"] == d]
            pred = sub["churn_prob_calib"].mean()
            act = sub["E"].mean()
            md_lines.append(f"| {d} | {pred:.4f} | {act:.4f} | {pred - act:+.4f} |")
            
    except Exception as e:
        md_lines.append(f"Could not load decisions for calibration: {e}")
        
    # 6. Proxy Policy / Uplift Metrics
    md_lines.append("\n## 6. Proxy Policy Metrics (T-Learner)\n")
    try:
        up_df = pd.read_csv(OUT / "phase4" / "uplift_df_T2_+sentiment.csv")
        md_lines.append("| Segment | % customers | Avg churn risk | Avg Monetary | tau_mean | tau_p5 | tau_p25 | tau_median | tau_p75 | tau_p95 |")
        md_lines.append("|---------|------------:|---------------:|-------------:|---------:|-------:|--------:|-----------:|--------:|--------:|")
        total = len(up_df)
        for seg in ["Persuadables", "Sure Things", "Sleeping Dogs", "Lost Causes"]:
            sub = up_df[up_df["uplift_segment"] == seg]
            if len(sub) == 0: continue
            pct = len(sub) / total * 100
            joined = sub.merge(dec_df[["CustomerID", "churn_prob_calib"]], on="CustomerID", how="left")
            risk = joined["churn_prob_calib"].mean()
            tau = sub["tau_hat"]
            mon = sub["Monetary"].median()
            md_lines.append(f"| {seg} | {pct:.1f}% | {risk:.3f} | ${mon:.0f} | {tau.mean():+.3f} | {tau.quantile(0.05):+.3f} | {tau.quantile(0.25):+.3f} | {tau.median():+.3f} | {tau.quantile(0.75):+.3f} | {tau.quantile(0.95):+.3f} |")
    except Exception as e:
        md_lines.append(f"Could not load uplift df: {e}")

    # 7. Sensitivity Grid
    md_lines.append("\n## 7. Sensitivity Grid Summary\n")
    try:
        with open(OUT / "phase5" / "sensitivity_metrics.json") as f:
            mets = json.load(f)
        md_lines.append(f"- Group-level win rate (Raw vs V1): {mets.get('group_win_rate_pct_raw', 0):.1f}%")
        md_lines.append(f"- Group-level win rate (Guardrail vs V1): {mets.get('group_win_rate_pct_guardrail', 0):.1f}%")
        md_lines.append(f"- Run-level win rate (Raw vs V1): {mets.get('run_win_rate_pct_raw', 0):.1f}%")
        md_lines.append(f"- Run-level win rate (Guardrail vs V1): {mets.get('run_win_rate_pct_guardrail', 0):.1f}%")
        md_lines.append(f"- Median profit lift (Raw): ${mets['median_lift_raw']:,.2f}")
        md_lines.append(f"- Median profit lift (Guardrail): ${mets['median_lift_guardrail']:,.2f}")
        md_lines.append(f"- Worst-case profit lift (Raw): ${mets['worst_lift_raw']:,.2f}")
        md_lines.append(f"- Worst-case profit lift (Guardrail): ${mets['worst_lift_guardrail']:,.2f}\n")
        
        grid = pd.read_csv(OUT / "phase5" / "sensitivity_grid.csv")
        md_lines.append("### By Voucher Cost")
        md_lines.append("| Voucher cost | Win% (Raw) | Win% (Guardrail) | Median (Raw) | Median (Guardrail) | Worst (Raw) | Worst (Guardrail) |")
        md_lines.append("|-------------:|-----------:|-----------------:|-------------:|-------------------:|------------:|------------------:|")
        for c in grid["voucher_cost"].unique():
            sub = grid[grid["voucher_cost"] == c]
            win_r = sub["win_rate_pct_raw"].mean()
            win_g = sub["win_rate_pct_guardrail"].mean()
            med_r = sub["median_lift_raw"].median()
            med_g = sub["median_lift_guardrail"].median()
            wst_r = sub["worst_lift_raw"].min()
            wst_g = sub["worst_lift_guardrail"].min()
            md_lines.append(f"| {c} | {win_r:.1f}% | {win_g:.1f}% | ${med_r:,.0f} | ${med_g:,.0f} | ${wst_r:,.0f} | ${wst_g:,.0f} |")
            
        md_lines.append("\n### By Response Rate Base")
        md_lines.append("| Response rate | Win% (Raw) | Win% (Guardrail) | Median (Raw) | Median (Guardrail) | Worst (Raw) | Worst (Guardrail) |")
        md_lines.append("|--------------:|-----------:|-----------------:|-------------:|-------------------:|------------:|------------------:|")
        for r in grid["response_rate_base"].unique():
            sub = grid[grid["response_rate_base"] == r]
            win_r = sub["win_rate_pct_raw"].mean()
            win_g = sub["win_rate_pct_guardrail"].mean()
            med_r = sub["median_lift_raw"].median()
            med_g = sub["median_lift_guardrail"].median()
            wst_r = sub["worst_lift_raw"].min()
            wst_g = sub["worst_lift_guardrail"].min()
            md_lines.append(f"| {r} | {win_r:.1f}% | {win_g:.1f}% | ${med_r:,.0f} | ${med_g:,.0f} | ${wst_r:,.0f} | ${wst_g:,.0f} |")
            
        md_lines.append("\n### By Noise Sigma")
        md_lines.append("| Noise Sigma | Win% (Raw) | Win% (Guardrail) | Median (Raw) | Median (Guardrail) | Worst (Raw) | Worst (Guardrail) |")
        md_lines.append("|------------:|-----------:|-----------------:|-------------:|-------------------:|------------:|------------------:|")
        for s in grid["noise_sigma"].unique():
            sub = grid[grid["noise_sigma"] == s]
            win_r = sub["win_rate_pct_raw"].mean()
            win_g = sub["win_rate_pct_guardrail"].mean()
            med_r = sub["median_lift_raw"].median()
            med_g = sub["median_lift_guardrail"].median()
            wst_r = sub["worst_lift_raw"].min()
            wst_g = sub["worst_lift_guardrail"].min()
            md_lines.append(f"| {s} | {win_r:.1f}% | {win_g:.1f}% | ${med_r:,.0f} | ${med_g:,.0f} | ${wst_r:,.0f} | ${wst_g:,.0f} |")
    except Exception as e:
        md_lines.append(f"Could not load sensitivity grid: {e}")
        
    # 8. Error Analysis (Top Examples)
    md_lines.append("\n## 8. Error Analysis Examples\n")
    try:
        # Join df and dec_df for predictions
        full = df.merge(dec_df[["CustomerID", "churn_prob_calib", "tau_hat"]], on="CustomerID", how="left")
        
        md_lines.append("### Top loyal customers with failure event among churned cases")
        sub1 = full[(full["loyal_customer_flag"] == 1) & (full["latest_fail_severity"] > 0)].sort_values("churn_prob_calib", ascending=False).head(5)
        for _, row in sub1.iterrows():
            md_lines.append(f"- ID: {row['CustomerID']} | Freq: {row.get('Frequency', '')} | Fail Severity: {row.get('latest_fail_severity', '')} | Pred Risk: {row.get('churn_prob_calib', ''):.3f} | Actual E: {row['E']}")
            
        md_lines.append("\n### Top 5 False Positives (High risk but E=0)")
        sub2 = full[full["E"] == 0].sort_values("churn_prob_calib", ascending=False).head(5)
        for _, row in sub2.iterrows():
            md_lines.append(f"- ID: {row['CustomerID']} | Pred Risk: {row.get('churn_prob_calib', ''):.3f} | Freq: {row.get('Frequency', '')} | Sent Latest: {row.get('sentiment_latest', '')}")
            
    except Exception as e:
        md_lines.append(f"Could not generate error analysis: {e}")

    # Write to file
    out_file = OUT / "V4_DIAGNOSTICS.md"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\nSuccessfully generated and saved full diagnostics to {out_file} !")

if __name__ == "__main__":
    generate_diagnostics()
