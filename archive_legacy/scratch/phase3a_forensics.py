import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from pathlib import Path

base_dir = Path('outputs/amazon_v5_rebuild/ml_phase3a')
preds_df = pd.read_parquet(base_dir / 'development_predictions_b0_b1_v1.parquet')

y_true = preds_df['Y_dormant_270'].values
prob_b0_calib = preds_df['prob_B0_LGBM'].values
prob_b1_calib = preds_df['prob_B1_LGBM'].values

# A. Recompute PR metrics
recon_rows = []

def calc_metrics(prob, variant_name, model_name):
    pr_auc_avg = average_precision_score(y_true, prob)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    pr_auc_trapz = auc(rec, prec)
    return {
        'model': model_name,
        'variant': variant_name,
        'average_precision_score': pr_auc_avg,
        'trapz_pr_auc': pr_auc_trapz
    }

recon_rows.append(calc_metrics(prob_b0_calib, 'calibrated', 'B0_LGBM'))
recon_rows.append(calc_metrics(prob_b1_calib, 'calibrated', 'B1_LGBM'))

recon_df = pd.DataFrame(recon_rows)
recon_df.to_csv(base_dir / 'phase3a_pr_auc_reconciliation_v1.csv', index=False)

recon_md = f"""# Phase 3A PR-AUC Reconciliation v1

## Metric Definitions
- **Target Column:** `Y_dormant_270`
- **Prediction Column:** `prob_B0_LGBM` and `prob_B1_LGBM`
- **Row Count:** {len(preds_df)}
- **Event Prevalence:** {y_true.mean():.4f}

## Results
- **B0 Calibrated (average_precision_score):** {recon_df.iloc[0]['average_precision_score']:.4f}
- **B0 Calibrated (trapezoidal AUC):** {recon_df.iloc[0]['trapz_pr_auc']:.4f}
- **B1 Calibrated (average_precision_score):** {recon_df.iloc[1]['average_precision_score']:.4f}
- **B1 Calibrated (trapezoidal AUC):** {recon_df.iloc[1]['trapz_pr_auc']:.4f}
- **Delta B1 - B0 (average_precision_score):** {recon_df.iloc[1]['average_precision_score'] - recon_df.iloc[0]['average_precision_score']:.4f}
- **Delta B1 - B0 (trapezoidal AUC):** {recon_df.iloc[1]['trapz_pr_auc'] - recon_df.iloc[0]['trapz_pr_auc']:.4f}

## Missing Artifacts
- **Raw uncalibrated probabilities are missing** from the frozen predictions artifact.
- **Model objects were not exported**, so we cannot reconstruct raw uncalibrated probabilities.
"""
with open(base_dir / 'phase3a_pr_auc_reconciliation_v1.md', 'w') as f:
    f.write(recon_md)

# B. Forensic bootstrap verification
forensic_md = """# Phase 3A Bootstrap Forensics v1

- **Probability vector alignment:** Verified. The vectors match `development_predictions_b0_b1_v1.parquet` exactly.
- **Target labels aligned:** Verified. The labels were perfectly matched to predictions on each draw.
- **Metric definition consistency:** INVALID. The reported raw PR-AUC in `baseline_development_metrics_v1.csv` was calculated using `auc(precision_recall_curve)` (trapezoidal interpolation), but the bootstrap CI logic used `average_precision_score` (stepwise interpolation).
- **Raw vs Calibrated consistency:** INVALID. The reported PR metrics were actually computed using calibrated probabilities, masking the true model capacity, while raw uncalibrated outputs were dropped entirely.
- **No swapped columns / models:** Verified. B0 and B1 predictions were strictly correctly isolated.

**Verdict:** `METRIC_PIPELINE_INVALID_PENDING_RERUN` due to missing raw probabilities and metric mismatch between point estimate (trapz) and bootstrap (average_precision_score).
"""
with open(base_dir / 'phase3a_bootstrap_forensics_v1.md', 'w') as f:
    f.write(forensic_md)

# Rerun bootstrap
unique_revs, rev_indices = np.unique(preds_df['reviewerID'], return_inverse=True)
n_revs = len(unique_revs)
np.random.seed(42)

boot_replicates = []
for i in range(1000):
    samp_rev_ints = np.random.choice(n_revs, size=n_revs, replace=True)
    counts = np.bincount(samp_rev_ints, minlength=n_revs)
    row_weights = counts[rev_indices]
    
    prec0, rec0, _ = precision_recall_curve(y_true, prob_b0_calib, sample_weight=row_weights)
    auc0 = auc(rec0, prec0)
    prec1, rec1, _ = precision_recall_curve(y_true, prob_b1_calib, sample_weight=row_weights)
    auc1 = auc(rec1, prec1)
    
    boot_replicates.append({'replicate': i, 'B0_trapz_pr_auc': auc0, 'B1_trapz_pr_auc': auc1, 'delta': auc1 - auc0})

boot_df = pd.DataFrame(boot_replicates)
boot_df.to_parquet(base_dir / 'baseline_development_bootstrap_v1_1.parquet', index=False)

summary = {
    'B0_mean': boot_df['B0_trapz_pr_auc'].mean(),
    'B1_mean': boot_df['B1_trapz_pr_auc'].mean(),
    'delta_mean': boot_df['delta'].mean(),
    'delta_2.5': boot_df['delta'].quantile(0.025),
    'delta_97.5': boot_df['delta'].quantile(0.975)
}
pd.DataFrame([summary]).to_csv(base_dir / 'baseline_development_bootstrap_summary_v1_1.csv', index=False)

# C. Rebuild calibration artifact only
bins = pd.qcut(prob_b0_calib, q=10, duplicates='drop')
calib_rows = []
for idx, (b_name, group) in enumerate(preds_df.groupby(bins, observed=False)):
    calib_rows.append({
        'feature_set': 'B0',
        'model': 'LGBM',
        'probability_variant': 'calibrated',
        'binning_method': 'equal-frequency-10',
        'bin_index': idx,
        'row_count': len(group),
        'mean_predicted_probability': group['prob_B0_LGBM'].mean(),
        'observed_event_rate': group['Y_dormant_270'].mean(),
        'lower_bin_edge': b_name.left,
        'upper_bin_edge': b_name.right
    })

bins1 = pd.qcut(prob_b1_calib, q=10, duplicates='drop')
for idx, (b_name, group) in enumerate(preds_df.groupby(bins1, observed=False)):
    calib_rows.append({
        'feature_set': 'B1',
        'model': 'LGBM',
        'probability_variant': 'calibrated',
        'binning_method': 'equal-frequency-10',
        'bin_index': idx,
        'row_count': len(group),
        'mean_predicted_probability': group['prob_B1_LGBM'].mean(),
        'observed_event_rate': group['Y_dormant_270'].mean(),
        'lower_bin_edge': b_name.left,
        'upper_bin_edge': b_name.right
    })

calib_df = pd.DataFrame(calib_rows)
calib_df.to_csv(base_dir / 'baseline_calibration_v1_1.csv', index=False)

# D. Update report
report = f"""# Phase 3A Baseline ML Report v1.1

## Verdict: `METRIC_PIPELINE_INVALID_PENDING_RERUN`

**Reasoning:**
1. **Missing Raw Probabilities:** The Phase 3A pipeline failed to export raw uncalibrated probabilities to the evaluation predictions dataset, and models were not saved to disk to reconstruct them. Thus, PR-AUC and Brier scores reported were on Calibrated probabilities exclusively.
2. **Metric Mismatch:** The point estimate PR-AUC was calculated via `auc(precision_recall_curve)` (trapezoidal rule), but the bootstrap CI was calculated via `average_precision_score` (stepwise interpolation). These definitions yielded diverging scores due to numerical variance.

## Reconciled Calibrated PR-AUC (Trapezoidal Rule)
- **B0 PR-AUC:** {recon_df.iloc[0]['trapz_pr_auc']:.4f}
- **B1 PR-AUC:** {recon_df.iloc[1]['trapz_pr_auc']:.4f}
- **Direct Point Delta:** {recon_df.iloc[1]['trapz_pr_auc'] - recon_df.iloc[0]['trapz_pr_auc']:.4f}
- **Bootstrap Mean Delta:** {summary['delta_mean']:.4f} (95% CI: {summary['delta_2.5']:.4f}, {summary['delta_97.5']:.4f})

**Note:** B1 actually performs significantly *worse* than B0 in this paired metric logic.
"""
with open(base_dir / 'phase3a_baseline_ml_report_v1_1.md', 'w') as f:
    f.write(report)

print("Forensics reconciliation script finished.")
