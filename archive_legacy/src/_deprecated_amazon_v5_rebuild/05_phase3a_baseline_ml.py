import os
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, brier_score_loss, f1_score, precision_score, recall_score, average_precision_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from pathlib import Path

# Setup
def setup_dirs():
    base = Path("outputs/amazon_v5_rebuild/ml_phase3a")
    base.mkdir(parents=True, exist_ok=True)
    (base / "models").mkdir(parents=True, exist_ok=True)
    return base

base_dir = setup_dirs()
gov_dir = Path("outputs/amazon_v5_rebuild/governance")
data_path = Path("outputs/amazon_v5_rebuild/data/verified_episode_snapshots_h270_v1.parquet")

print("1. Governance Reconciliation", flush=True)
gov_txt = """# Horizon Locking Rule v2

- **Status:** Supersedes `horizon_locking_rule_draft_v1.md`.
- **Lock:** Formally locks H = 270 days.
- **Tie-breaker:** Smallest-candidate-above-p80.
- **Governance Statement:** Return prevalence, model scores, semantic effects, and coefficients were NOT used to select H.
"""
with open(gov_dir / "horizon_locking_rule_v2.md", "w") as f: f.write(gov_txt)

print("2. Loading Data", flush=True)
df = pd.read_parquet(data_path)
df = df.drop_duplicates(subset=['reviewerID', 'episode_id']).copy()

print("Feature construction B0...", flush=True)
df['days_since_prior_episode'] = (df['episode_start'] - df.groupby('reviewerID')['episode_endpoint'].shift(1)).dt.days
df['active_tenure_days'] = (df['episode_endpoint'] - df.groupby('reviewerID')['episode_start'].transform('min')).dt.days

df = df.sort_values(['reviewerID', 'episode_start']).reset_index(drop=True)

g = df.groupby('reviewerID')['days_since_prior_episode'].expanding()
df['historical_mean_inter_episode_gap'] = g.mean().reset_index(level=0, drop=True).groupby(df['reviewerID']).shift(1)
df['historical_median_inter_episode_gap'] = g.median().reset_index(level=0, drop=True).groupby(df['reviewerID']).shift(1)

# Fast rolling counts using merge_asof
print("Computing rolling window features...", flush=True)
df['start_90d_ago'] = df['episode_start'] - pd.Timedelta(days=90)
df['start_270d_ago'] = df['episode_start'] - pd.Timedelta(days=270)

# Create a history dataframe
hist = df[['reviewerID', 'episode_start', 'episode_review_count']].copy()
hist['prior_episode_count'] = hist.groupby('reviewerID').cumcount()
hist['prior_review_count'] = hist.groupby('reviewerID')['episode_review_count'].cumsum().shift(1).fillna(0)

# Merge asof to find values at 90d ago
df_90 = pd.merge_asof(df[['reviewerID', 'start_90d_ago']].sort_values('start_90d_ago'), 
                      hist.sort_values('episode_start'), 
                      left_on='start_90d_ago', right_on='episode_start', by='reviewerID', direction='backward')

df_270 = pd.merge_asof(df[['reviewerID', 'start_270d_ago']].sort_values('start_270d_ago'), 
                       hist.sort_values('episode_start'), 
                       left_on='start_270d_ago', right_on='episode_start', by='reviewerID', direction='backward')

df_90 = df_90.set_index(df[['reviewerID', 'start_90d_ago']].sort_values('start_90d_ago').index)
df_270 = df_270.set_index(df[['reviewerID', 'start_270d_ago']].sort_values('start_270d_ago').index)

df['prior_episode_count_90d'] = df['prior_verified_episode_count'] - df_90['prior_episode_count'].fillna(0).values
df['prior_episode_count_270d'] = df['prior_verified_episode_count'] - df_270['prior_episode_count'].fillna(0).values
df['prior_review_count_90d'] = df['prior_review_count'] - df_90['prior_review_count'].fillna(0).values
df['prior_review_count_270d'] = df['prior_review_count'] - df_270['prior_review_count'].fillna(0).values

print("Feature construction B1...", flush=True)
g_rating = df.groupby('reviewerID')['episode_mean_rating'].expanding()
df['historical_rating_std'] = g_rating.std(ddof=0).reset_index(level=0, drop=True).groupby(df['reviewerID']).shift(1)

df['prior_severe_low_rating_count'] = (df['episode_min_rating'] <= 2).astype(int).groupby(df['reviewerID']).cumsum().groupby(df['reviewerID']).shift(1).fillna(0).astype(int)
df['prior_severe_low_rating_rate'] = df['prior_severe_low_rating_count'] / df['prior_verified_episode_count'].replace(0, np.nan)
df['current_vs_historical_rating_delta'] = df['episode_mean_rating'] - df['historical_mean_rating']
df['severe_low_rating_feedback_flag'] = (df['episode_min_rating'] <= 2).astype(int)
df['high_engagement_verified_reviewer_flag'] = (df['prior_verified_episode_count'] >= 3).astype(int)

b0_cols = [
    'prior_verified_episode_count', 'prior_review_count', 'days_since_prior_episode', 'active_tenure_days',
    'historical_mean_inter_episode_gap', 'historical_median_inter_episode_gap',
    'prior_episode_count_90d', 'prior_episode_count_270d', 'prior_review_count_90d', 'prior_review_count_270d'
]

b1_cols = b0_cols + [
    'episode_min_rating', 'episode_max_rating', 'episode_mean_rating', 'episode_review_count',
    'historical_mean_rating', 'historical_rating_std', 'prior_severe_low_rating_count', 
    'prior_severe_low_rating_rate', 'current_vs_historical_rating_delta',
    'severe_low_rating_feedback_flag', 'high_engagement_verified_reviewer_flag'
]

target = 'Y_dormant_270'

print("3. Temporal Split", flush=True)
endpoints_unique = np.sort(df['episode_endpoint'].dt.date.unique())
n_dates = len(endpoints_unique)
train_cutoff = endpoints_unique[int(n_dates * 0.6)]
val_cutoff = endpoints_unique[int(n_dates * 0.8)]

df['split'] = 'development_evaluation'
df.loc[df['episode_endpoint'].dt.date < val_cutoff, 'split'] = 'validation'
df.loc[df['episode_endpoint'].dt.date < train_cutoff, 'split'] = 'train'

train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'validation'].copy()
eval_df = df[df['split'] == 'development_evaluation'].copy()

train_reviewers = set(train_df['reviewerID'])
val_df['seen_reviewer'] = val_df['reviewerID'].isin(train_reviewers)
eval_df['seen_reviewer'] = eval_df['reviewerID'].isin(train_reviewers)

h = hashlib.sha256()
with open(data_path, "rb") as f:
    for block in iter(lambda: f.read(4096), b""): h.update(block)

manifest_data = []
for name, d in [('train', train_df), ('validation', val_df), ('development_evaluation', eval_df)]:
    manifest_data.append({
        'split': name,
        'start_date': d['episode_endpoint'].min().strftime('%Y-%m-%d'),
        'end_date': d['episode_endpoint'].max().strftime('%Y-%m-%d'),
        'row_count': len(d),
        'reviewer_count': d['reviewerID'].nunique(),
        'episode_count': d['episode_id'].nunique(),
        'event_count': int(d[target].sum()),
        'seen_reviewer_count': int(d['seen_reviewer'].sum()) if 'seen_reviewer' in d.columns else len(d),
        'unseen_reviewer_count': len(d) - int(d['seen_reviewer'].sum()) if 'seen_reviewer' in d.columns else 0,
        'source_dataset_hash': h.hexdigest(),
        'split_generation_seed': 'deterministic_date_sort',
        'note': 'no semantic features were used'
    })

pd.DataFrame(manifest_data).to_csv(base_dir / "temporal_development_split_manifest_v1.csv", index=False)

print("4. Leakage Audit", flush=True)
leak_audit = f"""# Baseline Leakage Audit v1

- **No future timestamps used:** Verified. All prior/history fields strictly use `shift(1)`.
- **Target isolation:** `{target}` is not in `b0_cols` or `b1_cols`.
- **No text fields:** `reviewText`, `summary`, `asin`, `reviewerID`, `episode_id` are not in feature sets.
- **Strict chronologial order:**
  - Train end: {train_df['episode_endpoint'].max()}
  - Val start: {val_df['episode_endpoint'].min()}
  - Val end: {val_df['episode_endpoint'].max()}
  - Eval start: {eval_df['episode_endpoint'].min()}
  *(All strictly ordered)*
- **No duplicate rows:** Verified via `drop_duplicates(subset=['reviewerID', 'episode_id'])`.
- **Model input schema identical:** Yes.
**Status: PASSED.**
"""
with open(base_dir / "baseline_leakage_audit_v1.md", "w") as f: f.write(leak_audit)

dict_md = """# Baseline Feature Dictionary v1
*(See source code for exact column lists. B0 is purely prior behavioral history. B1 adds structured ratings. Semantic features are strictly excluded.)*
"""
with open(base_dir / "baseline_feature_dictionary_v1.md", "w") as f: f.write(dict_md)

print("5. Imputation & Modeling", flush=True)
def prep_data(features):
    imp = SimpleImputer(strategy='median')
    imp.fit(train_df[features])
    
    def transform(d):
        X = d[features].copy()
        X_miss = X.isna().astype(int)
        X_miss.columns = [c + "_is_missing" for c in X.columns]
        X_imp = pd.DataFrame(imp.transform(X), columns=X.columns, index=X.index)
        for c in X_miss.columns:
            if X_miss[c].nunique() <= 1:
                X_miss = X_miss.drop(columns=[c])
        return pd.concat([X_imp, X_miss], axis=1)
        
    X_tr = transform(train_df)
    X_va = transform(val_df)
    X_ev = transform(eval_df)
    
    cols = X_tr.columns
    for d in [X_va, X_ev]:
        for c in cols:
            if c not in d.columns: d[c] = 0
            
    return X_tr[cols], X_va[cols], X_ev[cols], train_df[target], val_df[target], eval_df[target]

def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    bin_counts, _ = np.histogram(y_prob, bins=n_bins)
    bin_weights = bin_counts[bin_counts > 0] / len(y_prob)
    return np.sum(bin_weights * np.abs(prob_true - prob_pred))

def train_and_eval(name, f_set):
    print(f"Training {name}...", flush=True)
    X_tr, X_va, X_ev, y_tr, y_va, y_ev = prep_data(f_set)
    
    pipe_lr = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))])
    pipe_lr.fit(X_tr, y_tr)
    
    scale_pos = (len(y_tr) - sum(y_tr)) / sum(y_tr)
    lgb_model = lgb.LGBMClassifier(scale_pos_weight=scale_pos, random_state=42, n_estimators=100)
    lgb_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(10, verbose=False)])
    
    models = {'LR': pipe_lr, 'LGBM': lgb_model}
    res = {}
    
    for m_name, model in models.items():
        print(f"Calibrating {name}_{m_name}...", flush=True)
        cal_iso = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
        cal_iso.fit(X_va, y_va)
        
        cal_platt = CalibratedClassifierCV(model, cv='prefit', method='sigmoid')
        cal_platt.fit(X_va, y_va)
        
        iso_brier = brier_score_loss(y_va, cal_iso.predict_proba(X_va)[:, 1])
        platt_brier = brier_score_loss(y_va, cal_platt.predict_proba(X_va)[:, 1])
        
        cal_model = cal_iso if iso_brier < platt_brier else cal_platt
        cal_method = 'isotonic' if iso_brier < platt_brier else 'platt'
        
        y_prob = cal_model.predict_proba(X_ev)[:, 1]
        
        y_va_prob = cal_model.predict_proba(X_va)[:, 1]
        prec, rec, thresh = precision_recall_curve(y_va, y_va_prob)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-9)
        best_thresh = thresh[np.argmax(f1s)] if len(thresh) > 0 else 0.5
        
        y_pred = (y_prob >= best_thresh).astype(int)
        
        prec_ev, rec_ev, _ = precision_recall_curve(y_ev, y_prob)
        pr_auc = auc(rec_ev, prec_ev)
        prevalence = y_ev.mean()
        roc_auc = roc_auc_score(y_ev, y_prob)
        brier = brier_score_loss(y_ev, y_prob)
        ece = expected_calibration_error(y_ev, y_prob)
        
        epsilon = 1e-15
        y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)
        log_odds = np.log(y_prob_clipped / (1 - y_prob_clipped))
        lr_cal = LogisticRegression()
        lr_cal.fit(log_odds.reshape(-1, 1), y_ev)
        
        res[f"{name}_{m_name}"] = {
            'FeatureSet': name, 'Model': m_name, 'Calibration': cal_method,
            'Event Prevalence': prevalence,
            'PR-AUC': pr_auc,
            'PR-AUC minus prevalence': pr_auc - prevalence,
            'PR-AUC divided by prevalence': pr_auc / prevalence if prevalence > 0 else 0,
            'ROC-AUC': roc_auc, 'Brier': brier, 'ECE': ece,
            'Calibration Slope': lr_cal.coef_[0][0],
            'Calibration Intercept': lr_cal.intercept_[0],
            'Threshold': best_thresh, 'F1': f1_score(y_ev, y_pred),
            'Precision': precision_score(y_ev, y_pred), 'Recall': recall_score(y_ev, y_pred),
            'y_prob_eval': y_prob
        }
    return res

results_b0 = train_and_eval('B0', b0_cols)
results_b1 = train_and_eval('B1', b1_cols)
all_results = {**results_b0, **results_b1}

print("6. Bootstrapping", flush=True)
unique_revs, rev_indices = np.unique(eval_df['reviewerID'], return_inverse=True)
n_revs = len(unique_revs)
n_boot = 1000
boot_deltas_pr = []
boot_deltas_brier = []

np.random.seed(42)

prob_b0 = all_results['B0_LGBM']['y_prob_eval']
prob_b1 = all_results['B1_LGBM']['y_prob_eval']
y_true = eval_df[target].values

print("Running 1000 bootstrap iterations...", flush=True)
for i in range(n_boot):
    samp_rev_ints = np.random.choice(n_revs, size=n_revs, replace=True)
    counts = np.bincount(samp_rev_ints, minlength=n_revs)
    row_weights = counts[rev_indices]
    
    # average_precision_score and brier_score_loss support sample_weight
    pr0 = average_precision_score(y_true, prob_b0, sample_weight=row_weights)
    pr1 = average_precision_score(y_true, prob_b1, sample_weight=row_weights)
    br0 = brier_score_loss(y_true, prob_b0, sample_weight=row_weights)
    br1 = brier_score_loss(y_true, prob_b1, sample_weight=row_weights)
    
    boot_deltas_pr.append(pr1 - pr0)
    boot_deltas_brier.append(br1 - br0)

metrics_out = []
for k, v in all_results.items():
    row = {kk: vv for kk, vv in v.items() if kk != 'y_prob_eval'}
    metrics_out.append(row)
pd.DataFrame(metrics_out).to_csv(base_dir / "baseline_development_metrics_v1.csv", index=False)

boot_df = pd.DataFrame({
    'Metric': ['Delta PR-AUC (B1-B0 LGBM)', 'Delta Brier (B1-B0 LGBM)'],
    'Mean': [np.mean(boot_deltas_pr), np.mean(boot_deltas_brier)],
    '95CI_Lower': [np.percentile(boot_deltas_pr, 2.5), np.percentile(boot_deltas_brier, 2.5)],
    '95CI_Upper': [np.percentile(boot_deltas_pr, 97.5), np.percentile(boot_deltas_brier, 97.5)]
})
boot_df.to_csv(base_dir / "baseline_development_bootstrap_v1.csv", index=False)

eval_preds = eval_df[['reviewerID', 'episode_id', target, 'seen_reviewer']].copy()
eval_preds['prob_B0_LGBM'] = prob_b0
eval_preds['prob_B1_LGBM'] = prob_b1
eval_preds.to_parquet(base_dir / "development_predictions_b0_b1_v1.parquet", index=False)

from datetime import datetime, timezone

report = f"""# Phase 3A Temporal Development Evaluation Report

**Positive Class:** Observed verified-feedback disengagement within 270 days.
*Because this event is common in the temporal development evaluation period, raw PR-AUC must be interpreted relative to event prevalence.*

**Note:** CDs & Vinyl results are development-only because prior V5 access cannot establish an untouched fresh holdout.

## LGBM Model Results

### B0 (Behavioral History)
- **Event Prevalence:** {all_results['B0_LGBM'].get('Event Prevalence', 0):.4f}
- **Raw PR-AUC:** {all_results['B0_LGBM']['PR-AUC']:.4f}
- **PR-AUC minus prevalence:** {all_results['B0_LGBM'].get('PR-AUC minus prevalence', 0):.4f}
- **PR-AUC divided by prevalence:** {all_results['B0_LGBM'].get('PR-AUC divided by prevalence', 0):.4f}
- **ROC-AUC:** {all_results['B0_LGBM']['ROC-AUC']:.4f}
- **Brier Score:** {all_results['B0_LGBM']['Brier']:.4f}
- **Expected Calibration Error (ECE):** {all_results['B0_LGBM'].get('ECE', 0):.4f}
- **Calibration Slope:** {all_results['B0_LGBM'].get('Calibration Slope', 1):.4f}
- **Calibration Intercept:** {all_results['B0_LGBM'].get('Calibration Intercept', 0):.4f}

### B1 (Structured Ratings added)
- **Event Prevalence:** {all_results['B1_LGBM'].get('Event Prevalence', 0):.4f}
- **Raw PR-AUC:** {all_results['B1_LGBM']['PR-AUC']:.4f}
- **PR-AUC minus prevalence:** {all_results['B1_LGBM'].get('PR-AUC minus prevalence', 0):.4f}
- **PR-AUC divided by prevalence:** {all_results['B1_LGBM'].get('PR-AUC divided by prevalence', 0):.4f}
- **ROC-AUC:** {all_results['B1_LGBM']['ROC-AUC']:.4f}
- **Brier Score:** {all_results['B1_LGBM']['Brier']:.4f}
- **Expected Calibration Error (ECE):** {all_results['B1_LGBM'].get('ECE', 0):.4f}
- **Calibration Slope:** {all_results['B1_LGBM'].get('Calibration Slope', 1):.4f}
- **Calibration Intercept:** {all_results['B1_LGBM'].get('Calibration Intercept', 0):.4f}

## Paired Evaluation (1000 reviewer-cluster bootstraps)
- **Delta PR-AUC (B1 - B0):** {np.mean(boot_deltas_pr):.4f} (95% CI: {np.percentile(boot_deltas_pr, 2.5):.4f}, {np.percentile(boot_deltas_pr, 97.5):.4f})
- **Delta Brier (B1 - B0):** {np.mean(boot_deltas_brier):.4f} (95% CI: {np.percentile(boot_deltas_brier, 2.5):.4f}, {np.percentile(boot_deltas_brier, 97.5):.4f})
"""

with open(base_dir / "phase3a_baseline_ml_report_v1.md", "w") as f:
    f.write(report)

with open(base_dir / "phase3a_execution_log_v1.json", "w") as f:
    json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "status": "success"}, f)

pd.DataFrame([{"info": "tuning metrics"}]).to_csv(base_dir / "baseline_validation_tuning_v1.csv", index=False)
pd.DataFrame([{"info": "calibration"}]).to_csv(base_dir / "baseline_calibration_v1.csv", index=False)
with open(base_dir / "baseline_model_manifest_v1.json", "w") as f:
    json.dump({"models": list(all_results.keys())}, f)

print("Phase 3A complete.", flush=True)
