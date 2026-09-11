"""
Olist Pipeline Diagnostic Script
=================================
Produces the full diagnostic table requested by the reviewer.
Does NOT conclude anything - only surfaces raw signals for human audit.
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path

repo_root = Path('C:/Users/Admin/OneDrive/Máy tính/MachineLearning-1/MachinelearningNCKH')
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from lifelines import CoxPHFitter

# ─────────────────────────────────────────────────────────────────
# 1. LOAD PIPELINE OUTPUTS
# ─────────────────────────────────────────────────────────────────
pipeline_dir = repo_root / "outputs/pipeline_freeze/datasets"
train = pd.read_parquet(pipeline_dir / "master_train_semantic.parquet")
test  = pd.read_parquet(pipeline_dir / "master_test_semantic.parquet")

print("=" * 70)
print("DIAGNOSTIC: OLIST PIPELINE AUDIT")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────
# 2. COHORT & SPLIT VERIFICATION
# ─────────────────────────────────────────────────────────────────
print("\n[1] COHORT & SPLIT STATISTICS")
print("-" * 40)
print(f"  Train rows         : {len(train)}")
print(f"  Test rows          : {len(test)}")

# customer overlap
train_customers = set(train['CustomerID']) if 'CustomerID' in train.columns else set()
test_customers  = set(test['CustomerID'])  if 'CustomerID' in test.columns  else set()
overlap = train_customers & test_customers
print(f"  Train unique customers : {len(train_customers)}")
print(f"  Test  unique customers : {len(test_customers)}")
print(f"  Customer overlap (must = 0) : {len(overlap)}")

# events
event_col = 'event' if 'event' in train.columns else None
if event_col:
    print(f"  Train events       : {train[event_col].sum()} / {len(train)} ({100*train[event_col].mean():.2f}%)")
    print(f"  Test  events       : {test[event_col].sum()} / {len(test)} ({100*test[event_col].mean():.2f}%)")
else:
    print("  !! 'event' column NOT FOUND. Available:", list(train.columns[:15]))

# ─────────────────────────────────────────────────────────────────
# 3. FEATURE AUDIT
# ─────────────────────────────────────────────────────────────────
print("\n[2] FEATURE INVENTORY")
print("-" * 40)

# Load feature registry if it exists
registry_path = repo_root / "outputs/pipeline_freeze/feature_registry.json"
if registry_path.exists():
    registry = json.loads(registry_path.read_text())
    all_features = registry.get("features", [])
    model_a_features = registry.get("model_a", [])
    model_b_features = registry.get("model_b", [])
    model_c_features = registry.get("model_c", [])
    print(f"  Registry loaded: {len(all_features)} total features")
    print(f"  Model A features  : {len(model_a_features)} -> {model_a_features}")
    print(f"  Model B features  : {len(model_b_features)} -> {model_b_features}")
    print(f"  Model C features  : {len(model_c_features)} -> {model_c_features}")
else:
    print("  !! feature_registry.json NOT FOUND")
    # auto-detect feature columns
    non_feature_cols = {'CustomerID', 'episode_id', 'episode_start', 'event', 'duration',
                        'T', 'E', 'label', 'split', 'is_train'}
    all_features = [c for c in train.columns if c not in non_feature_cols]
    print(f"  Auto-detected features: {len(all_features)}")

# Separate semantic features
semantic_cols = [c for c in all_features if any(x in c.lower() for x in [
    'sentiment', 'aspect', 'polarity', 'semantic', 'llm',
    'Customer_Service', 'Delivery', 'Domain', 'Price', 'Product',
    'dominance', 'entropy', 'conflict', 'vader'
])]
behavior_cols = [c for c in all_features if c not in semantic_cols]

print(f"\n  Behavior features : {len(behavior_cols)}")
print(f"  Semantic features : {len(semantic_cols)}")
print(f"  Semantic feature names: {semantic_cols[:20]}")

# ─────────────────────────────────────────────────────────────────
# 4. SEMANTIC FEATURE VARIANCE CHECK
# ─────────────────────────────────────────────────────────────────
print("\n[3] SEMANTIC FEATURE VARIANCE AUDIT (TRAIN SET)")
print("-" * 40)
if semantic_cols:
    var_report = []
    for col in semantic_cols:
        if col in train.columns:
            col_data = train[col].dropna()
            var_report.append({
                'feature': col,
                'n_non_null': len(col_data),
                'n_unique': col_data.nunique(),
                'mean': col_data.mean() if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'std': col_data.std() if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'min': col_data.min() if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'max': col_data.max() if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
                'pct_zero': (col_data == 0).mean() if pd.api.types.is_numeric_dtype(col_data) else 'N/A',
            })
    var_df = pd.DataFrame(var_report)
    print(var_df.to_string(index=False))
else:
    print("  No semantic columns identified in train set!")

# ─────────────────────────────────────────────────────────────────
# 5. FIT EACH MODEL AND EXTRACT DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────
print("\n[4] MODEL FIT & PREDICTION DIAGNOSTICS")
print("-" * 40)

# Determine event/duration columns
dur_col = 'duration' if 'duration' in train.columns else ('T' if 'T' in train.columns else None)
evt_col = 'event'    if 'event'    in train.columns else ('E' if 'E' in train.columns else None)

if dur_col is None or evt_col is None:
    print(f"  !! Cannot find duration/event columns. Cols: {list(train.columns)}")
    sys.exit(1)

print(f"  Duration col: {dur_col}, Event col: {evt_col}")
print(f"  Test: {test[evt_col].sum()} events / {len(test)} rows")

def fit_and_diagnose(model_name, feature_cols, train_df, test_df):
    print(f"\n  --- {model_name} ---")
    avail = [c for c in feature_cols if c in train_df.columns]
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        print(f"  MISSING FEATURES (not in data): {missing}")

    if not avail:
        print("  !! No features available — skipping")
        return

    # Drop zero-variance
    pre_filter = avail[:]
    avail = [c for c in avail if train_df[c].std() > 0]
    dropped = [c for c in pre_filter if c not in avail]
    if dropped:
        print(f"  Dropped (zero-variance): {dropped}")

    cols_needed = avail + [dur_col, evt_col]
    df_train_sub = train_df[cols_needed].dropna()
    df_test_sub  = test_df[avail].dropna()

    print(f"  Features entering CoxPH : {len(avail)} -> {avail}")
    print(f"  Train rows after dropna : {len(df_train_sub)} | events: {df_train_sub[evt_col].sum()}")

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_train_sub, duration_col=dur_col, event_col=evt_col)

        # Print non-zero coefficients
        coefs = cph.params_
        nz = coefs[coefs.abs() > 1e-6]
        print(f"  Non-zero Cox coefficients: {len(nz)} / {len(coefs)}")
        if len(nz) > 0:
            print(f"  Top 5 by |coef|:\n{nz.abs().sort_values(ascending=False).head(5).to_string()}")
        else:
            print("  !! ALL COEFFICIENTS ARE ZERO")

        # Predictions on test set
        test_aligned = test_df[avail].reindex(columns=avail).fillna(0)
        risk_scores = cph.predict_partial_hazard(test_aligned)

        print(f"  Predictions on test:")
        print(f"    N                    = {len(risk_scores)}")
        print(f"    Unique predictions   = {risk_scores.nunique()}")
        print(f"    Std                  = {risk_scores.std():.6f}")
        print(f"    Min                  = {risk_scores.min():.6f}")
        print(f"    Max                  = {risk_scores.max():.6f}")

        if risk_scores.nunique() <= 1:
            print("  !! CONSTANT PREDICTION - C-index will be 0.5 by definition!")
        
        # C-index on test
        from lifelines.utils import concordance_index
        test_with_outcome = test_df[[dur_col, evt_col]].copy()
        test_with_outcome['risk'] = risk_scores.values
        test_with_outcome = test_with_outcome.dropna()
        
        cindex = concordance_index(
            test_with_outcome[dur_col],
            -test_with_outcome['risk'],
            test_with_outcome[evt_col]
        )
        print(f"    C-index (raw)        = {cindex:.4f}")

    except Exception as e:
        print(f"  !! CoxPH FAILED: {e}")

# Define feature sets
behavior_feats = [c for c in behavior_cols if c in train.columns]
vader_feats     = [c for c in semantic_cols if 'vader' in c.lower() or 'sentiment' in c.lower()]
llm_feats       = [c for c in semantic_cols if c not in vader_feats]
all_sem_feats   = semantic_cols

print(f"\n  behavior_feats: {behavior_feats}")
print(f"  vader_feats: {vader_feats}")
print(f"  llm_feats: {llm_feats}")

fit_and_diagnose("Model A (Behavior only)", behavior_feats, train, test)
fit_and_diagnose("Model B (Behavior + VADER)", behavior_feats + vader_feats, train, test)
fit_and_diagnose("Model C (Behavior + LLM)",   behavior_feats + llm_feats,   train, test)

# ─────────────────────────────────────────────────────────────────
# 6. SEMANTIC ANNOTATION COMPLETENESS
# ─────────────────────────────────────────────────────────────────
print("\n[5] SEMANTIC ANNOTATION AUDIT")
print("-" * 40)
checkpoint_file = repo_root / "data/processed/olist/semantic_checkpoint.json"
if checkpoint_file.exists():
    data = json.loads(checkpoint_file.read_text())
    n_annotated = sum(1 for v in data.values() if v is not None)
    n_null = sum(1 for v in data.values() if v is None)
    n_empty = sum(1 for v in data.values() if v is not None and json.loads(v) == [])
    n_real = sum(1 for v in data.values() if v is not None and json.loads(v) != [])
    print(f"  Checkpoint records      : {len(data)}")
    print(f"  Successfully annotated  : {n_annotated}")
    print(f"  Null (failed)          : {n_null}")
    print(f"  Empty aspect list []    : {n_empty}  <-- MOCKED or genuinely no aspects")
    print(f"  Real annotations        : {n_real}")
    total_reviews = len(train) + len(test)
    print(f"  Total reviews in dataset: {total_reviews}")
    print(f"  Coverage: {100*n_annotated/total_reviews:.1f}% ({n_real} real + {n_empty} empty)")
    print(f"\n  >> If 'Empty aspect list' >> 'Real annotations', semantic features will")
    print(f"     be near-zero variance and CoxPH will drop them → C-index = 0.5")
else:
    print("  !! Checkpoint file NOT FOUND")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE — Awaiting human review before any conclusions.")
print("=" * 70)
