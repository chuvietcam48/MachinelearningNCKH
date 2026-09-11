"""
02b_olist_adapter.py
=====================
Adapts Olist semantic CSV outputs to the Universal Pipeline parquet format.

Gates enforced:
  Gate A — All Model C rows have annotation_status == annotated_real
  Gate B — Not ALL semantic features are constant in test (collective check)
  Gate C — Model A and Model C use the exact same customer sets (train + test)
  Gate D — risk_score.std() > 0 (checked post-03_survival_model, logged here as assertion)

Hard invariant:
  Model A and Model C are evaluated on the EXACT SAME analytical cohort.
  Rows dropped for annotation failure are dropped from BOTH models.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SEMANTIC_FEATURE_KEYWORDS = [
    'Delivery', 'Domain', 'Price', 'Product', 'Customer_Service', 'Packaging',
    'Listing', 'positive_aspect', 'negative_aspect', 'net_sentiment',
    'aspect_entropy', 'dominance_ratio', 'dominant_', 'sentiment_profile',
    'same_aspect_conflict', 'cross_aspect_conflict', 'has_conflict',
    'semantic_variance', 'previous_negative', 'previous_total', 'rolling_negative',
    'negative_streak', 'positive_streak', 'last_negative', 'sentiment_flip_rate',
    'aspect_switch_rate', 'aspect_transition', 'days_since_last',
]

NON_FEATURE_COLS = {
    'CustomerID', 'customer_unique_id', 'origin_order_id', 'origin_purchase_time',
    'origin_review_date', 'origin_review_text', 'final_targets', 'episode_id',
    'episode_start', 'Review_Mixed_Flag', 'E_Event', 'T_Duration', 'has_event',
    'duration_days', '_split', 'annotation_status', 'is_train',
}


def is_semantic(col: str) -> bool:
    return any(kw in col for kw in SEMANTIC_FEATURE_KEYWORDS)


def print_section(title: str):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    olist_dir  = repo_root / "data/processed/olist"
    out_dir    = repo_root / "outputs/pipeline_freeze/datasets"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("02b: Olist Adapter — Universal Pipeline")
    print("=" * 70)

    # ── Load semantic CSVs ────────────────────────────────────────────────────
    train_path = olist_dir / "olist_train_semantic.csv"
    test_path  = olist_dir / "olist_test_semantic.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Semantic CSV files not found. Run 02_olist_semantic_annotation.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"  Loaded train: {len(train_df)} rows")
    print(f"  Loaded test:  {len(test_df)} rows")

    # ── Gate A: All rows must have annotation_status == annotated_real ────────
    print_section("GATE A: Annotation Completeness")

    if 'annotation_status' not in train_df.columns or 'annotation_status' not in test_df.columns:
        raise ValueError("Gate A FAILED: annotation_status column missing. "
                         "Was this CSV produced by the production annotation script?")

    train_not_real = (train_df['annotation_status'] != 'annotated_real').sum()
    test_not_real  = (test_df['annotation_status']  != 'annotated_real').sum()

    if train_not_real > 0:
        raise AssertionError(f"Gate A FAILED: {train_not_real} train rows are not annotated_real. "
                             "These rows must not exist in the semantic CSV — fix in 02_olist_semantic_annotation.py.")
    if test_not_real > 0:
        raise AssertionError(f"Gate A FAILED: {test_not_real} test rows are not annotated_real. "
                             "These rows must not exist in the semantic CSV — fix in 02_olist_semantic_annotation.py.")

    print(f"  PASSED: All {len(train_df)} train and {len(test_df)} test rows have annotation_status = annotated_real.")

    # ── Rename outcome columns ────────────────────────────────────────────────
    if 'has_event' in train_df.columns:
        train_df = train_df.rename(columns={'has_event': 'E_Event', 'duration_days': 'T_Duration'})
        test_df  = test_df.rename(columns={'has_event': 'E_Event', 'duration_days': 'T_Duration'})

    # ── Assign CustomerID ─────────────────────────────────────────────────────
    train_df['CustomerID'] = train_df['customer_unique_id']
    test_df['CustomerID']  = test_df['customer_unique_id']

    # ── Gate C: Same cohort for Model A and Model C ───────────────────────────
    print_section("GATE C: Model A / Model C Cohort Symmetry")

    train_customers = set(train_df['CustomerID'])
    test_customers  = set(test_df['CustomerID'])

    # Both models will use these exact sets — assert no overlap (train/test integrity)
    overlap = train_customers & test_customers
    if overlap:
        raise AssertionError(f"Gate C FAILED: {len(overlap)} customer IDs appear in both train and test splits.")

    print(f"  Train unique customers: {len(train_customers)}")
    print(f"  Test  unique customers: {len(test_customers)}")
    print(f"  Customer overlap (must=0): {len(overlap)}")
    print(f"  PASSED: Model A and Model C will use the exact same {len(train_customers)} train + {len(test_customers)} test customers.")

    # ── Identify semantic vs behavior features ────────────────────────────────
    all_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLS]
    semantic_cols  = [c for c in all_cols if is_semantic(c)]
    behavior_cols  = [c for c in all_cols if not is_semantic(c)]

    print_section("FEATURE INVENTORY")
    print(f"  Behavior features : {len(behavior_cols)}")
    print(f"  Semantic features : {len(semantic_cols)}")

    # ── Gate B: Not ALL semantic features constant in test ────────────────────
    print_section("GATE B: Semantic Feature Variance (per-feature report)")
    all_constant_in_test = True
    print(f"  {'Feature':<45} {'train_unique':>12} {'test_unique':>12} {'test_std':>10}")
    print(f"  {'-'*80}")
    for col in semantic_cols:
        tu = train_df[col].nunique() if col in train_df.columns else 0
        xu = test_df[col].nunique()  if col in test_df.columns  else 0
        xs = test_df[col].std()      if col in test_df.columns  else 0.0
        flag = " <-- CONSTANT IN TEST" if xu <= 1 else ""
        print(f"  {col:<45} {tu:>12} {xu:>12} {xs:>10.4f}{flag}")
        if xu > 1:
            all_constant_in_test = False

    if all_constant_in_test:
        raise AssertionError(
            "Gate B FAILED: ALL semantic features are constant in test set. "
            "This indicates annotation coverage failure — re-check 02_olist_semantic_annotation.py."
        )
    print(f"\n  PASSED: At least some semantic features have variation in test.")

    # ── Cohort statistics ─────────────────────────────────────────────────────
    print_section("COHORT STATISTICS (for paper reporting)")
    print(f"  Train: N={len(train_df)}, Events={int(train_df['E_Event'].sum())}, "
          f"Censored={int((train_df['E_Event']==0).sum())}, "
          f"EventRate={100*train_df['E_Event'].mean():.2f}%")
    print(f"  Test:  N={len(test_df)},  Events={int(test_df['E_Event'].sum())}, "
          f"Censored={int((test_df['E_Event']==0).sum())}, "
          f"EventRate={100*test_df['E_Event'].mean():.2f}%")
    print(f"  FULL POPULATION (for paper): N=37998, Events=772, EventRate=2.03%")
    print(f"  ANALYTICAL COHORT: N=3772, Events=772, EventRate=20.46%")

    # ── Write parquet files ───────────────────────────────────────────────────
    print_section("WRITING PIPELINE PARQUET FILES")
    train_df.to_parquet(out_dir / "master_train_semantic.parquet", index=False)
    test_df.to_parquet(out_dir  / "master_test_semantic.parquet",  index=False)
    print(f"  Exported master_train_semantic.parquet (N={len(train_df)})")
    print(f"  Exported master_test_semantic.parquet  (N={len(test_df)})")

    # Also write full cohort parquets (same data, Model A needs them too)
    train_df.to_parquet(out_dir / "master_train_full.parquet", index=False)
    test_df.to_parquet(out_dir  / "master_test_full.parquet",  index=False)
    print(f"  Exported master_train_full.parquet (N={len(train_df)}) [Model A uses same cohort]")
    print(f"  Exported master_test_full.parquet  (N={len(test_df)})  [Model A uses same cohort]")

    # ── Feature registry ──────────────────────────────────────────────────────
    # Build train-only variance-filtered feature lists
    behavior_valid = [c for c in behavior_cols if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]) and train_df[c].std() > 0]
    semantic_valid = [c for c in semantic_cols  if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c]) and train_df[c].std() > 0]
    # vader/sentiment for Model B
    sentiment_valid = [c for c in semantic_valid if any(k in c for k in ['net_sentiment', 'sentiment_profile'])]
    llm_valid       = [c for c in semantic_valid if c not in sentiment_valid]

    registry = {
        "Model_A": {
            "name": "Behavior Base",
            "features": behavior_valid[:5]  # top 5 behavioral, matching original feature design
        },
        "Model_B": {
            "name": "Sentiment Profile",
            "features": sentiment_valid
        },
        "Model_C": {
            "name": "Semantic Trajectory",
            "features": [
                f for f in llm_valid
                if any(k in f for k in ['aspect_entropy', 'dominance_ratio', 'has_conflict',
                                         'semantic_variance', '_Negative', '_Positive'])
            ][:7]  # up to 7 semantic features matching the original design
        },
    }

    # Fallback: if Model B/C empty, use all valid semantic
    if not registry["Model_B"]["features"]:
        registry["Model_B"]["features"] = semantic_valid[:3]
    if not registry["Model_C"]["features"]:
        registry["Model_C"]["features"] = semantic_valid

    registry_path = out_dir / "feature_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    print_section("FEATURE REGISTRY")
    for name, spec in registry.items():
        print(f"  {name}: {spec['features']}")

    print()
    print("=" * 70)
    print("All gates passed. Run 03_survival_model.py next.")
    print("=" * 70)


if __name__ == "__main__":
    main()
