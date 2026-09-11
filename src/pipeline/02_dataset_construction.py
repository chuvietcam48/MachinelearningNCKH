import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset_registry import get_dataset
from src.framework.behavior_engine import BehaviorEngine

def main():
    features_dir = repo_root / "data" / "artifacts" / "semantic_labels"
    out_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # We no longer hardcode paths to deprecated folders
    input_path = features_dir / "verified_episode_snapshots_h270_v1.parquet"
    sentiment_path = features_dir / "episode_sentiment_features.parquet"
    
    print("02: Dataset Construction & Feature Registry (Pure Orchestration)")
    print("-" * 50)
    
    print(f"Loading episode snapshots from {input_path}...")
    try:
        df = pd.read_parquet(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)
        
    initial_len = len(df)
    
    # Standardize schema (Schema Mapper logic for this offline script)
    if 'reviewerID' in df.columns:
        df = df.rename(columns={'reviewerID': 'CustomerID'})
    
    # 1. Eligibility Filters
    print("\nApplying Eligibility Filters...")
    df = df.dropna(subset=['episode_start', 'episode_endpoint'])
    df['episode_duration'] = (df['episode_endpoint'] - df['episode_start']).dt.total_seconds() / (24 * 3600)
    df = df[df['episode_duration'] >= 0]
    
    user_counts = df['CustomerID'].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df['CustomerID'].isin(valid_users)]
    df = df.sort_values(['CustomerID', 'episode_start']).reset_index(drop=True)
    print(f"Rows after filtering: {len(df):,} (Dropped {initial_len - len(df):,})")
    
    # 2. Framework Behavior Engine (Delegated logic)
    print("Delegating to Framework Behavior Engine...")
    behavior_engine = BehaviorEngine()
    df = behavior_engine.process(df)
    
    # 3. Load Semantic Annotations (Pre-engineered)
    # The offline pipeline assumes semantic features are already built by 01 script or SemanticProvider
    semantic_features_path = repo_root / "outputs" / "pipeline_freeze" / "features" / "semantic_features_engineered.parquet"
    if semantic_features_path.exists():
        print(f"Loading semantic features from {semantic_features_path}...")
        df_semantic = pd.read_parquet(semantic_features_path)
        if 'reviewerID' in df_semantic.columns:
            df_semantic = df_semantic.rename(columns={'reviewerID': 'CustomerID'})
            
        from src.framework.feature_fusion import FeatureFusionEngine
        fusion = FeatureFusionEngine()
        df = fusion.merge(df, df_semantic)
    
    # 4. Merge Traditional Sentiment (Frozen Artifact)
    print("\nMerging Traditional Sentiment Features...")
    try:
        sentiment_df = pd.read_parquet(sentiment_path)
        df = df.merge(sentiment_df, on="episode_id", how="left")
    except Exception as e:
        print(f"Warning: Could not load traditional sentiment ({e}). Skipping.")
        
    sentiment_cols = [
        'sentiment_positive_sentences', 'sentiment_negative_sentences', 
        'sentiment_total_sentences', 'sentiment_mean_compound', 
        'sentiment_mixed_intensity', 'sentiment_positive_ratio', 'sentiment_negative_ratio'
    ]
    
    # 5. Temporal Split (70-15-15)
    print("\nComputing Temporal Split (70-15-15)...")
    q_70 = df['episode_start'].quantile(0.70)
    q_85 = df['episode_start'].quantile(0.85)
    
    df_train = df[df['episode_start'] <= q_70].copy()
    df_val = df[(df['episode_start'] > q_70) & (df['episode_start'] <= q_85)].copy()
    df_test = df[df['episode_start'] > q_85].copy()
    
    # QC: Impute missing sentiment with Train medians
    if all(c in df_train.columns for c in sentiment_cols):
        medians = df_train[sentiment_cols].median()
        df_train[sentiment_cols] = df_train[sentiment_cols].fillna(medians)
        df_val[sentiment_cols] = df_val[sentiment_cols].fillna(medians)
        df_test[sentiment_cols] = df_test[sentiment_cols].fillna(medians)
    
    print(f"Train size: {len(df_train):,}")
    print(f"Val size: {len(df_val):,}")
    print(f"Test size: {len(df_test):,}")
    
    # 6. Feature Registry
    print("\nBuilding Feature Registry...")
    behavioral_features = [
        'prior_verified_episode_count', 'prior_review_count',
        'days_since_previous_episode', 'review_frequency',
        'episode_min_rating', 'recent_low_rating_count',
        'episode_duration', 'customer_lifetime'
    ]
    
    static_semantic_features = [
        'Domain_Experience_Positive', 'Domain_Experience_Negative',
        'Product_Performance_Usability_Positive', 'Product_Performance_Usability_Negative',
        'Product_Condition_Quality_Positive', 'Product_Condition_Quality_Negative',
        'Packaging_Presentation_Positive', 'Packaging_Presentation_Negative',
        'Delivery_Fulfillment_Positive', 'Delivery_Fulfillment_Negative',
        'Customer_Service_Returns_Positive', 'Customer_Service_Returns_Negative',
        'Listing_Expectation_Compatibility_Positive', 'Listing_Expectation_Compatibility_Negative',
        'Price_Value_Positive', 'Price_Value_Negative'
    ]
    
    dynamic_semantic_features = [
        'negative_streak', 'positive_streak', 'sentiment_flip_rate', 
        'aspect_switch_rate', 'aspect_transition_pattern', 
        'days_since_last_negative', 'previous_negative_count',
        'previous_total_aspects', 'rolling_negative_ratio',
        'sentiment_profile', 'same_aspect_conflict', 'cross_aspect_conflict',
        'has_conflict', 'semantic_variance', 'net_sentiment', 
        'aspect_entropy', 'dominance_ratio'
    ]
    
    feature_registry = {
        "Model_A": {
            "description": "Baseline (Behavioral Features Only)",
            "features": behavioral_features
        },
        "Model_B": {
            "description": "Behavior + Traditional Sentiment",
            "features": behavioral_features + sentiment_cols if all(c in df_train.columns for c in sentiment_cols) else behavioral_features
        },
        "Model_C": {
            "description": "Behavior + LLM Semantic Trajectory (Static + Dynamic)",
            "features": behavioral_features + static_semantic_features + dynamic_semantic_features
        }
    }
    
    with open(out_dir / "feature_registry.json", "w") as f:
        json.dump(feature_registry, f, indent=4)
        
    # 7. Dual-Cohort Output
    print("\nSaving Dual-Cohort Master Splits...")
    # Full Population
    df_train.to_parquet(out_dir / "master_train_full.parquet", index=False)
    df_val.to_parquet(out_dir / "master_val_full.parquet", index=False)
    df_test.to_parquet(out_dir / "master_test_full.parquet", index=False)
    
    # Semantic Cohort
    if 'semantic_available' in df_train.columns:
        sem_train = df_train[df_train['semantic_available'] == 1].copy()
        sem_train.to_parquet(out_dir / "master_train_semantic.parquet", index=False)
        
        sem_val = df_val[df_val['semantic_available'] == 1].copy()
        sem_val.to_parquet(out_dir / "master_val_semantic.parquet", index=False)
        
        sem_test = df_test[df_test['semantic_available'] == 1].copy()
        sem_test.to_parquet(out_dir / "master_test_semantic.parquet", index=False)
        
        print(f"Semantic Train size: {len(sem_train):,}")
    
    print("02: Dataset Construction completed successfully.")

if __name__ == "__main__":
    main()
