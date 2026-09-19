import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate9_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate9_baseline"
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    
    print("Gate 10.1 & 10.2: Feature Merging, QC & Registry")
    print("-" * 50)
    
    # 1. Load Data
    print("Loading Gate 9 Baseline splits (contains Behavior + Semantic)...")
    try:
        train_df = pd.read_parquet(gate9_dir / "baseline_train.parquet")
        val_df = pd.read_parquet(gate9_dir / "baseline_val.parquet")
        test_df = pd.read_parquet(gate9_dir / "baseline_test.parquet")
    except Exception as e:
        print(f"Error loading baseline splits: {e}")
        sys.exit(1)
        
    print("Loading Gate 10.0 Traditional Sentiment Features...")
    try:
        sentiment_df = pd.read_parquet(gate10_dir / "episode_sentiment_features.parquet")
    except Exception as e:
        print(f"Error loading sentiment features: {e}")
        sys.exit(1)
        
    # 2. Merge Features
    print("Merging Sentiment features into splits...")
    train_merged = train_df.merge(sentiment_df, on="episode_id", how="left")
    val_merged = val_df.merge(sentiment_df, on="episode_id", how="left")
    test_merged = test_df.merge(sentiment_df, on="episode_id", how="left")
    
    # 3. QC & Imputation
    print("Performing QC and missing value imputation for Sentiment features...")
    sentiment_cols = [
        'sentiment_positive_sentences', 'sentiment_negative_sentences', 
        'sentiment_total_sentences', 'sentiment_mean_compound', 
        'sentiment_mixed_intensity', 'sentiment_positive_ratio', 
        'sentiment_negative_ratio'
    ]
    
    # Compute median from TRAIN only to avoid data leakage
    medians = train_merged[sentiment_cols].median()
    
    # Fill NAs
    train_merged[sentiment_cols] = train_merged[sentiment_cols].fillna(medians)
    val_merged[sentiment_cols] = val_merged[sentiment_cols].fillna(medians)
    test_merged[sentiment_cols] = test_merged[sentiment_cols].fillna(medians)
    
    # Check for any remaining nulls in sentiment
    assert train_merged[sentiment_cols].isnull().sum().sum() == 0, "Nulls remain in train sentiment"
    
    # 4. Feature Registry
    print("Building Feature Registry...")
    behavioral_features = [
        'episode_review_count', 'prior_verified_episode_count', 'prior_review_count',
        'days_since_previous_episode', 'review_frequency',
        'episode_mean_rating', 'episode_min_rating', 'episode_max_rating', 
        'historical_mean_rating', 'recent_low_rating_count',
        'episode_duration', 'customer_lifetime'
    ]
    
    # Organize into Static and Dynamic per user feedback
    static_semantic_features = [
        'Domain_Experience_Positive', 'Domain_Experience_Negative',
        'Product_Performance_Usability_Positive', 'Product_Performance_Usability_Negative',
        'Quality_Reliability_Positive', 'Quality_Reliability_Negative',
        'Price_Value_Positive', 'Price_Value_Negative',
        'Delivery_Packaging_Positive', 'Delivery_Packaging_Negative',
        'Customer_Service_Support_Positive', 'Customer_Service_Support_Negative',
        'Brand_Trust_Loyalty_Positive', 'Brand_Trust_Loyalty_Negative',
        'Features_Design_Positive', 'Features_Design_Negative'
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
            "features": behavioral_features + sentiment_cols
        },
        "Model_C": {
            "description": "Behavior + LLM Semantic Trajectory (Static + Dynamic)",
            "features": behavioral_features + static_semantic_features + dynamic_semantic_features
        }
    }
    
    registry_path = gate10_dir / "feature_registry.json"
    with open(registry_path, "w") as f:
        json.dump(feature_registry, f, indent=4)
        
    print(f"Feature Registry saved to {registry_path.name}")
    
    # 5. Save Merged Master Splits (Dual-Cohort)
    print("Saving Dual-Cohort Master Splits for Gate 10...")
    
    # Full Population Cohort
    train_merged.to_parquet(gate10_dir / "master_train_full.parquet", index=False)
    val_merged.to_parquet(gate10_dir / "master_val_full.parquet", index=False)
    test_merged.to_parquet(gate10_dir / "master_test_full.parquet", index=False)
    
    # Semantic Cohort (semantic_available == 1)
    if 'semantic_available' in train_merged.columns:
        train_merged[train_merged['semantic_available'] == 1].to_parquet(gate10_dir / "master_train_semantic.parquet", index=False)
        val_merged[val_merged['semantic_available'] == 1].to_parquet(gate10_dir / "master_val_semantic.parquet", index=False)
        test_merged[test_merged['semantic_available'] == 1].to_parquet(gate10_dir / "master_test_semantic.parquet", index=False)
        sem_train_size = len(train_merged[train_merged['semantic_available'] == 1])
    else:
        print("Warning: semantic_available column missing. Semantic cohort not generated.")
        sem_train_size = 0
    
    print(f"Full Train size: {len(train_merged):,} | Semantic Train size: {sem_train_size:,}")
    print("Gate 10.1 & 10.2 completed successfully.")

if __name__ == "__main__":
    main()
