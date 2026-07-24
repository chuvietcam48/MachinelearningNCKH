import pandas as pd
import numpy as np
from pathlib import Path
import json

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    features_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "features"
    gate9_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate9_baseline"
    gate9_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = features_dir / "episode_semantic_features_v1.parquet"
    
    print("Gate 9.1: Data Preparation & Target Construction")
    print("-" * 30)
    
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    
    initial_len = len(df)
    
    # ---------------------------------------------------------
    # Gate 9.0: Eligibility Filter
    # ---------------------------------------------------------
    print("\n[Gate 9.0] Applying Eligibility Filters...")
    
    # 1. Drop rows missing episode_start or episode_endpoint
    df = df.dropna(subset=['episode_start', 'episode_endpoint'])
    
    # 2. Episode Duration < 0 (Data error)
    df['episode_duration'] = (df['episode_endpoint'] - df['episode_start']).dt.total_seconds() / (24 * 3600)
    df = df[df['episode_duration'] >= 0]
    
    # 3. Only keep users with > 1 episode (Need history)
    # Count episodes per reviewer
    user_counts = df['reviewerID'].value_counts()
    valid_users = user_counts[user_counts > 1].index
    df = df[df['reviewerID'].isin(valid_users)]
    
    # 4. Sort chronologically
    df = df.sort_values(['reviewerID', 'episode_start']).reset_index(drop=True)
    
    print(f"Rows after filtering: {len(df):,} (Dropped {initial_len - len(df):,})")
    
    # ---------------------------------------------------------
    # Gate 9.1: Target Construction
    # ---------------------------------------------------------
    print("\n[Gate 9.1] Constructing Survival Targets...")
    # T (Duration): next_episode_start - episode_endpoint
    # E (Event): 1 - Y_dormant_270
    
    # For calculation, if next_episode_start is NaT (which means dormant), duration is 270.
    df['T_Duration'] = np.where(
        pd.isna(df['next_episode_start']),
        270.0,
        (df['next_episode_start'] - df['episode_endpoint']).dt.total_seconds() / (24 * 3600)
    )
    # Clip at 270 (Right-censoring limit)
    df['T_Duration'] = df['T_Duration'].clip(upper=270.0)
    # E (Event)
    df['E_Event'] = 1 - df['Y_dormant_270']
    
    # ---------------------------------------------------------
    # Gate 9.3: Advanced Baseline Features
    # ---------------------------------------------------------
    print("\n[Gate 9.3] Engineering Advanced Baseline Features...")
    
    # We already have:
    # Behavioral: episode_review_count, prior_verified_episode_count, prior_review_count
    # Rating: episode_mean_rating, episode_min_rating, episode_max_rating, historical_mean_rating
    # Temporal: episode_duration
    
    # Let's add:
    # 1. days_since_previous_episode
    # We need to shift episode_endpoint to get the previous one
    df['prev_episode_endpoint'] = df.groupby('reviewerID')['episode_endpoint'].shift(1)
    df['days_since_previous_episode'] = (df['episode_start'] - df['prev_episode_endpoint']).dt.total_seconds() / (24 * 3600)
    df['days_since_previous_episode'] = df['days_since_previous_episode'].fillna(-1.0) # For first episodes
    
    # 2. customer_lifetime (from first episode_start to current episode_endpoint)
    first_episode_start = df.groupby('reviewerID')['episode_start'].transform('min')
    df['customer_lifetime'] = (df['episode_endpoint'] - first_episode_start).dt.total_seconds() / (24 * 3600)
    
    # 3. review_frequency
    df['review_frequency'] = df['prior_review_count'] / (df['customer_lifetime'] + 1.0)
    
    # 4. recent_low_rating_count (We don't have the raw reviews here, so we approximate using historical_mean_rating)
    # Since we can't easily count exact 1-2 star reviews from history without joining the 100M row table, 
    # we will use the `episode_min_rating` of previous episodes.
    df['is_low_rating_episode'] = (df['episode_min_rating'] <= 2).astype(int)
    df['recent_low_rating_count'] = df.groupby('reviewerID')['is_low_rating_episode'].cumsum().shift(1).fillna(0)
    
    # Drop temp columns
    df = df.drop(columns=['prev_episode_endpoint', 'is_low_rating_episode'])
    
    # ---------------------------------------------------------
    # Gate 9.2: Temporal Split
    # ---------------------------------------------------------
    print("\n[Gate 9.2] Computing Cutoff Dates for Temporal Split (70-15-15)...")
    
    q_70 = df['episode_start'].quantile(0.70)
    q_85 = df['episode_start'].quantile(0.85)
    
    print(f"Train Cutoff: <= {q_70.date()}")
    print(f"Val Cutoff: {q_70.date()} to {q_85.date()}")
    print(f"Test Cutoff: > {q_85.date()}")
    
    df_train = df[df['episode_start'] <= q_70].copy()
    df_val = df[(df['episode_start'] > q_70) & (df['episode_start'] <= q_85)].copy()
    df_test = df[df['episode_start'] > q_85].copy()
    
    print(f"Train size: {len(df_train):,} ({len(df_train)/len(df):.1%})")
    print(f"Val size: {len(df_val):,} ({len(df_val)/len(df):.1%})")
    print(f"Test size: {len(df_test):,} ({len(df_test)/len(df):.1%})")
    
    # Save Split Config
    config = {
        "split_dates": {
            "train_end": str(q_70),
            "val_end": str(q_85)
        },
        "duration_clip": 270,
        "feature_version": "Gate9_v1",
        "seed": 42
    }
    with open(gate9_dir / "experiment_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    # Save Datasets
    print("\nSaving splits...")
    df_train.to_parquet(gate9_dir / "baseline_train.parquet", index=False)
    df_val.to_parquet(gate9_dir / "baseline_val.parquet", index=False)
    df_test.to_parquet(gate9_dir / "baseline_test.parquet", index=False)
    
    print("Gate 9.1 Data Prep completed.")

if __name__ == "__main__":
    main()
