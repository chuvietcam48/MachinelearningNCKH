import pandas as pd
import numpy as np
from pathlib import Path
import json

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate7_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate7_pipeline"
    gate8_features_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "features"
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    
    print("Gate 8 & Semantic Data Audit")
    print("="*60)
    
    # 1. Audit Gate 7 (How many labels exist?)
    print("\n[Question 1] Gate 7 LLM Annotation Count:")
    # Look for the final llm results parquet in gate7_pipeline
    llm_files = list(gate7_dir.glob("llm_results_*.parquet"))
    total_labels = 0
    gate7_episodes = set()
    if not llm_files:
        print("  -> Could not find llm_results_*.parquet in gate7_pipeline")
    else:
        for f in llm_files:
            try:
                df = pd.read_parquet(f)
                total_labels += len(df)
                if 'episode_id' in df.columns:
                    gate7_episodes.update(df['episode_id'].unique())
            except Exception as e:
                pass
        print(f"  -> Found {len(llm_files)} annotation files.")
        print(f"  -> Total LLM labels generated: {total_labels}")
        print(f"  -> Total Unique Episodes with LLM labels: {len(gate7_episodes)}")
        
    # 2. Audit Gate 8.1 (Merge)
    print("\n[Question 2] Gate 8 Merge Check:")
    g8_merge = gate8_features_dir / "episode_semantic_merged.parquet"
    if not g8_merge.exists():
        print(f"  -> {g8_merge.name} not found.")
    else:
        df_g8 = pd.read_parquet(g8_merge)
        semantic_avail = df_g8['semantic_available'].sum() if 'semantic_available' in df_g8.columns else "Column not found"
        print(f"  -> Total episodes in Gate 8: {len(df_g8)}")
        print(f"  -> Episodes with semantic_available == 1: {semantic_avail}")
        if len(gate7_episodes) > 0:
            print(f"  -> Loss during merge: {len(gate7_episodes) - semantic_avail} episodes lost.")
            
    # 3. Audit Gate 8.2 (Feature Distributions on semantic_available == 1)
    print("\n[Question 3] Distributions when semantic_available == 1:")
    g8_engineered = gate8_features_dir / "episode_semantic_engineered.parquet"
    if not g8_engineered.exists():
        print(f"  -> {g8_engineered.name} not found.")
    else:
        df_eng = pd.read_parquet(g8_engineered)
        if 'semantic_available' in df_eng.columns:
            subset = df_eng[df_eng['semantic_available'] == 1]
            print(f"  -> Subset size: {len(subset)}")
            
            check_feats = ['negative_streak', 'has_conflict', 'net_sentiment', 'previous_negative_count']
            print("  -> Zeros % on THIS subset:")
            for f in check_feats:
                if f in subset.columns:
                    pct_0 = (subset[f] == 0).mean() * 100
                    print(f"     * {f}: {pct_0:.1f}% zeros")
                    
            # 5. Cohort Trajectory check
            print("\n[Question 5] How many customers in semantic subset have >= 2 episodes?")
            if 'reviewerID' in subset.columns:
                counts = subset['reviewerID'].value_counts()
                multi_ep = (counts >= 2).sum()
                print(f"  -> Customers with >= 2 labeled episodes: {multi_ep} out of {len(counts)} total customers in subset.")
        else:
            print("  -> semantic_available column missing.")
            
    # 4. Audit Logic (Answered by code review)
    print("\n[Question 4] Was Trajectory calculated before or after missing fill?")
    print("  -> Code Review shows: Trajectory is calculated AFTER merge in Gate 8.2.")
    print("  -> Since missing semantic data means 'targets'='[]', the profile defaults to 0 (Neutral).")
    print("  -> If profile is 0, neg_streak and pos_streak are RESET TO 0.")
    print("  -> Therefore, any missing episode in the middle of a customer's history destroys their streak!")
    print("="*60)

if __name__ == "__main__":
    main()
