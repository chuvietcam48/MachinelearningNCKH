import pandas as pd
import numpy as np
import json
from pathlib import Path
import scipy.stats as stats

ASPECTS = [
    "Domain_Experience",
    "Product_Performance_Usability",
    "Quality_Reliability",
    "Price_Value",
    "Delivery_Packaging",
    "Customer_Service_Support",
    "Brand_Trust_Loyalty",
    "Features_Design"
]
ASPECT_TO_ID = {aspect: idx for idx, aspect in enumerate(ASPECTS)}
ASPECT_TO_ID["Unknown"] = -1

def parse_semantic_targets(row):
    targets_str = row['final_targets']
    if not isinstance(targets_str, str) or targets_str == '[]':
        return []
    try:
        return json.loads(targets_str)
    except:
        return []

def compute_episode_features(df):
    print("Computing episode-level semantic features...")
    
    # Pre-allocate lists for fast assignment
    presence_feats = {f"{asp}_{pol}": [] for asp in ASPECTS for pol in ["Positive", "Negative"]}
    
    pos_counts, neg_counts = [], []
    dom_pos_ids, dom_neg_ids = [], []
    same_aspect_conflicts, cross_aspect_conflicts = [], []
    aspect_entropies, dominance_ratios = [], []
    sentiment_profiles = []
    semantic_variances = []
    
    for idx, row in df.iterrows():
        targets = parse_semantic_targets(row)
        
        # Track presence
        asp_counts = {asp: {'Positive': 0, 'Negative': 0} for asp in ASPECTS}
        
        pos_c, neg_c = 0, 0
        polarities_val = []
        
        for t in targets:
            asp = t.get('aspect', 'Unknown')
            pol = t.get('polarity', 'Unknown')
            if asp in asp_counts and pol in ['Positive', 'Negative']:
                asp_counts[asp][pol] += 1
                if pol == 'Positive': 
                    pos_c += 1
                    polarities_val.append(1)
                else: 
                    neg_c += 1
                    polarities_val.append(-1)
                    
        pos_counts.append(pos_c)
        neg_counts.append(neg_c)
        
        # Presence features
        for asp in ASPECTS:
            presence_feats[f"{asp}_Positive"].append(1 if asp_counts[asp]['Positive'] > 0 else 0)
            presence_feats[f"{asp}_Negative"].append(1 if asp_counts[asp]['Negative'] > 0 else 0)
            
        # Conflict features
        same_conflict = 0
        for asp in ASPECTS:
            if asp_counts[asp]['Positive'] > 0 and asp_counts[asp]['Negative'] > 0:
                same_conflict += 1
        same_aspect_conflicts.append(same_conflict)
        cross_aspect_conflicts.append(1 if (pos_c > 0 and neg_c > 0 and same_conflict == 0) else 0)
        
        # Profile
        if pos_c > 0 and neg_c == 0: profile = 1  # Pos only
        elif neg_c > 0 and pos_c == 0: profile = 2  # Neg only
        elif pos_c > 0 and neg_c > 0: profile = 3  # Mixed
        else: profile = 0  # Neutral
        sentiment_profiles.append(profile)
        
        # Variance
        semantic_variances.append(np.var(polarities_val) if len(polarities_val) > 0 else 0.0)
        
        # Dominant Aspects (first encountered for simplicity, or most frequent if we counted)
        # We find the aspect with max Pos/Neg
        dom_pos_id, dom_neg_id = -1, -1
        max_pos, max_neg = 0, 0
        asp_freqs = []
        
        for asp in ASPECTS:
            p_cnt = asp_counts[asp]['Positive']
            n_cnt = asp_counts[asp]['Negative']
            total_cnt = p_cnt + n_cnt
            if total_cnt > 0:
                asp_freqs.append(total_cnt)
            if p_cnt > max_pos:
                max_pos = p_cnt
                dom_pos_id = ASPECT_TO_ID[asp]
            if n_cnt > max_neg:
                max_neg = n_cnt
                dom_neg_id = ASPECT_TO_ID[asp]
                
        dom_pos_ids.append(dom_pos_id)
        dom_neg_ids.append(dom_neg_id)
        
        # Entropy & Dominance
        if len(asp_freqs) > 0:
            probs = np.array(asp_freqs) / sum(asp_freqs)
            aspect_entropies.append(stats.entropy(probs))
            dominance_ratios.append(max(asp_freqs) / sum(asp_freqs))
        else:
            aspect_entropies.append(0.0)
            dominance_ratios.append(0.0)

    # Assign to dataframe
    for k, v in presence_feats.items():
        df[k] = v
        
    df['positive_aspect_count'] = pos_counts
    df['negative_aspect_count'] = neg_counts
    df['net_sentiment'] = df['positive_aspect_count'] - df['negative_aspect_count']
    df['positive_minus_negative_ratio'] = df['net_sentiment'] / (df['positive_aspect_count'] + df['negative_aspect_count'] + 1e-9)
    
    total_aspects = df['positive_aspect_count'] + df['negative_aspect_count']
    df['positive_ratio'] = df['positive_aspect_count'] / (total_aspects + 1e-9)
    df['negative_ratio'] = df['negative_aspect_count'] / (total_aspects + 1e-9)
    
    df['unique_aspect_count'] = df[[f"{asp}_Positive" for asp in ASPECTS] + [f"{asp}_Negative" for asp in ASPECTS]].sum(axis=1)
    df['aspect_entropy'] = aspect_entropies
    df['dominance_ratio'] = dominance_ratios
    
    df['dominant_positive_aspect_id'] = dom_pos_ids
    df['dominant_negative_aspect_id'] = dom_neg_ids
    
    df['sentiment_profile'] = sentiment_profiles
    df['same_aspect_conflict'] = same_aspect_conflicts
    df['cross_aspect_conflict'] = cross_aspect_conflicts
    df['has_conflict'] = ((df['same_aspect_conflict'] > 0) | (df['cross_aspect_conflict'] > 0)).astype(int)
    
    df['semantic_variance'] = semantic_variances
    
    return df

def compute_historical_features(df):
    print("Computing historical trajectory features (t-1 strictly) using fast single-pass arrays...")
    # Sort chronologically
    if 'episode_start' in df.columns:
        df = df.sort_values(by=['reviewerID', 'episode_start', 'episode_id']).reset_index(drop=True)
    else:
        df = df.sort_values(by=['reviewerID', 'episode_id']).reset_index(drop=True)

    # 1. Rolling counts and ratios (t-1)
    df['previous_negative_count'] = df.groupby('reviewerID')['negative_aspect_count'].cumsum().shift(1).fillna(0)
    
    total_aspects = df['positive_aspect_count'] + df['negative_aspect_count']
    df['previous_total_aspects'] = total_aspects.groupby(df['reviewerID']).cumsum().shift(1).fillna(0)
    df['rolling_negative_ratio'] = df['previous_negative_count'] / (df['previous_total_aspects'] + 1e-9)
    
    # 2. Fast Single-Pass for Stateful Features
    reviewer_ids = df['reviewerID'].values
    profiles = df['sentiment_profile'].values
    neg_asp_ids = df['dominant_negative_aspect_id'].values
    
    has_time = 'episode_start' in df.columns
    if has_time:
        # Convert to unix seconds for fast diff
        times = df['episode_start'].astype('int64').values // 10**9

    n = len(df)
    
    # Outputs
    out_neg_streaks = np.zeros(n, dtype=int)
    out_pos_streaks = np.zeros(n, dtype=int)
    out_last_neg_asps = np.full(n, -1, dtype=int)
    out_flip_rates = np.zeros(n, dtype=float)
    out_switch_rates = np.zeros(n, dtype=float)
    out_days_since = np.full(n, -1.0, dtype=float)
    out_transitions = np.full(n, -1, dtype=int)
    
    # State variables
    cur_reviewer = None
    neg_streak = 0
    pos_streak = 0
    last_neg_asp = -1
    flips = 0
    switches = 0
    last_profile = 0
    neg_episodes = 0
    last_neg_time = -1
    ep_index = 0
    
    for i in range(n):
        rev = reviewer_ids[i]
        prof = profiles[i]
        neg_asp = neg_asp_ids[i]
        if has_time:
            t = times[i]
            
        if rev != cur_reviewer:
            # Reset state for new reviewer
            cur_reviewer = rev
            neg_streak = 0
            pos_streak = 0
            last_neg_asp = -1
            flips = 0
            switches = 0
            last_profile = 0
            neg_episodes = 0
            last_neg_time = -1
            ep_index = 0
            
        # 1. Output state (t-1)
        out_neg_streaks[i] = neg_streak
        out_pos_streaks[i] = pos_streak
        out_last_neg_asps[i] = last_neg_asp
        
        out_flip_rates[i] = flips / max(1, ep_index)
        out_switch_rates[i] = switches / max(1, neg_episodes - 1) if neg_episodes > 1 else 0.0
        
        if last_neg_asp != -1 and neg_asp != -1:
            out_transitions[i] = last_neg_asp * 100 + neg_asp
            
        if has_time:
            if last_neg_time != -1:
                out_days_since[i] = (t - last_neg_time) / (24 * 3600.0)
            
        # 2. Update state for next step (t)
        ep_index += 1
        
        if prof in [2, 3]: # Has negative
            neg_streak += 1
            pos_streak = 0
            neg_episodes += 1
            if has_time:
                last_neg_time = t
        elif prof == 1: # Pos only
            pos_streak += 1
            neg_streak = 0
        # If prof == 0, we implement Semantic State Persistence (Carry Forward).
        # We do not reset pos_streak or neg_streak to 0.
            
        if last_profile != 0 and prof != 0 and last_profile != prof:
            flips += 1
        if prof != 0:
            last_profile = prof
            
        if neg_asp != -1:
            if last_neg_asp != -1 and neg_asp != last_neg_asp:
                switches += 1
            last_neg_asp = neg_asp

    df['negative_streak'] = out_neg_streaks
    df['positive_streak'] = out_pos_streaks
    df['last_negative_aspect_id'] = out_last_neg_asps
    df['sentiment_flip_rate'] = out_flip_rates
    df['aspect_switch_rate'] = out_switch_rates
    df['aspect_transition_pattern'] = out_transitions
    if has_time:
        df['days_since_last_negative'] = out_days_since
        
    return df

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    features_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "features"
    input_path = features_dir / "episode_semantic_merged.parquet"
    output_path = features_dir / "episode_semantic_engineered.parquet"

    print("Gate 8.2: Semantic Feature Engineering")
    print("-" * 30)

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run Gate 8.1 first.")
        return

    df = pd.read_parquet(input_path)
    
    # We only compute heavy features where semantic_available == 1 to save time,
    # but for historical features, we need the full timeline.
    # Since semantic_available == 0 means no semantic labels, those rows will have neutral/0 profiles natively.
    # We can process the whole DataFrame. Missing semantic targets are already '[]'.
    
    df = compute_episode_features(df)
    df = compute_historical_features(df)

    print(f"Saving engineered features to {output_path}...")
    df.to_parquet(output_path, index=False)
    print("Gate 8.2 completed successfully.")

if __name__ == "__main__":
    main()
