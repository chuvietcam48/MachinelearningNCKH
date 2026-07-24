import logging
import pandas as pd
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)

ASPECTS = [
    "Domain_Experience",
    "Product_Performance_Usability",
    "Product_Condition_Quality",
    "Packaging_Presentation",
    "Delivery_Fulfillment",
    "Customer_Service_Returns",
    "Listing_Expectation_Compatibility",
    "Price_Value",
]
ASPECT_TO_ID = {aspect: idx for idx, aspect in enumerate(ASPECTS)}
ASPECT_TO_ID["Unknown"] = -1

import json

def parse_semantic_targets(row):
    targets_str = row['final_targets']
    if pd.isna(targets_str) or not isinstance(targets_str, str) or targets_str == '[]':
        return []
    try:
        return json.loads(targets_str)
    except:
        return []

class SemanticEngine:
    """
    Layer 2: Semantic Extension Engine
    This module encapsulates the state-aware semantic extraction logic.
    It computes aspect presence, aspect transitions, and conflict signals.
    """
    def __init__(self):
        pass

    def compute_episode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing episode-level semantic features...")
        presence_feats = {f"{asp}_{pol}": [] for asp in ASPECTS for pol in ["Positive", "Negative"]}
        pos_counts, neg_counts, dom_pos_ids, dom_neg_ids = [], [], [], []
        same_aspect_conflicts, cross_aspect_conflicts = [], []
        aspect_entropies, dominance_ratios, sentiment_profiles, semantic_variances = [], [], [], []
        
        for _, row in df.iterrows():
            targets = parse_semantic_targets(row)
            asp_counts = {asp: {'Positive': 0, 'Negative': 0} for asp in ASPECTS}
            pos_c, neg_c = 0, 0
            polarities_val = []
            
            for t in targets:
                asp = t.get('aspect', 'Unknown')
                pol = t.get('polarity', 'Unknown')
                if asp in asp_counts and pol in ['Positive', 'Negative']:
                    asp_counts[asp][pol] += 1
                    if pol == 'Positive': 
                        pos_c += 1; polarities_val.append(1)
                    else: 
                        neg_c += 1; polarities_val.append(-1)
                        
            pos_counts.append(pos_c)
            neg_counts.append(neg_c)
            
            for asp in ASPECTS:
                presence_feats[f"{asp}_Positive"].append(1 if asp_counts[asp]['Positive'] > 0 else 0)
                presence_feats[f"{asp}_Negative"].append(1 if asp_counts[asp]['Negative'] > 0 else 0)
                
            same_conflict = sum(1 for asp in ASPECTS if asp_counts[asp]['Positive'] > 0 and asp_counts[asp]['Negative'] > 0)
            same_aspect_conflicts.append(same_conflict)
            cross_aspect_conflicts.append(1 if (pos_c > 0 and neg_c > 0 and same_conflict == 0) else 0)
            
            if pos_c > 0 and neg_c == 0: profile = 1
            elif neg_c > 0 and pos_c == 0: profile = 2
            elif pos_c > 0 and neg_c > 0: profile = 3
            else: profile = 0
            sentiment_profiles.append(profile)
            
            semantic_variances.append(np.var(polarities_val) if polarities_val else 0.0)
            
            dom_pos_id, dom_neg_id = -1, -1
            max_pos, max_neg = 0, 0
            asp_freqs = []
            
            for asp in ASPECTS:
                p_cnt, n_cnt = asp_counts[asp]['Positive'], asp_counts[asp]['Negative']
                if p_cnt + n_cnt > 0: asp_freqs.append(p_cnt + n_cnt)
                if p_cnt > max_pos: max_pos = p_cnt; dom_pos_id = ASPECT_TO_ID[asp]
                if n_cnt > max_neg: max_neg = n_cnt; dom_neg_id = ASPECT_TO_ID[asp]
                    
            dom_pos_ids.append(dom_pos_id)
            dom_neg_ids.append(dom_neg_id)
            
            if asp_freqs:
                probs = np.array(asp_freqs) / sum(asp_freqs)
                aspect_entropies.append(stats.entropy(probs))
                dominance_ratios.append(max(asp_freqs) / sum(asp_freqs))
            else:
                aspect_entropies.append(0.0)
                dominance_ratios.append(0.0)

        for k, v in presence_feats.items(): df[k] = v
        df['positive_aspect_count'] = pos_counts
        df['negative_aspect_count'] = neg_counts
        df['net_sentiment'] = df['positive_aspect_count'] - df['negative_aspect_count']
        total_aspects = df['positive_aspect_count'] + df['negative_aspect_count']
        df['positive_minus_negative_ratio'] = df['net_sentiment'] / (total_aspects + 1e-9)
        df['positive_ratio'] = df['positive_aspect_count'] / (total_aspects + 1e-9)
        df['negative_ratio'] = df['negative_aspect_count'] / (total_aspects + 1e-9)
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

    def compute_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Computing historical trajectory features using state persistence...")
        has_time = 'episode_start' in df.columns
        if has_time:
            df = df.sort_values(by=['CustomerID', 'episode_start', 'episode_id']).reset_index(drop=True)
            times = df['episode_start'].astype('int64').values // 10**9
        else:
            df = df.sort_values(by=['CustomerID', 'episode_id']).reset_index(drop=True)

        df['previous_negative_count'] = (
            df.groupby('CustomerID')['negative_aspect_count']
            .transform(lambda s: s.cumsum().shift(1))
            .fillna(0)
        )
        total_aspects = df['positive_aspect_count'] + df['negative_aspect_count']
        df['previous_total_aspects'] = (
            total_aspects.groupby(df['CustomerID'])
            .transform(lambda s: s.cumsum().shift(1))
            .fillna(0)
        )
        df['rolling_negative_ratio'] = df['previous_negative_count'] / (df['previous_total_aspects'] + 1e-9)
        
        n = len(df)
        out_neg_streaks = np.zeros(n, dtype=int)
        out_pos_streaks = np.zeros(n, dtype=int)
        out_last_neg_asps = np.full(n, -1, dtype=int)
        out_flip_rates = np.zeros(n, dtype=float)
        out_switch_rates = np.zeros(n, dtype=float)
        out_days_since = np.full(n, -1.0, dtype=float)
        out_transitions = np.full(n, -1, dtype=int)
        
        cur_reviewer = None
        neg_streak, pos_streak = 0, 0
        last_neg_asp = -1
        flips, switches, last_profile = 0, 0, 0
        neg_episodes, last_neg_time, ep_index = 0, -1, 0
        
        rev_ids = df['CustomerID'].values
        profiles = df['sentiment_profile'].values
        neg_asp_ids = df['dominant_negative_aspect_id'].values
        
        for i in range(n):
            rev = rev_ids[i]
            prof = profiles[i]
            neg_asp = neg_asp_ids[i]
            if has_time: t = times[i]
                
            if rev != cur_reviewer:
                cur_reviewer = rev
                neg_streak, pos_streak = 0, 0
                last_neg_asp = -1
                flips, switches, last_profile = 0, 0, 0
                neg_episodes, last_neg_time, ep_index = 0, -1, 0
                
            out_neg_streaks[i] = neg_streak
            out_pos_streaks[i] = pos_streak
            out_last_neg_asps[i] = last_neg_asp
            
            out_flip_rates[i] = flips / max(1, ep_index)
            out_switch_rates[i] = switches / max(1, neg_episodes - 1) if neg_episodes > 1 else 0.0
            
            if last_neg_asp != -1 and neg_asp != -1:
                out_transitions[i] = last_neg_asp * 100 + neg_asp
                
            if has_time and last_neg_time != -1:
                out_days_since[i] = (t - last_neg_time) / (24 * 3600.0)
                
            ep_index += 1
            
            if prof in [2, 3]:
                neg_streak += 1
                pos_streak = 0
                neg_episodes += 1
                if has_time: last_neg_time = t
            elif prof == 1:
                pos_streak += 1
                neg_streak = 0
                
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
        if has_time: df['days_since_last_negative'] = out_days_since
        else: df['days_since_last_negative'] = -1.0
        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes semantic feature engineering.
        """
        logger.info("Executing Semantic Extension Engine...")
        if len(df) == 0:
            return df
        
        # Check if it has targets
        if 'final_targets' not in df.columns:
            logger.warning("No 'final_targets' column found. Returning untouched.")
            return df

        df = self.compute_episode_features(df)
        df = self.compute_historical_features(df)
        return df
