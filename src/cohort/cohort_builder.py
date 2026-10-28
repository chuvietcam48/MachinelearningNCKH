"""
src/study4/cohort_builder.py
===========================
Clean Cohort Construction for Study 4 (Master Plan Final Section 8.1).
Replaces the contaminated legacy DSIT cohort:
1. Episode grouping: reviews of same customer < 14 days apart are grouped into an episode.
2. Outcome-blind eligibility:
   - Must have >= 1 prior episode before episode endpoint f_i.
   - Follow-up window: dataset_max_date - f_i >= 270 days.
   - NO FILTERING on future episodes (fixes DSIT conditioning on outcome).
3. Event & Duration:
   - Next episode within <= 270 days -> event = 1, T = days to next episode (>= 14).
   - Censored -> event = 0, T = 271.
4. Causal behavioral features computed strictly at f_i.
5. Chronological 60/20/20 train/val/test split by f_i.
"""

import gzip
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cohort_builder")


def stream_amazon_reviews(json_gz_path: str, max_reviews: Optional[int] = None) -> pd.DataFrame:
    """Stream raw review records from Amazon json.gz file."""
    logger.info(f"Streaming reviews from: {json_gz_path}")
    records = []
    with gzip.open(json_gz_path, "rt", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_reviews is not None and idx >= max_reviews:
                break
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                reviewer_id = obj.get("reviewerID")
                unix_time = obj.get("unixReviewTime")
                rating = obj.get("overall")
                verified = obj.get("verified", False)
                text = obj.get("reviewText", "")
                summary = obj.get("summary", "")
                
                if reviewer_id and unix_time is not None and rating is not None:
                    records.append({
                        "reviewer_id": str(reviewer_id),
                        "unix_time": int(unix_time),
                        "date": pd.to_datetime(int(unix_time), unit="s"),
                        "rating": float(rating),
                        "verified": bool(verified),
                        "review_text": str(text) if text else "",
                        "summary": str(summary) if summary else "",
                    })
            except Exception:
                continue
                
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} valid review rows across {df['reviewer_id'].nunique()} unique customers.")
    return df


def build_customer_episodes(df_reviews: pd.DataFrame, episode_gap_days: int = 14) -> List[Dict[str, Any]]:
    """Group consecutive reviews by the same customer separated by < 14 days into episodes."""
    logger.info("Grouping reviews into customer episodes (< 14 days gap)...")
    episodes = []
    
    # Sort reviews by reviewer and timestamp
    df_sorted = df_reviews.sort_values(["reviewer_id", "unix_time"]).reset_index(drop=True)
    
    for reviewer_id, group in df_sorted.groupby("reviewer_id"):
        dates = group["date"].values
        ratings = group["rating"].values
        verifieds = group["verified"].values
        texts = group["review_text"].values
        summaries = group["summary"].values
        
        n_rev = len(dates)
        if n_rev == 0:
            continue
            
        current_ep_start = dates[0]
        current_ep_end = dates[0]
        current_ratings = [ratings[0]]
        current_verifieds = [verifieds[0]]
        current_texts = [texts[0]]
        current_summaries = [summaries[0]]
        
        cust_episodes = []
        
        for i in range(1, n_rev):
            # Anchored window rule matching verified_episode_snapshots:
            # Reviews within 14 days of current_ep_start belong to the same episode
            time_from_start = (dates[i] - current_ep_start) / np.timedelta64(1, "D")
            if time_from_start <= episode_gap_days:
                # Same episode
                current_ep_end = dates[i]
                current_ratings.append(ratings[i])
                current_verifieds.append(verifieds[i])
                current_texts.append(texts[i])
                current_summaries.append(summaries[i])
            else:
                # Close current episode
                cust_episodes.append({
                    "reviewer_id": reviewer_id,
                    "ep_start": pd.to_datetime(current_ep_start),
                    "ep_end": pd.to_datetime(current_ep_end),
                    "ratings": current_ratings,
                    "verifieds": current_verifieds,
                    "texts": current_texts,
                    "summaries": current_summaries,
                })
                # Start new episode anchored at dates[i]
                current_ep_start = dates[i]
                current_ep_end = dates[i]
                current_ratings = [ratings[i]]
                current_verifieds = [verifieds[i]]
                current_texts = [texts[i]]
                current_summaries = [summaries[i]]
                
        # Append final episode
        cust_episodes.append({
            "reviewer_id": reviewer_id,
            "ep_start": pd.to_datetime(current_ep_start),
            "ep_end": pd.to_datetime(current_ep_end),
            "ratings": current_ratings,
            "verifieds": current_verifieds,
            "texts": current_texts,
            "summaries": current_summaries,
        })
        
        episodes.extend(cust_episodes)
        
    logger.info(f"Built {len(episodes)} total customer episodes.")
    return episodes


def extract_clean_cohort(
    episodes: List[Dict[str, Any]],
    dataset_max_date: pd.Timestamp,
    followup_window_days: int = 270,
    sample_size: Optional[int] = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Extract clean outcome-blind cohort adhering to Master Plan Section 8.1:
    - Must have >= 1 prior episode before f_i (endpoint of index episode).
    - Follow-up window: dataset_max_date - f_i >= 270 days.
    - NO FILTERING on future episodes.
    - Outcome: next episode in <= 270 days -> event=1, T=gap; censored -> event=0, T=271.
    """
    logger.info("Filtering eligible episodes (>=1 prior episode & >=270d follow-up window)...")
    
    # Organize episodes by reviewer
    from collections import defaultdict
    by_reviewer = defaultdict(list)
    for ep in episodes:
        by_reviewer[ep["reviewer_id"]].append(ep)
        
    for r in by_reviewer:
        by_reviewer[r].sort(key=lambda x: x["ep_end"])
        
    eligible_rows = []
    
    for r, eps in by_reviewer.items():
        n_eps = len(eps)
        if n_eps < 2:
            continue
            
        for ep_idx in range(1, n_eps):
            index_ep = eps[ep_idx]
            f_i = index_ep["ep_end"]
            
            # Check follow-up window: dataset_max_date - f_i >= 270
            remaining_days = (dataset_max_date - f_i).total_seconds() / 86400.0
            if remaining_days < followup_window_days:
                continue  # Not enough follow-up window
                
            # Outcome evaluation: look at next episode (if any)
            if ep_idx + 1 < n_eps:
                next_ep = eps[ep_idx + 1]
                gap_to_next = (next_ep["ep_start"] - f_i).total_seconds() / 86400.0
                if gap_to_next <= followup_window_days:
                    event = 1  # Re-engaged within 270 days
                    duration = max(14, int(np.round(gap_to_next)))
                else:
                    event = 0  # Censored (next episode occurred after 270 days)
                    duration = 271
            else:
                event = 0  # Censored (no next episode observed)
                duration = 271
                
            # Causal behavioral features strictly at f_i
            prior_eps = eps[:ep_idx]
            days_since_prev = (index_ep["ep_start"] - prior_eps[-1]["ep_end"]).total_seconds() / 86400.0
            
            # Customer lifetime
            cust_first_d = prior_eps[0]["ep_start"]
            cust_lifetime_days = max(1.0, (f_i - cust_first_d).total_seconds() / 86400.0)
            
            # Counts
            prior_rev_count = sum(len(p["ratings"]) for p in prior_eps)
            prior_verified_ep_count = sum(1 for p in prior_eps if any(p["verifieds"]))
            
            # Ratings history
            all_prior_ratings = [rt for p in prior_eps for rt in p["ratings"]]
            hist_mean_rating = float(np.mean(all_prior_ratings)) if all_prior_ratings else 3.0
            
            # Index episode features
            ep_duration = max(0.0, (index_ep["ep_end"] - index_ep["ep_start"]).total_seconds() / 86400.0)
            ep_ratings = index_ep["ratings"]
            ep_min_rating = float(np.min(ep_ratings))
            ep_max_rating = float(np.max(ep_ratings))
            ep_mean_rating = float(np.mean(ep_ratings))
            ep_rev_count = len(ep_ratings)
            
            # Review frequency (reviews per 30 days)
            total_revs_to_date = prior_rev_count + ep_rev_count
            review_freq = float(total_revs_to_date / (cust_lifetime_days / 30.0))
            
            # Recent low ratings (in last 90 days before f_i)
            # Checked across all prior reviews
            recent_low_count = sum(1 for p in prior_eps for i, rt in enumerate(p["ratings"]) if rt <= 2.0 and (f_i - p["ep_end"]).total_seconds() / 86400.0 <= 90.0)
            if ep_min_rating <= 2.0:
                recent_low_count += sum(1 for rt in ep_ratings if rt <= 2.0)
                
            eligible_rows.append({
                "reviewer_id": r,
                "episode_idx": ep_idx,
                "f_i": f_i,
                "event": event,
                "duration": duration,
                "churn_270": int(1 - event),
                # 12 Behavioral features (Model A)
                "episode_duration": ep_duration,
                "days_since_previous_episode": days_since_prev,
                "prior_verified_episode_count": prior_verified_ep_count,
                "prior_review_count": prior_rev_count,
                "review_frequency": review_freq,
                "episode_min_rating": ep_min_rating,
                "episode_max_rating": ep_max_rating,
                "episode_mean_rating": ep_mean_rating,
                "episode_review_count": ep_rev_count,
                "historical_mean_rating": hist_mean_rating,
                "recent_low_rating_count": recent_low_count,
                "customer_lifetime": cust_lifetime_days,
                # Texts for LLM extraction
                "episode_texts": " \n ".join(index_ep["texts"]),
                "episode_summaries": " \n ".join(index_ep["summaries"]),
            })
            
    df_clean = pd.DataFrame(eligible_rows)
    logger.info(f"Total eligible clean episodes: {len(df_clean)}")
    
    if len(df_clean) == 0:
        return df_clean
        
    logger.info(f"Re-engagement event rate: {df_clean['event'].mean():.4f} (Churn rate = {df_clean['churn_270'].mean():.4f})")
    
    # Outcome-blind sampling if sample_size specified
    if sample_size is not None and len(df_clean) > sample_size:
        logger.info(f"Outcome-blind random sampling {sample_size} episodes (seed={seed})...")
        df_clean = df_clean.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        
    # Chronological 60/20/20 train/val/test split by f_i
    df_clean = df_clean.sort_values("f_i").reset_index(drop=True)
    n = len(df_clean)
    n_train = int(0.60 * n)
    n_val = int(0.20 * n)
    
    df_clean["split"] = "train"
    df_clean.loc[n_train:n_train + n_val - 1, "split"] = "val"
    df_clean.loc[n_train + n_val:, "split"] = "test"
    
    split_counts = df_clean["split"].value_counts().to_dict()
    logger.info(f"Split distribution: {split_counts}")
    for sp in ["train", "val", "test"]:
        sub = df_clean[df_clean["split"] == sp]
        logger.info(f"  {sp.upper()} ({len(sub)} eps): event_rate={sub['event'].mean():.4f}, date range: {sub['f_i'].min().date()} to {sub['f_i'].max().date()}")
        
    return df_clean
