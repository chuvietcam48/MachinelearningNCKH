import hashlib
import numpy as np
import pandas as pd
from pathlib import Path

def setup_directories(base_dir="outputs/amazon_v5_rebuild"):
    """Creates the standard directory structure for the V5 rebuild pipeline."""
    dirs = ["data", "governance", "reports", "annotation", "feasibility"]
    for d in dirs:
        Path(f"{base_dir}/{d}").mkdir(parents=True, exist_ok=True)
    return Path(base_dir)

def get_sha256(file_path):
    """Computes the SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            h.update(byte_block)
    return h.hexdigest()

def apply_14_day_episode_rule(df, reviewer_col='reviewerID', date_col='unixReviewTime'):
    """
    Applies the strict 14-day episode rule:
    - Sorts by reviewer and date
    - Starts episode at t0
    - Any review <= t0 + 14 days stays in the episode
    - > t0 + 14 days starts a new episode
    
    Expects df to be sorted by (reviewer_col, date_col).
    date_col must be in seconds.
    """
    reviewer_ids = df[reviewer_col].values
    dates = df[date_col].values
    
    episode_ids = np.zeros(len(df), dtype=np.int32)
    ep_starts = np.zeros(len(df), dtype=np.int64)

    if len(df) == 0:
        return df

    curr_rev = reviewer_ids[0]
    ep_start = dates[0]
    curr_ep = 0
    episode_ids[0] = 0
    ep_starts[0] = ep_start

    fourteen_days_sec = 14 * 24 * 3600

    for i in range(1, len(df)):
        if reviewer_ids[i] != curr_rev:
            curr_rev = reviewer_ids[i]
            ep_start = dates[i]
            curr_ep += 1
        else:
            if (dates[i] - ep_start) > fourteen_days_sec:
                ep_start = dates[i]
                curr_ep += 1
        episode_ids[i] = curr_ep
        ep_starts[i] = ep_start
        
    df['episode_id'] = episode_ids
    df['episode_start_unix'] = ep_starts
    df['episode_start'] = pd.to_datetime(ep_starts, unit='s')
    
    return df
