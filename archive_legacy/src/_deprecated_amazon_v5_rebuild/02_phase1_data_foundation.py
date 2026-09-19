import gzip
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from src.amazon_v5_rebuild.utils import setup_directories, get_sha256, apply_14_day_episode_rule

def run_phase1(data_path="data/amazon_reviews/CDs_and_Vinyl.json.gz"):
    out_dir = setup_directories()
    out_data = out_dir / "data"
    out_gov = out_dir / "governance"
    out_rep = out_dir / "reports"
    
    print("Hashing raw source...")
    raw_hash = get_sha256(data_path)
    with open(out_data / "raw_source_manifest_v1.json", "w") as f:
        json.dump({"file": data_path, "sha256": raw_hash, "timestamp": datetime.now(timezone.utc).isoformat()}, f, indent=2)

    print("Streaming raw data for Phase 1...")
    rows = []
    row_idx = 0
    with gzip.open(data_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            row_idx += 1
            if not line: continue
            try:
                rec = json.loads(line)
                if rec.get("verified", False) == True:
                    rows.append({
                        "raw_source_row_id": row_idx,
                        "reviewerID": rec.get("reviewerID"),
                        "unixReviewTime": rec.get("unixReviewTime"),
                        "overall": rec.get("overall"),
                        "asin": rec.get("asin"),
                        "reviewText": str(rec.get("reviewText", "")),
                        "summary": str(rec.get("summary", ""))
                    })
            except:
                pass

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["reviewerID", "unixReviewTime", "overall"])
    df["review_timestamp"] = pd.to_datetime(df["unixReviewTime"], unit="s")
    df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)
    T_max = df["review_timestamp"].max()

    print("Applying 14-day rule...")
    df = apply_14_day_episode_rule(df)

    endpoints = df.groupby(['reviewerID', 'episode_id'])['review_timestamp'].max().reset_index()
    endpoints.rename(columns={'review_timestamp': 'episode_endpoint'}, inplace=True)
    df = df.merge(endpoints, on=['reviewerID', 'episode_id'])

    membership_cols = ['reviewerID', 'episode_id', 'episode_start', 'episode_endpoint', 'review_timestamp', 'asin', 'overall', 'reviewText', 'summary', 'raw_source_row_id']
    df[membership_cols].to_parquet(out_data / "verified_episode_review_membership_v1.parquet", index=False)

    print("Aggregating episodes...")
    episodes = df.groupby(['reviewerID', 'episode_id']).agg(
        episode_start=('episode_start', 'first'),
        episode_endpoint=('episode_endpoint', 'first'),
        episode_min_rating=('overall', 'min'),
        episode_max_rating=('overall', 'max'),
        episode_mean_rating=('overall', 'mean'),
        episode_review_count=('overall', 'size')
    ).reset_index()
    episodes = episodes.sort_values(['reviewerID', 'episode_start']).reset_index(drop=True)

    episodes['prior_verified_episode_count'] = episodes.groupby('reviewerID').cumcount()
    episodes['prior_review_count'] = episodes.groupby('reviewerID')['episode_review_count'].cumsum().shift(1).fillna(0).astype(int)

    grouped = episodes.groupby('reviewerID')
    cum_sum = grouped['episode_mean_rating'].transform(lambda x: (x * episodes.loc[x.index, 'episode_review_count']).cumsum().shift(1))
    cum_count = grouped['episode_review_count'].transform(lambda x: x.cumsum().shift(1))
    episodes['historical_mean_rating'] = cum_sum / cum_count

    episodes.to_parquet(out_data / "verified_review_feedback_episodes_v1.parquet", index=False)

    print("Building H=270 snapshots...")
    H = 270
    episodes['followup_time'] = (T_max - episodes['episode_endpoint']).dt.days
    snapshots = episodes[episodes['followup_time'] >= H].copy()

    episodes['next_episode_start'] = episodes.groupby('reviewerID')['episode_start'].shift(-1)
    snap_next_start = episodes.loc[snapshots.index, 'next_episode_start']
    snapshots['next_episode_start'] = snap_next_start

    snapshots['days_to_next'] = (snapshots['next_episode_start'] - snapshots['episode_endpoint']).dt.days
    snapshots['Y_dormant_270'] = ((snapshots['days_to_next'].isna()) | (snapshots['days_to_next'] > 270)).astype(int)
    snapshots = snapshots.drop(columns=['followup_time', 'days_to_next'])
    snapshots.to_parquet(out_data / "verified_episode_snapshots_h270_v1.parquet", index=False)

    print("Phase 1 Data Foundation built.")

if __name__ == "__main__":
    run_phase1()
