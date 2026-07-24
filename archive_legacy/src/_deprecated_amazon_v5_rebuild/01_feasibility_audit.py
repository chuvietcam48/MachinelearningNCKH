import gzip
import json
import pandas as pd
from datetime import datetime
from src.amazon_v5_rebuild.utils import setup_directories, apply_14_day_episode_rule

def run_feasibility_audit(data_path="data/amazon_reviews/CDs_and_Vinyl.json.gz"):
    out_dir = setup_directories()
    out_feas = out_dir / "feasibility"
    
    print("Starting feasibility audit cell extraction...")
    rows = []
    with gzip.open(data_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                if rec.get("verified", False) == True:
                    rows.append({
                        "reviewerID": rec.get("reviewerID"),
                        "unixReviewTime": rec.get("unixReviewTime"),
                        "overall": rec.get("overall")
                    })
            except:
                pass

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["reviewerID", "unixReviewTime", "overall"])
    df["date"] = pd.to_datetime(df["unixReviewTime"], unit="s")
    df = df.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)

    T_max = df["date"].max()
    print(f"Verified rows loaded: {len(df)}")

    # Episode Logic
    df = apply_14_day_episode_rule(df)

    episodes = df.groupby(['reviewerID', 'episode_id']).agg(
        episode_end=('date', 'max'),
        episode_min_rating=('overall', 'min')
    ).reset_index()

    episodes = episodes.sort_values(['reviewerID', 'episode_end']).reset_index(drop=True)
    episodes['prior_verified_episode_count'] = episodes.groupby('reviewerID').cumcount()

    # H=270
    H = 270
    episodes['followup_time'] = (T_max - episodes['episode_end']).dt.days
    eligible_ep = episodes[episodes['followup_time'] >= H].copy()

    def cross_tabulate(prior_col, rating_col, prior_thresh, rating_thresh):
        ep_data = eligible_ep.copy()
        ep_data['is_prior'] = ep_data['prior_verified_episode_count'] >= prior_thresh
        ep_data['is_severe'] = ep_data['episode_min_rating'] <= rating_thresh
        
        counts = ep_data.groupby(['is_prior', 'is_severe']).size().reset_index(name='snapshot_count')
        all_combs = pd.DataFrame([(False, False), (False, True), (True, False), (True, True)], columns=['is_prior', 'is_severe'])
        merged = all_combs.merge(counts, on=['is_prior', 'is_severe'], how='left').fillna(0)
        merged['snapshot_count'] = merged['snapshot_count'].astype(int)
        return merged

    prim_matrix = cross_tabulate('prior_verified_episode_count', 'episode_min_rating', 3, 2)
    prim_matrix = prim_matrix.rename(columns={'is_prior': 'prior_verified_episode_count >=3', 'is_severe': 'episode_min_rating <=2'})
    
    sens_matrix = cross_tabulate('prior_verified_episode_count', 'episode_min_rating', 2, 3)
    sens_matrix = sens_matrix.rename(columns={'is_prior': 'prior_verified_episode_count >=2', 'is_severe': 'episode_min_rating <=3'})

    final_csv = pd.concat([prim_matrix, sens_matrix], axis=1)
    final_csv.to_csv(out_feas / "repeat_engagement_severe_feedback_cell_counts_v1.csv", index=False)
    print("Feasibility audit complete. Cells exported.")

if __name__ == "__main__":
    run_feasibility_audit()
