import pandas as pd
import numpy as np
from src.amazon_v5_rebuild.utils import setup_directories

def run_phase2a():
    out_dir = setup_directories()
    out_ann = out_dir / "annotation"
    
    print("Generating Samples for Phase 2A...")
    mem_path = out_dir / "data" / "verified_episode_review_membership_v1.parquet"
    if not mem_path.exists():
        print("Run Phase 1 first.")
        return

    df = pd.read_parquet(mem_path)
    df['text_len'] = df['reviewText'].astype(str).str.len()
    df_valid = df[df['text_len'] > 10].copy()

    np.random.seed(42)
    unique_reviewers = df_valid['reviewerID'].unique()
    np.random.shuffle(unique_reviewers)

    calib_reviewers = unique_reviewers[:5000]
    blind_reviewers = unique_reviewers[5000:10000]

    df_calib_pool = df_valid[df_valid['reviewerID'].isin(calib_reviewers)].copy()
    df_blind_pool = df_valid[df_valid['reviewerID'].isin(blind_reviewers)].copy()

    def stratified_sample(df_pool, n):
        df_pool['rating_strata'] = df_pool['overall'].astype(int)
        samples = []
        per_strata = max(1, n // 5)
        for r in range(1, 6):
            subset = df_pool[df_pool['rating_strata'] == r]
            if len(subset) > per_strata:
                samples.append(subset.sample(n=per_strata, random_state=42))
            else:
                samples.append(subset)
        sampled = pd.concat(samples)
        if len(sampled) < n:
            remainder = df_pool[~df_pool.index.isin(sampled.index)]
            sampled = pd.concat([sampled, remainder.sample(n=n-len(sampled), random_state=42)])
        return sampled.head(n)

    calib_sample = stratified_sample(df_calib_pool, 120)
    blind_sample = stratified_sample(df_blind_pool, 200)

    calib_sample.to_csv(out_ann / "calibration_sample_manifest_v1.csv", index=False)
    blind_sample.to_csv(out_ann / "blind_gold_sample_manifest_v1.csv", index=False)

    template_cols = [
        "reviewerID", "episode_id", "raw_source_row_id", "reviewText", "summary", "overall",
        "annotator_id", "annotation_round",
        "adjudicated_aspect1", "adjudicated_polarity1", "adjudicated_evidence1",
        "adjudicated_aspect2", "adjudicated_polarity2", "adjudicated_evidence2",
        "adjudicated_mixed_flag", "adjudication_note"
    ]

    template_df = blind_sample.copy()
    for c in template_cols[6:]:
        template_df[c] = ""
    template_df[template_cols].to_csv(out_ann / "human_annotation_template_v1.csv", index=False)
    print("Phase 2A Annotation Samples generated.")

if __name__ == "__main__":
    run_phase2a()
