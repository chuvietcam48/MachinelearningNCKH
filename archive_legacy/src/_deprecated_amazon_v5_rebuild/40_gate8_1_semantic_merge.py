import pandas as pd
import json
from pathlib import Path

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "data"
    annotation_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "annotation" / "mass_workspace"
    features_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    episode_table_path = data_dir / "verified_episode_snapshots_h270_v1.parquet"
    qa_labels_path = annotation_dir / "mass_results_v224_qa.jsonl"
    output_path = features_dir / "episode_semantic_merged.parquet"

    print("Gate 8.1: Semantic Merge")
    print("-" * 30)

    # 1. Load Episode Table
    print(f"Loading episode table from {episode_table_path}...")
    df_episodes = pd.read_parquet(episode_table_path)
    print(f"Loaded {len(df_episodes):,} episodes.")

    # Create key for joining
    df_episodes['annotation_item_id'] = df_episodes['reviewerID'].astype(str) + '_' + df_episodes['episode_id'].astype(str)

    # 2. Load Semantic Labels
    print(f"Loading semantic labels from {qa_labels_path}...")
    labels = []
    if qa_labels_path.exists():
        with open(qa_labels_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                labels.append({
                    'annotation_item_id': data['annotation_item_id'],
                    'final_targets': json.dumps(data.get('final_targets', [])),
                    'Review_Mixed_Flag': data.get('Review_Mixed_Flag', False)
                })
    else:
        print(f"WARNING: {qa_labels_path} not found!")
    
    df_labels = pd.DataFrame(labels)
    print(f"Loaded {len(df_labels):,} semantic labels.")

    # 3. Merge
    print("Merging semantic labels into episode table...")
    df_merged = df_episodes.merge(df_labels, on='annotation_item_id', how='left')

    # Create semantic_available flag
    df_merged['semantic_available'] = df_merged['final_targets'].notna().astype(int)
    
    # Fill missing values for the merged columns
    df_merged['final_targets'] = df_merged['final_targets'].fillna('[]')
    df_merged['Review_Mixed_Flag'] = df_merged['Review_Mixed_Flag'].fillna(False)

    print(f"Semantic available for {df_merged['semantic_available'].sum():,} episodes.")
    print(f"Total merged rows: {len(df_merged):,}")

    # 4. Save
    print(f"Saving merged data to {output_path}...")
    df_merged.to_parquet(output_path, index=False)
    print("Gate 8.1 completed successfully.")

if __name__ == "__main__":
    main()
