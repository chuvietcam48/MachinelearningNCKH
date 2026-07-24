import pandas as pd
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.framework.semantic_engine import SemanticEngine

def merge_llm_annotations(episode_table_path, qa_labels_path):
    print("Merging Frozen LLM Annotations into Episode Table...")
    df_episodes = pd.read_parquet(episode_table_path)
    
    # Standardize ID column
    if 'reviewerID' in df_episodes.columns:
        df_episodes = df_episodes.rename(columns={'reviewerID': 'CustomerID'})
        
    df_episodes['annotation_item_id'] = df_episodes['CustomerID'].astype(str) + '_' + df_episodes['episode_id'].astype(str)

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
    
    df_labels = pd.DataFrame(labels)
    if len(df_labels) > 0:
        df_merged = df_episodes.merge(df_labels, on='annotation_item_id', how='left')
    else:
        df_merged = df_episodes.copy()
        df_merged['final_targets'] = '[]'
        df_merged['Review_Mixed_Flag'] = False

    df_merged['semantic_available'] = df_merged['final_targets'].notna().astype(int)
    df_merged['final_targets'] = df_merged['final_targets'].fillna('[]')
    df_merged['Review_Mixed_Flag'] = df_merged['Review_Mixed_Flag'].fillna(False)
    return df_merged

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    data_dir = repo_root / "data" / "artifacts" / "semantic_labels"
    out_dir = repo_root / "outputs" / "pipeline_freeze" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("01: Semantic Feature Engineering")
    df = merge_llm_annotations(
        data_dir / "verified_episode_snapshots_h270_v1.parquet",
        data_dir / "mass_results_v224_qa.jsonl"
    )
    
    # Delegating complex state-persistence tracking and conflict resolution to the abstract framework layer
    semantic_engine = SemanticEngine()
    df = semantic_engine.process(df)

    out_path = out_dir / "semantic_features_engineered.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
