import os
import logging
import pandas as pd
from typing import Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class SemanticProvider:
    """
    Abstract base class for providing semantic data to the framework.
    """
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

class AmazonSemanticProvider(SemanticProvider):
    """
    Loads Amazon's Frozen Annotations (Derived Artifacts) for the Semantic Plugin.
    """
    def load(self) -> pd.DataFrame:
        logger.info("AmazonSemanticProvider: Loading frozen semantic annotations...")
        repo_root = Path(__file__).resolve().parent.parent.parent
        
        # Load the frozen LLM QA JSONL which contains the 'final_targets'
        llm_path = repo_root / "data" / "artifacts" / "semantic_labels" / "mass_results_v224_qa.jsonl"
        
        # Load the episode mapping (to link annotation_item_id to CustomerID and episode_id)
        episode_path = repo_root / "data" / "artifacts" / "semantic_labels" / "verified_episode_snapshots_h270_v1.parquet"
        
        if not llm_path.exists() or not episode_path.exists():
            logger.warning("Frozen annotations not found in data/artifacts/semantic_labels/")
            return pd.DataFrame()
            
        df_episodes = pd.read_parquet(episode_path)
        # Standardize ID right away
        if 'reviewerID' in df_episodes.columns:
            df_episodes = df_episodes.rename(columns={'reviewerID': 'CustomerID'})
            
        df_episodes['annotation_item_id'] = df_episodes['CustomerID'].astype(str) + '_' + df_episodes['episode_id'].astype(str)

        labels = []
        with open(llm_path, 'r', encoding='utf-8') as f:
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
        
        # Load traditional sentiment as well if available
        sentiment_path = repo_root / "data" / "artifacts" / "semantic_labels" / "episode_sentiment_features.parquet"
        if sentiment_path.exists():
            sentiment_df = pd.read_parquet(sentiment_path)
            df_merged = df_merged.merge(sentiment_df, on="episode_id", how="left")
            
        return df_merged

class PluginManager:
    """
    Central hub for activating and managing framework plugins.
    Decides whether a dataset uses standard behavior only or behavior + semantic.
    """
    def __init__(self, dataset_name: str, force_semantic: bool = False):
        self.dataset_name = dataset_name.lower()
        self.force_semantic = force_semantic
        
    def get_semantic_provider(self) -> Optional[SemanticProvider]:
        """
        Returns the appropriate Semantic Provider for the dataset, or None if not applicable.
        """
        if self.dataset_name == "amazon" and self.force_semantic:
            return AmazonSemanticProvider()
            
        # In the future, we could add ShopeeProvider, HospitalProvider, etc.
        return None
