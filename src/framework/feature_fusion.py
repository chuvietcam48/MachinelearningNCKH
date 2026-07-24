import logging
import pandas as pd

logger = logging.getLogger(__name__)

class FeatureFusionEngine:
    """
    Layer 2: Feature Fusion Engine
    Combines Behavioral Features with Semantic Features (if available).
    """
    def __init__(self):
        pass

    def merge(self, df_behavior: pd.DataFrame, df_semantic: pd.DataFrame) -> pd.DataFrame:
        """
        Safely merges semantic signals into the behavioral cohort.
        """
        if df_semantic is None or len(df_semantic) == 0:
            logger.info("No semantic data provided. Using behavioral features exclusively.")
            return df_behavior
            
        logger.info("Fusing Behavioral and Semantic Features via Feature Registry Pattern...")
        
        merge_keys = ['CustomerID']
        if 'episode_id' in df_behavior.columns and 'episode_id' in df_semantic.columns:
            merge_keys.append('episode_id')

        semantic_cols = [
            c for c in df_semantic.columns
            if c in merge_keys or c not in df_behavior.columns
        ]
        df_semantic = df_semantic[semantic_cols].copy()

        df_merged = pd.merge(df_behavior, df_semantic, on=merge_keys, how='left')

        new_cols = [c for c in df_semantic.columns if c not in merge_keys]
        numeric_cols = [
            c for c in new_cols
            if c in df_merged.columns and pd.api.types.is_numeric_dtype(df_merged[c])
        ]
        if numeric_cols:
            df_merged[numeric_cols] = df_merged[numeric_cols].fillna(0)

        if 'final_targets' in df_merged.columns:
            df_merged['final_targets'] = df_merged['final_targets'].fillna('[]')
        if 'Review_Mixed_Flag' in df_merged.columns:
            df_merged['Review_Mixed_Flag'] = df_merged['Review_Mixed_Flag'].fillna(False)
        
        return df_merged
