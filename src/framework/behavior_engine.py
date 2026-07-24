import logging
import pandas as pd
from src.feature_engine import build_customer_features

logger = logging.getLogger(__name__)

class BehaviorEngine:
    """
    Layer 2: Behavior Engine
    Extracts purely behavioral signals (RFM, Inter-purchase time, churn tags)
    from the standardized dataset.
    """
    def __init__(self, snapshot_date=None, tau=90):
        self.snapshot_date = snapshot_date
        self.tau = tau

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes behavioral feature engineering.
        Delegates to the existing robust logic in src.feature_engine,
        or handles pre-aggregated episode data dynamically.
        """
        logger.info("Executing Behavior Engine...")
        if len(df) == 0:
            logger.warning("Empty dataframe passed to BehaviorEngine.")
            return df
            
        if 'episode_start' in df.columns:
            # Handle Amazon episode-based behavior logic
            return self._extract_episode_behavior(df)
            
        # Standard retail format (InvoiceDate, TotalSpend)
        df_features = build_customer_features(df, snapshot=self.snapshot_date, tau=self.tau)
        
        if 'T' in df_features.columns and 'T_Duration' not in df_features.columns:
            df_features = df_features.rename(columns={'T': 'T_Duration', 'E': 'E_Event'})
            
        return df_features

    def _extract_episode_behavior(self, df: pd.DataFrame) -> pd.DataFrame:
        import numpy as np
        logger.info("Extracting episode-based behavior features...")
        # Survival targets
        df['T_Duration'] = np.where(
            pd.isna(df.get('next_episode_start')),
            270.0,
            (df.get('next_episode_start') - df['episode_endpoint']).dt.total_seconds() / (24 * 3600)
        ) if 'next_episode_start' in df.columns else 270.0
        df['T_Duration'] = df['T_Duration'].clip(upper=270.0)
        
        if 'Y_dormant_270' in df.columns:
            df['E_Event'] = df['Y_dormant_270'].astype(int)
        else:
            df['E_Event'] = 0

        # Advanced behavior
        df['prev_episode_endpoint'] = df.groupby('CustomerID')['episode_endpoint'].shift(1)
        df['days_since_previous_episode'] = (df['episode_start'] - df['prev_episode_endpoint']).dt.total_seconds() / (24 * 3600)
        df['days_since_previous_episode'] = df['days_since_previous_episode'].fillna(-1.0)
        
        first_episode_start = df.groupby('CustomerID')['episode_start'].transform('min')
        df['customer_lifetime'] = (df['episode_endpoint'] - first_episode_start).dt.total_seconds() / (24 * 3600)
        df['review_frequency'] = df.get('prior_review_count', 0) / (df['customer_lifetime'] + 1.0)
        
        if 'episode_min_rating' in df.columns:
            df['is_low_rating_episode'] = (df['episode_min_rating'] <= 2).astype(int)
            df['recent_low_rating_count'] = (
                df.groupby('CustomerID')['is_low_rating_episode']
                .transform(lambda s: s.cumsum().shift(1))
                .fillna(0)
            )
            df = df.drop(columns=['prev_episode_endpoint', 'is_low_rating_episode'])
        else:
            df = df.drop(columns=['prev_episode_endpoint'])
            
        # Handle historical_mean_rating NaN (missing because first episode / no rating history)
        if 'historical_mean_rating' in df.columns:
            df['historical_mean_rating_missing'] = df['historical_mean_rating'].isna().astype(int)
            df['historical_mean_rating'] = df['historical_mean_rating'].fillna(0.0)
            
        return df
