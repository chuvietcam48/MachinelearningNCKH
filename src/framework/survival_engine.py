import logging
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter

logger = logging.getLogger(__name__)

class SurvivalEngine:
    """
    Layer 3: Survival Engine
    Fits survival models (CoxPH) dynamically depending on the features provided.
    """
    def __init__(self, penalizer: float = 0.1, l1_ratio: float = 0.0):
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio

    def fit(self, df_train: pd.DataFrame, duration_col='T_Duration', event_col='E_Event') -> CoxPHFitter:
        """
        Fits a CoxPH Model. Automatically drops zero-variance columns.
        """
        logger.info(f"Fitting CoxPH Model (Penalizer: {self.penalizer})")
        
        # Drop columns that shouldn't be in the model (identifiers, datetimes, targets)
        drop_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'episode_id', 'Y_dormant_270']
        df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
        
        # Keep only numeric columns for CoxPH
        df_train = df_train.select_dtypes(include=[np.number])
        
        # Drop zero variance columns to prevent convergence issues
        variances = df_train.var(numeric_only=True)
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            logger.info(f"Dropping {len(zero_var_cols)} zero-variance features.")
            df_train = df_train.drop(columns=zero_var_cols)
            
        cph = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        cph.fit(df_train, duration_col=duration_col, event_col=event_col)
        
        return cph
