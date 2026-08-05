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

    def fit(self, df_train: pd.DataFrame, duration_col='T_Duration', event_col='E_Event', check_ph=False) -> CoxPHFitter:
        """
        Fits a CoxPH Model. Automatically drops zero-variance columns and applies Cluster-Robust SE.
        """
        logger.info(f"Fitting CoxPH Model (Penalizer: {self.penalizer})")
        
        # We need to keep CustomerID for cluster robust standard errors
        cluster_col_actual = None
        if 'CustomerID' in df_train.columns:
            cluster_col_actual = 'CustomerID'
            
        # Drop columns that shouldn't be in the model (identifiers, datetimes, targets)
        drop_cols = ['InvoiceNo', 'InvoiceDate', 'episode_id', 'Y_dormant_270']
        
        # Keep only numeric columns + cluster_col
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_keep = numeric_cols
        if cluster_col_actual and cluster_col_actual not in cols_to_keep:
            cols_to_keep.append(cluster_col_actual)
            
        df_train = df_train[cols_to_keep].copy()
        df_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
        
        # Drop zero variance columns to prevent convergence issues (exclude cluster_col from check)
        check_var_cols = [c for c in df_train.columns if c != cluster_col_actual]
        variances = df_train[check_var_cols].var(numeric_only=True)
        zero_var_cols = variances[variances == 0].index.tolist()
        if zero_var_cols:
            logger.info(f"Dropping {len(zero_var_cols)} zero-variance features.")
            df_train = df_train.drop(columns=zero_var_cols)
            
        cph = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
        
        if cluster_col_actual:
            logger.info(f"Applying Cluster-Robust Standard Errors using cluster_col='{cluster_col_actual}'")
            cph.fit(df_train, duration_col=duration_col, event_col=event_col, robust=True, cluster_col=cluster_col_actual)
        else:
            cph.fit(df_train, duration_col=duration_col, event_col=event_col)
            
        if check_ph:
            logger.info("Running Proportional Hazards (PH) assumption test...")
            import sys
            import os
            os.makedirs('outputs/benchmark', exist_ok=True)
            try:
                with open('outputs/benchmark/ph_test_report.txt', 'w') as f:
                    sys.stdout = f
                    cph.check_assumptions(df_train, p_value_threshold=0.05, show_plots=False)
                    sys.stdout = sys.__stdout__
                logger.info("PH test completed. Report saved to outputs/benchmark/ph_test_report.txt")
            except Exception as e:
                sys.stdout = sys.__stdout__
                logger.warning(f"PH test encountered an issue: {e}")
        
        return cph
