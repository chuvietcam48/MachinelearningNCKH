import os
import logging
import pandas as pd
import numpy as np
import warnings
from lifelines import CoxPHFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
warnings.filterwarnings("ignore")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

def run_diagnostics():
    from src.dataset_registry import get_dataset
    from src.framework.dataset_adapter import DatasetAdapter
    from src.framework.behavior_engine import BehaviorEngine
    from src.framework.survival_engine import SurvivalEngine
    
    # 1. Load Data
    logging.info("--- 1. Loading Baseline Data (CDNOW) ---")
    info = get_dataset("cdnow")
    adapter = DatasetAdapter("cdnow")
    df_raw = adapter.load_data()
    
    # 2. Behavior Engine
    logging.info("--- 2. Building Features ---")
    snapshot_date = info.snapshot_fn(df_raw)
    engine = BehaviorEngine(snapshot_date=snapshot_date)
    df_features = engine.process(df_raw)
    
    # Drop identifiers
    drop_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'episode_id']
    df_train = df_features.drop(columns=[c for c in drop_cols if c in df_features.columns])
    df_train = df_train.select_dtypes(include=[np.number]).dropna()
    
    # 3. Variance & VIF
    logging.info("\n--- 3. Feature Diagnostics (VIF) ---")
    X = df_train.drop(columns=['T_Duration', 'E_Event'])
    # Add constant for VIF
    X['const'] = 1
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    for idx, row in vif_data.iterrows():
        if row['feature'] != 'const':
            logging.info(f"{row['feature']:<30}: {row['VIF']:.2f}")
    
    # 4. Survival Diagnostics (Schoenfeld, PH)
    logging.info("\n--- 4. Survival Diagnostics (CoxPH Assumptions) ---")
    # Sample down to 2000 rows for speed of Schoenfeld test
    df_sample = df_train.sample(n=min(2000, len(df_train)), random_state=42)
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(df_sample, duration_col='T_Duration', event_col='E_Event')
    
    logging.info(f"Concordance Index: {cph.concordance_index_:.4f}")
    
    # Check assumptions (this outputs to stdout/stderr natively)
    logging.info("\nChecking Proportional Hazards Assumption...")
    results = cph.check_assumptions(df_sample, p_value_threshold=0.05, show_plots=False)
    
    logging.info("\nDiagnostics Complete.")

if __name__ == "__main__":
    run_diagnostics()
