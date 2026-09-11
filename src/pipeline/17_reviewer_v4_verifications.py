import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main():
    print("17: Reviewer v4 Verifications (C3, C5, B1)")
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    
    train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
    test_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
    full_df = pd.concat([train_df, test_df])
    
    print("--- C5: Cohort Size Verification ---")
    print(f"Train Episodes: {len(train_df)}")
    print(f"Test Episodes: {len(test_df)}")
    print(f"Total Episodes (Train + Test): {len(full_df)}")
    
    total_unique_customers = full_df['CustomerID'].nunique()
    print(f"Total Unique Customers in Full Cohort: {total_unique_customers}")
    print(f"Episodes per Customer (Mean): {len(full_df) / total_unique_customers:.4f}")
    print("-" * 40)
    
    print("--- C3: True Univariate HR (Dominance Ratio) ---")
    full_df['dom_high'] = (full_df['dominance_ratio'] >= 1.0).astype(float)
    dom_1 = sum(full_df['dom_high'] == 1.0)
    dom_not_1 = sum(full_df['dom_high'] == 0.0)
    print(f"Dominance = 1.0 (Single-Aspect): {dom_1} ({dom_1/len(full_df)*100:.1f}%)")
    print(f"Dominance < 1.0 (Multi-Aspect): {dom_not_1} ({dom_not_1/len(full_df)*100:.1f}%)")
    
    cph = CoxPHFitter(penalizer=0.001)
    try:
        cph.fit(full_df[['dom_high', 'T_Duration', 'E_Event']], duration_col='T_Duration', event_col='E_Event')
        univariate_hr = cph.hazard_ratios_['dom_high']
        print(f"True Univariate HR for Dominance >= 1.0: {univariate_hr:.4f}")
    except Exception as e:
        print(f"Failed to fit CPH: {e}")
    print("-" * 40)

if __name__ == "__main__":
    main()
