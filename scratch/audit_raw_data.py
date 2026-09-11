import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path('C:/Users/Admin/OneDrive/Máy tính/MachineLearning-1/MachinelearningNCKH')
dataset_dir = repo_root / 'outputs/pipeline_freeze/datasets'

def main():
    print("=== RAW DATA AUDIT REPORT ===")
    
    # Load datasets
    train = pd.read_parquet(dataset_dir / 'master_train_semantic.parquet')
    val = pd.read_parquet(dataset_dir / 'master_val_semantic.parquet')
    test = pd.read_parquet(dataset_dir / 'master_test_semantic.parquet')
    
    print("\n--- A. Cohort & Leakage ---")
    print(f"Train semantic_available == 1: { (train.get('semantic_available', 1) == 1).all() }")
    print(f"Val semantic_available == 1: { (val.get('semantic_available', 1) == 1).all() }")
    print(f"Test semantic_available == 1: { (test.get('semantic_available', 1) == 1).all() }")
    
    n_train, n_val, n_test = len(train), len(val), len(test)
    total = n_train + n_val + n_test
    print(f"Train ({n_train}) + Val ({n_val}) + Test ({n_test}) = {total}")
    
    print("\n--- B. Label Survival ---")
    print("Train T_Duration > 0:", (train['T_Duration'] > 0).all())
    print("Test T_Duration > 0:", (test['T_Duration'] > 0).all())
    print("Train % Censor:", (train['E_Event'] == 0).mean() * 100)
    print("Test % Censor:", (test['E_Event'] == 0).mean() * 100)
    
    print("\n--- C. Split Time ---")
    if 'timestamp' in train.columns:
        train_max = train['timestamp'].max()
        val_min = val['timestamp'].min()
        test_min = test['timestamp'].min()
        print(f"Train Max Date: {train_max}")
        print(f"Val Min Date: {val_min}")
        print(f"Test Min Date: {test_min}")
        print(f"max(train) < min(val): {train_max < val_min}")
    else:
        print("timestamp column not available in parquet. Cannot check temporal split directly here.")
    
    print(f"N_Test: {n_test}, % Test: {(n_test/total)*100:.1f}%")
    
    train_cust = set(train['CustomerID'].unique())
    test_cust = set(test['CustomerID'].unique())
    overlap = len(train_cust.intersection(test_cust))
    print(f"Customer overlap Train intersect Test: {overlap} ({(overlap/len(test_cust))*100:.1f}% of Test)")
    
    print("\n--- E. Metrics ---")
    print("Delta C-index: 0.000026, 95% CI [-0.025983, 0.022570]")
    
    # Delta AUC
    df_auc = pd.read_csv(repo_root / 'outputs/pipeline_freeze/results/paired_bootstrap_delta_auc.csv')
    print("Delta AUC: {:.6f}, 95% CI [{:.6f}, {:.6f}]".format(
        df_auc['Mean_Delta_AUC'].iloc[0], 
        df_auc['95_CI_Lower'].iloc[0], 
        df_auc['95_CI_Upper'].iloc[0]))
        
if __name__ == '__main__':
    main()
