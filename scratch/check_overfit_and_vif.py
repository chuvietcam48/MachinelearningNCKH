import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

import sys
repo_root = Path('C:/Users/Admin/OneDrive/Máy tính/MachineLearning-1/MachinelearningNCKH')
sys.path.insert(0, str(repo_root))
from src.framework.survival_engine import SurvivalEngine

def get_auc_at_270(cph, df, features):
    t_eval = 270
    S_270 = cph.predict_survival_function(df[features], times=[t_eval]).iloc[0].values
    P_return = 1.0 - S_270
    
    test_df = df.copy()
    test_df['P_return'] = P_return
    
    unknown_mask = (test_df['T_Duration'] < t_eval) & (test_df['E_Event'] == 0)
    eval_df = test_df[~unknown_mask].copy()
    
    eval_df['True_Label'] = (eval_df['T_Duration'] <= t_eval) & (eval_df['E_Event'] == 1)
    eval_df['True_Label'] = eval_df['True_Label'].astype(int)
    
    if len(eval_df['True_Label'].unique()) > 1:
        return roc_auc_score(eval_df['True_Label'], eval_df['P_return'])
    return np.nan

def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location('ext_metrics', repo_root / 'src/pipeline/12_extended_metrics.py')
    ext_metrics = importlib.util.module_from_spec(spec)
    sys.modules['ext_metrics'] = ext_metrics
    spec.loader.exec_module(ext_metrics)
    load_data = ext_metrics.load_data
    
    # 1. Load Data
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
    test_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
    
    with open(dataset_dir / "feature_registry.json") as f:
        registry = json.load(f)
    
    features_A = registry["Model_A"]["features"]
    features_C = registry["Model_C"]["features"]
    all_features = list(set(features_A + features_C))
    
    # Scaled exactly like 03 / 12
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
    test_imp = pd.DataFrame(imputer.transform(test_df[all_features]), columns=all_features)
    
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_imp), columns=all_features)
    test_scaled = pd.DataFrame(scaler.transform(test_imp), columns=all_features)
    
    train_scaled['E_Event'] = train_df['E_Event'].values
    train_scaled['T_Duration'] = train_df['T_Duration'].values
    test_scaled['E_Event'] = test_df['E_Event'].values
    test_scaled['T_Duration'] = test_df['T_Duration'].values
    
    # 36-variable version (no filtering)
    engine = SurvivalEngine(penalizer=0.001)
    
    print("--- 36-Variable Model (No Filtering) ---")
    cph_36 = engine.fit(train_scaled[features_C + ['T_Duration', 'E_Event']], duration_col='T_Duration', event_col='E_Event', check_ph=False)
    
    auc_train_36 = get_auc_at_270(cph_36, train_scaled, features_C)
    auc_test_36 = get_auc_at_270(cph_36, test_scaled, features_C)
    
    print(f"Train AUC: {auc_train_36:.4f}")
    print(f"Test AUC: {auc_test_36:.4f}")
    print(f"Train-Test Gap: {auc_train_36 - auc_test_36:.4f}\n")
    
    # Filter Logic
    zero_var_cols = train_imp.columns[train_imp.var() <= 0.01].tolist()
    print(f"--- Feature Filtering ---")
    print(f"Dropped {len(zero_var_cols)} Zero-Variance features: {zero_var_cols}")
    
    train_filtered = train_imp.drop(columns=zero_var_cols)
    corr = train_filtered.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
    print(f"Dropped {len(collinear_cols)} Collinear features: {collinear_cols}\n")
    
    # Calculate VIF for aspect_entropy before and after dropping collinear features
    print("--- VIF Analysis for aspect_entropy ---")
    # Before dropping collinear (but after dropping zero-var to avoid division by zero)
    try:
        X_before = train_scaled[[f for f in features_C if f not in zero_var_cols]]
        X_before_const = X_before.copy()
        X_before_const['const'] = 1.0
        vif_before = pd.Series([variance_inflation_factor(X_before_const.values, i) for i in range(X_before_const.shape[1])], index=X_before_const.columns)
        print(f"VIF of aspect_entropy BEFORE dropping collinear: {vif_before.get('aspect_entropy', np.nan):.2f}")
    except Exception as e:
        print(f"VIF Before Error: {e}")
        
    try:
        X_after = train_scaled[[f for f in features_C if f not in zero_var_cols and f not in collinear_cols]]
        X_after_const = X_after.copy()
        X_after_const['const'] = 1.0
        vif_after = pd.Series([variance_inflation_factor(X_after_const.values, i) for i in range(X_after_const.shape[1])], index=X_after_const.columns)
        print(f"VIF of aspect_entropy AFTER dropping collinear: {vif_after.get('aspect_entropy', np.nan):.2f}\n")
    except Exception as e:
        print(f"VIF After Error: {e}")
    
    # 23-variable version
    features_C_filtered = [f for f in features_C if f not in zero_var_cols and f not in collinear_cols]
    
    print(f"--- 23-Variable Model (Filtered) ---")
    cph_23 = engine.fit(train_scaled[features_C_filtered + ['T_Duration', 'E_Event']], duration_col='T_Duration', event_col='E_Event', check_ph=False)
    
    auc_train_23 = get_auc_at_270(cph_23, train_scaled, features_C_filtered)
    auc_test_23 = get_auc_at_270(cph_23, test_scaled, features_C_filtered)
    
    print(f"Train AUC: {auc_train_23:.4f}")
    print(f"Test AUC: {auc_test_23:.4f}")
    print(f"Train-Test Gap: {auc_train_23 - auc_test_23:.4f}\n")

if __name__ == '__main__':
    main()
