import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.framework.survival_engine import SurvivalEngine
from lifelines import CoxPHFitter

def main():
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("09: Robust vs Classical Standard Errors Comparison")
    print("=" * 70)
    
    with open(dataset_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    all_features = registry["Model_C"]["features"]
    
    train_path = dataset_dir / "master_train_semantic.parquet"
    train_df = pd.read_parquet(train_path)
    
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
    
    zero_var_cols = train_imp.columns[train_imp.var() <= 0.01]
    train_imp = train_imp.drop(columns=zero_var_cols)
    
    corr = train_imp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
    train_imp = train_imp.drop(columns=collinear_cols)
    
    remaining_features = list(train_imp.columns)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imp)
    train_X = pd.DataFrame(train_scaled, columns=remaining_features)
    train_X['E_Event'] = train_df['E_Event'].values
    train_X['T_Duration'] = train_df['T_Duration'].values
    train_X['CustomerID'] = train_df['CustomerID'].values
    
    # 1. Fit Classical (No Robust SE)
    print("Fitting Classical CoxPH (No cluster robust)...")
    cph_classic = CoxPHFitter(penalizer=0.01)
    df_classic = train_X[remaining_features + ['E_Event', 'T_Duration']].copy()
    cph_classic.fit(df_classic, duration_col='T_Duration', event_col='E_Event')
    se_classic = cph_classic.standard_errors_
    
    # 2. Fit Robust (With cluster_col)
    print("Fitting Robust CoxPH (With cluster robust)...")
    cph_robust = CoxPHFitter(penalizer=0.01)
    df_robust = train_X[remaining_features + ['E_Event', 'T_Duration', 'CustomerID']].copy()
    cph_robust.fit(df_robust, duration_col='T_Duration', event_col='E_Event', cluster_col='CustomerID', robust=True)
    se_robust = cph_robust.standard_errors_
    
    # Compare
    results = []
    for feat in remaining_features:
        if feat in se_classic.index and feat in se_robust.index:
            sc = se_classic.loc[feat]
            sr = se_robust.loc[feat]
            results.append({
                "Feature": feat,
                "Classical_SE": sc,
                "Robust_SE": sr,
                "Diff_Absolute": abs(sr - sc),
                "Diff_Percentage": f"{((sr - sc)/sc)*100:.2f}%"
            })
            
    df_res = pd.DataFrame(results).sort_values(by="Diff_Absolute", ascending=False)
    df_res.to_csv(res_dir / "se_comparison.csv", index=False)
    
    print("\nSTANDARD ERROR COMPARISON TABLE (Top 10 differences):")
    print(df_res.head(10).to_markdown(index=False))
    print("\nSaved full table to results/se_comparison.csv")

if __name__ == "__main__":
    main()
