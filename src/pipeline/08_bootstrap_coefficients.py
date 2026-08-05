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

def main():
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("08: Bootstrap Coefficient Stability (Model C)")
    print("=" * 70)
    
    with open(dataset_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    all_features = registry["Model_C"]["features"]
    
    train_path = dataset_dir / "master_train_semantic.parquet"
    if not train_path.exists():
        print("Required dataset not found.")
        return
        
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
    if 'CustomerID' in train_df.columns:
        train_X['CustomerID'] = train_df['CustomerID'].values
        
    survival_engine = SurvivalEngine(penalizer=0.01)
    
    train_cols = remaining_features + ['E_Event', 'T_Duration']
    if 'CustomerID' in train_X.columns:
        train_cols.append('CustomerID')
        
    n_boot = 50
    print(f"Running {n_boot} bootstrap iterations for Coefficient Stability...")
    
    boot_coefs = {feat: [] for feat in remaining_features}
    
    for i in range(n_boot):
        # Sample with replacement
        boot_idx = np.random.choice(len(train_X), len(train_X), replace=True)
        fit_df = train_X[train_cols].iloc[boot_idx].copy()
        
        try:
            # check_ph=False to avoid clutter
            cph = survival_engine.fit(fit_df, duration_col='T_Duration', event_col='E_Event', check_ph=False)
            
            summary = cph.summary
            for feat in remaining_features:
                if feat in summary.index:
                    boot_coefs[feat].append(summary.loc[feat, 'coef'])
                    
        except Exception as e:
            # Convergence issues on some bootstrap samples are normal
            pass
            
    print("\nBootstrapping Complete. Calculating Hazard Ratios and 95% CI...")
    
    results = []
    for feat in remaining_features:
        if len(boot_coefs[feat]) > 0:
            coef_array = np.array(boot_coefs[feat])
            # Convert coefficients to Hazard Ratios (exp(coef))
            hr_array = np.exp(coef_array)
            
            hr_mean = np.mean(hr_array)
            ci_lower = np.percentile(hr_array, 2.5)
            ci_upper = np.percentile(hr_array, 97.5)
            
            results.append({
                "Feature": feat,
                "Mean_HR": hr_mean,
                "95%_CI_Lower": ci_lower,
                "95%_CI_Upper": ci_upper,
                "Significant": "Yes" if (ci_lower > 1.0 or ci_upper < 1.0) else "No"
            })
            
    df_res = pd.DataFrame(results).sort_values(by="Mean_HR", ascending=False)
    df_res.to_csv(res_dir / "bootstrap_hazard_ratios.csv", index=False)
    
    print("\nFINAL BOOTSTRAP STABILITY TABLE (Top 15 Features):")
    print(df_res.head(15).to_markdown(index=False))
    print("\nSaved full table to results/bootstrap_hazard_ratios.csv")

if __name__ == "__main__":
    main()
