import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lifelines
from lifelines.statistics import proportional_hazard_test
import warnings

def load_data(repo_root):
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
    test_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
    
    with open(dataset_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    features_A = registry["Model_A"]["features"]
    features_C = registry["Model_C"]["features"]
    all_features = list(set(features_A + features_C))
    
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
    test_imp = pd.DataFrame(imputer.transform(test_df[all_features]), columns=all_features)
    
    zero_var_cols = train_imp.columns[train_imp.var() <= 0.01]
    train_imp = train_imp.drop(columns=zero_var_cols)
    test_imp = test_imp.drop(columns=zero_var_cols)
        
    corr = train_imp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
    train_imp = train_imp.drop(columns=collinear_cols)
    test_imp = test_imp.drop(columns=collinear_cols)
        
    remaining_features = list(train_imp.columns)
        
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imp)
    test_scaled = scaler.transform(test_imp)
    
    train_X = pd.DataFrame(train_scaled, columns=remaining_features)
    test_X = pd.DataFrame(test_scaled, columns=remaining_features)
    
    train_X['E_Event'] = train_df['E_Event'].values
    train_X['T_Duration'] = train_df['T_Duration'].values
    train_X['CustomerID'] = train_df['CustomerID'].values
    
    test_X['E_Event'] = test_df['E_Event'].values
    test_X['T_Duration'] = test_df['T_Duration'].values
    test_X['CustomerID'] = test_df['CustomerID'].values
    
    active_A = [f for f in features_A if f in remaining_features]
    active_C = [f for f in features_C if f in remaining_features]
    
    return train_X, test_X, active_A, active_C

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    model_dir = repo_root / "outputs" / "pipeline_freeze" / "models"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    
    print("12: Extended Metrics Computation (PH, Uno's C, IBS, Bootstrap)")
    train_X, test_X, active_A, active_C = load_data(repo_root)
    
    cph_A = joblib.load(model_dir / "model_a_semantic.pkl")
    cph_C = joblib.load(model_dir / "model_c_semantic.pkl")
    
    metrics = {}
    
    # 1. Proportional Hazards Test (Schoenfeld)
    print("Running PH test on Model C...")
    train_cols_C = active_C + ['E_Event', 'T_Duration']
    # Add small noise to avoid tied times error in lifelines PH test
    df_ph = train_X[train_cols_C].copy()
    if 'CustomerID' in df_ph.columns:
        df_ph = df_ph.drop(columns=['CustomerID'])
    df_ph['T_Duration'] += np.random.uniform(0, 1e-4, size=len(df_ph))
    try:
        ph_test = proportional_hazard_test(cph_C, df_ph, time_transform='rank')
        
        metrics['ph_test_global_p_value'] = float(ph_test.p_value[0]) if isinstance(ph_test.p_value, (list, np.ndarray, pd.Series)) else float(ph_test.p_value)
        
        ph_test_df = pd.DataFrame({
            'test_statistic': ph_test.test_statistic,
            'p_value': ph_test.p_value
        })
        ph_test_df.to_csv(res_dir / "ph_test_results.csv")
        print(f"Global PH Test p-value: {metrics['ph_test_global_p_value']:.4f}")
    except Exception as e:
        print(f"PH test failed: {e}")
    
    # 2. Uno's C and IBS (using scikit-survival if available, otherwise skip)
    try:
        from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
        y_train = np.array(list(zip(train_X['E_Event'], train_X['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
        y_test = np.array(list(zip(test_X['E_Event'], test_X['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
        
        # Test times for IBS (strictly within training support to avoid zero censoring probability)
        tau = np.percentile(y_train['Duration'], 90)
        
        # Filter test set to ensure times are strictly less than max train time
        max_train_time = y_train['Duration'].max() - 1
        mask = (y_test['Duration'] < max_train_time) | (y_test['Event'] == True)
        
        # Actually sksurv has tau for concordance
        pred_A = cph_A.predict_partial_hazard(test_X[active_A]).values
        pred_C = cph_C.predict_partial_hazard(test_X[active_C]).values
        uno_c_A = concordance_index_ipcw(y_train, y_test, pred_A, tau=tau)[0]
        uno_c_C = concordance_index_ipcw(y_train, y_test, pred_C, tau=tau)[0]
        
        # times for IBS must be within training follow up
        min_time = y_test['Duration'].min() + 1
        max_time = tau
        times = np.linspace(min_time, max_time, 20)
        
        surv_A = cph_A.predict_survival_function(test_X[active_A], times=times).T.values
        ibs_A = integrated_brier_score(y_train, y_test, surv_A, times)
        
        surv_C = cph_C.predict_survival_function(test_X[active_C], times=times).T.values
        ibs_C = integrated_brier_score(y_train, y_test, surv_C, times)
        
        metrics['unos_c_A'] = float(uno_c_A)
        metrics['unos_c_C'] = float(uno_c_C)
        metrics['ibs_A'] = float(ibs_A)
        metrics['ibs_C'] = float(ibs_C)
        print(f"Uno's C - Model A: {uno_c_A:.4f}, Model C: {uno_c_C:.4f}")
        print(f"IBS - Model A: {ibs_A:.4f}, Model C: {ibs_C:.4f}")
        
    except ImportError:
        print("scikit-survival not installed, skipping Uno's C and IBS.")
    except Exception as e:
        print(f"Uno's C / IBS calculation failed: {e}")
        
    # 3. Bootstrap HR (Customer-level)
    print("Running Customer-level Bootstrap HR...")
    n_boot = 100
    seed = 42
    np.random.seed(seed)
    metrics['bootstrap_seed'] = seed
    metrics['bootstrap_iterations'] = n_boot
    
    customers = train_X['CustomerID'].unique()
    coefs = []
    
    import sys
    sys.path.insert(0, str(repo_root))
    from src.framework.survival_engine import SurvivalEngine
    
    engine = SurvivalEngine(penalizer=0.001)
    
    for i in range(n_boot):
        boot_custs = np.random.choice(customers, size=len(customers), replace=True)
        boot_idx = []
        for c in boot_custs:
            boot_idx.extend(train_X[train_X['CustomerID'] == c].index.tolist())
            
        boot_df = train_X.iloc[boot_idx].copy()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph_boot = engine.fit(boot_df[train_cols_C], duration_col='T_Duration', event_col='E_Event', check_ph=False)
                coefs.append(cph_boot.params_)
        except Exception:
            pass
            
    if coefs:
        coef_df = pd.DataFrame(coefs)
        hr_df = np.exp(coef_df)
        
        summary = []
        for col in hr_df.columns:
            mean_hr = hr_df[col].mean()
            ci_lower = hr_df[col].quantile(0.025)
            ci_upper = hr_df[col].quantile(0.975)
            sig = "Yes" if ci_lower > 1.0 or ci_upper < 1.0 else "No"
            
            summary.append({
                "Feature": col,
                "Mean_HR": mean_hr,
                "95%_CI_Lower": ci_lower,
                "95%_CI_Upper": ci_upper,
                "Significant": sig
            })
            
        boot_res = pd.DataFrame(summary).sort_values("Mean_HR", ascending=False)
        boot_res.to_csv(res_dir / "bootstrap_hazard_ratios_extended.csv", index=False)
        print("Bootstrap HR exported.")
        
    with open(res_dir / "model_metrics_extended.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("\nExtended metrics computation complete.")

if __name__ == "__main__":
    main()
