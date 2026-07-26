import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import scipy.stats
from src.framework.survival_engine import SurvivalEngine

def main():
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    model_dir = repo_root / "outputs" / "pipeline_freeze" / "models"
    pred_dir = repo_root / "outputs" / "pipeline_freeze" / "predictions"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("03: Survival Model Training & Inference")
    print("=" * 70)
    
    with open(dataset_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    features_A = registry["Model_A"]["features"]
    features_B = registry["Model_B"]["features"]
    features_C = registry["Model_C"]["features"]
    
    all_features = list(set(features_A + features_B + features_C))
    
    cohorts = [
        ("Full", "master_train_full.parquet", "master_test_full.parquet"),
        ("Semantic", "master_train_semantic.parquet", "master_test_semantic.parquet")
    ]
    
    results_info_gain = []
    
    # Initialize Framework Engine
    survival_engine = SurvivalEngine(penalizer=0.001)
    
    for cohort_name, train_file, test_file in cohorts:
        print(f"\nProcessing Cohort: {cohort_name}")
        print("-" * 50)
        
        train_path = dataset_dir / train_file
        test_path = dataset_dir / test_file
        
        if not train_path.exists():
            print(f"Skipping {cohort_name} (File not found)")
            continue
            
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        imputer = SimpleImputer(strategy='median')
        train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
        test_imp = pd.DataFrame(imputer.transform(test_df[all_features]), columns=all_features)
        
        zero_var_cols = train_imp.columns[train_imp.var() <= 0.01]
        if len(zero_var_cols) > 0:
            print(f"  Dropping {len(zero_var_cols)} low-variance features: {list(zero_var_cols)}")
            train_imp = train_imp.drop(columns=zero_var_cols)
            test_imp = test_imp.drop(columns=zero_var_cols)
            
        # Drop highly collinear features
        corr = train_imp.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        collinear_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
        if len(collinear_cols) > 0:
            print(f"  Dropping {len(collinear_cols)} collinear features: {collinear_cols}")
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
        
        test_X['E_Event'] = test_df['E_Event'].values
        test_X['T_Duration'] = test_df['T_Duration'].values
        
        models = {}
        log_likelihoods = {}
        aic_scores = {}
        
        for name, feats in [("Model_A", features_A), ("Model_B", features_B), ("Model_C", features_C)]:
            active_feats = [f for f in feats if f in remaining_features]
            print(f"  Training {name} ({len(active_feats)} features)...")
            
            train_cols = active_feats + ['E_Event', 'T_Duration']
            
            try:
                fit_df = train_X[train_cols].copy()
                
                # Delegating to Framework Component
                cph = survival_engine.fit(fit_df, duration_col='T_Duration', event_col='E_Event')
                
                models[name] = cph
                log_likelihoods[name] = cph.log_likelihood_
                aic_scores[name] = cph.AIC_partial_
                
                # Save Model
                with open(model_dir / f"{name.lower()}_{cohort_name.lower()}.pkl", "wb") as f:
                    joblib.dump(cph, f)
                
                # Predict
                pred = cph.predict_partial_hazard(test_X[active_feats])
                pred_df = test_df[['episode_id', 'E_Event', 'T_Duration']].copy()
                pred_df['risk_score'] = pred.values
                pred_df.to_parquet(pred_dir / f"prediction_{name.lower()}_{cohort_name.lower()}.parquet", index=False)
                
                # Export Hazard Ratios for Model C
                if name == "Model_C":
                    summary = cph.summary
                    hr_df = pd.DataFrame({
                        'Feature': summary.index,
                        'Hazard_Ratio': summary['exp(coef)'],
                        'CI_Lower': summary['exp(coef) lower 95%'],
                        'CI_Upper': summary['exp(coef) upper 95%'],
                        'p_value': summary['p']
                    }).sort_values(by='Hazard_Ratio', ascending=False)
                    hr_df.to_csv(res_dir / f"hazard_ratios_{cohort_name.lower()}.csv", index=False)
                    
            except Exception as e:
                print(f"    -> Failed to fit {name}: {e}")
                
        # Information Gain (Model C vs Model A)
        if "Model_A" in models and "Model_C" in models:
            ll_A = log_likelihoods["Model_A"]
            ll_C = log_likelihoods["Model_C"]
            
            lr_stat = 2 * (ll_C - ll_A)
            df_deg = len(features_C) - len(features_A)
            p_val = scipy.stats.chi2.sf(lr_stat, df_deg)
            
            results_info_gain.append({
                'Cohort': cohort_name,
                'AIC_ModelA': aic_scores["Model_A"],
                'AIC_ModelC': aic_scores["Model_C"],
                'AIC_Improvement': aic_scores["Model_A"] - aic_scores["Model_C"],
                'LRT_Statistic': lr_stat,
                'p_value': p_val
            })
            
    if results_info_gain:
        pd.DataFrame(results_info_gain).to_csv(res_dir / "information_gain.csv", index=False)
        print("\nExported Gate 10.6 Information Gain.")

if __name__ == "__main__":
    main()
