import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import scipy.stats

# We use lifelines here to get statistical properties (p-values, CI, AIC)
from lifelines import CoxPHFitter

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    
    print("Gate 10.3 & 10.5 & 10.6: Multidimensional Training & Interpretability")
    print("=" * 70)
    
    # 1. Load Feature Registry
    with open(gate10_dir / "feature_registry.json", "r") as f:
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
    
    for cohort_name, train_file, test_file in cohorts:
        print(f"\nProcessing Cohort: {cohort_name}")
        print("-" * 50)
        
        train_path = gate10_dir / train_file
        test_path = gate10_dir / test_file
        
        if not train_path.exists():
            print(f"Skipping {cohort_name} (File not found)")
            continue
            
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        
        # Preprocessing (Impute & Scale)
        imputer = SimpleImputer(strategy='median')
        
        train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
        test_imp = pd.DataFrame(imputer.transform(test_df[all_features]), columns=all_features)
        
        # Drop Zero Variance columns to prevent NaN standard scaling and lifelines crashes
        zero_var_cols = train_imp.columns[train_imp.var() <= 1e-9]
        if len(zero_var_cols) > 0:
            print(f"  Dropping {len(zero_var_cols)} zero-variance features: {list(zero_var_cols)}")
            train_imp = train_imp.drop(columns=zero_var_cols)
            test_imp = test_imp.drop(columns=zero_var_cols)
            
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
            
            # Increase penalizer slightly to handle complete separation in sparse cohorts
            cph = CoxPHFitter(penalizer=0.1)
            train_cols = active_feats + ['E_Event', 'T_Duration']
            
            try:
                cph.fit(train_X[train_cols], duration_col='T_Duration', event_col='E_Event', fit_options={"step_size": 0.5})
                models[name] = cph
                log_likelihoods[name] = cph.log_likelihood_
                aic_scores[name] = cph.AIC_partial_
                
                # Predict risk scores
                pred = cph.predict_partial_hazard(test_X[active_feats])
                
                pred_df = test_df[['episode_id', 'E_Event', 'T_Duration']].copy()
                pred_df['risk_score'] = pred.values
                pred_df.to_parquet(gate10_dir / f"prediction_{name.lower()}_{cohort_name.lower()}.parquet", index=False)
                
                # Gate 10.5: Export Hazard Ratios for Model C
                if name == "Model_C":
                    summary = cph.summary
                    
                    hr_df = pd.DataFrame({
                        'Feature': summary.index,
                        'Hazard Ratio': summary['exp(coef)'],
                        'CI_Lower': summary['exp(coef) lower 95%'],
                        'CI_Upper': summary['exp(coef) upper 95%'],
                        'p_value': summary['p']
                    })
                    
                    # Sort by Hazard Ratio
                    hr_df = hr_df.sort_values(by='Hazard Ratio', ascending=False)
                    hr_df.to_csv(gate10_dir / f"hazard_ratios_{cohort_name.lower()}.csv", index=False)
                    print(f"    -> Exported Hazard Ratios to hazard_ratios_{cohort_name.lower()}.csv")
                    
            except Exception as e:
                print(f"    -> Failed to fit {name}: {e}")
                
        # Gate 10.6: Information Gain (Model C vs Model A)
        if "Model_A" in models and "Model_C" in models:
            ll_A = log_likelihoods["Model_A"]
            ll_C = log_likelihoods["Model_C"]
            
            lr_stat = 2 * (ll_C - ll_A)
            df = len(features_C) - len(features_A)
            p_val = scipy.stats.chi2.sf(lr_stat, df)
            
            results_info_gain.append({
                'Cohort': cohort_name,
                'AIC_ModelA': aic_scores["Model_A"],
                'AIC_ModelC': aic_scores["Model_C"],
                'AIC_Improvement': aic_scores["Model_A"] - aic_scores["Model_C"],
                'LRT_Statistic': lr_stat,
                'p_value': p_val
            })
            
    # Export Information Gain
    if results_info_gain:
        ig_df = pd.DataFrame(results_info_gain)
        ig_df.to_csv(gate10_dir / "information_gain.csv", index=False)
        print("\nExported Gate 10.6 Information Gain to information_gain.csv")
        print(ig_df)

    print("\nMultidimensional Training Complete!")

if __name__ == "__main__":
    main()
