import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from lifelines import CoxPHFitter
import scipy.stats

def run_model_robustness(train_X, features_A, features_C):
    print("\n(A) Model Robustness: Varying Cox Penalizer")
    results = []
    penalties = [0.01, 0.1, 0.5]
    
    for p in penalties:
        cph_A = CoxPHFitter(penalizer=p)
        cph_C = CoxPHFitter(penalizer=p)
        try:
            cph_A.fit(train_X[features_A + ['E_Event', 'T_Duration']], duration_col='T_Duration', event_col='E_Event', fit_options={"step_size": 0.5})
            cph_C.fit(train_X[features_C + ['E_Event', 'T_Duration']], duration_col='T_Duration', event_col='E_Event', fit_options={"step_size": 0.5})
            
            lr_stat = 2 * (cph_C.log_likelihood_ - cph_A.log_likelihood_)
            df_deg = len(features_C) - len(features_A)
            p_val = scipy.stats.chi2.sf(lr_stat, df_deg)
            
            results.append({"Penalizer": p, "LRT_Statistic": lr_stat, "p_value": p_val})
        except:
            pass
    return pd.DataFrame(results)

def run_data_robustness(train_X, features_A, features_C):
    print("\n(B) Data Robustness: Sparse Semantic Coverage")
    results = []
    coverages = [0.1, 0.2, 0.5, 1.0]
    
    for cov in coverages:
        if cov < 1.0:
            sample_X = train_X.sample(frac=cov, random_state=42)
        else:
            sample_X = train_X.copy()
            
        cph_A = CoxPHFitter(penalizer=0.1)
        cph_C = CoxPHFitter(penalizer=0.1)
        try:
            cph_A.fit(sample_X[features_A + ['E_Event', 'T_Duration']], duration_col='T_Duration', event_col='E_Event', fit_options={"step_size": 0.5})
            cph_C.fit(sample_X[features_C + ['E_Event', 'T_Duration']], duration_col='T_Duration', event_col='E_Event', fit_options={"step_size": 0.5})
            
            lr_stat = 2 * (cph_C.log_likelihood_ - cph_A.log_likelihood_)
            df_deg = len(features_C) - len(features_A)
            p_val = scipy.stats.chi2.sf(lr_stat, df_deg)
            
            results.append({"Semantic_Coverage": f"{int(cov*100)}%", "LRT_Statistic": lr_stat, "p_value": p_val})
        except:
            pass
    return pd.DataFrame(results)

def run_policy_robustness(df):
    print("\n(C) Policy Robustness: Intervention Cost Ratios")
    results = []
    ratios = [1, 2, 3]
    
    for r in ratios:
        # Re-calc priority score with scaled cost
        scores = (df['risk_score'] * df['CLV_Proxy'] * df['Success_Prob']) / ((df['Intervention_Cost'] * r) + 0.1)
        
        # What is the median priority score for customers requiring action?
        active_mask = df['Intervention_Cost'] > 0
        med_score = scores[active_mask].median() if active_mask.sum() > 0 else 0
        results.append({"Cost_Ratio": f"{r}x", "Median_Active_Priority_Score": med_score})
        
    return pd.DataFrame(results)

def get_action_and_cost(risk_level, value_level, reason):
    if risk_level == "Low": return "No Action Needed", 0.0, 0.0
    if risk_level == "Medium" and value_level == "High": return "Retention Reminder", 1.0, 0.15
    if risk_level == "High":
        if reason == "Delivery": return "Agent + Logistics Recovery", 20.0, 0.55 if value_level == "High" else ("Shipping Coupon", 5.0, 0.45)
        elif reason == "Price": return "Discount Voucher", 10.0, 0.40
        elif reason == "Product" and value_level == "High": return "Replacement / Warranty", 50.0, 0.70
        elif reason == "Product" and value_level == "Low": return "Discount Voucher", 10.0, 0.40
        elif reason == "Mixed": return "Human Escalation", 15.0, 0.65
        else: return "Generic Behavioral Campaign", 5.0, 0.20
    return "No Action Needed", 0.0, 0.0

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    pred_dir = repo_root / "outputs" / "pipeline_freeze" / "predictions"
    sens_dir = repo_root / "outputs" / "pipeline_freeze" / "results" / "sensitivity"
    sens_dir.mkdir(parents=True, exist_ok=True)
    
    print("06: Sensitivity & Robustness Analysis")
    print("=" * 70)
    
    # Load for Model & Data Robustness
    try:
        train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
        with open(dataset_dir / "feature_registry.json", "r") as f:
            registry = json.load(f)
    except:
        print("Missing required files for sensitivity analysis.")
        return
        
    features_A = registry["Model_A"]["features"]
    features_C = registry["Model_C"]["features"]
    all_features = list(set(features_A + features_C))
    
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
    zero_var_cols = train_imp.columns[train_imp.var() <= 1e-9]
    if len(zero_var_cols) > 0: train_imp = train_imp.drop(columns=zero_var_cols)
    remaining_features = list(train_imp.columns)
    
    scaler = StandardScaler()
    train_X = pd.DataFrame(scaler.fit_transform(train_imp), columns=remaining_features)
    train_X['E_Event'] = train_df['E_Event'].values
    train_X['T_Duration'] = train_df['T_Duration'].values
    
    active_A = [f for f in features_A if f in remaining_features]
    active_C = [f for f in features_C if f in remaining_features]
    
    # Execute A and B
    df_A = run_model_robustness(train_X, active_A, active_C)
    df_A.to_csv(sens_dir / "model_robustness_penalizer.csv", index=False)
    
    df_B = run_data_robustness(train_X, active_A, active_C)
    df_B.to_csv(sens_dir / "data_robustness_coverage.csv", index=False)
    
    # Execute C
    try:
        pred_df = pd.read_parquet(pred_dir / "prediction_model_c_semantic.parquet")
        feat_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
        df_policy = pd.merge(pred_df, feat_df.drop(columns=['E_Event', 'T_Duration'], errors='ignore'), on="episode_id")
        
        # Re-calc policy params
        p50 = np.percentile(df_policy['risk_score'], 50)
        p80 = np.percentile(df_policy['risk_score'], 80)
        df_policy['Risk_Level'] = df_policy['risk_score'].apply(lambda x: "High" if x >= p80 else ("Medium" if x >= p50 else "Low"))
        
        df_policy['CLV_Proxy'] = df_policy['prior_verified_episode_count'] + 1
        val_p50 = np.percentile(df_policy['CLV_Proxy'], 50)
        df_policy['Value_Level'] = df_policy['CLV_Proxy'].apply(lambda x: "High" if x >= val_p50 else "Low")
        
        def get_reason(row):
            if row.get('has_conflict', 0) == 1: return "Mixed"
            if row.get('rolling_negative_ratio', 0) > 0: return "Accumulated Negativity"
            return "Unknown"
            
        df_policy['Semantic_Attribution'] = df_policy.apply(get_reason, axis=1)
        df_policy['Intervention_Cost'] = df_policy.apply(lambda r: get_action_and_cost(r['Risk_Level'], r['Value_Level'], r['Semantic_Attribution'])[1], axis=1)
        df_policy['Success_Prob'] = df_policy.apply(lambda r: get_action_and_cost(r['Risk_Level'], r['Value_Level'], r['Semantic_Attribution'])[2], axis=1)
        
        df_C = run_policy_robustness(df_policy)
        df_C.to_csv(sens_dir / "policy_robustness_cost.csv", index=False)
    except Exception as e:
        print(f"Policy robustness failed: {e}")
        
    print("\nSensitivity Analysis Complete!")

if __name__ == "__main__":
    main()
