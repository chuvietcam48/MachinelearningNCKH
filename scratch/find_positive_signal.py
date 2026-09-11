import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

def get_auc_at_t(cph, df, features, t_eval):
    S_t = cph.predict_survival_function(df[features], times=[t_eval]).iloc[0].values
    P_return = 1.0 - S_t
    
    test_df = df.copy()
    test_df['P_return'] = P_return
    
    unknown_mask = (test_df['T_Duration'] < t_eval) & (test_df['E_Event'] == 0)
    eval_df = test_df[~unknown_mask].copy()
    
    eval_df['True_Label'] = (eval_df['T_Duration'] <= t_eval) & (eval_df['E_Event'] == 1)
    eval_df['True_Label'] = eval_df['True_Label'].astype(int)
    
    if len(eval_df['True_Label'].unique()) > 1:
        return roc_auc_score(eval_df['True_Label'], eval_df['P_return']), len(eval_df)
    return np.nan, 0

def main():
    import json, sys
    repo_root = Path('C:/Users/Admin/OneDrive/Máy tính/MachineLearning-1/MachinelearningNCKH')
    sys.path.insert(0, str(repo_root))
    from src.framework.survival_engine import SurvivalEngine
    
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
    test_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
    
    with open(dataset_dir / "feature_registry.json") as f:
        registry = json.load(f)
    
    features_A = registry["Model_A"]["features"]
    features_C = registry["Model_C"]["features"]
    
    # Reload the filtered features exactly as pipeline did
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    all_features = list(set(features_A + features_C))
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
    test_imp = pd.DataFrame(imputer.transform(test_df[all_features]), columns=all_features)
    
    zero_var_cols = train_imp.columns[train_imp.var() <= 0.01].tolist()
    train_filtered = train_imp.drop(columns=zero_var_cols)
    corr = train_filtered.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
    
    final_features_C = [f for f in features_C if f not in zero_var_cols and f not in collinear_cols]
    final_features_A = [f for f in features_A if f not in zero_var_cols and f not in collinear_cols]
    
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_imp), columns=all_features)
    test_scaled = pd.DataFrame(scaler.transform(test_imp), columns=all_features)
    
    for df_scaled, df_orig in [(train_scaled, train_df), (test_scaled, test_df)]:
        df_scaled['E_Event'] = df_orig['E_Event'].values
        df_scaled['T_Duration'] = df_orig['T_Duration'].values
        df_scaled['prior_review_count'] = df_orig['prior_review_count'].values
    
    engine = SurvivalEngine(penalizer=0.001)
    
    cph_A = engine.fit(train_scaled[final_features_A + ['T_Duration', 'E_Event']], duration_col='T_Duration', event_col='E_Event', check_ph=False)
    cph_C = engine.fit(train_scaled[final_features_C + ['T_Duration', 'E_Event']], duration_col='T_Duration', event_col='E_Event', check_ph=False)
    
    print("=== Subgroup Analysis (Cold-Start Customers) ===")
    
    for thresh in [0, 1, 2]:
        test_sub = test_scaled[test_scaled['prior_review_count'] <= thresh]
        
        if len(test_sub) > 0:
            auc_A, n = get_auc_at_t(cph_A, test_sub, final_features_A, 270)
            auc_C, _ = get_auc_at_t(cph_C, test_sub, final_features_C, 270)
            
            print(f"Prior Reviews <= {thresh} (N={n} eval cases):")
            print(f"  Model A AUC: {auc_A:.4f}")
            print(f"  Model C AUC: {auc_C:.4f}")
            if auc_A is not np.nan and auc_C is not np.nan:
                print(f"  Delta: {auc_C - auc_A:.4f}\n")
        else:
            print(f"Prior Reviews <= {thresh}: No data\n")

    print("=== Short-Term vs Long-Term Horizon ===")
    for t_eval in [30, 90, 180, 270, 365]:
        auc_A, n = get_auc_at_t(cph_A, test_scaled, final_features_A, t_eval)
        auc_C, _ = get_auc_at_t(cph_C, test_scaled, final_features_C, t_eval)
        print(f"AUC @ t={t_eval} (N={n} eval cases):")
        if auc_A is not np.nan and auc_C is not np.nan:
            print(f"  Model A: {auc_A:.4f} | Model C: {auc_C:.4f} | Delta: {auc_C - auc_A:.4f}")

if __name__ == '__main__':
    main()
