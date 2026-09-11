import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import json
import sys
import warnings
import importlib.util

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.framework.survival_engine import SurvivalEngine

# Import 12_extended_metrics.py
spec = importlib.util.spec_from_file_location('ext_metrics', repo_root / 'src/pipeline/12_extended_metrics.py')
ext_metrics = importlib.util.module_from_spec(spec)
sys.modules['ext_metrics'] = ext_metrics
spec.loader.exec_module(ext_metrics)
load_data = ext_metrics.load_data

def main():
    print("15: Episodic AUC Evaluation (Model A vs Model C)")
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    
    with open(dataset_dir / "feature_registry.json") as f:
        registry = json.load(f)
        
    train_X, test_X, active_A, active_C = load_data(repo_root)
    engine = SurvivalEngine(penalizer=0.001)
    
    results = []
    
    for model_name, features in [('Model_A', active_A), ('Model_C', active_C)]:
        train_cols = features + ['T_Duration', 'E_Event']
        
        cph = engine.fit(train_X[train_cols], duration_col='T_Duration', event_col='E_Event', check_ph=False)
        
        t_star = 60
        S_270 = cph.predict_survival_function(test_X[features], times=[t_star]).iloc[0].values
        P_return = 1.0 - S_270
        
        test_df = test_X.copy()
        test_df['P_return'] = P_return
        
        # Drop unknowns: Censored BEFORE 270 days.
        # If censored AT 270 days, we know they didn't return within 270 days, so label=0.
        unknown_mask = (test_df['T_Duration'] < t_star) & (test_df['E_Event'] == 0)
        eval_df = test_df[~unknown_mask].copy()
        
        eval_df['True_Label'] = (eval_df['T_Duration'] <= t_star) & (eval_df['E_Event'] == 1)
        eval_df['True_Label'] = eval_df['True_Label'].astype(int)
        
        auc = roc_auc_score(eval_df['True_Label'], eval_df['P_return'])
        print(f"{model_name} AUC at t={t_star}: {auc:.4f} (Evaluated on {len(eval_df)}/{len(test_df)} unambiguous episodes)")
        
        results.append({
            'Model': model_name,
            'AUC': auc,
            'Evaluated_Episodes': len(eval_df),
            'P_return': P_return,
            'eval_df': eval_df
        })
        
    res_A = results[0]
    res_C = results[1]
    
    print(f"\nDelta AUC (Model C - Model A): {res_C['AUC'] - res_A['AUC']:.4f}")
    
    # Bootstrap CI for Delta AUC
    print("Starting paired bootstrap for Delta AUC...")
    np.random.seed(42)
    deltas_auc = []
    
    # We bootstrap on the unambiguous evaluated indices
    # We must ensure both models are evaluated on the exact same indices
    # Since eval_df is derived identically for both (based on T_Duration and E_Event), they match
    y_eval = res_A['eval_df']['True_Label'].values
    prob_A = res_A['eval_df']['P_return'].values
    prob_C = res_C['eval_df']['P_return'].values
    
    for i in range(1000):
        idx = np.random.choice(len(y_eval), len(y_eval), replace=True)
        if len(np.unique(y_eval[idx])) < 2: 
            continue
        boot_A = roc_auc_score(y_eval[idx], prob_A[idx])
        boot_C = roc_auc_score(y_eval[idx], prob_C[idx])
        deltas_auc.append(boot_C - boot_A)
        
    from src.framework.stats import compute_percentile_ci
    ci_lower, ci_upper = compute_percentile_ci(deltas_auc)
    
    print(f"95% CI for Delta AUC: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("Does CI cross zero?", "Yes" if ci_lower < 0 and ci_upper > 0 else "No")
    
    out_df = pd.DataFrame([{
        'Model_A_AUC': res_A['AUC'],
        'Model_C_AUC': res_C['AUC'],
        'Delta_AUC': res_C['AUC'] - res_A['AUC'],
        '95_CI_Lower': ci_lower,
        '95_CI_Upper': ci_upper
    }])
    out_df.to_csv(repo_root / "outputs" / "pipeline_freeze" / "results" / "episodic_auc_270.csv", index=False)
    print("\nSaved episodic_auc_270.csv")

if __name__ == "__main__":
    main()
