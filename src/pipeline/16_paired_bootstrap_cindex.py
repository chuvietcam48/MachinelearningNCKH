import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import warnings
import importlib.util

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.framework.survival_engine import SurvivalEngine
from sksurv.metrics import concordance_index_ipcw

# Import 12_extended_metrics.py
spec = importlib.util.spec_from_file_location('ext_metrics', repo_root / 'src/pipeline/12_extended_metrics.py')
ext_metrics = importlib.util.module_from_spec(spec)
sys.modules['ext_metrics'] = ext_metrics
spec.loader.exec_module(ext_metrics)
load_data = ext_metrics.load_data

def main():
    print("16: Paired Bootstrap C-index (Model A vs Model C)")
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    
    train_X, test_X, active_A, active_C = load_data(repo_root)
    engine = SurvivalEngine(penalizer=0.001)
    
    train_cols_A = active_A + ['T_Duration', 'E_Event']
    train_cols_C = active_C + ['T_Duration', 'E_Event']
    
    cph_A = engine.fit(train_X[train_cols_A], duration_col='T_Duration', event_col='E_Event', check_ph=False)
    cph_C = engine.fit(train_X[train_cols_C], duration_col='T_Duration', event_col='E_Event', check_ph=False)
    
    y_train = np.array(list(zip(train_X['E_Event'], train_X['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
    tau = np.percentile(y_train['Duration'], 90)
    
    max_train_time = y_train['Duration'].max() - 1e-5
    
    n_boot = 1000 # Increased to 1000 for stability
    np.random.seed(42)
    
    delta_cs = []
    
    print("Starting bootstrap...")
    for i in range(n_boot):
        boot_idx = np.random.choice(len(test_X), len(test_X), replace=True)
        test_boot = test_X.iloc[boot_idx].copy()
        
        y_test_boot = np.array(list(zip(test_boot['E_Event'], test_boot['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
        
        mask = (y_test_boot['Duration'] < max_train_time) | (y_test_boot['Event'] == True)
        if not mask.any():
            continue
            
        test_boot = test_boot[mask]
        y_test_boot = y_test_boot[mask]
        
        try:
            pred_A = cph_A.predict_partial_hazard(test_boot[active_A]).values
            pred_C = cph_C.predict_partial_hazard(test_boot[active_C]).values
            
            c_A = concordance_index_ipcw(y_train, y_test_boot, pred_A, tau=tau)[0]
            c_C = concordance_index_ipcw(y_train, y_test_boot, pred_C, tau=tau)[0]
            
            delta_cs.append(c_C - c_A)
        except Exception as e:
            print(f"Iter {i} error: {e}")
            continue
            
    if not delta_cs:
        print("Error: No successful bootstrap iterations.")
        return
        
    delta_cs = np.array(delta_cs)
    mean_delta = np.mean(delta_cs)
    from src.framework.stats import compute_percentile_ci
    ci_lower, ci_upper = compute_percentile_ci(delta_cs)
    
    print(f"Completed {len(delta_cs)} successful iterations.")
    print(f"Paired Bootstrap Delta C-index (Model C - Model A): {mean_delta:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print("Does CI cross zero?", "Yes" if ci_lower < 0 and ci_upper > 0 else "No")
    
    res = pd.DataFrame([{
        'Mean_Delta_C': mean_delta,
        '95_CI_Lower': ci_lower,
        '95_CI_Upper': ci_upper
    }])
    res.to_csv(repo_root / "outputs" / "pipeline_freeze" / "results" / "paired_bootstrap_delta_cindex.csv", index=False)
    print("Saved paired_bootstrap_delta_cindex.csv")

if __name__ == "__main__":
    main()
