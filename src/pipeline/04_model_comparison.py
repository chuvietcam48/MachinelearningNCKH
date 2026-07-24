import pandas as pd
import numpy as np
import json
from pathlib import Path
from sksurv.metrics import concordance_index_censored
from joblib import Parallel, delayed

def compute_c_index_for_sample(y_test, pred_A, pred_B, pred_C, n_test_pool):
    idx = np.random.choice(n_test_pool, 10000, replace=True)
    if y_test['Event'][idx].sum() == 0 or y_test['Event'][idx].sum() == 10000:
        return None
        
    try:
        cA, _, _, _, _ = concordance_index_censored(y_test['Event'][idx], y_test['Duration'][idx], pred_A[idx])
        cB, _, _, _, _ = concordance_index_censored(y_test['Event'][idx], y_test['Duration'][idx], pred_B[idx])
        cC, _, _, _, _ = concordance_index_censored(y_test['Event'][idx], y_test['Duration'][idx], pred_C[idx])
        return (cA, cB, cC)
    except:
        return None

def evaluate_cohort(cohort_name, pred_dir):
    print(f"\nEvaluating Cohort: {cohort_name}")
    print("-" * 50)
    
    try:
        df_A = pd.read_parquet(pred_dir / f"prediction_model_a_{cohort_name.lower()}.parquet")
        df_B = pd.read_parquet(pred_dir / f"prediction_model_b_{cohort_name.lower()}.parquet")
        df_C = pd.read_parquet(pred_dir / f"prediction_model_c_{cohort_name.lower()}.parquet")
    except Exception as e:
        print(f"Predictions not found for {cohort_name}.")
        return None
        
    np.random.seed(42)
    n_total = len(df_A)
    sample_size = min(n_total, 20000)
    sub_idx = np.random.choice(n_total, sample_size, replace=False)
    
    y_test = np.array(
        list(zip(df_A['E_Event'].iloc[sub_idx], df_A['T_Duration'].iloc[sub_idx])),
        dtype=[('Event', '?'), ('Duration', '<f8')]
    )
    
    pred_A = df_A['risk_score'].iloc[sub_idx].values
    pred_B = df_B['risk_score'].iloc[sub_idx].values
    pred_C = df_C['risk_score'].iloc[sub_idx].values
    
    print(f"Calculating Baseline C-index on {len(y_test)} random samples (Fast Mode)...")
    try:
        orig_cA, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_A)
        orig_cB, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_B)
        orig_cC, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_C)
    except Exception as e:
        print(f"Failed to calculate c-index for {cohort_name}: {e}")
        return None
        
    n_boot = 1000
    n_test_pool = len(y_test)
    
    print(f"Running parallel bootstrap (1000 iter, sample size=10k)...")
    results = Parallel(n_jobs=4, verbose=0)( 
        delayed(compute_c_index_for_sample)(y_test, pred_A, pred_B, pred_C, n_test_pool) 
        for _ in range(n_boot)
    )
    
    valid_results = [r for r in results if r is not None]
    if len(valid_results) == 0:
        return None
        
    boot_c_A = np.array([r[0] for r in valid_results])
    boot_c_B = np.array([r[1] for r in valid_results])
    boot_c_C = np.array([r[2] for r in valid_results])
    
    ci_A = (np.percentile(boot_c_A, 2.5), np.percentile(boot_c_A, 97.5))
    ci_B = (np.percentile(boot_c_B, 2.5), np.percentile(boot_c_B, 97.5))
    ci_C = (np.percentile(boot_c_C, 2.5), np.percentile(boot_c_C, 97.5))
    
    print(f"\nFINAL RESULTS ({cohort_name})")
    print(f"Model A (Behavior)       : C-index = {orig_cA:.4f}  | 95% CI: [{ci_A[0]:.4f}, {ci_A[1]:.4f}]")
    print(f"Model B (+ VADER)        : C-index = {orig_cB:.4f}  | 95% CI: [{ci_B[0]:.4f}, {ci_B[1]:.4f}]")
    print(f"Model C (+ LLM Semantic) : C-index = {orig_cC:.4f}  | 95% CI: [{ci_C[0]:.4f}, {ci_C[1]:.4f}]")
    
    return {
        "Model_A": {"C-index": orig_cA, "C-index_95_CI": ci_A},
        "Model_B": {"C-index": orig_cB, "C-index_95_CI": ci_B},
        "Model_C": {"C-index": orig_cC, "C-index_95_CI": ci_C}
    }

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    pred_dir = repo_root / "outputs" / "pipeline_freeze" / "predictions"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("04: Model Comparison & Metric Evaluation")
    print("=" * 70)
    
    cohorts = ["Full", "Semantic"]
    all_metrics = {}
    
    for cohort in cohorts:
        res = evaluate_cohort(cohort, pred_dir)
        if res:
            all_metrics[cohort] = res
            
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        if isinstance(o, tuple): return list(o)
        raise TypeError
        
    with open(res_dir / "model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4, default=convert)
        
    print("\nMetrics saved to results/model_metrics.json")

if __name__ == "__main__":
    main()
