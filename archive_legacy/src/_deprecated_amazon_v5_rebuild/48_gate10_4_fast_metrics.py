import pandas as pd
import numpy as np
import json
from pathlib import Path
from sksurv.metrics import concordance_index_censored
from joblib import Parallel, delayed
import sys

def compute_c_index_for_sample(y_test, pred_A, pred_B, pred_C, n_test):
    idx = np.random.choice(n_test, n_test, replace=True)
    if y_test['Event'][idx].sum() == 0 or y_test['Event'][idx].sum() == n_test:
        return None
        
    try:
        cA, _, _, _, _ = concordance_index_censored(y_test['Event'][idx], y_test['Duration'][idx], pred_A[idx])
        cB, _, _, _, _ = concordance_index_censored(y_test['Event'][idx], y_test['Duration'][idx], pred_B[idx])
        cC, _, _, _, _ = concordance_index_censored(y_test['Event'][idx], y_test['Duration'][idx], pred_C[idx])
        return (cA, cB, cC)
    except:
        return None

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    
    print("Loading predictions...")
    try:
        df_A = pd.read_parquet(gate10_dir / "prediction_model_a.parquet")
        df_B = pd.read_parquet(gate10_dir / "prediction_model_b.parquet")
        df_C = pd.read_parquet(gate10_dir / "prediction_model_c.parquet")
    except Exception as e:
        print("Predictions not found. Please wait for the main script to finish training.")
        sys.exit(1)
        
    y_test = np.array(
        list(zip(df_A['E_Event'], df_A['T_Duration'])),
        dtype=[('Event', '?'), ('Duration', '<f8')]
    )
    
    pred_A = df_A['risk_score'].values
    pred_B = df_B['risk_score'].values
    pred_C = df_C['risk_score'].values
    
    # Fast Bootstrap
    n_boot = 1000
    n_test = len(y_test)
    
    print(f"Running parallel bootstrap for {n_boot} iterations on {n_test} samples...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(compute_c_index_for_sample)(y_test, pred_A, pred_B, pred_C, n_test) 
        for _ in range(n_boot)
    )
    
    valid_results = [r for r in results if r is not None]
    
    boot_c_A = np.array([r[0] for r in valid_results])
    boot_c_B = np.array([r[1] for r in valid_results])
    boot_c_C = np.array([r[2] for r in valid_results])
    
    # Calculate original C-index
    orig_cA, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_A)
    orig_cB, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_B)
    orig_cC, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_C)
    
    ci_A = (np.percentile(boot_c_A, 2.5), np.percentile(boot_c_A, 97.5))
    ci_B = (np.percentile(boot_c_B, 2.5), np.percentile(boot_c_B, 97.5))
    ci_C = (np.percentile(boot_c_C, 2.5), np.percentile(boot_c_C, 97.5))
    
    print("\n" + "="*60)
    print("FINAL GATE 10 RESULTS (C-INDEX & BOOTSTRAP)")
    print("="*60)
    print(f"Model A (Behavior)       : C-index = {orig_cA:.4f}  | 95% CI: [{ci_A[0]:.4f}, {ci_A[1]:.4f}]")
    print(f"Model B (+ VADER)        : C-index = {orig_cB:.4f}  | 95% CI: [{ci_B[0]:.4f}, {ci_B[1]:.4f}]")
    print(f"Model C (+ LLM Semantic) : C-index = {orig_cC:.4f}  | 95% CI: [{ci_C[0]:.4f}, {ci_C[1]:.4f}]")
    print("-" * 60)
    
    p_boot_B_vs_A = np.mean(boot_c_B <= boot_c_A)
    p_boot_C_vs_A = np.mean(boot_c_C <= boot_c_A)
    p_boot_C_vs_B = np.mean(boot_c_C <= boot_c_B)
    
    print("STATISTICAL SIGNIFICANCE (Empirical p-values):")
    print(f"Model B > Model A : p-value = {p_boot_B_vs_A:.4e}")
    print(f"Model C > Model A : p-value = {p_boot_C_vs_A:.4e}")
    print(f"Model C > Model B : p-value = {p_boot_C_vs_B:.4e}")
    print("="*60)
    
    metrics = {
        "Model_A": {"C-index": orig_cA, "C-index_95_CI": ci_A},
        "Model_B": {"C-index": orig_cB, "C-index_95_CI": ci_B, "Bootstrap_vs_A_pvalue": p_boot_B_vs_A},
        "Model_C": {"C-index": orig_cC, "C-index_95_CI": ci_C, "Bootstrap_vs_A_pvalue": p_boot_C_vs_A, "Bootstrap_vs_B_pvalue": p_boot_C_vs_B}
    }
    
    # Convert tuples to lists for JSON
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        if isinstance(o, tuple): return list(o)
        raise TypeError
        
    with open(gate10_dir / "gate10_metrics_fast.json", "w") as f:
        json.dump(metrics, f, indent=4, default=convert)
        
    print("Metrics saved to gate10_metrics_fast.json")

if __name__ == "__main__":
    main()
