import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main():
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("13: Evaluation Protocol Manifest")
    print("=" * 60)
    
    try:
        train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
        test_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    n_train_episodes = len(train_df)
    n_test_episodes = len(test_df)
    
    train_custs = set(train_df['CustomerID'].unique())
    test_custs = set(test_df['CustomerID'].unique())
    
    overlap_custs = train_custs.intersection(test_custs)
    overlap_pct = len(overlap_custs) / len(test_custs) if len(test_custs) > 0 else 0.0
    
    # Censoring rates
    train_censored = (train_df['E_Event'] == 0).mean()
    test_censored = (test_df['E_Event'] == 0).mean()
    
    n_events_test = test_df['E_Event'].sum()
    
    # t* used in Decision Support (median T on train)
    t_star = np.median(train_df['T_Duration'])
    
    # tau used in Retrospective
    try:
        df_retro = pd.read_csv(res_dir / "retrospective_vs_episodic_delta.csv")
        tau = float(df_retro['tau'].iloc[0])
        t0_retro = str(df_retro['t0'].iloc[0])
    except:
        tau = None
        t0_retro = None
        
    protocol = {
        "N_train_episodes": n_train_episodes,
        "N_test_episodes": n_test_episodes,
        "N_train_customers": len(train_custs),
        "N_test_customers": len(test_custs),
        "Customer_overlap_count": len(overlap_custs),
        "Customer_overlap_pct": overlap_pct,
        "Train_censoring_rate": float(train_censored),
        "Test_censoring_rate": float(test_censored),
        "N_events_test": int(n_events_test),
        "Decision_Support_t_star_days": float(t_star),
        "Retrospective_tau_days": float(tau) if tau else None,
        "Retrospective_t0": t0_retro,
        "Penalizer": 0.001,
        "Seed": 42
    }
    
    with open(res_dir / "eval_protocol.json", "w") as f:
        json.dump(protocol, f, indent=4)
        
    print(json.dumps(protocol, indent=4))
    print("\nEvaluation Protocol Manifest exported successfully.")

if __name__ == "__main__":
    main()
