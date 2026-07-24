import joblib, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.simulator import run_monte_carlo_simulation
from src.models import train_logistic
from src.policy import lr_intervention_decisions

DATASETS = ["UCI_tau124", "TAFENG_tau39", "CDNOW_tau181"]
MC_RESP_MEAN = 0.15
MC_COST_MEAN = 1.0

def main():
    for ds in DATASETS:
        print(f"Processing {ds}...")
        meta_path = f"outputs/{ds}/models/pipeline_meta.pkl"
        data_path = f"outputs/{ds}/models/processed_data.pkl"
        dec_path = f"outputs/{ds}/models/decisions.pkl"
        
        if not os.path.exists(meta_path):
            continue
            
        meta = joblib.load(meta_path)
        customer_df = joblib.load(data_path)
        w_decisions = joblib.load(dec_path)
        
        # Ensure index is CustomerID
        if "CustomerID" in customer_df.columns:
            cdf_idx = customer_df.set_index("CustomerID")
        else:
            cdf_idx = customer_df
            
        # Train LR and get decisions
        print("  Training LR...")
        _, lr_pipe, _ = train_logistic(cdf_idx)
        uplift = pd.Series(MC_RESP_MEAN, index=cdf_idx.index)
        pred_clv = w_decisions.set_index("CustomerID")["predicted_clv"]
        
        lr_decisions = lr_intervention_decisions(
            lr_pipe, cdf_idx, uplift,
            predicted_clv=pred_clv,
            p_response=MC_RESP_MEAN, cost_per_contact=MC_COST_MEAN,
            churn_prob_threshold=0.5
        )
        
        # Run Monte Carlo (uses updated simulator.py with penalty)
        print("  Running Monte Carlo...")
        mc_results = run_monte_carlo_simulation(
            df_decisions=w_decisions,
            n_iterations=1000,
            seed=42,
            lr_decisions=lr_decisions
        )
        
        # Update meta
        meta["monte_carlo_results"] = mc_results
        joblib.dump(meta, meta_path)
        print("  Updated pipeline_meta.pkl")

if __name__ == "__main__":
    main()
