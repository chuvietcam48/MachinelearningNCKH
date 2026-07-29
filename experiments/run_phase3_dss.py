import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
OUT_LEGACY = PROJECT_ROOT / "archive_legacy" / "outputs" / "amazon_cds_v3"
OUT_UNIVERSAL = PROJECT_ROOT / "outputs" / "pipeline_freeze"
EXP_OUT = PROJECT_ROOT / "experiments" / "outputs"

def simulate_dss(df, duration_col, event_col, risk_scores, cost_per_intervention=10, value_per_saved=100, intervention_success_rate=0.3, budget_limit=200):
    """
    Simulates a retention campaign DSS based on predicted risk.
    """
    # Create a dataframe for simulation
    sim_df = pd.DataFrame({
        "T": df[duration_col].values,
        "E": df[event_col].values,
        "Risk": risk_scores
    })
    
    # Targeting policy: target top N users based on budget
    max_targets = budget_limit
    
    sim_df = sim_df.sort_values(by="Risk", ascending=False).reset_index(drop=True)
    sim_df["Targeted"] = False
    sim_df.loc[:max_targets-1, "Targeted"] = True
    
    # Calculate metrics
    targeted_df = sim_df[sim_df["Targeted"]]
    
    # True At Risk: E=1 (or T < some threshold)
    # Since E=1 means churned, targeting them means they were correctly identified as at risk.
    true_positives = targeted_df["E"].sum()
    false_positives = len(targeted_df) - true_positives
    
    targeting_precision = true_positives / len(targeted_df) if len(targeted_df) > 0 else 0
    false_incentive_rate = false_positives / len(targeted_df) if len(targeted_df) > 0 else 0
    
    expected_saved = true_positives * intervention_success_rate
    cost = len(targeted_df) * cost_per_intervention
    benefit = expected_saved * value_per_saved
    
    expected_net_benefit = benefit - cost
    roi = (expected_net_benefit / cost) if cost > 0 else 0
    resource_allocation_efficiency = benefit / cost if cost > 0 else 0
    
    return {
        "Targeting_Precision": targeting_precision,
        "False_Incentive_Rate": false_incentive_rate,
        "Expected_Saved_Customers": expected_saved,
        "Cost": cost,
        "Benefit": benefit,
        "Expected_Net_Benefit": expected_net_benefit,
        "ROI": roi,
        "Resource_Allocation_Efficiency": resource_allocation_efficiency
    }

def run_phase3():
    print("--- Phase 3: EXP 3 Decision Support (Business Value) ---")
    
    cohort_df = pd.read_csv(EXP_OUT / "exact_cohort_intersect.csv")
    overlapping_customers = set(cohort_df["CustomerID"])
    
    legacy_file = OUT_LEGACY / "phase2" / "customer_augmented_v3.csv"
    df_legacy = pd.read_csv(legacy_file)
    df_legacy["T"] = df_legacy["T"].clip(lower=0.5)
    df_sem_legacy = df_legacy[df_legacy['CustomerID'].isin(overlapping_customers)].copy()
    
    train_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_train_semantic.parquet")
    val_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_val_semantic.parquet")
    test_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_test_semantic.parquet")
    sem_universal = pd.concat([train_u, val_u, test_u])
    df_sem_univ = sem_universal[sem_universal["CustomerID"].isin(overlapping_customers)].groupby("CustomerID").last().reset_index()
    
    llm_cols = [
        'net_sentiment', 'positive_minus_negative_ratio', 'aspect_entropy', 
        'semantic_variance', 'rolling_negative_ratio', 'negative_streak', 'sentiment_flip_rate'
    ]
    sem_agg = sem_universal.groupby('CustomerID')[llm_cols].last().reset_index()
    df_sem_legacy = pd.merge(df_sem_legacy, sem_agg, on="CustomerID", how="left")
    
    legacy_llm = ["Recency", "Frequency", "Monetary", "InterPurchaseTime", "GapDeviation"] + llm_cols
    univ_llm = [
        "episode_review_count", "prior_verified_episode_count", "prior_review_count", 
        "historical_mean_rating", "days_since_previous_episode", "customer_lifetime", 
        "review_frequency", "recent_low_rating_count"
    ] + llm_cols

    def get_risk_scores(df, features, duration_col, event_col):
        cols = [c for c in features if c in df.columns]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")), 
            ("var", VarianceThreshold(threshold=1e-5)),
            ("sc", StandardScaler())
        ])
        
        X_trans = pipe.fit_transform(df[cols])
        remaining_cols = np.array(cols)[pipe.named_steps["var"].get_support()]
        X_scaled = pd.DataFrame(X_trans, columns=remaining_cols, index=df.index)
        
        X_scaled[duration_col] = df[duration_col].values + np.random.uniform(0.01, 0.1, size=len(df))
        X_scaled[event_col] = df[event_col].values
        
        cph = CoxPHFitter(penalizer=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(X_scaled, duration_col=duration_col, event_col=event_col)
            # Risk score is simply the partial hazard
            return cph.predict_partial_hazard(X_scaled).values
        except Exception as e:
            print(f"Model fit failed: {e}")
            return np.zeros(len(X_scaled))

    print("\nFitting models and generating risk scores...")
    legacy_risk = get_risk_scores(df_sem_legacy, legacy_llm, "T", "E")
    univ_risk = get_risk_scores(df_sem_univ, univ_llm, "T_Duration", "E_Event")
    
    print("\nSimulating Decision Support System campaigns...")
    legacy_dss = simulate_dss(df_sem_legacy, "T", "E", legacy_risk)
    univ_dss = simulate_dss(df_sem_univ, "T_Duration", "E_Event", univ_risk)
    
    print("\nHistorical Framework DSS Results:")
    for k, v in legacy_dss.items():
        print(f"  {k}: {v:.2f}")
        
    print("\nUniversal Framework DSS Results:")
    for k, v in univ_dss.items():
        print(f"  {k}: {v:.2f}")
        
    dss_results = {
        "Historical_DSS": legacy_dss,
        "Universal_DSS": univ_dss
    }
    
    with open(EXP_OUT / "exp3_dss_evaluation.json", "w") as f:
        json.dump(dss_results, f, indent=4)
        
    print("\nPhase 3 Complete.")

if __name__ == "__main__":
    run_phase3()
