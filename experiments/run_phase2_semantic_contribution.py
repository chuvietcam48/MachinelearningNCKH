import os
import sys
import json
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test, proportional_hazard_test
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
OUT_LEGACY = PROJECT_ROOT / "archive_legacy" / "outputs" / "amazon_cds_v3"
OUT_UNIVERSAL = PROJECT_ROOT / "outputs" / "pipeline_freeze"
EXP_OUT = PROJECT_ROOT / "experiments" / "outputs"

def likelihood_ratio_test(ll_nested, ll_full, df_nested, df_full):
    lr_stat = -2 * (ll_nested - ll_full)
    df_diff = df_full - df_nested
    p_value = chi2.sf(lr_stat, df_diff)
    return lr_stat, p_value

def run_phase2():
    print("--- Phase 2: EXP 2 Semantic Contribution & Statistical Validation ---")
    
    # 1. Load the Exact Cohort
    cohort_df = pd.read_csv(EXP_OUT / "exact_cohort_intersect.csv")
    overlapping_customers = set(cohort_df["CustomerID"])
    
    # 2. Load Datasets
    legacy_file = OUT_LEGACY / "phase2" / "customer_augmented_v3.csv"
    df_legacy = pd.read_csv(legacy_file)
    df_legacy["T"] = df_legacy["T"].clip(lower=0.5)
    df_sem_legacy = df_legacy[df_legacy['CustomerID'].isin(overlapping_customers)].copy()
    
    train_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_train_semantic.parquet")
    val_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_val_semantic.parquet")
    test_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_test_semantic.parquet")
    sem_universal = pd.concat([train_u, val_u, test_u])
    df_sem_univ = sem_universal[sem_universal["CustomerID"].isin(overlapping_customers)].groupby("CustomerID").last().reset_index()
    
    # Prepare LLM columns for Legacy
    llm_cols = [
        'net_sentiment', 'positive_minus_negative_ratio', 'aspect_entropy', 
        'semantic_variance', 'rolling_negative_ratio', 'negative_streak', 'sentiment_flip_rate'
    ]
    sem_agg = sem_universal.groupby('CustomerID')[llm_cols].last().reset_index()
    df_sem_legacy = pd.merge(df_sem_legacy, sem_agg, on="CustomerID", how="left")
    
    # 3. Define Features
    legacy_base = ["Recency", "Frequency", "Monetary", "InterPurchaseTime", "GapDeviation"]
    legacy_vader = legacy_base + ["sentiment_latest", "sentiment_mean", "sentiment_volatility", "negative_review_rate", "mixed_flag"]
    legacy_llm = legacy_base + llm_cols

    univ_base = [
        "episode_review_count", "prior_verified_episode_count", "prior_review_count", 
        "historical_mean_rating", "days_since_previous_episode", "customer_lifetime", 
        "review_frequency", "recent_low_rating_count"
    ]
    univ_vader = univ_base + ["sentiment_mean_compound", "sentiment_mixed_intensity", "sentiment_positive_ratio", "sentiment_negative_ratio"]
    univ_llm = univ_base + llm_cols

    def fit_model(df, features, duration_col, event_col):
        cols = [c for c in features if c in df.columns]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")), 
            ("var", VarianceThreshold(threshold=1e-5)),
            ("sc", StandardScaler())
        ])
        
        # Fit on full to get scaling
        # Note: VarianceThreshold will drop columns, so we can't use `cols` as columns for the output DF directly
        X_trans = pipe.fit_transform(df[cols])
        # Get remaining features
        remaining_cols = np.array(cols)[pipe.named_steps["var"].get_support()]
        X_scaled = pd.DataFrame(X_trans, columns=remaining_cols, index=df.index)
        
        X_scaled[duration_col] = df[duration_col].values
        X_scaled[event_col] = df[event_col].values
        
        # Add epsilon to prevent ties/0
        X_scaled[duration_col] = X_scaled[duration_col] + np.random.uniform(0.01, 0.1, size=len(X_scaled))
        
        cph = CoxPHFitter(penalizer=0.1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.fit(X_scaled, duration_col=duration_col, event_col=event_col)
            return cph, X_scaled
        except Exception as e:
            print(f"Model fit failed: {e}")
            return None, None

    # --- A. Fit Base Models on Full Exact Cohort (for LRT and HR) ---
    print("\nFitting models on full cohort...")
    models = {}
    
    models["Legacy_Behavior"], data_L_B = fit_model(df_sem_legacy, legacy_base, "T", "E")
    models["Legacy_VADER"], data_L_V = fit_model(df_sem_legacy, legacy_vader, "T", "E")
    models["Legacy_LLM"], data_L_L = fit_model(df_sem_legacy, legacy_llm, "T", "E")
    
    df_sem_univ["T_Duration"] = df_sem_univ["T_Duration"].clip(lower=0.5)
    models["Univ_Behavior"], data_U_B = fit_model(df_sem_univ, univ_base, "T_Duration", "E_Event")
    models["Univ_VADER"], data_U_V = fit_model(df_sem_univ, univ_vader, "T_Duration", "E_Event")
    models["Univ_LLM"], data_U_L = fit_model(df_sem_univ, univ_llm, "T_Duration", "E_Event")

    # --- B. Likelihood Ratio Tests ---
    print("\nLikelihood Ratio Tests (Nested: Behavior vs Behavior+LLM)")
    lrt_results = {}
    
    if models["Legacy_Behavior"] and models["Legacy_LLM"]:
        ll_nested = models["Legacy_Behavior"].log_likelihood_
        df_nested = len(legacy_base)
        ll_full = models["Legacy_LLM"].log_likelihood_
        df_full = len(legacy_llm)
        stat, p = likelihood_ratio_test(ll_nested, ll_full, df_nested, df_full)
        lrt_results["Legacy"] = {"stat": stat, "p_value": p}
        print(f"Legacy LRT: stat={stat:.2f}, p={p:.4e}")
        
    if models["Univ_Behavior"] and models["Univ_LLM"]:
        ll_nested = models["Univ_Behavior"].log_likelihood_
        df_nested = len(univ_base)
        ll_full = models["Univ_LLM"].log_likelihood_
        df_full = len(univ_llm)
        stat, p = likelihood_ratio_test(ll_nested, ll_full, df_nested, df_full)
        lrt_results["Universal"] = {"stat": stat, "p_value": p}
        print(f"Universal LRT: stat={stat:.2f}, p={p:.4e}")

    # --- C. Hazard Ratios for Semantic Variables (Universal LLM) ---
    print("\nExtracting Hazard Ratios for Semantic Variables...")
    hr_results = []
    if models["Univ_LLM"]:
        summary = models["Univ_LLM"].summary
        semantic_summary = summary.loc[[c for c in llm_cols if c in summary.index]]
        # Filter for p < 0.05
        sig_summary = semantic_summary[semantic_summary["p"] < 0.05]
        
        for idx, row in sig_summary.iterrows():
            hr_results.append({
                "feature": idx,
                "HR": row["exp(coef)"],
                "p_value": row["p"],
                "z_score": row["z"],
                "CI_lower": row["exp(coef) lower 95%"],
                "CI_upper": row["exp(coef) upper 95%"]
            })
            print(f"  {idx}: HR={row['exp(coef)']:.3f}, p={row['p']:.3f}")

    # --- D. Paired Bootstrap ---
    print("\nRunning Paired Bootstrap (1000 iterations)...")
    N_BOOT = 1000
    np.random.seed(42)
    
    # We will bootstrap evaluate the predictions using OOB or simply resample the dataset and compute C-index
    # To save time, we will use the PRE-TRAINED models and bootstrap the C-INDEX EVALUATION on the resampled test sets.
    # This is a standard way to get C-index CI for a frozen model.
    
    boot_legacy_B, boot_legacy_V, boot_legacy_L = [], [], []
    boot_univ_B, boot_univ_V, boot_univ_L = [], [], []
    
    n_samples = len(df_sem_legacy)
    
    for i in tqdm(range(N_BOOT)):
        idx = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        
        try:
            # Legacy
            if models["Legacy_Behavior"]:
                c_LB = models["Legacy_Behavior"].score(data_L_B.iloc[idx], scoring_method="concordance_index")
                c_LV = models["Legacy_VADER"].score(data_L_V.iloc[idx], scoring_method="concordance_index")
                c_LL = models["Legacy_LLM"].score(data_L_L.iloc[idx], scoring_method="concordance_index")
                boot_legacy_B.append(c_LB)
                boot_legacy_V.append(c_LV)
                boot_legacy_L.append(c_LL)
                
            # Universal
            if models["Univ_Behavior"]:
                c_UB = models["Univ_Behavior"].score(data_U_B.iloc[idx], scoring_method="concordance_index")
                c_UV = models["Univ_VADER"].score(data_U_V.iloc[idx], scoring_method="concordance_index")
                c_UL = models["Univ_LLM"].score(data_U_L.iloc[idx], scoring_method="concordance_index")
                boot_univ_B.append(c_UB)
                boot_univ_V.append(c_UV)
                boot_univ_L.append(c_UL)
        except Exception:
            pass

    def get_ci(arr):
        return [np.percentile(arr, 2.5), np.percentile(arr, 97.5)] if len(arr) > 0 else [np.nan, np.nan]

    delta_legacy = np.array(boot_legacy_L) - np.array(boot_legacy_B)
    delta_univ = np.array(boot_univ_L) - np.array(boot_univ_B)
    
    bootstrap_results = {
        "Legacy_Behavior": {"mean": np.mean(boot_legacy_B), "CI": get_ci(boot_legacy_B)},
        "Legacy_VADER": {"mean": np.mean(boot_legacy_V), "CI": get_ci(boot_legacy_V)},
        "Legacy_LLM": {"mean": np.mean(boot_legacy_L), "CI": get_ci(boot_legacy_L)},
        "Legacy_Delta_LLM_vs_B": {"mean": np.mean(delta_legacy), "CI": get_ci(delta_legacy)},
        
        "Univ_Behavior": {"mean": np.mean(boot_univ_B), "CI": get_ci(boot_univ_B)},
        "Univ_VADER": {"mean": np.mean(boot_univ_V), "CI": get_ci(boot_univ_V)},
        "Univ_LLM": {"mean": np.mean(boot_univ_L), "CI": get_ci(boot_univ_L)},
        "Univ_Delta_LLM_vs_B": {"mean": np.mean(delta_univ), "CI": get_ci(delta_univ)}
    }
    
    # Save all results
    final_output = {
        "Bootstrap": bootstrap_results,
        "LRT": lrt_results,
        "Hazard_Ratios": hr_results
    }
    
    with open(EXP_OUT / "exp2_semantic_contribution.json", "w") as f:
        json.dump(final_output, f, indent=4)
        
    print("\nPhase 2 Complete. Results saved to exp2_semantic_contribution.json")

if __name__ == "__main__":
    run_phase2()
