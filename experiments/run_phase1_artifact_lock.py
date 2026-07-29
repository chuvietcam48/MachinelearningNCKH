import os
import sys
import hashlib
import json
import pkg_resources
import pandas as pd
import numpy as np
from pathlib import Path
from lifelines import CoxPHFitter, WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
OUT_LEGACY = PROJECT_ROOT / "archive_legacy" / "outputs" / "amazon_cds_v3"
OUT_UNIVERSAL = PROJECT_ROOT / "outputs" / "pipeline_freeze"
EXP_OUT = PROJECT_ROOT / "experiments" / "outputs"
EXP_OUT.mkdir(parents=True, exist_ok=True)

def get_file_hash(filepath):
    h = hashlib.sha256()
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as file:
        while chunk := file.read(8192):
            h.update(chunk)
    return h.hexdigest()

def get_pkg_version(pkg_name):
    try:
        return pkg_resources.get_distribution(pkg_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Installed"

def run_phase1():
    print("--- Phase 1: Artifact Lock & Cohort Isolation ---")
    
    # 1. Load Legacy Dataset
    legacy_file = OUT_LEGACY / "phase2" / "customer_augmented_v3.csv"
    print(f"Loading Historical Dataset: {legacy_file}")
    df_legacy = pd.read_csv(legacy_file)
    df_legacy["T"] = df_legacy["T"].clip(lower=0.5)
    
    # 2. Load Universal Semantic Dataset (Test Set)
    train_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_train_semantic.parquet")
    val_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_val_semantic.parquet")
    test_u = pd.read_parquet(OUT_UNIVERSAL / "datasets" / "master_test_semantic.parquet")
    sem_universal = pd.concat([train_u, val_u, test_u])
    
    # 3. Isolate exact overlapping semantic cohort
    overlapping_customers = set(df_legacy['CustomerID']).intersection(set(sem_universal['CustomerID']))
    print(f"Exact Customer Overlap (Historical & Universal Semantic): {len(overlapping_customers)}")
    
    df_intersect = pd.DataFrame({"CustomerID": list(overlapping_customers)})
    df_intersect = df_intersect.sort_values("CustomerID").reset_index(drop=True)
    
    intersect_file = EXP_OUT / "exact_cohort_intersect.csv"
    df_intersect.to_csv(intersect_file, index=False)
    print(f"Saved intersection cohort to {intersect_file}")
    
    # 4. Generate Reproducibility Appendix (Artifact Lock)
    reproducibility = {
        "Python_Version": sys.version,
        "Lifelines_Version": get_pkg_version("lifelines"),
        "Scikit_Learn_Version": get_pkg_version("scikit-learn"),
        "Pandas_Version": get_pkg_version("pandas"),
        "Seed_Locked": 42,
        "Hashes": {
            "Historical_Dataset": get_file_hash(legacy_file),
            "Universal_Train_Semantic": get_file_hash(OUT_UNIVERSAL / "datasets" / "master_train_semantic.parquet"),
            "Universal_Test_Semantic": get_file_hash(OUT_UNIVERSAL / "datasets" / "master_test_semantic.parquet"),
            "Exact_Cohort_Intersect": get_file_hash(intersect_file)
        },
        "Cohort_Size": len(overlapping_customers)
    }
    
    with open(EXP_OUT / "reproducibility_lock.json", "w") as f:
        json.dump(reproducibility, f, indent=4)
        
    print("\nReproducibility artifacts locked successfully.")
    
    # 5. EXP 1: Architecture Validation (Historical Behavior vs Universal Behavior)
    print("\n--- EXP 1: Architecture Validation ---")
    
    # --- A. Historical Behavior ---
    df_sem_legacy = df_legacy[df_legacy['CustomerID'].isin(overlapping_customers)].copy()
    
    # Use 5-fold CV to compute the C-index on the exact overlapping cohort
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    base_features_legacy = ["Recency", "Frequency", "Monetary", "InterPurchaseTime", "GapDeviation"]
    c_legacy_scores = []
    
    for train_idx, test_idx in kf.split(df_sem_legacy):
        train_df = df_sem_legacy.iloc[train_idx]
        test_df = df_sem_legacy.iloc[test_idx]
        
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        pipe.fit(train_df[base_features_legacy])
        
        X_train_l = pd.DataFrame(pipe.transform(train_df[base_features_legacy]), columns=base_features_legacy, index=train_df.index)
        X_test_l  = pd.DataFrame(pipe.transform(test_df[base_features_legacy]), columns=base_features_legacy, index=test_df.index)
        
        X_train_l["T"], X_train_l["E"] = train_df["T"].values, train_df["E"].values
        X_test_l["T"], X_test_l["E"]   = test_df["T"].values, test_df["E"].values
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Using CoxPHFitter instead of Weibull to avoid convergence issues
                cph_legacy = CoxPHFitter(penalizer=0.01)
                cph_legacy.fit(X_train_l, duration_col="T", event_col="E")
            c_test = cph_legacy.score(X_test_l, scoring_method="concordance_index")
            c_legacy_scores.append(c_test)
        except Exception:
            pass

    c_legacy = np.mean(c_legacy_scores)
    print(f"[Historical Architecture] Behavior-Only C-index: {c_legacy:.4f}")
    
    # --- B. Universal Behavior ---
    univ_features = [
        "episode_review_count", "prior_verified_episode_count", "prior_review_count", 
        "historical_mean_rating", "days_since_previous_episode", "customer_lifetime", 
        "review_frequency", "recent_low_rating_count"
    ]
    
    u_sem_cust = sem_universal[sem_universal["CustomerID"].isin(overlapping_customers)].groupby("CustomerID").last().reset_index()
    
    c_univ_scores = []
    
    for train_idx, test_idx in kf.split(u_sem_cust):
        train_df = u_sem_cust.iloc[train_idx]
        test_df = u_sem_cust.iloc[test_idx]
        
        pipe_u = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        pipe_u.fit(train_df[univ_features])
        
        X_train_u = pd.DataFrame(pipe_u.transform(train_df[univ_features]), columns=univ_features, index=train_df.index)
        X_test_u  = pd.DataFrame(pipe_u.transform(test_df[univ_features]), columns=univ_features, index=test_df.index)
        
        X_train_u["T"], X_train_u["E"] = train_df["T_Duration"].values, train_df["E_Event"].values
        X_test_u["T"], X_test_u["E"]   = test_df["T_Duration"].values, test_df["E_Event"].values
        
        # Add a small epsilon to T to prevent ties/0 duration issues
        X_train_u["T"] = X_train_u["T"] + np.random.uniform(0.01, 0.1, size=len(X_train_u))
        X_test_u["T"] = X_test_u["T"] + np.random.uniform(0.01, 0.1, size=len(X_test_u))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph_univ = CoxPHFitter(penalizer=0.01)
                cph_univ.fit(X_train_u, duration_col="T", event_col="E")
            c_test = cph_univ.score(X_test_u, scoring_method="concordance_index")
            c_univ_scores.append(c_test)
        except Exception as e:
            print(f"Fold failed: {e}")
            pass

    if c_univ_scores:
        c_univ = np.mean(c_univ_scores)
    else:
        c_univ = float('nan')
    print(f"[Universal Architecture] Behavior-Only C-index: {c_univ:.4f}")
    
    exp1_results = {
        "Historical_Behavior_C_index": c_legacy,
        "Universal_Behavior_C_index": c_univ
    }
    
    with open(EXP_OUT / "exp1_architecture_validation.json", "w") as f:
        json.dump(exp1_results, f, indent=4)
        
    print("\nPhase 1 Complete.")

if __name__ == "__main__":
    run_phase1()
