import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import numpy as np
import json
from pathlib import Path
import sys

def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    
    print("Loading data for Sensitivity Check: Temporal-only without episode_duration...")
    try:
        df_train = pd.read_parquet(data_dir / "master_train_semantic.parquet")
        df_test = pd.read_parquet(data_dir / "master_test_semantic.parquet")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure the pipeline has been run to generate the datasets.")
        sys.exit(1)

    temporal_features_6 = [
        'days_since_previous_episode',
        'prior_verified_episode_count',
        'prior_review_count',
        'review_frequency',
        'recent_low_rating_count',
        'customer_lifetime'
    ]

    print(f"Features used ({len(temporal_features_6)}):", temporal_features_6)
    print("Excluded feature: 'episode_duration'")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    # Impute missing values with median from train
    df_train[temporal_features_6] = df_train[temporal_features_6].fillna(df_train[temporal_features_6].median())
    df_test[temporal_features_6] = df_test[temporal_features_6].fillna(df_train[temporal_features_6].median())

    # Scale features
    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    
    df_train_scaled[temporal_features_6] = scaler.fit_transform(df_train[temporal_features_6])
    df_test_scaled[temporal_features_6] = scaler.transform(df_test[temporal_features_6])

    print("\nFitting CoxPH model (penalizer=0.001)...")
    cph = CoxPHFitter(penalizer=0.001)
    
    try:
        cph.fit(
            df_train_scaled[temporal_features_6 + ['T_Duration', 'E_Event']],
            duration_col='T_Duration',
            event_col='E_Event'
        )
    except Exception as e:
        print(f"Error fitting model: {e}")
        sys.exit(1)

    # Point estimate
    preds_test = cph.predict_partial_hazard(df_test_scaled)
    c_index_pt = concordance_index(df_test_scaled['T_Duration'], -preds_test, df_test_scaled['E_Event'])
    print(f"Point Estimate C-index: {c_index_pt:.4f}")

    # Bootstrap (500 iterations at customer level)
    print("\nRunning Bootstrap for 95% CI (500 iterations)...")
    n_bootstraps = 500
    np.random.seed(42)
    bootstrapped_c_indices = []

    unique_customers = df_test_scaled['CustomerID'].unique()
    
    for i in range(n_bootstraps):
        # Sample customers with replacement
        sampled_customers = np.random.choice(unique_customers, size=len(unique_customers), replace=True)
        # Reconstruct test set
        # Using a list comprehension to preserve duplicates
        sampled_rows = [df_test_scaled[df_test_scaled['CustomerID'] == cust] for cust in sampled_customers]
        df_boot = pd.concat(sampled_rows)
        
        try:
            boot_preds = cph.predict_partial_hazard(df_boot)
            c_idx = concordance_index(df_boot['T_Duration'], -boot_preds, df_boot['E_Event'])
            bootstrapped_c_indices.append(c_idx)
        except Exception:
            continue
            
        if (i+1) % 100 == 0:
            print(f" Completed {i+1}/500 iterations...")

    lo = np.percentile(bootstrapped_c_indices, 2.5)
    hi = np.percentile(bootstrapped_c_indices, 97.5)
    
    print("\n" + "="*50)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*50)
    print(f"Model: Temporal-only (excluding episode_duration)")
    print(f"Point C-index: {c_index_pt:.3f}")
    print(f"95% CI: [{lo:.3f}, {hi:.3f}]")
    print(f"Official phrase for manuscript:")
    print(f'"removing episode_duration from the temporal-only model changes discrimination from C=0.753 to C={c_index_pt:.3f} (95% CI [{lo:.3f}, {hi:.3f}])"')

if __name__ == "__main__":
    main()
