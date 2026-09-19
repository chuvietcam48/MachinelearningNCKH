import pandas as pd
import numpy as np
import json
from pathlib import Path

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, integrated_brier_score, cumulative_dynamic_auc
    from lifelines import WeibullAFTFitter, CoxPHFitter
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import joblib
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please run: pip install scikit-survival lifelines scikit-learn statsmodels")
    exit(1)

def get_structured_target(df):
    return np.array(
        list(zip(df['E_Event'].astype(bool), df['T_Duration'])),
        dtype=[('Event', '?'), ('Duration', '<f8')]
    )

def compute_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    # Add small noise to avoid singular matrix if perfectly collinear
    X_mat = X.values + np.random.normal(0, 1e-6, X.shape)
    vif_data["VIF"] = [variance_inflation_factor(X_mat, i) for i in range(X_mat.shape[1])]
    return vif_data

def bootstrap_c_index(model, X_test, y_test, n_iterations=100):
    scores = []
    n_size = len(X_test)
    for _ in range(n_iterations):
        indices = np.random.choice(n_size, size=n_size, replace=True)
        X_resampled = X_test.iloc[indices]
        y_resampled = y_test[indices]
        
        # Check if we have both events and censored to avoid errors
        if y_resampled['Event'].sum() > 0 and (~y_resampled['Event']).sum() > 0:
            pred = model.predict(X_resampled)
            c_index, _, _, _, _ = concordance_index_censored(y_resampled['Event'], y_resampled['Duration'], pred)
            scores.append(c_index)
            
    if scores:
        return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    return 0, 0, 0

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate9_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate9_baseline"
    
    print("Gate 9.2: Model Training & Evaluation")
    print("-" * 30)
    
    print("Loading datasets...")
    df_train = pd.read_parquet(gate9_dir / "baseline_train.parquet")
    df_val = pd.read_parquet(gate9_dir / "baseline_val.parquet")
    df_test = pd.read_parquet(gate9_dir / "baseline_test.parquet")
    
    # ---------------------------------------------------------
    # Baseline Feature Definition
    # ---------------------------------------------------------
    baseline_features = [
        # Behavioral
        'episode_review_count', 'prior_verified_episode_count', 'prior_review_count',
        'days_since_previous_episode', 'review_frequency',
        # Rating
        'episode_mean_rating', 'episode_min_rating', 'episode_max_rating', 
        'historical_mean_rating', 'recent_low_rating_count',
        # Temporal
        'episode_duration', 'customer_lifetime'
    ]
    
    print(f"Using {len(baseline_features)} baseline features.")
    
    X_train_raw = df_train[baseline_features]
    X_val_raw = df_val[baseline_features]
    X_test_raw = df_test[baseline_features]
    
    y_train = get_structured_target(df_train)
    y_val = get_structured_target(df_val)
    y_test = get_structured_target(df_test)
    
    # ---------------------------------------------------------
    # Gate 9.4: Preprocessing & Training
    # ---------------------------------------------------------
    print("\n[Gate 9.4] Preprocessing & Training Models...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    # Fit on train, transform all
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_train_scaled = scaler.fit_transform(X_train_imp)
    
    X_val_scaled = scaler.transform(imputer.transform(X_val_raw))
    X_test_scaled = scaler.transform(imputer.transform(X_test_raw))
    
    X_train = pd.DataFrame(X_train_scaled, columns=baseline_features)
    X_val = pd.DataFrame(X_val_scaled, columns=baseline_features)
    X_test = pd.DataFrame(X_test_scaled, columns=baseline_features)
    
    # Check VIF on Train
    print("\n[Gate 9.6] Checking VIF (Multicollinearity)...")
    vif_df = compute_vif(X_train)
    print(vif_df.sort_values(by="VIF", ascending=False).head(5))
    vif_df.to_csv(gate9_dir / "baseline_vif.csv", index=False)
    
    # 1. Scikit-Survival Cox PH
    print("\nTraining Scikit-Survival Cox PH...")
    cox_sk = CoxPHSurvivalAnalysis(alpha=1e-4)
    cox_sk.fit(X_train, y_train)
    
    # 2. Lifelines Cox PH (For Interpretation & PH Test)
    print("Training Lifelines Cox PH (for Diagnostics)...")
    train_lifelines = X_train.copy()
    train_lifelines['T_Duration'] = df_train['T_Duration'].values
    train_lifelines['E_Event'] = df_train['E_Event'].values
    
    cph_ll = CoxPHFitter(penalizer=0.01)
    # Using a sample to speed up fitting for diagnostics if dataset is huge, but here we run on all if fast enough
    # If 1.2M rows is too slow for lifelines, we sample 100k
    try:
        if len(train_lifelines) > 100000:
            cph_ll.fit(train_lifelines.sample(100000, random_state=42), duration_col='T_Duration', event_col='E_Event')
        else:
            cph_ll.fit(train_lifelines, duration_col='T_Duration', event_col='E_Event')
    except Exception as e:
        print(f"Lifelines CoxPH failed with penalizer=0.01: {e}\nRetrying with penalizer=0.1...")
        cph_ll = CoxPHFitter(penalizer=0.1)
        if len(train_lifelines) > 100000:
            cph_ll.fit(train_lifelines.sample(100000, random_state=42), duration_col='T_Duration', event_col='E_Event')
        else:
            cph_ll.fit(train_lifelines, duration_col='T_Duration', event_col='E_Event')
        
    print("Testing Proportional Hazards Assumption...")
    try:
        ph_test = cph_ll.check_assumptions(train_lifelines.sample(10000, random_state=42) if len(train_lifelines)>10000 else train_lifelines, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print(f"PH Test skipped/failed: {e}")
        
    cph_summary = cph_ll.summary
    cph_summary.to_csv(gate9_dir / "baseline_coefficients_cox.csv")
    
    # 3. Lifelines Weibull AFT
    print("Training Lifelines Weibull AFT...")
    aft_ll = WeibullAFTFitter(penalizer=0.01)
    try:
        if len(train_lifelines) > 100000:
            aft_ll.fit(train_lifelines.sample(100000, random_state=42), duration_col='T_Duration', event_col='E_Event')
        else:
            aft_ll.fit(train_lifelines, duration_col='T_Duration', event_col='E_Event')
    except Exception as e:
        print(f"Lifelines WeibullAFT failed with penalizer=0.01: {e}\nRetrying with penalizer=0.1...")
        aft_ll = WeibullAFTFitter(penalizer=0.1)
        if len(train_lifelines) > 100000:
            aft_ll.fit(train_lifelines.sample(100000, random_state=42), duration_col='T_Duration', event_col='E_Event')
        else:
            aft_ll.fit(train_lifelines, duration_col='T_Duration', event_col='E_Event')
        
    aft_summary = aft_ll.summary
    aft_summary.to_csv(gate9_dir / "baseline_coefficients_weibull.csv")
    
    # ---------------------------------------------------------
    # Gate 9.5: Advanced Evaluation
    # ---------------------------------------------------------
    print("\n[Gate 9.5] Evaluating on Test Set...")
    
    # C-Index (Scikit-Survival)
    pred_test = cox_sk.predict(X_test)
    c_index, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred_test)
    print(f"Test C-index: {c_index:.4f}")
    
    # Bootstrap CI for C-index
    print("Running Bootstrap for 95% CI (n=50 for speed)...")
    c_mean, c_lower, c_upper = bootstrap_c_index(cox_sk, X_test, y_test, n_iterations=50)
    print(f"Bootstrap C-index: {c_mean:.4f} [{c_lower:.4f} - {c_upper:.4f}]")
    
    # Time-dependent AUC
    times = np.array([90, 180, 260]) # Evaluated at 90, 180, 260 days (must be < max observed)
    try:
        auc, mean_auc = cumulative_dynamic_auc(y_train, y_test, pred_test, times)
        print(f"Time-dependent AUC (90d, 180d, 260d): {auc}")
        print(f"Mean Time-dependent AUC: {mean_auc:.4f}")
    except Exception as e:
        print(f"Could not compute Time-dependent AUC: {e}")
        auc, mean_auc = [], 0
        
    # Log-likelihood
    # Scikit-survival Cox returns log-likelihood
    ll = cox_sk.score(X_test, y_test)
    print(f"Test Score (Log-Likelihood approximation): {ll:.4f}")
    
    # ---------------------------------------------------------
    # Gate 9.6: Interpretation (Top HRs)
    # ---------------------------------------------------------
    print("\n[Gate 9.6] Interpretation...")
    hr_df = pd.DataFrame({
        'Feature': baseline_features,
        'Coef': cox_sk.coef_,
        'HR': np.exp(cox_sk.coef_)
    }).sort_values(by='HR', ascending=False)
    
    print("Top Positive Hazard Ratios (Increases Risk/Returns faster):")
    print(hr_df.head(3))
    print("\nTop Negative Hazard Ratios (Decreases Risk/Takes longer):")
    print(hr_df.tail(3))
    
    # ---------------------------------------------------------
    # Gate 9.7: Freeze Baseline
    # ---------------------------------------------------------
    print("\n[Gate 9.7] Freezing Models and Artifacts...")
    
    metrics = {
        "c_index": float(c_index),
        "c_index_95_ci": [float(c_lower), float(c_upper)],
        "time_dependent_auc": {str(t): float(a) for t, a in zip(times, auc)} if len(auc) > 0 else {},
        "mean_auc": float(mean_auc),
        "test_score": float(ll)
    }
    
    with open(gate9_dir / "baseline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Scikit-Survival Model & Preprocessors
    joblib.dump(cox_sk, gate9_dir / "baseline_model_cox.pkl")
    joblib.dump(imputer, gate9_dir / "baseline_imputer.pkl")
    joblib.dump(scaler, gate9_dir / "baseline_scaler.pkl")
    
    # Save Predictions
    df_test_out = df_test[['reviewerID', 'episode_id', 'T_Duration', 'E_Event']].copy()
    df_test_out['Risk_Score'] = pred_test
    df_test_out.to_parquet(gate9_dir / "baseline_predictions.parquet", index=False)
    
    print("\nBaseline Training completed successfully! Ready for Gate 10.")

if __name__ == "__main__":
    main()
