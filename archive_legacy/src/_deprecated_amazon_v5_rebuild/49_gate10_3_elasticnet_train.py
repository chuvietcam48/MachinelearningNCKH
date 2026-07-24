import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
import warnings

warnings.filterwarnings('ignore')

def get_structured_target(df):
    return np.array(
        list(zip(df['E_Event'], df['T_Duration'])),
        dtype=[('Event', '?'), ('Duration', '<f8')]
    )

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    
    print("Gate 10.3: Robust Training with ElasticNet Cox")
    print("=" * 60)
    print("Using L1/L2 penalty to automatically handle Perfect Multicollinearity.")
    
    # 1. Load Data
    print("\nLoading Master Splits and Feature Registry...")
    try:
        train_df = pd.read_parquet(gate10_dir / "master_train.parquet")
        val_df = pd.read_parquet(gate10_dir / "master_val.parquet")
        test_df = pd.read_parquet(gate10_dir / "master_test.parquet")
        
        with open(gate10_dir / "feature_registry.json", "r") as f:
            registry = json.load(f)
    except Exception as e:
        print(f"Error loading inputs: {e}")
        sys.exit(1)
        
    features_A = registry["Model_A"]["features"]
    features_B = registry["Model_B"]["features"]
    features_C = registry["Model_C"]["features"]
    
    # 2. Preprocessing
    print("Imputing and Scaling features...")
    all_features = list(set(features_A + features_B + features_C))
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    train_imp = imputer.fit_transform(train_df[all_features])
    train_scaled = scaler.fit_transform(train_imp)
    train_X = pd.DataFrame(train_scaled, columns=all_features)
    
    test_X = pd.DataFrame(scaler.transform(imputer.transform(test_df[all_features])), columns=all_features)
    
    y_train = get_structured_target(train_df)
    y_test = get_structured_target(test_df)
    
    # 3. Train Final Models with ElasticNet
    print("\nTraining Models with ElasticNet (l1_ratio=0.9, alpha=0.1)...")
    # l1_ratio=0.9 applies strong Lasso penalty to drop collinear features (like net_sentiment vs pos/neg)
    models = {}
    
    for name, feats in [("Model_A", features_A), ("Model_B", features_B), ("Model_C", features_C)]:
        print(f"  [{name}] Training on {len(feats)} features...")
        
        # Coxnet is highly optimized for high-dimensional collinear data
        clf = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[0.1], fit_baseline_model=True, max_iter=2000)
        clf.fit(train_X[feats], y_train)
        models[name] = clf
        
        # Predict on Test
        print(f"  [{name}] Predicting on Test set...")
        pred = clf.predict(test_X[feats])
        
        # Quick sanity check C-index
        c_index, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred)
        print(f"  [{name}] Base Test C-index: {c_index:.4f}")
        
        # Save predictions
        pred_df = test_df[['episode_id', 'E_Event', 'T_Duration']].copy()
        pred_df['risk_score'] = pred
        pred_df.to_parquet(gate10_dir / f"prediction_{name.lower()}.parquet", index=False)
        
    # 4. Freeze Models
    print("\nFreezing Model Artifacts...")
    for name in ["Model_A", "Model_B", "Model_C"]:
        joblib.dump(models[name], gate10_dir / f"{name.lower()}.pkl")
        
    joblib.dump(scaler, gate10_dir / "gate10_scaler.pkl")
    joblib.dump(imputer, gate10_dir / "gate10_imputer.pkl")
    
    print("\nTraining Complete! Predictions saved.")
    print("-> Next step: Run 'python src/amazon_v5_rebuild/48_gate10_4_fast_metrics.py' to calculate Bootstrap CIs.")

if __name__ == "__main__":
    main()
