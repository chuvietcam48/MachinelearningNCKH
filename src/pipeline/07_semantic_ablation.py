import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sksurv.metrics import concordance_index_censored
import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from src.framework.survival_engine import SurvivalEngine

def main():
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("07: Semantic Ablation Study")
    print("=" * 70)
    
    with open(dataset_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    features_base = registry["Model_A"]["features"]
    
    # Define Ablation Groups
    group_mean = ["net_sentiment", "sentiment_profile", "Domain_Experience_Positive", "Domain_Experience_Negative",
                  "Product_Performance_Usability_Positive", "Product_Performance_Usability_Negative",
                  "Product_Condition_Quality_Positive", "Product_Condition_Quality_Negative",
                  "Packaging_Presentation_Positive", "Packaging_Presentation_Negative",
                  "Delivery_Fulfillment_Positive", "Delivery_Fulfillment_Negative",
                  "Customer_Service_Returns_Positive", "Customer_Service_Returns_Negative",
                  "Listing_Expectation_Compatibility_Positive", "Listing_Expectation_Compatibility_Negative",
                  "Price_Value_Positive", "Price_Value_Negative"]
                  
    group_var = ["semantic_variance"]
    group_flip = ["sentiment_flip_rate", "aspect_switch_rate", "aspect_transition_pattern", "negative_streak", "positive_streak"]
    group_entropy = ["aspect_entropy", "dominance_ratio"]
    group_conflict = ["same_aspect_conflict", "cross_aspect_conflict", "has_conflict"]
    
    ablation_steps = [
        ("Behavior Only", features_base),
        ("+ Mean/Aspects", features_base + group_mean),
        ("+ Variance", features_base + group_mean + group_var),
        ("+ Flip Rate", features_base + group_mean + group_var + group_flip),
        ("+ Entropy", features_base + group_mean + group_var + group_flip + group_entropy),
        ("All (Full Semantic)", registry["Model_C"]["features"])
    ]
    
    train_path = dataset_dir / "master_train_semantic.parquet"
    test_path = dataset_dir / "master_test_semantic.parquet"
    
    if not train_path.exists():
        print("Required dataset not found.")
        return
        
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    all_features = registry["Model_C"]["features"]
    
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_df[all_features]), columns=all_features)
    test_imp = pd.DataFrame(imputer.transform(test_df[all_features]), columns=all_features)
    
    # Drop low variance
    zero_var_cols = train_imp.columns[train_imp.var() <= 0.01]
    train_imp = train_imp.drop(columns=zero_var_cols)
    test_imp = test_imp.drop(columns=zero_var_cols)
    
    # Drop collinear
    corr = train_imp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear_cols = [column for column in upper.columns if any(upper[column] > 0.9)]
    train_imp = train_imp.drop(columns=collinear_cols)
    test_imp = test_imp.drop(columns=collinear_cols)
    
    remaining_features = list(train_imp.columns)
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imp)
    test_scaled = scaler.transform(test_imp)
    
    train_X = pd.DataFrame(train_scaled, columns=remaining_features)
    test_X = pd.DataFrame(test_scaled, columns=remaining_features)
    
    train_X['E_Event'] = train_df['E_Event'].values
    train_X['T_Duration'] = train_df['T_Duration'].values
    if 'CustomerID' in train_df.columns:
        train_X['CustomerID'] = train_df['CustomerID'].values
        
    # Align T_Duration for test to calculate C-index
    test_df_eval = test_df.copy()
    test_df_eval.loc[test_df_eval['E_Event'] == 0, 'T_Duration'] = 270.0
    
    y_test = np.array(
        list(zip(test_df_eval['E_Event'], test_df_eval['T_Duration'])),
        dtype=[('Event', '?'), ('Duration', '<f8')]
    )
    
    survival_engine = SurvivalEngine(penalizer=0.001)
    
    results = []
    baseline_c = 0.0
    
    for step_name, feats in ablation_steps:
        active_feats = [f for f in feats if f in remaining_features]
        print(f"Training step: {step_name} ({len(active_feats)} features)...")
        
        train_cols = active_feats + ['E_Event', 'T_Duration']
        if 'CustomerID' in train_X.columns:
            train_cols.append('CustomerID')
            
        fit_df = train_X[train_cols].copy()
        
        try:
            cph = survival_engine.fit(fit_df, duration_col='T_Duration', event_col='E_Event', check_ph=False)
            pred = cph.predict_partial_hazard(test_X[active_feats]).values
            
            c_idx, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred)
            
            if step_name == "Behavior Only":
                baseline_c = c_idx
                
            results.append({
                "Features": step_name,
                "C-index": c_idx,
                "Delta_C": c_idx - baseline_c
            })
            print(f"  -> C-index: {c_idx:.4f} (Delta: +{c_idx - baseline_c:.4f})")
            
        except Exception as e:
            print(f"  -> Failed: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "semantic_ablation.csv", index=False)
    
    print("\nFINAL ABLATION TABLE:")
    print(df_res.to_markdown(index=False))
    print("\nSaved to results/semantic_ablation.csv")

if __name__ == "__main__":
    main()
