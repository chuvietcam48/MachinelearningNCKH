import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def main():
    dataset_dir = repo_root / "outputs" / "pipeline_freeze" / "datasets"
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("10: Retrospective Baseline Evaluation")
    print("=" * 60)
    
    try:
        train_df = pd.read_parquet(dataset_dir / "master_train_semantic.parquet")
        test_df = pd.read_parquet(dataset_dir / "master_test_semantic.parquet")
        val_df = pd.read_parquet(dataset_dir / "master_val_semantic.parquet")
        df_all = pd.concat([train_df, val_df, test_df], ignore_index=True)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
        
    with open(dataset_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    features_A = registry["Model_A"]["features"]
    features_C = registry["Model_C"]["features"]
    
    # Define t_0 as the 70th percentile of all episode_start dates
    t_0 = df_all['episode_start'].quantile(0.70)
    
    # Filter only episodes that happened before t_0
    df_past = df_all[df_all['episode_endpoint'] <= t_0].copy()
    
    # Compute tau (95th percentile of inter-event times in the past)
    tau = df_past[df_past['E_Event'] == 1]['T_Duration'].quantile(0.95)
    
    print(f"Computed t_0: {t_0}")
    print(f"Computed tau: {tau:.2f} days")
    
    # For each customer, get their LAST episode before t_0
    idx_last = df_past.groupby('CustomerID')['episode_start'].idxmax()
    df_snap = df_past.loc[idx_last].copy()
    
    # Define Label Y: interaction happens in (t_0, t_0 + tau]
    # next_interaction_date = episode_endpoint + T_Duration
    df_snap['next_interaction_date'] = df_snap['episode_endpoint'] + pd.to_timedelta(df_snap['T_Duration'], unit='D')
    
    def get_label(row):
        if row['E_Event'] == 0:
            return 0
        nid = row['next_interaction_date']
        t_max = t_0 + pd.Timedelta(days=tau)
        if t_0 < nid <= t_max:
            return 1
        return 0
        
    df_snap['Y_label'] = df_snap.apply(get_label, axis=1)
    
    print(f"Snapshot cohort size: {len(df_snap)}")
    print(f"Positive labels (churn/inactive): {len(df_snap) - df_snap['Y_label'].sum()}? Wait, Y=1 means RETURNED.")
    print(f"Returned within tau: {df_snap['Y_label'].sum()} ({df_snap['Y_label'].mean():.1%})")
    
    # Customer-level Train/Test Split (70/30)
    train_snap, test_snap = train_test_split(df_snap, test_size=0.3, random_state=42, stratify=df_snap['Y_label'])
    
    # Preprocessing
    all_features = list(set(features_A + features_C))
    imputer = SimpleImputer(strategy='median')
    train_imp = pd.DataFrame(imputer.fit_transform(train_snap[all_features]), columns=all_features)
    test_imp = pd.DataFrame(imputer.transform(test_snap[all_features]), columns=all_features)
    
    # Variance and collinearity
    zero_var = train_imp.columns[train_imp.var() <= 0.01]
    train_imp = train_imp.drop(columns=zero_var)
    test_imp = test_imp.drop(columns=zero_var)
    
    corr = train_imp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    collinear = [c for c in upper.columns if any(upper[c] > 0.9)]
    train_imp = train_imp.drop(columns=collinear)
    test_imp = test_imp.drop(columns=collinear)
    
    remaining = list(train_imp.columns)
    
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imp)
    test_scaled = scaler.transform(test_imp)
    
    train_X = pd.DataFrame(train_scaled, columns=remaining)
    test_X = pd.DataFrame(test_scaled, columns=remaining)
    
    y_train = train_snap['Y_label'].values
    y_test = test_snap['Y_label'].values
    
    # Train Logistic Regression
    act_A = [f for f in features_A if f in remaining]
    act_C = [f for f in features_C if f in remaining]
    
    clf_A = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_A.fit(train_X[act_A], y_train)
    pred_A = clf_A.predict_proba(test_X[act_A])[:, 1]
    auc_A = roc_auc_score(y_test, pred_A)
    
    clf_C = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_C.fit(train_X[act_C], y_train)
    pred_C = clf_C.predict_proba(test_X[act_C])[:, 1]
    auc_C = roc_auc_score(y_test, pred_C)
    
    delta_auc = auc_C - auc_A
    
    print(f"Retro-A (Behavior Only) AUC: {auc_A:.4f}")
    print(f"Retro-C (Behavior + Semantic) AUC: {auc_C:.4f}")
    print(f"Delta AUC: {delta_auc:+.4f}")
    
    res_df = pd.DataFrame([{
        "setting": "Retrospective Snapshot",
        "model": "Retro-A",
        "auc": auc_A,
        "delta_vs_behavior": 0.0,
        "n": len(df_snap),
        "tau": tau,
        "t0": str(t_0)
    }, {
        "setting": "Retrospective Snapshot",
        "model": "Retro-C",
        "auc": auc_C,
        "delta_vs_behavior": delta_auc,
        "n": len(df_snap),
        "tau": tau,
        "t0": str(t_0)
    }])
    
    res_df.to_csv(res_dir / "retrospective_vs_episodic_delta.csv", index=False)
    print("\nRetrospective baseline completed and exported.")

if __name__ == "__main__":
    main()
