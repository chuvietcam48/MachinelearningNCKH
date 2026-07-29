import os
import sys
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from lifelines import WeibullAFTFitter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
OUT = PROJECT_ROOT / "archive_legacy" / "outputs" / "amazon_cds_v3"

def run_legacy_llm():
    print("Loading legacy data...")
    df = pd.read_csv(OUT / "phase2" / "customer_augmented_v3.csv")
    df["T"] = df["T"].clip(lower=0.5)

    print("Loading Universal Semantic datasets...")
    # Load the exact semantic users used in Universal
    train_u = pd.read_parquet(PROJECT_ROOT / "outputs" / "pipeline_freeze" / "datasets" / "master_train_semantic.parquet")
    val_u = pd.read_parquet(PROJECT_ROOT / "outputs" / "pipeline_freeze" / "datasets" / "master_val_semantic.parquet")
    test_u = pd.read_parquet(PROJECT_ROOT / "outputs" / "pipeline_freeze" / "datasets" / "master_test_semantic.parquet")
    
    sem_df = pd.concat([train_u, val_u, test_u])
    
    # Check what columns exist
    llm_cols = [
        'net_sentiment', 'positive_minus_negative_ratio', 'aspect_entropy', 
        'semantic_variance', 'rolling_negative_ratio', 'negative_streak', 'sentiment_flip_rate'
    ]
    llm_cols = [c for c in llm_cols if c in sem_df.columns]
    
    # Aggregate semantic features to 1 per customer to match legacy format
    sem_agg = sem_df.groupby('CustomerID')[llm_cols].last().reset_index()
    
    # Identify exact overlap
    overlapping_customers = set(df['CustomerID']).intersection(set(sem_agg['CustomerID']))
    print(f"Exact Customer Overlap (Legacy & Universal Semantic): {len(overlapping_customers)}")
    
    if len(overlapping_customers) == 0:
        print("No overlap found. Cannot compute.")
        return

    # Filter to only overlapping customers
    df_sem = df[df['CustomerID'].isin(overlapping_customers)].copy()
    
    # Merge LLM features
    df_sem = pd.merge(df_sem, sem_agg, on="CustomerID", how="left")
    
    # Use standard 5-fold cross validation since the dataset is only ~500 users
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def eval_features(name, features):
        print(f"\n--- Evaluating {name} ---")
        cols = [c for c in features if c in df_sem.columns]
        
        c_scores = []
        for train_idx, test_idx in kf.split(df_sem):
            train_df = df_sem.iloc[train_idx]
            test_df = df_sem.iloc[test_idx]
            
            pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc",  StandardScaler())])
                             
            pipe.fit(train_df[cols])
            df_train_sc = pd.DataFrame(pipe.transform(train_df[cols]), columns=cols, index=train_df.index)
            df_test_sc  = pd.DataFrame(pipe.transform(test_df[cols]), columns=cols, index=test_df.index)
            
            df_train_sc["T"] = train_df["T"].values
            df_train_sc["E"] = train_df["E"].values
            df_test_sc["T"]  = test_df["T"].values
            df_test_sc["E"]  = test_df["E"].values

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    waf = WeibullAFTFitter(penalizer=0.01)
                    waf.fit(df_train_sc[cols + ["T", "E"]], duration_col="T", event_col="E")
                    
                c_test = waf.score(df_test_sc[cols + ["T", "E"]], scoring_method="concordance_index")
                c_scores.append(c_test)
            except Exception as e:
                pass
                
        if len(c_scores) > 0:
            print(f"Mean C-OOS (5-fold) = {np.mean(c_scores):.4f}")
        else:
            print("Failed to fit any folds.")

    base_features = ["Recency", "Frequency", "Monetary", "InterPurchaseTime", "GapDeviation"]
    vader_features = base_features + ["sentiment_latest", "sentiment_mean", "sentiment_volatility", "negative_review_rate", "mixed_flag"]
    llm_features = base_features + llm_cols

    eval_features("Legacy Behavior", base_features)
    eval_features("Legacy + VADER", vader_features)
    eval_features("Legacy + LLM", llm_features)

if __name__ == "__main__":
    run_legacy_llm()
