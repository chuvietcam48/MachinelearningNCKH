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

def run_ablation():
    print("Loading data...")
    df = pd.read_csv(OUT / "phase2" / "customer_augmented_v3.csv")
    df["T"] = df["T"].clip(lower=0.5)

    dates = df["first_purchase_date"].sort_values().unique()
    n = len(dates)
    idx_val = int(n * 0.7)
    idx_test = int(n * 0.85)

    val_start_date = pd.to_datetime(dates[idx_val])
    test_start_date = pd.to_datetime(dates[idx_test])

    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    train_df = df[df["first_purchase_date"] < val_start_date].copy()
    val_df   = df[(df["first_purchase_date"] >= val_start_date) & (df["first_purchase_date"] < test_start_date)].copy()
    test_df  = df[df["first_purchase_date"] >= test_start_date].copy()
    
    def eval_features(name, features):
        print(f"\n--- Evaluating {name} ---")
        cols = [c for c in features if c in df.columns]
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
                         
        pipe.fit(train_df[cols])
        df_train_sc = pd.DataFrame(pipe.transform(train_df[cols]), columns=cols, index=train_df.index)
        df_test_sc  = pd.DataFrame(pipe.transform(test_df[cols]), columns=cols, index=test_df.index)
        
        df_train_sc["T"] = train_df["T"].values
        df_train_sc["E"] = train_df["E"].values
        df_test_sc["T"]  = test_df["T"].values
        df_test_sc["E"]  = test_df["E"].values

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waf = WeibullAFTFitter(penalizer=0.01)
            waf.fit(df_train_sc[cols + ["T", "E"]], duration_col="T", event_col="E")
            
        c_train = waf.concordance_index_
        c_test = waf.score(df_test_sc[cols + ["T", "E"]], scoring_method="concordance_index")
        print(f"C-OOS = {c_test:.4f}")

    eval_features("Model_A (Original)", ["Recency", "Frequency", "Monetary", "InterPurchaseTime", "GapDeviation"])
    eval_features("Model_A (Drop Recency)", ["Frequency", "Monetary", "InterPurchaseTime", "GapDeviation"])
    eval_features("Model_A (Drop Recency & IPT)", ["Frequency", "Monetary", "GapDeviation"])
    eval_features("Model_A (Drop Recency, IPT, Frequency)", ["Monetary", "GapDeviation"])

if __name__ == "__main__":
    run_ablation()
