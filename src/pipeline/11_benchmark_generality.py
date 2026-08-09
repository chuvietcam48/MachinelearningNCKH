import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings

import sys
repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dataset_registry import get_dataset
from src.framework.dataset_adapter import DatasetAdapter
from src.framework.behavior_engine import BehaviorEngine
from src.framework.survival_engine import SurvivalEngine
from sksurv.metrics import concordance_index_censored

def main():
    res_dir = repo_root / "outputs" / "pipeline_freeze" / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    
    print("11: Multi-Dataset Benchmark Generality")
    print("=" * 60)
    
    datasets = ["cdnow", "uci", "tafeng"]
    results = []
    
    for ds_name in datasets:
        print(f"\nProcessing {ds_name.upper()}...")
        try:
            adapter = DatasetAdapter(dataset_name=ds_name)
            df_raw = adapter.load_data()
            registry_info = get_dataset(ds_name)
            snapshot_date = registry_info.snapshot_fn(df_raw)
            
            behavior_engine = BehaviorEngine(snapshot_date=snapshot_date)
            df = behavior_engine.process(df_raw)
            
            # For generic datasets, BehaviorEngine outputs 1 row per customer at the snapshot date.
            # Thus, we do a randomized out-of-sample split (e.g., 80/20) as in run_framework.py
            df_train = df.sample(frac=0.8, random_state=42).copy()
            df_test = df.drop(df_train.index).copy()
            
            # Extract features (exclude ID and time columns)
            exclude_cols = ['CustomerID', 'E_Event', 'T_Duration']
            features = [c for c in df_train.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df_train[c])]
            
            imputer = SimpleImputer(strategy='median')
            train_imp = pd.DataFrame(imputer.fit_transform(df_train[features]), columns=features)
            test_imp = pd.DataFrame(imputer.transform(df_test[features]), columns=features)
            
            # Drop zero variance
            zero_var = train_imp.columns[train_imp.var() <= 0.01]
            train_imp = train_imp.drop(columns=zero_var)
            test_imp = test_imp.drop(columns=zero_var)
            
            # Drop collinear
            corr = train_imp.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            collinear = [column for column in upper.columns if any(upper[column] > 0.9)]
            train_imp = train_imp.drop(columns=collinear)
            test_imp = test_imp.drop(columns=collinear)
            
            remaining = list(train_imp.columns)
            
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train_imp)
            test_scaled = scaler.transform(test_imp)
            
            train_X = pd.DataFrame(train_scaled, columns=remaining)
            test_X = pd.DataFrame(test_scaled, columns=remaining)
            
            train_X['E_Event'] = df_train['E_Event'].values
            train_X['T_Duration'] = df_train['T_Duration'].values
            
            test_X['E_Event'] = df_test['E_Event'].values
            test_X['T_Duration'] = df_test['T_Duration'].values
            
            survival_engine = SurvivalEngine(penalizer=0.001)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph = survival_engine.fit(train_X, duration_col='T_Duration', event_col='E_Event', check_ph=False)
                
            pred = cph.predict_partial_hazard(test_X[remaining]).values
            
            y_test = np.array(list(zip(test_X['E_Event'], test_X['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
            c_idx, _, _, _, _ = concordance_index_censored(y_test['Event'], y_test['Duration'], pred)
            
            results.append({
                "Dataset": ds_name.upper(),
                "C-index (OOS)": c_idx,
                "Train_Episodes": len(train_X),
                "Test_Episodes": len(test_X),
                "Features_Used": len(remaining)
            })
            print(f"  -> {ds_name.upper()} C-index (OOS): {c_idx:.4f}")
            
        except Exception as e:
            print(f"  -> Error processing {ds_name}: {e}")
            
    df_res = pd.DataFrame(results)
    df_res.to_csv(res_dir / "benchmark_episodic_cox.csv", index=False)
    print("\nBenchmark Evaluation Complete.")
    print(df_res.to_markdown(index=False))

if __name__ == "__main__":
    main()
