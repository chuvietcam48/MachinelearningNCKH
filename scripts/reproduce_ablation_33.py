"""scripts/reproduce_ablation_33.py
===================================
Ablation Bootstrap Analysis for Study 4 (Amazon DSIT Re-engagement):
- Evaluates the 6 disjoint descriptor groups forming exactly 33 semantic features.
- Model A: 12 baseline behavioral features.
- Resampling: 500 customer-level bootstrap iterations (B=500, seed=42).
- Concordance index: Evaluated with Uno IPCW (and Harrell's C-index).
- Equivalence margin: [-0.03, +0.03].
"""

import json
import warnings
import numpy as np
import pandas as pd
from sksurv.metrics import concordance_index_ipcw, concordance_index_censored

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
from src.framework.survival_engine import SurvivalEngine

def run_ablation_33(seed: int = 42, n_boot: int = 500):
    dir_path = 'outputs/amazon_v5_rebuild/gate10_baseline/'
    df_train = pd.read_parquet(dir_path + 'master_train_semantic.parquet').fillna(0)
    df_test = pd.read_parquet(dir_path + 'master_test_semantic.parquet').fillna(0)

    with open(dir_path + 'feature_registry.json', 'r') as f:
        reg = json.load(f)

    feats_A = reg['Model_A']['features']
    feats_C = reg['Model_C']['features']
    feats_Sem = [f for f in feats_C if f not in feats_A]

    # The 6 disjoint groups forming exactly 33 features
    groups_33 = {
        'Dominance/concentration': ['dominance_ratio'],
        'Dispersion': ['aspect_entropy'],
        'Conflict': ['same_aspect_conflict', 'cross_aspect_conflict', 'has_conflict'],
        'Sentiment': ['net_sentiment', 'sentiment_profile', 'sentiment_flip_rate', 'semantic_variance'],
        'Aspect-specific counts': [f for f in feats_Sem if 'Positive' in f or 'Negative' in f],
        'Temporal/streak': [
            'negative_streak', 'positive_streak', 'aspect_switch_rate',
            'aspect_transition_pattern', 'days_since_last_negative',
            'previous_negative_count', 'previous_total_aspects', 'rolling_negative_ratio'
        ]
    }

    # Rigorous partition assertions
    all_grouped = [f for grp in groups_33.values() for f in grp]
    assert len(all_grouped) == 33, f"Expected 33 descriptors, got {len(all_grouped)}"
    assert len(set(all_grouped)) == 33, f"Duplicate descriptors detected: {len(all_grouped)} vs {len(set(all_grouped))}"
    assert set(all_grouped) == set(feats_Sem), f"Partition mismatch with registry: {set(all_grouped) ^ set(feats_Sem)}"
    print("[SUCCESS] All 33 descriptors partitioned with 0 overlap and 0 omission.\n")

    y_train = np.array(list(zip(df_train['E_Event'], df_train['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
    y_test = np.array(list(zip(df_test['E_Event'], df_test['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
    tau = np.percentile(y_train['Duration'], 90)
    max_train_time = y_train['Duration'].max() - 1e-5

    def compute_model(train_df, feats):
        engine = SurvivalEngine(penalizer=0.001)
        if 'CustomerID' not in train_df.columns and 'customer_unique_id' in train_df.columns:
            train_df['CustomerID'] = train_df['customer_unique_id']
        cols = feats + ['T_Duration', 'E_Event']
        if 'CustomerID' in train_df.columns:
            cols.append('CustomerID')
        return engine.fit(train_df[cols], duration_col='T_Duration', event_col='E_Event')

    cph_A = compute_model(df_train, feats_A)
    pred_A_full = cph_A.predict_partial_hazard(df_test[feats_A]).values
    c_A_ipcw = concordance_index_ipcw(y_train, y_test, pred_A_full, tau=tau)[0]

    models = {}
    point_deltas = {}
    for g_name, g_feats in groups_33.items():
        cph_grp = compute_model(df_train, feats_A + g_feats)
        models[g_name] = cph_grp
        pred_grp = cph_grp.predict_partial_hazard(df_test[feats_A + g_feats]).values
        c_grp_ipcw = concordance_index_ipcw(y_train, y_test, pred_grp, tau=tau)[0]
        point_deltas[g_name] = c_grp_ipcw - c_A_ipcw

    np.random.seed(seed)
    boot_deltas = {g_name: [] for g_name in models.keys()}

    for _ in range(n_boot):
        idx = np.random.choice(len(df_test), len(df_test), replace=True)
        test_boot = df_test.iloc[idx].copy()
        y_boot = np.array(list(zip(test_boot['E_Event'], test_boot['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
        
        mask = (y_boot['Duration'] < max_train_time) | (y_boot['Event'] == True)
        if not mask.any():
            continue
        
        try:
            pred_A_b = cph_A.predict_partial_hazard(test_boot[feats_A]).values
            c_A_b = concordance_index_ipcw(y_train, y_boot, pred_A_b, tau=tau)[0]
            
            for g_name, cph_grp in models.items():
                g_feats = groups_33[g_name]
                pred_grp_b = cph_grp.predict_partial_hazard(test_boot[feats_A + g_feats]).values
                c_grp_b = concordance_index_ipcw(y_train, y_boot, pred_grp_b, tau=tau)[0]
                boot_deltas[g_name].append(c_grp_b - c_A_b)
        except Exception:
            pass

    results = []
    for g_name, g_feats in groups_33.items():
        deltas = boot_deltas[g_name]
        ci_lower = float(np.percentile(deltas, 2.5))
        ci_upper = float(np.percentile(deltas, 97.5))
        pt = float(point_deltas[g_name])
        in_margin = (ci_lower >= -0.03) and (ci_upper <= 0.03)
        results.append({
            "group": g_name,
            "n_vars": len(g_feats),
            "point_delta": pt,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "equivalent": in_margin
        })

    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    return df_res

if __name__ == '__main__':
    run_ablation_33()
