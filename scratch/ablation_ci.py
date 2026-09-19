import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
from sksurv.metrics import concordance_index_ipcw
from lifelines import CoxPHFitter
import json
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')

from src.framework.survival_engine import SurvivalEngine

dir_path = 'outputs/amazon_v5_rebuild/gate10_baseline/'
df_train = pd.read_parquet(dir_path + 'master_train_semantic.parquet').fillna(0)
df_test = pd.read_parquet(dir_path + 'master_test_semantic.parquet').fillna(0)

with open(dir_path + 'feature_registry.json', 'r') as f:
    reg = json.load(f)

feats_A = reg['Model_A']['features']
feats_C = reg['Model_C']['features']
feats_Sem = [f for f in feats_C if f not in feats_A]

y_train = np.array(list(zip(df_train['E_Event'], df_train['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
y_test = np.array(list(zip(df_test['E_Event'], df_test['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
tau = np.percentile(y_train['Duration'], 90)
max_train_time = y_train['Duration'].max() - 1e-5

def compute_model(train_df, feats):
    engine = SurvivalEngine(penalizer=0.001)
    if 'CustomerID' not in train_df.columns and 'customer_unique_id' in train_df.columns:
        train_df['CustomerID'] = train_df['customer_unique_id']
    cols = feats + ['T_Duration', 'E_Event']
    if 'CustomerID' in train_df.columns: cols.append('CustomerID')
    return engine.fit(train_df[cols], duration_col='T_Duration', event_col='E_Event')

cph_A = compute_model(df_train, feats_A)

groups = {
    'dominance/concentration': ['dominance_ratio'],
    'dispersion': ['aspect_entropy'],
    'conflict': ['same_aspect_conflict', 'cross_aspect_conflict', 'has_conflict'],
    'sentiment': ['net_sentiment', 'sentiment_profile', 'sentiment_flip_rate', 'rolling_negative_ratio'],
    'aspect-specific counts': [f for f in feats_Sem if 'Positive' in f or 'Negative' in f],
    'temporal/streak': ['negative_streak', 'positive_streak', 'aspect_switch_rate', 'aspect_transition_pattern', 'days_since_last_negative', 'previous_negative_count', 'previous_total_aspects']
}

models = {}
for g_name, g_feats in groups.items():
    curr_feats = [f for f in g_feats if f in feats_Sem]
    if curr_feats:
        models[g_name] = compute_model(df_train, feats_A + curr_feats)
        
np.random.seed(42)
n_boot = 500
boot_deltas = {g_name: [] for g_name in models.keys()}
point_deltas = {}

for g_name, cph_grp in models.items():
    curr_feats = [f for f in groups[g_name] if f in feats_Sem]
    pred_A = cph_A.predict_partial_hazard(df_test[feats_A]).values
    pred_grp = cph_grp.predict_partial_hazard(df_test[feats_A + curr_feats]).values
    c_A_full = concordance_index_ipcw(y_train, y_test, pred_A, tau=tau)[0]
    c_grp_full = concordance_index_ipcw(y_train, y_test, pred_grp, tau=tau)[0]
    point_deltas[g_name] = c_grp_full - c_A_full

for i in range(n_boot):
    idx = np.random.choice(len(df_test), len(df_test), replace=True)
    test_boot = df_test.iloc[idx].copy()
    y_boot = np.array(list(zip(test_boot['E_Event'], test_boot['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
    mask = (y_boot['Duration'] < max_train_time) | (y_boot['Event'] == True)
    if not mask.any(): continue
    
    try:
        pred_A = cph_A.predict_partial_hazard(test_boot[feats_A]).values
        c_A_b = concordance_index_ipcw(y_train, y_boot, pred_A, tau=tau)[0]
        
        for g_name, cph_grp in models.items():
            curr_feats = [f for f in groups[g_name] if f in feats_Sem]
            pred_grp = cph_grp.predict_partial_hazard(test_boot[feats_A + curr_feats]).values
            c_grp_b = concordance_index_ipcw(y_train, y_boot, pred_grp, tau=tau)[0]
            boot_deltas[g_name].append(c_grp_b - c_A_b)
    except:
        pass

print("=== ABLATION BOOTSTRAP CI (Delta C-index) ===")
for g_name in models.keys():
    deltas = boot_deltas[g_name]
    ci_lower = np.percentile(deltas, 2.5)
    ci_upper = np.percentile(deltas, 97.5)
    print(f"{g_name}: Point Delta = {point_deltas[g_name]:.4f}, 95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")
