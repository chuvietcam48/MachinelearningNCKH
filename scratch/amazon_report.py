import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings
from sksurv.metrics import concordance_index_ipcw
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
import json
import sys
import os
sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')

out_file = open('scratch/amazon_report.txt', 'w', encoding='utf-8')
def pr(txt):
    print(txt)
    out_file.write(txt + '\n')

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

from src.framework.survival_engine import SurvivalEngine

def compute_metrics(train_df, test_df, feats, cph_model=None):
    if cph_model is None:
        engine = SurvivalEngine(penalizer=0.001)
        # Use CustomerID if exists, else customer_unique_id mapped to CustomerID
        if 'CustomerID' not in train_df.columns and 'customer_unique_id' in train_df.columns:
            train_df['CustomerID'] = train_df['customer_unique_id']
            test_df['CustomerID'] = test_df['customer_unique_id']
        cols = feats + ['T_Duration', 'E_Event']
        if 'CustomerID' in train_df.columns: cols.append('CustomerID')
        cph_model = engine.fit(train_df[cols], duration_col='T_Duration', event_col='E_Event')
    
    pred = cph_model.predict_partial_hazard(test_df[feats]).values
    c_idx = concordance_index_ipcw(y_train, y_test, pred, tau=tau)[0]
    
    S_t = cph_model.predict_survival_function(test_df[feats], times=[270]).iloc[0].values
    P_return = 1.0 - S_t
    mask = ~((test_df['T_Duration'] < 270) & (test_df['E_Event'] == 0))
    eval_df = test_df[mask].copy()
    y_true = ((eval_df['T_Duration'] <= 270) & (eval_df['E_Event'] == 1)).astype(int)
    auc = roc_auc_score(y_true, P_return[mask])
    return c_idx, auc, cph_model

# Fit Base Models
pr("--- FITTING BASE MODELS ---")
c_A, auc_A, cph_A = compute_metrics(df_train, df_test, feats_A)
c_C, auc_C, cph_C = compute_metrics(df_train, df_test, feats_C)

# 1. Semantic-only model
pr("\n## 1. Semantic-only Model")
c_Sem, auc_Sem, cph_Sem = compute_metrics(df_train, df_test, feats_Sem)
pr(f"Semantic-only C-index: {c_Sem:.4f}")
pr(f"Semantic-only AUC@270: {auc_Sem:.4f}")

# Bootstrap for Semantic-only
np.random.seed(42)
n_boot = 500
boot_c_Sem = []
boot_auc_Sem = []
for i in range(n_boot):
    idx = np.random.choice(len(df_test), len(df_test), replace=True)
    test_boot = df_test.iloc[idx].copy()
    y_boot = np.array(list(zip(test_boot['E_Event'], test_boot['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
    mask = (y_boot['Duration'] < max_train_time) | (y_boot['Event'] == True)
    if not mask.any(): continue
    
    try:
        pred_Sem = cph_Sem.predict_partial_hazard(test_boot[feats_Sem]).values
        c_val = concordance_index_ipcw(y_train, y_boot, pred_Sem, tau=tau)[0]
        
        S_t = cph_Sem.predict_survival_function(test_boot[feats_Sem], times=[270]).iloc[0].values
        P_return = 1.0 - S_t
        eval_mask = ~((test_boot['T_Duration'] < 270) & (test_boot['E_Event'] == 0))
        eval_df = test_boot[eval_mask]
        y_true = ((eval_df['T_Duration'] <= 270) & (eval_df['E_Event'] == 1)).astype(int)
        if len(np.unique(y_true)) > 1:
            a_val = roc_auc_score(y_true, P_return[eval_mask])
            boot_c_Sem.append(c_val)
            boot_auc_Sem.append(a_val)
    except:
        pass

pr(f"Semantic-only C-index 95% CI: [{np.percentile(boot_c_Sem, 2.5):.4f}, {np.percentile(boot_c_Sem, 97.5):.4f}]")
pr(f"Semantic-only AUC 95% CI: [{np.percentile(boot_auc_Sem, 2.5):.4f}, {np.percentile(boot_auc_Sem, 97.5):.4f}]")


# 2. Correlation Matrix + LRT
pr("\n## 2. Correlation & LRT")
corr_matrix = df_train[feats_A + feats_Sem].corr(method='spearman')
high_corr = []
for f_sem in feats_Sem:
    for f_beh in feats_A:
        r = corr_matrix.loc[f_sem, f_beh]
        if abs(r) > 0.15:
            high_corr.append((f_sem, f_beh, r))
high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
pr("Top Spearman Correlations (|r| > 0.15) between Semantic and Behavioral:")
for f_s, f_b, r in high_corr[:10]:
    pr(f"  {f_s} vs {f_b}: {r:.3f}")

ll_A = cph_A.log_likelihood_
ll_C = cph_C.log_likelihood_
LRT_stat = -2 * (ll_A - ll_C)
df_diff = len(feats_C) - len(feats_A)
p_val = stats.chi2.sf(LRT_stat, df_diff)
pr(f"LRT Statistic: {LRT_stat:.2f}, df: {df_diff}, p-value: {p_val:.4e}")


# 3. Absolute CI for Model A and Model C
pr("\n## 3. Absolute CI for Models A and C")
boot_c_A, boot_c_C, boot_c_delta = [], [], []
boot_auc_A, boot_auc_C, boot_auc_delta = [], [], []

for i in range(n_boot):
    idx = np.random.choice(len(df_test), len(df_test), replace=True)
    test_boot = df_test.iloc[idx].copy()
    y_boot = np.array(list(zip(test_boot['E_Event'], test_boot['T_Duration'])), dtype=[('Event', '?'), ('Duration', '<f8')])
    mask = (y_boot['Duration'] < max_train_time) | (y_boot['Event'] == True)
    if not mask.any(): continue
    
    try:
        pred_A = cph_A.predict_partial_hazard(test_boot[feats_A]).values
        pred_C = cph_C.predict_partial_hazard(test_boot[feats_C]).values
        c_A_b = concordance_index_ipcw(y_train, y_boot, pred_A, tau=tau)[0]
        c_C_b = concordance_index_ipcw(y_train, y_boot, pred_C, tau=tau)[0]
        
        eval_mask = ~((test_boot['T_Duration'] < 270) & (test_boot['E_Event'] == 0))
        eval_df = test_boot[eval_mask]
        y_true = ((eval_df['T_Duration'] <= 270) & (eval_df['E_Event'] == 1)).astype(int)
        if len(np.unique(y_true)) > 1:
            P_A = 1.0 - cph_A.predict_survival_function(test_boot[feats_A], times=[270]).iloc[0].values
            P_C = 1.0 - cph_C.predict_survival_function(test_boot[feats_C], times=[270]).iloc[0].values
            a_A_b = roc_auc_score(y_true, P_A[eval_mask])
            a_C_b = roc_auc_score(y_true, P_C[eval_mask])
            
            boot_c_A.append(c_A_b)
            boot_c_C.append(c_C_b)
            boot_c_delta.append(c_C_b - c_A_b)
            boot_auc_A.append(a_A_b)
            boot_auc_C.append(a_C_b)
            boot_auc_delta.append(a_C_b - a_A_b)
    except: pass

pr(f"Model A C-index: {c_A:.4f} 95% CI: [{np.percentile(boot_c_A, 2.5):.4f}, {np.percentile(boot_c_A, 97.5):.4f}]")
pr(f"Model C C-index: {c_C:.4f} 95% CI: [{np.percentile(boot_c_C, 2.5):.4f}, {np.percentile(boot_c_C, 97.5):.4f}]")
pr(f"Model A AUC: {auc_A:.4f} 95% CI: [{np.percentile(boot_auc_A, 2.5):.4f}, {np.percentile(boot_auc_A, 97.5):.4f}]")
pr(f"Model C AUC: {auc_C:.4f} 95% CI: [{np.percentile(boot_auc_C, 2.5):.4f}, {np.percentile(boot_auc_C, 97.5):.4f}]")

ci_c_lower, ci_c_upper = np.percentile(boot_c_delta, 2.5), np.percentile(boot_c_delta, 97.5)
ci_a_lower, ci_a_upper = np.percentile(boot_auc_delta, 2.5), np.percentile(boot_auc_delta, 97.5)


# 4. Equivalence Test (TOST)
pr("\n## 4. Equivalence Test (TOST)")
pr(f"Delta C-index 95% CI: [{ci_c_lower:.4f}, {ci_c_upper:.4f}]")
pr(f"Equivalence Margin: [-0.03, +0.03]")
c_equiv = (ci_c_lower >= -0.03) and (ci_c_upper <= 0.03)
pr(f"C-index Equivalence Reached: {c_equiv}")

pr(f"Delta AUC 95% CI: [{ci_a_lower:.4f}, {ci_a_upper:.4f}]")
pr(f"Equivalence Margin: [-0.04, +0.04]")
a_equiv = (ci_a_lower >= -0.04) and (ci_a_upper <= 0.04)
pr(f"AUC Equivalence Reached: {a_equiv}")


# 5. Feature List
pr("\n## 5. Feature List (Table II)")
pr("Behavioral Features:")
for f in feats_A: pr("  - " + f)
pr("\nSemantic Features:")
for f in feats_Sem: pr("  - " + f)


# 6. Check verified_purchase
pr("\n## 6. Check verified_purchase")
try:
    df_ver = pd.read_parquet('outputs/amazon_v5_rebuild/data/verified_review_feedback_episodes_v1.parquet')
    if 'verified_purchase' in df_ver.columns:
        pr(df_ver['verified_purchase'].value_counts(dropna=False).to_string())
    else:
        pr("verified_purchase column not found.")
except Exception as e:
    pr(f"Error reading verifications: {e}")


# 7. Ablation by Semantic Groups
pr("\n## 7. Semantic Ablation")
groups = {
    'Dominance/Concentration': ['dominance_ratio'],
    'Dispersion': ['aspect_entropy'],
    'Conflict': ['same_aspect_conflict', 'cross_aspect_conflict', 'has_conflict'],
    'Sentiment': ['net_sentiment', 'sentiment_profile', 'sentiment_flip_rate', 'rolling_negative_ratio'],
    'Aspect Counts': [f for f in feats_Sem if 'Positive' in f or 'Negative' in f],
    'Temporal/Streak': ['negative_streak', 'positive_streak', 'aspect_switch_rate', 'aspect_transition_pattern', 'days_since_last_negative', 'previous_negative_count', 'previous_total_aspects']
}

for g_name, g_feats in groups.items():
    # Filter features that exist in feats_Sem
    curr_feats = [f for f in g_feats if f in feats_Sem]
    if not curr_feats: continue
    c_grp, _, _ = compute_metrics(df_train, df_test, feats_A + curr_feats)
    delta_c = c_grp - c_A
    pr(f"{g_name} ({len(curr_feats)} feats) Delta C-index: {delta_c:.4f}")

# 8. Olist MDE Calculation
pr("\n## 8. Olist MDE")
# Approximate MDE based on standard error.
# If SE is roughly CI_width / (2 * 1.96), then MDE = 2.8 * SE for 80% power and alpha=0.05.
olist_ci_width = 0.0821 - (-0.2187)
olist_se = olist_ci_width / (2 * 1.96)
olist_mde = olist_se * (1.96 + 0.84) # 2.8 multiplier
pr(f"Olist CI Width: {olist_ci_width:.4f}")
pr(f"Approx Olist SE: {olist_se:.4f}")
pr(f"Calculated MDE (80% power, alpha=0.05): {olist_mde:.4f}")

out_file.close()
