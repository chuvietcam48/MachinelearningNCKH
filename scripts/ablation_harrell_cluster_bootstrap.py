"""scripts/ablation_harrell_cluster_bootstrap.py
=================================================
Semantic-descriptor group ablation (Table IV / Fig. 2) using the procedure
stated in the paper:

  * Discrimination : Harrell concordance index (sksurv.concordance_index_censored)
  * Resampling     : customer-level (reviewerID) cluster bootstrap, B = 500, seed = 42
  * Models         : Model A (12 behavioural features) vs Model A + group,
                     lifelines CoxPH via SurvivalEngine(penalizer=0.001), fit on train only
  * Margin         : +/- 0.03 on Delta C

For transparency the script also recomputes the legacy procedure used by
scratch/ablation_ci.py (commit 5924510): Uno IPCW C-index (tau = 90th pct of
train durations) with row-level resampling.

Outputs (results/ is tracked in git):
  results/ablation_33_groups.json
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw

sys.path.insert(0, os.path.abspath('.'))
warnings.filterwarnings('ignore')
from src.framework.survival_engine import SurvivalEngine  # noqa: E402

SEED = 42
N_BOOT = 500
MARGIN = 0.03
DATA_DIR = 'outputs/amazon_v5_rebuild/gate10_baseline/'
OUT_PATH = 'results/ablation_33_groups.json'


def harrell(event, time, risk):
    return concordance_index_censored(event.astype(bool), time.astype(float), risk)[0]


def main():
    df_train = pd.read_parquet(DATA_DIR + 'master_train_semantic.parquet').fillna(0)
    df_test = pd.read_parquet(DATA_DIR + 'master_test_semantic.parquet').fillna(0)
    with open(DATA_DIR + 'feature_registry.json', 'r') as f:
        reg = json.load(f)

    feats_A = reg['Model_A']['features']
    feats_C = reg['Model_C']['features']
    feats_Sem = [f for f in feats_C if f not in feats_A]

    groups = {
        'Full semantic block': list(feats_Sem),
        'Sentiment': ['net_sentiment', 'sentiment_profile', 'sentiment_flip_rate', 'semantic_variance'],
        'Aspect-specific counts': [f for f in feats_Sem if 'Positive' in f or 'Negative' in f],
        'Temporal/streak': ['negative_streak', 'positive_streak', 'aspect_switch_rate',
                            'aspect_transition_pattern', 'days_since_last_negative',
                            'previous_negative_count', 'previous_total_aspects',
                            'rolling_negative_ratio'],
        'Dominance/concentration': ['dominance_ratio'],
        'Dispersion': ['aspect_entropy'],
        'Conflict': ['same_aspect_conflict', 'cross_aspect_conflict', 'has_conflict'],
    }

    # Partition checks (6 disjoint groups == the 33 semantic descriptors)
    parts = [f for g, fs in groups.items() if g != 'Full semantic block' for f in fs]
    assert len(feats_Sem) == 33, len(feats_Sem)
    assert len(parts) == 33 and len(set(parts)) == 33, 'overlap or wrong count'
    assert set(parts) == set(feats_Sem), set(parts) ^ set(feats_Sem)
    for g in ('sentiment_profile', 'aspect_transition_pattern'):
        assert g in df_train.columns, g  # single column each (no one-hot expansion)

    def fit(feats):
        eng = SurvivalEngine(penalizer=0.001)
        return eng.fit(df_train[feats + ['T_Duration', 'E_Event']],
                       duration_col='T_Duration', event_col='E_Event')

    cph_A = fit(feats_A)
    models = {g: fit(feats_A + fs) for g, fs in groups.items()}

    ev = df_test['E_Event'].values.astype(bool)
    tm = df_test['T_Duration'].values.astype(float)
    risk_A = cph_A.predict_partial_hazard(df_test[feats_A]).values
    risk = {g: models[g].predict_partial_hazard(df_test[feats_A + fs]).values
            for g, fs in groups.items()}

    # ---- Harrell C + customer-cluster bootstrap (paper procedure) ----------
    cA = harrell(ev, tm, risk_A)
    point = {g: harrell(ev, tm, risk[g]) - cA for g in groups}

    cust = df_test['reviewerID'].values
    uniq = np.unique(cust)
    rows_of = {c: np.where(cust == c)[0] for c in uniq}
    rng = np.random.RandomState(SEED)
    # 270-day return label: all censoring in the test split is administrative at
    # t* = 270 d, so label = E_Event with no exclusions. 1 - S(270) is monotone in
    # the partial hazard, so AUC is computed on the partial hazard directly.
    assert (tm[~ev] == 270).all(), 'non-administrative censoring present'
    lab = (ev & (tm <= 270)).astype(int)
    rC = risk['Full semantic block']
    auc_point = roc_auc_score(lab, rC) - roc_auc_score(lab, risk_A)
    boot_auc = []

    boot = {g: [] for g in groups}
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_of[c] for c in pick])
        cA_b = harrell(ev[idx], tm[idx], risk_A[idx])
        for g in groups:
            boot[g].append(harrell(ev[idx], tm[idx], risk[g][idx]) - cA_b)
        boot_auc.append(roc_auc_score(lab[idx], rC[idx]) - roc_auc_score(lab[idx], risk_A[idx]))

    # ---- Legacy procedure (ablation_ci.py): IPCW + row bootstrap ------------
    y_tr = np.array(list(zip(df_train['E_Event'].astype(bool), df_train['T_Duration'])),
                    dtype=[('Event', '?'), ('Duration', '<f8')])
    y_te = np.array(list(zip(ev, tm)), dtype=[('Event', '?'), ('Duration', '<f8')])
    tau = np.percentile(y_tr['Duration'], 90)
    ipcw = lambda y, r: concordance_index_ipcw(y_tr, y, r, tau=tau)[0]  # noqa: E731
    point_ipcw = {g: ipcw(y_te, risk[g]) - ipcw(y_te, risk_A) for g in groups}
    np.random.seed(SEED)
    boot_ipcw = {g: [] for g in groups}
    for _ in range(N_BOOT):
        idx = np.random.choice(len(df_test), len(df_test), replace=True)
        try:
            a = ipcw(y_te[idx], risk_A[idx])
            for g in groups:
                boot_ipcw[g].append(ipcw(y_te[idx], risk[g][idx]) - a)
        except Exception:
            pass

    out = {
        'procedure': {
            'metric': 'Harrell C (sksurv.concordance_index_censored)',
            'bootstrap': 'customer-level cluster bootstrap on test split (reviewerID)',
            'B': N_BOOT, 'seed': SEED, 'margin': MARGIN,
            'n_test_episodes': int(len(df_test)), 'n_test_customers': int(len(uniq)),
            'n_test_events': int(ev.sum()),
            'C_model_A_harrell': float(cA),
        },
        'groups': [],
    }
    for g, fs in groups.items():
        lo, hi = np.percentile(boot[g], [2.5, 97.5])
        lo_i, hi_i = np.percentile(boot_ipcw[g], [2.5, 97.5])
        out['groups'].append({
            'group': g, 'n_vars': len(fs),
            'delta_c': float(point[g]), 'ci_lower': float(lo), 'ci_upper': float(hi),
            'within_margin': bool(lo >= -MARGIN and hi <= MARGIN),
            'n_boot_ok': len(boot[g]),
            'legacy_ipcw_row': {'delta_c': float(point_ipcw[g]), 'ci_lower': float(lo_i),
                                'ci_upper': float(hi_i), 'n_boot_ok': len(boot_ipcw[g])},
        })

    lo_a, hi_a = np.percentile(boot_auc, [2.5, 97.5])
    out['full_block_auc270'] = {
        'n_vars': 33, 'delta_auc': float(auc_point), 'ci_lower': float(lo_a),
        'ci_upper': float(hi_a), 'margin': 0.04,
        'within_margin': bool(lo_a >= -0.04 and hi_a <= 0.04), 'n_boot_ok': len(boot_auc),
    }

    os.makedirs('results', exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out['procedure'], indent=2))
    print(f"\n{'group':26s} {'n':>3s} | {'Harrell dC':>10s} {'95% CI':>20s} in? | "
          f"{'IPCW dC':>8s} {'95% CI (legacy)':>20s}")
    for r in out['groups']:
        L = r['legacy_ipcw_row']
        print(f"{r['group']:26s} {r['n_vars']:3d} | {r['delta_c']:+10.4f} "
              f"[{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}] {'Y' if r['within_margin'] else 'N'}   | "
              f"{L['delta_c']:+8.4f} [{L['ci_lower']:+.4f}, {L['ci_upper']:+.4f}]")
    a = out['full_block_auc270']
    print(f"\nFull block dAUC(270d): {a['delta_auc']:+.4f} [{a['ci_lower']:+.4f}, {a['ci_upper']:+.4f}]")
    print(f'\nSaved {OUT_PATH}')


if __name__ == '__main__':
    main()
