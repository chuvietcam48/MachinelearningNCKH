"""
E1: CDNOW Behavioral Baseline — Final Run for Export
=====================================================
Runs the framework on CDNOW, captures Cox summary and intervention 
distribution, saves to results/E1_cdnow/.
"""
import sys
import os
import io
import json
import numpy as np
import pandas as pd
import logging

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.framework.dataset_adapter import DatasetAdapter
from src.framework.behavior_engine import BehaviorEngine
from src.framework.survival_engine import SurvivalEngine
from src.framework.decision_support_engine import DecisionSupportEngine
from src.dataset_registry import get_dataset

OUT_DIR = os.path.join(ROOT, 'results', 'E1_cdnow')
os.makedirs(OUT_DIR, exist_ok=True)

# Capture all output
log_lines = []
class TeeLogger:
    def __init__(self, original):
        self.original = original
    def write(self, msg):
        self.original.write(msg)
        if msg.strip():
            log_lines.append(msg)
    def flush(self):
        self.original.flush()

old_stdout = sys.stdout
sys.stdout = TeeLogger(old_stdout)

# === Run Pipeline ===
adapter = DatasetAdapter(dataset_name='cdnow')
df_raw = adapter.load_data()
registry_info = get_dataset('cdnow')
snapshot_date = registry_info.snapshot_fn(df_raw)

behavior_engine = BehaviorEngine(snapshot_date=snapshot_date)
df_behavior = behavior_engine.process(df_raw)

survival_engine = SurvivalEngine(
    penalizer=adapter.config['survival_model']['penalizer'],
    l1_ratio=adapter.config['survival_model']['l1_ratio']
)

df_train = df_behavior.sample(frac=0.8, random_state=42).copy()
df_test = df_behavior.drop(df_train.index).copy()

cph = survival_engine.fit(df_train)

# Capture Cox summary
summary_buf = io.StringIO()
cph.print_summary(columns=['coef', 'exp(coef)', 'se(coef)', 'z', 'p', '-log2(p)'])
# lifelines prints to stdout, so capture it differently
import contextlib
summary_str = io.StringIO()
with contextlib.redirect_stdout(summary_str):
    cph.print_summary(columns=['coef', 'exp(coef)', 'se(coef)', 'z', 'p', '-log2(p)'])
cox_text = summary_str.getvalue()

# Save Cox summary
with open(os.path.join(OUT_DIR, 'cox_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(cox_text)
    f.write(f"\n\nAIC: {cph.AIC_partial_:.2f}")
    f.write(f"\nConcordance: {cph.concordance_index_:.4f}")

# Save HR table
hr_df = cph.summary[['exp(coef)', 'exp(coef) lower 95%', 'exp(coef) upper 95%', 'p']].copy()
hr_df.columns = ['Hazard_Ratio', 'CI_Lower', 'CI_Upper', 'p_value']
hr_df.index.name = 'Feature'
hr_df.to_csv(os.path.join(OUT_DIR, 'hazard_ratios.csv'))

# Decision Engine
df_test_numeric = df_test.select_dtypes(include=[np.number])
valid_cols = [c for c in df_test_numeric.columns if c in cph.params_.index]
risk_scores = cph.predict_partial_hazard(df_test_numeric[valid_cols])

dss_engine = DecisionSupportEngine(
    high_risk_threshold=adapter.config['decision_support']['high_risk_threshold']
)
df_policy = dss_engine.route_customers(df_test, risk_scores)

# Save intervention distribution
intervention_counts = df_policy['Recommended_Intervention'].value_counts()
intervention_counts.to_csv(os.path.join(OUT_DIR, 'intervention_distribution.csv'))

# Save metrics
metrics = {
    'dataset': 'CDNOW',
    'n_customers': len(df_behavior),
    'n_train': len(df_train),
    'n_test': len(df_test),
    'seed': 42,
    'AIC': cph.AIC_partial_,
    'concordance_index': cph.concordance_index_,
    'n_features': len(cph.params_),
}
with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

sys.stdout = old_stdout

# Save run log
with open(os.path.join(OUT_DIR, 'run_log.txt'), 'w', encoding='utf-8') as f:
    f.write('\n'.join(str(l) for l in log_lines))

print(f"\n✅ E1 (CDNOW) export complete → results/E1_cdnow/")
print(f"   AIC: {cph.AIC_partial_:.2f}")
print(f"   C-index: {cph.concordance_index_:.4f}")
print(f"   Interventions: {dict(intervention_counts)}")
