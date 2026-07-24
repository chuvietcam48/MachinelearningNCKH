"""
Export Results for Paper
========================
Reads from outputs/pipeline_freeze/ and old pipeline outputs.
Generates paper-ready tables in results/tables/ and raw copies in results/raw/.

This script does NOT modify any source data or models.
"""
import os
import sys
import json
import shutil
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FREEZE_DIR = os.path.join(ROOT, 'outputs', 'pipeline_freeze')
RESULTS_DIR = os.path.join(ROOT, 'results')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')

for d in [TABLES_DIR, METRICS_DIR, RAW_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# 1. Copy raw metrics
# ─────────────────────────────────────────────────────────────────
src_metrics = os.path.join(FREEZE_DIR, 'results', 'model_metrics.json')
if os.path.exists(src_metrics):
    shutil.copy2(src_metrics, os.path.join(METRICS_DIR, 'model_metrics.json'))
    print("[1] Copied model_metrics.json → results/metrics/")

    # Also split into per-model files for results/raw/
    with open(src_metrics, 'r') as f:
        metrics = json.load(f)
    for cohort, models in metrics.items():
        for model_name, vals in models.items():
            fname = f"{model_name.lower()}_{cohort.lower()}.json"
            with open(os.path.join(RAW_DIR, fname), 'w') as f:
                json.dump({"cohort": cohort, "model": model_name, **vals}, f, indent=2)
    print(f"[1] Split into {len(os.listdir(RAW_DIR))} per-model JSON files → results/raw/")

# ─────────────────────────────────────────────────────────────────
# 2. Table 1: C-index Comparison (all models × all cohorts)
# ─────────────────────────────────────────────────────────────────
if os.path.exists(src_metrics):
    with open(src_metrics, 'r') as f:
        metrics = json.load(f)
    
    rows = []
    for cohort in ['Full', 'Semantic']:
        for model in ['Model_A', 'Model_B', 'Model_C']:
            m = metrics[cohort][model]
            ci = m['C-index_95_CI']
            rows.append({
                'Cohort': cohort,
                'Model': model.replace('_', ' '),
                'C-index': f"{m['C-index']:.4f}",
                'CI_Lower': f"{ci[0]:.4f}",
                'CI_Upper': f"{ci[1]:.4f}",
                'C-index_Display': f"{m['C-index']:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"
            })
    
    with open(os.path.join(TABLES_DIR, 'table1_c_index.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Cohort', 'Model', 'C-index', 'CI_Lower', 'CI_Upper', 'C-index_Display'])
        w.writeheader()
        w.writerows(rows)
    print("[2] Generated table1_c_index.csv")

# ─────────────────────────────────────────────────────────────────
# 3. Table 2: Information Gain (AIC / LRT)
# ─────────────────────────────────────────────────────────────────
src_ig = os.path.join(FREEZE_DIR, 'results', 'information_gain.csv')
if os.path.exists(src_ig):
    shutil.copy2(src_ig, os.path.join(TABLES_DIR, 'table2_information_gain.csv'))
    print("[3] Copied table2_information_gain.csv")

# ─────────────────────────────────────────────────────────────────
# 4. Table 3: Top Hazard Ratios (Semantic Cohort)
# ─────────────────────────────────────────────────────────────────
src_hr = os.path.join(FREEZE_DIR, 'results', 'hazard_ratios_semantic.csv')
if os.path.exists(src_hr):
    import pandas as pd
    df_hr = pd.read_csv(src_hr)
    # Top 10 by absolute distance from HR=1
    df_hr['distance_from_1'] = abs(df_hr['Hazard_Ratio'] - 1)
    df_top = df_hr.nlargest(10, 'distance_from_1').drop(columns=['distance_from_1'])
    df_top.to_csv(os.path.join(TABLES_DIR, 'table3_hazard_ratios_top10.csv'), index=False)
    
    # Also save full HR table
    df_hr.drop(columns=['distance_from_1']).to_csv(
        os.path.join(TABLES_DIR, 'table3_hazard_ratios_full.csv'), index=False)
    print("[4] Generated table3_hazard_ratios_top10.csv + full")

# ─────────────────────────────────────────────────────────────────
# 5. Table 4: Policy Routing Matrix
# ─────────────────────────────────────────────────────────────────
src_policy = os.path.join(FREEZE_DIR, 'results', 'policy', 'policy_routing_matrix.csv')
if os.path.exists(src_policy):
    shutil.copy2(src_policy, os.path.join(TABLES_DIR, 'table4_policy_routing.csv'))
    print("[5] Copied table4_policy_routing.csv")

# ─────────────────────────────────────────────────────────────────
# 6. Table 5: Sensitivity Analysis
# ─────────────────────────────────────────────────────────────────
src_sens = os.path.join(FREEZE_DIR, 'results', 'sensitivity', 'model_robustness_penalizer.csv')
if os.path.exists(src_sens):
    shutil.copy2(src_sens, os.path.join(TABLES_DIR, 'table5_sensitivity_penalizer.csv'))
    print("[6] Copied table5_sensitivity_penalizer.csv")

src_data_rob = os.path.join(FREEZE_DIR, 'results', 'sensitivity', 'data_robustness_coverage.csv')
if os.path.exists(src_data_rob):
    shutil.copy2(src_data_rob, os.path.join(TABLES_DIR, 'table5b_sensitivity_coverage.csv'))

src_policy_rob = os.path.join(FREEZE_DIR, 'results', 'sensitivity', 'policy_robustness_cost.csv')
if os.path.exists(src_policy_rob):
    shutil.copy2(src_policy_rob, os.path.join(TABLES_DIR, 'table5c_sensitivity_policy.csv'))

# ─────────────────────────────────────────────────────────────────
# 7. Copy frozen experiment registry
# ─────────────────────────────────────────────────────────────────
src_reg = os.path.join(FREEZE_DIR, 'experiment_registry.json')
if os.path.exists(src_reg):
    shutil.copy2(src_reg, os.path.join(METRICS_DIR, 'frozen_experiment_registry.json'))
    print("[7] Copied frozen_experiment_registry.json")

# ─────────────────────────────────────────────────────────────────
# 8. Copy full HR tables for both cohorts
# ─────────────────────────────────────────────────────────────────
src_hr_full = os.path.join(FREEZE_DIR, 'results', 'hazard_ratios_full.csv')
if os.path.exists(src_hr_full):
    shutil.copy2(src_hr_full, os.path.join(RAW_DIR, 'hazard_ratios_full_cohort.csv'))
    print("[8] Copied hazard_ratios_full_cohort.csv → results/raw/")

print("\n" + "=" * 50)
print(" EXPORT COMPLETE")
print("=" * 50)
print(f"\nTables: {len(os.listdir(TABLES_DIR))} files in results/tables/")
print(f"Metrics: {len(os.listdir(METRICS_DIR))} files in results/metrics/")
print(f"Raw: {len(os.listdir(RAW_DIR))} files in results/raw/")
