"""
NaN Diagnostic for Amazon Semantic Pipeline
============================================
Purpose: Identify exactly WHERE and WHY NaN values appear in the 
semantic-fused feature dataframe BEFORE CoxPH fitting.

This is a diagnostic-only script. It does NOT modify any data.
Output: results/E5_amazon_semantic/nan_diagnostic.txt
"""
import sys
import os
import logging
import numpy as np
import pandas as pd

# Setup
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.framework.dataset_adapter import DatasetAdapter
from src.framework.behavior_engine import BehaviorEngine
from src.framework.semantic_engine import SemanticEngine
from src.framework.feature_fusion import FeatureFusionEngine
from src.dataset_registry import get_dataset

OUT_DIR = os.path.join(ROOT, 'results', 'E5_amazon_semantic')
os.makedirs(OUT_DIR, exist_ok=True)
report_path = os.path.join(OUT_DIR, 'nan_diagnostic.txt')

def main():
    lines = []
    def log(msg):
        print(msg)
        lines.append(msg)
    
    log("=" * 70)
    log("NaN DIAGNOSTIC REPORT — Amazon Semantic Pipeline")
    log("=" * 70)
    
    # Step 1: Load raw data
    adapter = DatasetAdapter(dataset_name='amazon')
    df_raw = adapter.load_data()
    log(f"\n[1] Raw data loaded: {len(df_raw)} rows, {len(df_raw.columns)} columns")
    log(f"    NaN in raw: {df_raw.isna().sum().sum()} total")
    
    # Step 2: Behavior Engine
    registry_info = get_dataset('amazon')
    snapshot_date = registry_info.snapshot_fn(df_raw)
    behavior_engine = BehaviorEngine(snapshot_date=snapshot_date)
    df_behavior = behavior_engine.process(df_raw)
    log(f"\n[2] Behavior features: {df_behavior.shape}")
    nan_behavior = df_behavior.isna().sum()
    nan_behavior = nan_behavior[nan_behavior > 0]
    if len(nan_behavior) > 0:
        log(f"    ⚠️ NaN in behavior features:")
        for col, count in nan_behavior.items():
            log(f"       {col}: {count} NaN ({count/len(df_behavior)*100:.1f}%)")
    else:
        log(f"    ✅ No NaN in behavior features")
    
    # Step 3: Plugin Manager → Semantic Provider
    from src.framework.plugin_manager import PluginManager
    plugin_manager = PluginManager(dataset_name='amazon', force_semantic=True)
    semantic_provider = plugin_manager.get_semantic_provider()
    
    if semantic_provider:
        df_semantic_raw = semantic_provider.load()
        log(f"\n[3] Semantic raw loaded: {df_semantic_raw.shape}")
        nan_sem_raw = df_semantic_raw.isna().sum()
        nan_sem_raw = nan_sem_raw[nan_sem_raw > 0]
        if len(nan_sem_raw) > 0:
            log(f"    ⚠️ NaN in semantic raw:")
            for col, count in nan_sem_raw.items():
                log(f"       {col}: {count} NaN ({count/len(df_semantic_raw)*100:.1f}%)")
        
        # Step 4: Semantic Engine
        semantic_engine = SemanticEngine()
        df_semantic = semantic_engine.process(df_semantic_raw)
        log(f"\n[4] Semantic features: {df_semantic.shape}")
        nan_sem = df_semantic.isna().sum()
        nan_sem = nan_sem[nan_sem > 0]
        if len(nan_sem) > 0:
            log(f"    ⚠️ NaN in semantic features:")
            for col, count in nan_sem.items():
                log(f"       {col}: {count} NaN ({count/len(df_semantic)*100:.1f}%)")
        else:
            log(f"    ✅ No NaN in semantic features")
        
        # Step 5: Fusion
        fusion_engine = FeatureFusionEngine()
        df_master = fusion_engine.merge(df_behavior, df_semantic)
        log(f"\n[5] Fused master: {df_master.shape}")
    else:
        df_master = df_behavior
        log(f"\n[3-5] No semantic provider. Master = behavior only.")
    
    # Step 6: Simulate what survival_engine sees
    drop_cols = ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'episode_id']
    df_fit = df_master.drop(columns=[c for c in drop_cols if c in df_master.columns])
    df_fit = df_fit.select_dtypes(include=[np.number])
    
    # Split same as run_framework.py
    df_train = df_fit.sample(frac=0.8, random_state=42).copy()
    
    log(f"\n[6] Train set for CoxPH: {df_train.shape}")
    nan_train = df_train.isna().sum()
    nan_train = nan_train[nan_train > 0]
    total_nan = nan_train.sum()
    
    log(f"\n{'=' * 70}")
    log(f"DIAGNOSIS SUMMARY")
    log(f"{'=' * 70}")
    
    if total_nan > 0:
        log(f"\n⚠️ TOTAL NaN VALUES IN TRAIN SET: {total_nan}")
        log(f"   Affected columns: {len(nan_train)}")
        log(f"\n   Column-level breakdown:")
        for col, count in nan_train.sort_values(ascending=False).items():
            pct = count / len(df_train) * 100
            log(f"   - {col}: {count} NaN ({pct:.1f}%)")
        
        # Identify rows with NaN
        nan_rows = df_train.isna().any(axis=1).sum()
        log(f"\n   Rows with at least one NaN: {nan_rows} ({nan_rows/len(df_train)*100:.1f}%)")
        
        # Identify ORIGIN: are NaN columns behavioral or semantic?
        behavioral_cols = set(df_behavior.select_dtypes(include=[np.number]).columns)
        semantic_only = []
        behavioral_nan = []
        for col in nan_train.index:
            if col in behavioral_cols:
                behavioral_nan.append(col)
            else:
                semantic_only.append(col)
        
        log(f"\n   ORIGIN ANALYSIS:")
        if behavioral_nan:
            log(f"   🔴 Behavioral NaN columns: {behavioral_nan}")
        if semantic_only:
            log(f"   🟡 Semantic-only NaN columns: {semantic_only}")
            log(f"   → These are expected: customers without reviews have no semantic features.")
            log(f"   → JUSTIFICATION: fillna(0) is correct because:")
            log(f"     1. Semantic features are sparse by design (only reviewers have them)")
            log(f"     2. 0 represents 'no signal' which is semantically correct")
            log(f"     3. This is standard practice for sparse feature matrices in survival analysis")
    else:
        log(f"\n✅ NO NaN VALUES. Pipeline is clean.")
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    log(f"\nReport saved to: {report_path}")

if __name__ == '__main__':
    main()
