import pytest
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "pipeline_freeze"

def test_dataset_integrity():
    """Verify Master Dataset was generated and contains no critical NaNs."""
    df_path = OUTPUT_DIR / "datasets" / "master_test_semantic.parquet"
    assert df_path.exists(), "Master test dataset not found."
    
    df = pd.read_parquet(df_path)
    
    # Assert critical columns exist
    assert 'E_Event' in df.columns
    assert 'T_Duration' in df.columns
    assert 'risk_score' not in df.columns # Risk score should only be in predictions
    
    # Assert no NaNs in targets
    assert not df['E_Event'].isna().any()
    assert not df['T_Duration'].isna().any()
    assert len(df) > 0, "Test set is empty."

def test_hazard_ratio_consistency():
    """Verify Hazard Ratios do not contain infinite values or NaNs."""
    hr_path = OUTPUT_DIR / "results" / "hazard_ratios_semantic.csv"
    assert hr_path.exists(), "Hazard ratios not found."
    
    hr = pd.read_csv(hr_path)
    
    assert not hr['Hazard_Ratio'].isna().any(), "NaN found in Hazard Ratios"
    assert not hr['p_value'].isna().any(), "NaN found in p-values"
    assert (hr['Hazard_Ratio'] > 0).all(), "Hazard Ratios must be positive"

def test_policy_routing_validity():
    """Ensure the Decision Support Engine assigns logical policies."""
    policy_path = OUTPUT_DIR / "results" / "policy" / "policy_routing_matrix.csv"
    assert policy_path.exists(), "Policy routing matrix not found."
    
    policy_df = pd.read_csv(policy_path)
    
    # Assert No Action Needed exists (since most customers shouldn't churn)
    assert 'No Action Needed' in policy_df['Recommended_Intervention'].values
    
    # Total customers routed should equal something reasonable
    assert policy_df['Customers'].sum() > 0

def test_sensitivity_results_exist():
    """Check that Gate 15 Sensitivity Analysis outputs are generated."""
    sens_dir = OUTPUT_DIR / "results" / "sensitivity"
    assert (sens_dir / "model_robustness_penalizer.csv").exists()
    assert (sens_dir / "data_robustness_coverage.csv").exists()
    assert (sens_dir / "policy_robustness_cost.csv").exists()

def test_experiment_registry():
    """Verify the experiment registry and hash checks are present."""
    assert (OUTPUT_DIR / "experiment_registry.json").exists()
    assert (OUTPUT_DIR / "validation_report.md").exists()
    assert (OUTPUT_DIR / "publication_checklist.md").exists()
