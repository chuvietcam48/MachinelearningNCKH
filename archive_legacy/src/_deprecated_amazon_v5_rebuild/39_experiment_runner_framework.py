"""
Experiment Runner Framework (Ablation Study)
=============================================
Purpose: Orchestrate the end-to-end ablation experiments for the paper.
         Runs the baseline models, semantic models, and outputs publication-ready tables.

Currently, this is a FRAMEWORK skeleton. The semantic feature sets are placeholders
until Gate 7 is fully completed and Gate 10 is implemented.

Workflow:
1. Run Baseline (Gate 9)
2. Run Semantic (Gate 10) - To be implemented
3. Compile Results
4. Export Markdown & CSV tables for the paper
"""

import json
from pathlib import Path
import pandas as pd
import datetime

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR = Path("outputs/amazon_v5_rebuild/experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mocked paths for the results of individual gates
GATE9_RESULTS = Path("outputs/amazon_v5_rebuild/gate9_survival_baseline/gate9_survival_baseline_results.csv")
GATE10_RESULTS = Path("outputs/amazon_v5_rebuild/gate10_semantic_survival/gate10_semantic_survival_results.csv")

# ── 1. Run Baseline ────────────────────────────────────────────────────────
def run_baseline():
    print("Running Baseline (Gate 9)...")
    # In practice, you might import and run the module directly:
    # from amazon_v5_rebuild.37_gate9_survival_baseline import run as run_gate9
    # run_gate9()
    
    if not GATE9_RESULTS.exists():
        print(f"  [WARN] Baseline results not found at {GATE9_RESULTS}. Please run Gate 9 first.")
        return pd.DataFrame()
        
    df = pd.read_csv(GATE9_RESULTS)
    df['Experiment'] = 'Baseline'
    return df

# ── 2. Run Semantic / Ablation ─────────────────────────────────────────────
def run_semantic_ablation():
    print("Running Semantic & Ablation Models (Gate 10)...")
    # from amazon_v5_rebuild.40_gate10_semantic_survival import run as run_gate10
    # run_gate10()
    
    if not GATE10_RESULTS.exists():
        print(f"  [INFO] Semantic results not found at {GATE10_RESULTS}. (Expected if Gate 10 is not yet implemented)")
        return pd.DataFrame()
        
    df = pd.read_csv(GATE10_RESULTS)
    df['Experiment'] = 'Semantic'
    return df

# ── 3. Export Tables ───────────────────────────────────────────────────────
def export_tables(results_df: pd.DataFrame):
    if results_df.empty:
        print("No results to export.")
        return
        
    print("Exporting publication-ready tables...")
    
    # Sort by model and feature set for clean presentation
    table = results_df.copy()
    if 'c_index' in table.columns:
        table = table.sort_values(by=['model', 'c_index'], ascending=[True, False])
    
    csv_path = OUT_DIR / "ablation_results_summary.csv"
    md_path = OUT_DIR / "ablation_results_summary.md"
    
    table.to_csv(csv_path, index=False)
    
    # Generate Markdown Table
    md_content = f"# Ablation Study Results\n\nGenerated on: {datetime.datetime.now().isoformat()}\n\n"
    md_content += table.to_markdown(index=False)
    md_content += "\n\n*Note: C-Index measures the model's ability to correctly rank survival times.*"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"  Saved CSV: {csv_path}")
    print(f"  Saved MD : {md_path}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=== Starting Experiment Runner ===\n")
    
    baseline_df = run_baseline()
    semantic_df = run_semantic_ablation()
    
    all_results = pd.concat([baseline_df, semantic_df], ignore_index=True)
    
    print("\n=== Compilation ===")
    export_tables(all_results)
    
    print("\n=== Experiment Runner Complete ===")

if __name__ == "__main__":
    main()
