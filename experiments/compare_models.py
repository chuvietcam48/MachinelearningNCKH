import json
import pandas as pd
from pathlib import Path

# Paths
E0_DIR = Path("experiments/E0_original_framework/outputs")
UNIVERSAL_METRICS = Path("outputs/pipeline_freeze/results/model_metrics.json")
E0_METRICS = E0_DIR / "e0_metrics.json"

# Output for the paper table
PAPER_TABLES_DIR = Path("experiments/tables")
PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=== Extracting E0 (Historical Baseline) Metrics ===")
    with open(E0_METRICS, "r") as f:
        e0_data = json.load(f)["results"]
    
    # E0 models
    e0_model_a = e0_data.get("Model_A_RFM", {})
    e0_model_f = e0_data.get("Model_F_All", {})
    
    print(f"E0 Model A (Behavioral): C-OOS = {e0_model_a.get('c_oos', 'N/A')}")
    print(f"E0 Model F (Full): C-OOS = {e0_model_f.get('c_oos', 'N/A')}")
    
    print("\n=== Extracting Universal Framework (E1-E4) Metrics ===")
    with open(UNIVERSAL_METRICS, "r") as f:
        universal_data = json.load(f)
        
    u_full = universal_data.get("Full", {})
    u_semantic = universal_data.get("Semantic", {})
    
    # E1: Universal Behavioral
    e1_model_a = u_full.get("Model_A", {})
    # E3: Universal Semantic
    e3_model_c = u_semantic.get("Model_C", {})
    
    print(f"E1 Model A (Behavioral): C-index = {e1_model_a.get('C-index', 'N/A'):.4f}")
    print(f"E3 Model C (Semantic): C-index = {e3_model_c.get('C-index', 'N/A'):.4f}")
    
    # Constructing Table 1: End-to-End Prediction Performance
    table1 = f"""
| Pipeline             | Preprocessing      | Model Features             | C-Index (OOS) |
|----------------------|--------------------|----------------------------|--------------:|
| Historical (E0)      | Legacy Pipeline    | Behavioral Only (Model A)  | {e0_model_a.get('c_oos'):.4f}        |
| Historical (E0)      | Legacy Pipeline    | Behavioral + NLP (Model F) | {e0_model_f.get('c_oos'):.4f}        |
| Universal (E1)       | Universal Pipeline | Behavioral Only            | {e1_model_a.get('C-index'):.4f}        |
| Universal (E3)       | Universal Pipeline | Behavioral + LLM           | {e3_model_c.get('C-index'):.4f}        |
"""
    
    print("\n=== Generated Table 1 ===")
    print(table1)
    
    with open(PAPER_TABLES_DIR / "Table1_Prediction_Performance.md", "w") as f:
        f.write(table1.strip())

if __name__ == "__main__":
    main()
