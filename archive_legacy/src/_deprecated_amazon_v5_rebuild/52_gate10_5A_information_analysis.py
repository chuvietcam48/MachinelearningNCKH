import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import sys

def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    gate10_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline"
    artifact_dir = repo_root / "outputs" / "amazon_v5_rebuild" / "gate10_baseline" / "analysis"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data for Gate 10.5A Information Analysis...")
    train_df = pd.read_parquet(gate10_dir / "master_train.parquet")
    
    with open(gate10_dir / "feature_registry.json", "r") as f:
        registry = json.load(f)
        
    semantic_feats = registry["Model_C"]["features"][12:] # Top 10 semantic
    behavioral_feats = registry["Model_A"]["features"]
    
    report = ["# Gate 10.5A: Semantic vs Behavioral Information Analysis\n"]
    
    # 1. Sparsity / Distribution Check
    report.append("## 1. Sparsity of Semantic Features")
    report.append("What percentage of the dataset actually has non-zero semantic trajectory?\n")
    report.append("| Feature | % Zeros | Mean | Max |")
    report.append("|---------|---------|------|-----|")
    for f in semantic_feats:
        zeros_pct = (train_df[f] == 0).mean() * 100
        mean_val = train_df[f].mean()
        max_val = train_df[f].max()
        report.append(f"| {f} | {zeros_pct:.1f}% | {mean_val:.4f} | {max_val:.2f} |")
        
    # 2. Correlation with Ratings
    report.append("\n## 2. Redundancy (Correlation with Ratings & Behavior)")
    report.append("How much of the Semantic signal is already captured by ratings?\n")
    target_behaviors = ['episode_mean_rating', 'episode_min_rating', 'recent_low_rating_count', 'prior_review_count']
    
    header = "| Semantic Feature | " + " | ".join(target_behaviors) + " |"
    report.append(header)
    report.append("|---" + "|---" * len(target_behaviors) + "|")
    
    for sf in semantic_feats:
        corrs = []
        for bf in target_behaviors:
            # Spearman correlation to capture non-linear redundancy
            c = train_df[sf].corr(train_df[bf], method='spearman')
            corrs.append(f"{c:.3f}")
        report.append(f"| {sf} | " + " | ".join(corrs) + " |")
        
    # 3. Model C Coefficients (Hazard Ratios)
    report.append("\n## 3. CoxPH Coefficients (Hazard Ratios) from Model C")
    report.append("Did the model actually assign weight to Semantic features, or were they shrunk near zero?\n")
    try:
        model_c = joblib.load(gate10_dir / "model_c.pkl")
        coefs = model_c.coef_
        all_feats = registry["Model_C"]["features"]
        
        report.append("| Feature | Type | Coefficient | Hazard Ratio (exp(coef)) |")
        report.append("|---------|------|-------------|--------------------------|")
        
        # Sort by absolute coefficient size
        feat_importance = []
        for i, f in enumerate(all_feats):
            ftype = "Semantic" if f in semantic_feats else "Behavioral"
            feat_importance.append((f, ftype, coefs[i], np.exp(coefs[i])))
            
        feat_importance.sort(key=lambda x: abs(x[2]), reverse=True)
        
        for f, ftype, coef, hr in feat_importance:
            report.append(f"| {f} | {ftype} | {coef:.5f} | {hr:.5f} |")
            
    except Exception as e:
        report.append(f"Could not load model_c.pkl to extract coefficients: {e}")
        
    report_content = "\n".join(report)
    
    with open(artifact_dir / "gate10_5A_information_analysis.md", "w") as f:
        f.write(report_content)
        
    print(f"Analysis saved to: {artifact_dir / 'gate10_5A_information_analysis.md'}")

if __name__ == "__main__":
    main()
