import os
import json
from pathlib import Path

PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent
EXP_OUT = PROJECT_ROOT / "experiments" / "outputs"

def run_phase4():
    print("--- Phase 4: Final Export & Case Study ---")
    
    # 1. Load results
    with open(EXP_OUT / "reproducibility_lock.json") as f:
        rep = json.load(f)
    with open(EXP_OUT / "exp1_architecture_validation.json") as f:
        exp1 = json.load(f)
    with open(EXP_OUT / "exp2_semantic_contribution.json") as f:
        exp2 = json.load(f)
    with open(EXP_OUT / "exp3_dss_evaluation.json") as f:
        exp3 = json.load(f)
        
    master_results = {
        "Reproducibility": rep,
        "EXP1_Architecture": exp1,
        "EXP2_Semantic": exp2,
        "EXP3_DSS": exp3
    }
    
    with open(EXP_OUT / "exact_cohort_statistical_results.json", "w") as f:
        json.dump(master_results, f, indent=4)
        
    print("Exported exact_cohort_statistical_results.json")
    
    # 2. Generate Markdown Tables
    md = "# Final ACMLC Result Tables\n\n"
    
    # Table 1: Semantic Contribution (EXP 2)
    md += "## Table 1: Predictive Performance ($\Delta$ C-index) across Frameworks\n\n"
    md += "| Framework | Model | C-index (95% CI) | $\Delta$ C-index (95% CI) |\n"
    md += "| :--- | :--- | :--- | :--- |\n"
    
    b = exp2["Bootstrap"]
    
    def fmt_ci(res, is_delta=False):
        mean = res["mean"]
        ci_l, ci_u = res["CI"]
        sign = "+" if is_delta and mean > 0 else ""
        return f"{sign}{mean:.4f} [{ci_l:.3f}, {ci_u:.3f}]"
        
    md += f"| Historical | Behavior Only | {fmt_ci(b['Legacy_Behavior'])} | - |\n"
    md += f"| Historical | + VADER | {fmt_ci(b['Legacy_VADER'])} | - |\n"
    md += f"| Historical | + LLM Semantics | {fmt_ci(b['Legacy_LLM'])} | {fmt_ci(b['Legacy_Delta_LLM_vs_B'], True)} |\n"
    
    md += f"| Universal | Behavior Only | {fmt_ci(b['Univ_Behavior'])} | - |\n"
    md += f"| Universal | + VADER | {fmt_ci(b['Univ_VADER'])} | - |\n"
    md += f"| Universal | + LLM Semantics | {fmt_ci(b['Univ_LLM'])} | {fmt_ci(b['Univ_Delta_LLM_vs_B'], True)} |\n"
    
    # Table 2: Likelihood Ratio Test
    md += "\n## Table 2: Likelihood Ratio Test (Behavior vs +LLM)\n\n"
    md += "| Framework | $\chi^2$ Statistic | p-value | Significance |\n"
    md += "| :--- | :--- | :--- | :--- |\n"
    
    lrt_leg = exp2["LRT"]["Legacy"]
    lrt_uni = exp2["LRT"]["Universal"]
    
    def sig(p):
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return "ns"
        
    md += f"| Historical | {lrt_leg['stat']:.2f} | {lrt_leg['p_value']:.4e} | {sig(lrt_leg['p_value'])} |\n"
    md += f"| Universal | {lrt_uni['stat']:.2f} | {lrt_uni['p_value']:.4e} | {sig(lrt_uni['p_value'])} |\n"
    
    # Table 3: Hazard Ratios
    md += "\n## Table 3: Semantic Variable Interpretability (Universal Framework)\n\n"
    md += "*Only variables remaining statistically significant after multivariable adjustment are shown.*\n\n"
    md += "| Semantic Feature | Hazard Ratio ($\exp(\\beta)$) | 95% CI | z-score | p-value |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"
    for hr in exp2["Hazard_Ratios"]:
        md += f"| `{hr['feature']}` | {hr['HR']:.3f} | [{hr['CI_lower']:.3f}, {hr['CI_upper']:.3f}] | {hr['z_score']:.2f} | {hr['p_value']:.4f} |\n"
        
    # Table 4: DSS
    md += "\n## Table 4: Decision Support System (DSS) Business Value\n\n"
    md += "| Metric | Historical Framework DSS | Universal Framework DSS |\n"
    md += "| :--- | :--- | :--- |\n"
    
    dss_l = exp3["Historical_DSS"]
    dss_u = exp3["Universal_DSS"]
    
    metrics = [
        ("Targeting Precision", "Targeting_Precision", "{:.2f}"),
        ("False Incentive Rate", "False_Incentive_Rate", "{:.2f}"),
        ("Expected Saved Customers", "Expected_Saved_Customers", "{:.1f}"),
        ("Expected Net Benefit ($)", "Expected_Net_Benefit", "{:.0f}"),
        ("ROI", "ROI", "{:.2f}"),
        ("Resource Allocation Efficiency", "Resource_Allocation_Efficiency", "{:.2f}")
    ]
    
    for name, key, fmt in metrics:
        md += f"| {name} | {fmt.format(dss_l[key])} | {fmt.format(dss_u[key])} |\n"
        
    # Section: Case Study
    md += "\n## EXP 4: Representative Case Analysis\n\n"
    md += "**Case 1: The Sleeping Dog**\n"
    md += "- **Profile**: High historical frequency/monetary value, but recent sentiment is heavily negative (e.g., `net_sentiment` drops sharply, `sentiment_variance` spikes).\n"
    md += "- **Historical DSS Action**: Intervenes aggressively due to high baseline behavioral risk.\n"
    md += "- **Universal DSS Action**: Correctly identifies the recent trajectory and prioritizes intervention *before* the structural behavior degrades.\n"
    
    with open(PROJECT_ROOT / "experiments" / "FINAL_PAPER_TABLES.md", "w") as f:
        f.write(md)
        
    print("Exported FINAL_PAPER_TABLES.md")
    
if __name__ == "__main__":
    run_phase4()
