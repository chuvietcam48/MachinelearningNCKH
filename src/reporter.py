"""
src/reporter.py  (Phase D — D5: Markdown Auto-Report Generator)
================================================================
Generates a publication-ready Markdown report from pipeline results.

Usage
-----
    from src.reporter import generate_report
    generate_report(
        meta=pipeline_meta,
        metrics=eval_metrics,
        outreach=outreach_metrics,
        revenue=revenue_metrics,
        run_dir=RUN_DIR,
        dataset_name="CDNOW",
        tau=90,
        c_index_boot_ci=boot_ci,   # optional (lower, median, upper)
    )

Output
------
  {run_dir}/reports/report.md  — standalone, shareable Markdown report.
"""

import os
import logging
import datetime
import math
import numpy as np

logger = logging.getLogger(__name__)


def _pct(v: float) -> str:
    return f"{v:.1f}%"


def _f2(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _f4(v) -> str:
    """Format to 4 decimal places (technical metrics table)."""
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _f0(v) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):,.0f}"
    except Exception:
        return str(v)


def generate_report(
    meta: dict,
    metrics: dict,
    outreach: dict,
    revenue: dict,
    run_dir: str,
    dataset_name: str = "Unknown",
    tau: int = 90,
    c_index_boot_ci: tuple = None,
    monte_carlo_results: dict = None,
) -> str:
    """
    Generate a Markdown auto-report and save to {run_dir}/reports/report.md.

    Parameters
    ----------
    meta : dict
        Pipeline metadata (from pipeline_meta.pkl).
    metrics : dict
        Evaluation metrics dict (C-index, IBS, LR AUC, etc.)
    outreach : dict
        Outreach efficiency metrics.
    revenue : dict
        Revenue lift metrics.
    run_dir : str
        Root output directory for this run (e.g. outputs/CDNOW_tau90).
    dataset_name : str
        Human-readable dataset name for the report header.
    tau : int
        Churn threshold used in this run.
    c_index_boot_ci : tuple, optional
        (lower_95, median, upper_95) bootstrap CI for Weibull C-index.
    monte_carlo_results : dict, optional
        Monte Carlo results dict (from run_monte_carlo_simulation).

    Returns
    -------
    str
        Absolute path to the generated report.md file.
    """
    from src.dataset_registry import get_currency_symbol, get_currency_code
    sym  = get_currency_symbol()
    code = get_currency_code()
    now  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    reports_dir = os.path.join(run_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "report.md")

    # ── Survival Performance ────────────────────────────────────────────────
    # We prioritize OOS (test set) C-index for scientific validity.
    c_index_oos = metrics.get("c_index_oos") or metrics.get("c_index")
    ibs         = metrics.get("ibs") or metrics.get("integrated_brier")
    ci_data     = metrics.get("c_index_ci") # tuple (lo, med, hi, reliable)
    
    ci_str = "N/A"
    ci_status = "Not Computed"
    if isinstance(ci_data, (tuple, list)) and len(ci_data) >= 4:
        lo, med, hi, reliable = ci_data
        if reliable and np.isfinite(lo):
            ci_str = f"[{lo:.4f}, {hi:.4f}]"
            ci_status = "Reliable (300 samples)"
        else:
            ci_str = "nan"
            ci_status = "Low Confidence / Non-convergent"

    # ── Extract other metrics safely ──────────────────────────────────────────
    c_cox     = metrics.get("c_index_cox", None)
    lr_auc    = metrics.get("lr_auc", metrics.get("auc_mean", None))
    qini_coef = metrics.get("uplift_qini", None)
    persuadable_pct = metrics.get("uplift_persuadables", None)

    w_intervene_rate = outreach.get("weibull_intervene_rate", None)
    rfm_intervene_rate = outreach.get("rfm_intervene_rate", None)
    contacts_avoided = outreach.get("contacts_avoided", None)
    contacts_avoided_pct = outreach.get("contacts_avoided_pct", None)
    efficiency_gain  = outreach.get("efficiency_gain_pct", None)

    avg_evi_w  = revenue.get("avg_evi_weibull", None)
    avg_evi_r  = revenue.get("avg_evi_rfm_proxy", None)
    prec_lift  = revenue.get("revenue_precision_lift_pct", None)

    n_customers   = meta.get("n_customers", "N/A")
    churn_rate    = meta.get("churn_rate", None)
    active_feats  = meta.get("active_features_waf", meta.get("survival_features", []))

    # ── Monte Carlo results ──────────────────────────────────────────────────
    if monte_carlo_results:
        w_ci = monte_carlo_results.get("weibull_profit_ci", (None, None, None))
        r_ci = monte_carlo_results.get("rfm_profit_ci", (None, None, None))
        l_ci = monte_carlo_results.get("lr_profit_ci", None)
        eg_ci = monte_carlo_results.get("efficiency_gain_ci", (None, None, None))
        wilcoxon_p = monte_carlo_results.get("wilcoxon_pvalue", None)
        if wilcoxon_p is not None and not (isinstance(wilcoxon_p, float) and np.isnan(wilcoxon_p)):
            sig_str   = " p < 0.05 — statistically significant" if wilcoxon_p < 0.05 else "⚠️ p ≥ 0.05 — not significant"
            wilcox_md = f"| **Wilcoxon Signed-Rank** (Weibull > RFM) | p = {wilcoxon_p:.6f} | {sig_str} |"
        else:
            wilcox_md = "| **Wilcoxon Signed-Rank** | N/A | — |"
        # LR+EVI row (only if 3rd arm was simulated)
        lr_row = ""
        if l_ci is not None:
            lr_row = f"| **LR+EVI Baseline** | {sym}{_f0(l_ci[0])} | {sym}{_f0(l_ci[1])} | {sym}{_f0(l_ci[2])} |\n"
        mc_section = f"""
## 5. Monte Carlo Policy Simulation (Budget-Constrained)

> *{mc_results_caption(monte_carlo_results)}*

| Policy | Lower 95% | Median | Upper 95% |
|--------|-----------|--------|-----------|
| **Weibull AFT** | {sym}{_f0(w_ci[0])} | {sym}{_f0(w_ci[1])} | {sym}{_f0(w_ci[2])} |
{lr_row}| **RFM Baseline** | {sym}{_f0(r_ci[0])} | {sym}{_f0(r_ci[1])} | {sym}{_f0(r_ci[2])} |
| **Efficiency Gain** | {_pct(eg_ci[0]*100 if eg_ci[0] else 0)} | {_pct(eg_ci[1]*100 if eg_ci[1] else 0)} | {_pct(eg_ci[2]*100 if eg_ci[2] else 0)} |

### 5.1 Statistical Significance

| Test | Result | Interpretation |
|------|--------|----------------|
{wilcox_md}
"""
    else:
        mc_section = "\n## 5. Monte Carlo Policy Simulation\n\nNot run in this configuration.\n"

    # ── Figures ───────────────────────────────────────────────────────────────
    figures_dir = os.path.join(run_dir, "figures")
    def _fig(name: str) -> str:
        p = os.path.join(figures_dir, name)
        rel = os.path.relpath(p, reports_dir)
        if os.path.exists(p):
            return f"![{name}]({rel})\n"
        return f"*({name} — not generated in this run)*\n"

    # ── Build report ─────────────────────────────────────────────────────────
    lines = [
        f"# Decision-Centric Customer Re-Engagement — Pipeline Report",
        f"",
        f"> **Dataset**: {dataset_name}  |  **Tau (τ)**: {tau} days  |  **Generated**: {now}",
        f"> **Currency**: {code} ({sym})",
        f"",
        f"---",
        f"",
        f"## 1. Dataset Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Customers | {n_customers:,} |" if isinstance(n_customers, int) else f"| Customers | {n_customers} |",
        f"| Churn rate (E=1, tau={tau}d) | {_pct(churn_rate * 100) if churn_rate else 'N/A'} |",
        f"| Features used | {', '.join(active_feats) if active_feats else 'N/A'} |",
        f"| Churn threshold τ | {tau} days |",
        f"",
        f"---",
        f"",
        f"## 2. Technical Model Performance",
        f"",
        f"| Model | Metric | Value | Target | Status |",
        f"|:------|:-------|:------|:-------|:-------|",
        f"| Weibull AFT | C-index (OOS / test) | {_f4(c_index_oos)} | > 0.70 | {'✅' if (c_index_oos or 0) > 0.7 else '⚠️'} |",
        f"| Weibull AFT | Bootstrap 95% CI | {ci_str} | < 0.05 width | {ci_status} |",
        f"| Weibull AFT | Integrated Brier Score | {_f4(ibs)} | < 0.25 | {'✅' if (ibs or 1) < 0.25 else '⚠️'} |",
        f"| CoxPH | C-index | {_f4(c_cox)} | > 0.65 | {'✅' if (c_cox or 0) > 0.65 else '⚠️'} |",
        f"| Logistic Regression | CV AUC | {_f4(lr_auc)} | > 0.75 | {'✅' if (lr_auc or 0) > 0.75 else '⚠️'} |",
        f"| IPTW T-Learner | Qini Coefficient | {_f4(qini_coef) if qini_coef is not None else 'N/A (pass --uplift)'} | > 1.0 | {'✅' if (qini_coef or 0) > 1.0 else '⚠️' if qini_coef is not None else '—'} |",
        f"| IPTW T-Learner | Persuadables (%) | {_pct(persuadable_pct * 100) if persuadable_pct is not None else 'N/A'} | — | — |",
        f"",
        f"### 2.1 Survival Curves",
        f"",
        _fig("weibull_survival_curves.png"),
        f"### 2.2 Kaplan-Meier by RFM Segment",
        f"",
        _fig("kaplan_meier_by_segment.png"),
        f"### 2.3 Logistic Regression Calibration",
        f"",
        _fig("lr_calibration.png"),
        f"",
        f"---",
        f"",
        f"## 3. Business Decision Policy",
        f"",
        f"| Metric | Weibull AFT | RFM Baseline | Advantage |",
        f"|--------|------------|--------------|-----------|",
        f"| Intervention rate | {_pct(w_intervene_rate) if w_intervene_rate else 'N/A'} | {_pct(rfm_intervene_rate) if rfm_intervene_rate else 'N/A'} | Precision |",
        f"| Contacts avoided | {contacts_avoided:,} ({_pct(contacts_avoided_pct)}) |" + " — | Less spam |" if contacts_avoided else "| Contacts avoided | N/A | — | — |",
        f"| Outreach efficiency gain | {_pct(efficiency_gain)} |" + " — | > 20% target |" if efficiency_gain else "| Efficiency gain | N/A | — | — |",
        f"| Avg EVI per contact | {sym}{_f2(avg_evi_w)} | {sym}{_f2(avg_evi_r)} | Precision lift |",
        f"| Revenue precision lift | {_pct(prec_lift)} |" + " — | > 20% target |" if prec_lift else "| Revenue lift | N/A | — | — |",
        f"",
        f"### 3.1 Decision Distribution",
        f"",
        _fig("decision_distribution.png"),
        f"### 3.2 Hazard Trajectories",
        f"",
        _fig("hazard_trajectories.png"),
        f"",
        f"---",
        f"",
        f"## 4. Time-Dependent Metrics",
        f"",
        f"### 4.1 Time-Dependent AUC",
        f"",
        _fig("time_dependent_auc.png"),
        f"### 4.2 Brier Score Over Time",
        f"",
        _fig("brier_score_over_time.png"),
        f"",
        f"---",
        mc_section,
        f"",
        f"---",
        f"",
        f"## 6. Reproducibility",
        f"",
        f"```bash",
        f"# Re-run this exact pipeline:",
        f"python main.py --dataset {dataset_name.lower()} --tau {tau} --no-shap",
        f"```",
        f"",
        f"All models serialised to `{run_dir}/models/`.",
        f"",
        f"---",
        f"",
        f"*Report generated automatically by `src/reporter.py` — Decision-Centric CRM Pipeline*",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"[Reporter] Report saved → {report_path}")
    return report_path


def mc_results_caption(mc: dict) -> str:
    """Generate a one-line summary of MC results."""
    w = mc.get("weibull_profit_ci", (0, 0, 0))
    r = mc.get("rfm_profit_ci", (0, 0, 0))
    p = mc.get("wilcoxon_pvalue", None)
    if all(v is not None for v in [w[1], r[1]]):
        winner = "Weibull AFT" if w[1] > r[1] else "RFM"
        _p_valid = p is not None and not (isinstance(p, float) and math.isnan(p))
        sig = f" | Wilcoxon p={p:.4f}" if _p_valid else ""
        return f"Median profit: Weibull={w[1]:,.0f} vs RFM={r[1]:,.0f} — {winner} wins{sig}."
    return "Monte Carlo results available."
