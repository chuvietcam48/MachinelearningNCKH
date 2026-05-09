"""
src/metrics/business_metrics.py
=================================
Realistic Business Metrics for Retention Campaign Evaluation.

Metrics
-------
CAC_retention    : Customer Acquisition (Retention) Cost
                   = campaign_spend / customers_retained
                   Lower is better.

ROI              : Return on Investment
                   = (revenue_saved - campaign_cost) / campaign_cost × 100%
                   Positive ROI = campaign is profitable.

Payback_Period   : Months until cumulative profit turns positive.
                   From multi-period simulation results.

Retention_Rate   : % of at-risk customers still active after campaign.

Cohort_Retention : Retention rate by customer cohort × time period matrix
                   (analogous to a SaaS cohort retention heatmap).
                   Rows = RFM cohorts or first-purchase cohorts.
                   Columns = periods 1..K.

ARPU_uplift      : Average Revenue Per User uplift vs no-intervention.

Break_Even_K     : Minimum K contacts at which Weibull policy breaks even
                   under budget constraint.
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# BusinessMetricsCalculator
# =============================================================================

class BusinessMetricsCalculator:
    """
    Computes and visualizes realistic business metrics from pipeline outputs.

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions().
        Required cols: CustomerID, decision, Monetary, survival, evi, hazard_now.
    customer_df : pd.DataFrame
        Customer features indexed by CustomerID.
    mc_results : dict
        Output of run_monte_carlo_simulation().
    advsim_results : dict, optional
        Output of run_advanced_simulation().  Used for payback period and
        cumulative CLV tracking.
    cost_per_contact : float
        Cost per marketing contact (MU).
    p_response : float
        Assumed campaign response rate.
    dataset_label : str
    """

    def __init__(
        self,
        weibull_decisions: pd.DataFrame,
        customer_df: pd.DataFrame,
        mc_results: dict,
        advsim_results: Optional[dict] = None,
        cost_per_contact: float = 1.0,
        p_response: float = 0.15,
        dataset_label: str = "",
    ):
        self.decisions        = weibull_decisions.copy()
        self.customer_df      = customer_df.copy()
        self.mc_results       = mc_results
        self.advsim_results   = advsim_results
        self.cost             = cost_per_contact
        self.p_response       = p_response
        self.dataset_label    = dataset_label

        # Core decision arrays
        self._intervene_mask  = (weibull_decisions["decision"] == "INTERVENE").values
        self._monetary        = weibull_decisions["Monetary"].fillna(0).values.astype(float)
        self._churn_prob      = np.clip(
            1.0 - weibull_decisions["survival"].fillna(0.5).values, 0, 1
        )
        self._evi             = weibull_decisions["evi"].fillna(0).values.astype(float)
        self._n_intervene     = int(self._intervene_mask.sum())
        self._n_total         = len(weibull_decisions)

    # =========================================================================
    # 1. CAC (Customer Retention Cost)
    # =========================================================================

    def compute_cac_retention(self) -> dict:
        """
        CAC_retention = campaign_spend / expected_customers_retained

        expected_retained = n_contacts × p_response × (% truly at-risk)
        """
        n_contact      = self._n_intervene
        campaign_spend = n_contact * self.cost

        # Expected retentions: contacted × response_rate × P(would have churned)
        churn_intervene = self._churn_prob[self._intervene_mask]
        expected_retained = n_contact * self.p_response * churn_intervene.mean()

        cac = campaign_spend / max(expected_retained, 0.01)

        # Comparison: RFM CAC
        n_rfm_funded    = self.mc_results.get("n_rfm_funded", int(self._n_total * 0.4)) if self.mc_results else int(self._n_total * 0.4)
        rfm_n_contact   = int(n_rfm_funded)
        rfm_spend       = rfm_n_contact * self.cost
        n_rfm_sleeping  = self.mc_results.get("n_rfm_sleeping_dogs", 0) if self.mc_results else 0
        rfm_retained    = rfm_n_contact * self.p_response * 0.5
        rfm_cac         = rfm_spend / max(rfm_retained, 0.01)

        return {
            "n_intervene":         n_contact,
            "campaign_spend":      round(campaign_spend, 2),
            "expected_retained":   round(expected_retained, 2),
            "CAC_retention":       round(cac, 2),
            "n_rfm_funded":        rfm_n_contact,
            "rfm_spend":           round(rfm_spend, 2),
            "rfm_CAC_retention":   round(rfm_cac, 2),
            "cac_reduction_pct":   round((1 - cac / max(rfm_cac, 0.01)) * 100, 2),
        }

    # =========================================================================
    # 2. ROI
    # =========================================================================

    def compute_roi(self, periods: int = 6) -> dict:
        """
        ROI = (revenue_retained - campaign_cost) / campaign_cost × 100%

        revenue_retained = sum(CLV_i × p_response × churn_prob_i)
                           for INTERVENE customers,
                           scaled by (periods / total_lifetime_periods).
        """
        m_intervene  = self._monetary[self._intervene_mask]
        cp_intervene = self._churn_prob[self._intervene_mask]

        # Revenue saved from retention (expected)
        revenue_retained = (m_intervene * self.p_response * cp_intervene).sum()
        campaign_cost    = self._n_intervene * self.cost

        roi = (revenue_retained - campaign_cost) / max(campaign_cost, 0.01) * 100

        # Weibull median profit from MC (1-period)
        mc_w_median = self.mc_results.get("weibull_profit_ci", (0, 0, 0))[1] if self.mc_results else 0
        mc_r_median = self.mc_results.get("rfm_profit_ci",    (0, 0, 0))[1] if self.mc_results else 0

        return {
            "revenue_retained":    round(revenue_retained, 2),
            "campaign_cost":       round(campaign_cost, 2),
            "ROI_pct":             round(roi, 2),
            "ROI_profitable":      roi > 0,
            "MC_weibull_profit":   round(mc_w_median, 2),
            "MC_rfm_profit":       round(mc_r_median, 2),
            "MC_profit_ratio":     round(mc_w_median / max(abs(mc_r_median), 1), 2),
        }

    # =========================================================================
    # 3. Payback Period (from multi-period simulation)
    # =========================================================================

    def compute_payback_period(self) -> dict:
        """
        From advanced simulation: find first period where cumulative Weibull
        profit turns positive (break-even month).
        """
        if self.advsim_results is None:
            logger.warning("[BusinessMetrics] advsim_results not provided — payback period N/A.")
            return {"payback_period": None, "note": "Run advanced simulation first."}

        sim_results = self.advsim_results.get("results", {})
        realistic   = sim_results.get("realistic", [])
        by_policy   = {r.policy: r for r in realistic}
        w_res       = by_policy.get("Weibull")

        if w_res is None:
            return {"payback_period": None, "note": "Weibull results not found."}

        cum_profit   = w_res.cumulative_profit_median
        cum_profit_lo = w_res.cumulative_profit_lo
        cum_profit_hi = w_res.cumulative_profit_hi

        # Find first period with positive cumulative profit
        positive_periods = np.where(cum_profit > 0)[0]
        payback_month    = int(positive_periods[0] + 1) if len(positive_periods) > 0 else None

        # Conservative payback (using lower CI)
        positive_lo   = np.where(cum_profit_lo > 0)[0]
        payback_cons  = int(positive_lo[0] + 1) if len(positive_lo) > 0 else None

        r_res = by_policy.get("RFM")
        rfm_payback = None
        if r_res is not None:
            rfm_positive = np.where(r_res.cumulative_profit_median > 0)[0]
            rfm_payback  = int(rfm_positive[0] + 1) if len(rfm_positive) > 0 else None

        return {
            "payback_period_median_months": payback_month,
            "payback_period_conservative":  payback_cons,
            "rfm_payback_period":           rfm_payback,
            "weibull_faster_payback":       (
                payback_month is not None and rfm_payback is not None
                and payback_month <= rfm_payback
            ),
            "final_cumulative_profit":      round(float(cum_profit[-1]), 2),
        }

    # =========================================================================
    # 4. Retention Rate & ARPU Uplift
    # =========================================================================

    def compute_retention_metrics(self) -> dict:
        """
        Retention Rate = % of INTERVENE customers who do NOT churn
                         after receiving the campaign.
        ARPU Uplift    = delta revenue per active user between Weibull and no-intervention.
        """
        # Retention rate for intervene group
        churn_i  = self._churn_prob[self._intervene_mask]
        # Without intervention: natural churn rate
        natural_retention = 1.0 - churn_i.mean()
        # With intervention: churn reduced by p_response × some retention effect
        campaign_retention = natural_retention + self.p_response * churn_i.mean()
        campaign_retention = min(campaign_retention, 1.0)

        retention_lift = campaign_retention - natural_retention

        # ARPU = total revenue / active customers
        m_all   = self._monetary.mean()
        m_ret   = self._monetary[self._intervene_mask].mean()

        return {
            "natural_retention_rate":   round(natural_retention * 100, 2),
            "campaign_retention_rate":  round(campaign_retention * 100, 2),
            "retention_lift_pp":        round(retention_lift * 100, 2),
            "avg_monetary_intervene":   round(m_ret, 2),
            "avg_monetary_all":         round(m_all, 2),
            "high_value_concentration": round(m_ret / max(m_all, 0.01), 2),
        }

    # =========================================================================
    # 5. Cohort Retention Heatmap
    # =========================================================================

    def compute_cohort_retention_matrix(
        self,
        n_cohorts: int = 4,
        n_periods: int = 6,
    ) -> pd.DataFrame:
        """
        Build a cohort × period retention matrix using Weibull survival predictions.

        Cohorts are defined by RFM Monetary quartile (proxy for customer tenure value).
        Retention in period k = average S(base_T + k × 30) for cohort members.

        Parameters
        ----------
        n_cohorts : int
            Number of customer cohorts (default: 4 quartiles).
        n_periods : int
            Number of forward periods to compute (default: 6 months).

        Returns
        -------
        pd.DataFrame
            Index = cohort labels, columns = "Month_1" .. "Month_K",
            values = retention rate (%) — 100% at month 0 baseline.
        """
        monetary = self.decisions["Monetary"].fillna(0).values
        quantile_cuts = np.quantile(monetary, np.linspace(0, 1, n_cohorts + 1))
        quantile_cuts[-1] += 1  # include max

        cohort_labels = []
        cohort_masks  = []
        for i in range(n_cohorts):
            lo = quantile_cuts[i]
            hi = quantile_cuts[i + 1]
            mask = (monetary >= lo) & (monetary < hi)
            label = f"Q{i+1} ({lo:.0f}–{hi:.0f} MU)"
            cohort_labels.append(label)
            cohort_masks.append(mask)

        # Use survival values from the decisions table as period-0 retention
        # Approximate per-period retention using the Weibull survival function
        # S(t_0 + k * 30) ≈ S(t_0)^(1 + k * hazard_rate)  -- simplified extrapolation
        surv_0 = np.clip(self.decisions["survival"].fillna(0.5).values, 0, 1)
        hazard  = np.clip(self.decisions["hazard_now"].fillna(0).values, 0, 0.5)

        rows = {}
        for label, mask in zip(cohort_labels, cohort_masks):
            if mask.sum() < 2:
                continue
            row = [100.0]  # period 0 = 100% retained
            s0   = surv_0[mask].mean()
            h0   = hazard[mask].mean()
            for k in range(1, n_periods + 1):
                # Simplified Weibull extrapolation: S(t+k) ≈ S(t) × exp(-h × k × 30)
                s_k = s0 * np.exp(-h0 * k * 30)
                row.append(round(s_k * 100, 2))
            rows[label] = row

        cols = ["Month_0"] + [f"Month_{k}" for k in range(1, n_periods + 1)]
        df   = pd.DataFrame(rows, index=cols).T
        df.index.name = "Cohort"
        return df

    # =========================================================================
    # 6. Break-Even K (minimum contacts for positive ROI)
    # =========================================================================

    def compute_break_even_k(self) -> dict:
        """
        Find the minimum number of INTERVENE contacts K such that:
          sum_top_K(EVI_i) > 0
        where customers are ranked by EVI descending.
        """
        evi_intervene = self._evi[self._intervene_mask]
        evi_sorted    = np.sort(evi_intervene)[::-1]  # best EVI first
        cum_evi       = np.cumsum(evi_sorted)
        positive_k    = np.where(cum_evi > 0)[0]
        break_even_k  = int(positive_k[0] + 1) if len(positive_k) > 0 else None

        return {
            "break_even_k":         break_even_k,
            "total_intervene_pool": self._n_intervene,
            "full_pool_total_evi":  round(float(self._evi[self._intervene_mask].sum()), 2),
            "avg_evi_intervene":    round(float(evi_intervene.mean()), 2),
        }

    # =========================================================================
    # Main compute: all metrics
    # =========================================================================

    def compute_all(self) -> dict:
        """Return all business metrics in a single dict."""
        return {
            "CAC":       self.compute_cac_retention(),
            "ROI":       self.compute_roi(),
            "Payback":   self.compute_payback_period(),
            "Retention": self.compute_retention_metrics(),
            "BreakEven": self.compute_break_even_k(),
        }

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_cohort_retention_heatmap(
        self,
        n_cohorts: int = 4,
        n_periods: int = 6,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Publication-quality cohort retention heatmap.
        Rows = CLV cohorts, Columns = months, Values = retention %.
        """
        matrix = self.compute_cohort_retention_matrix(n_cohorts, n_periods)
        if matrix.empty:
            return plt.figure()

        fig, ax = plt.subplots(figsize=(max(8, n_periods + 2), max(4, n_cohorts + 1)))

        # Drop Month_0 (all 100%)
        plot_matrix = matrix.drop(columns=["Month_0"], errors="ignore")
        vals = plot_matrix.values.astype(float)

        im = ax.imshow(vals, cmap="YlOrRd_r", aspect="auto",
                       vmin=max(0, vals.min() - 5), vmax=100)

        ax.set_xticks(range(len(plot_matrix.columns)))
        ax.set_xticklabels(plot_matrix.columns, fontsize=10)
        ax.set_yticks(range(len(plot_matrix.index)))
        ax.set_yticklabels(plot_matrix.index, fontsize=10)

        # Annotate cells
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals[i, j]
                color = "white" if v < 50 else "black"
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Retention Rate (%)", fraction=0.03, pad=0.04)
        ax.set_xlabel("Month After Campaign", fontsize=12)
        ax.set_ylabel("Customer Cohort (by CLV Quartile)", fontsize=12)
        ax.set_title(
            f"Cohort Retention Heatmap — {self.dataset_label}\n"
            f"(Weibull survival model projection)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[BusinessMetrics] Cohort heatmap saved -> %s", save_path)
        return fig

    def plot_business_summary(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        2×2 summary dashboard: CAC, ROI, Payback, Break-Even.
        """
        all_metrics = self.compute_all()
        cac  = all_metrics["CAC"]
        roi  = all_metrics["ROI"]
        pay  = all_metrics["Payback"]
        bev  = all_metrics["BreakEven"]
        ret  = all_metrics["Retention"]

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(f"Business Metrics Dashboard — {self.dataset_label}",
                     fontsize=14, fontweight="bold")

        # ── [0,0] CAC comparison ────────────────────────────────────────────────
        ax = axes[0, 0]
        policies   = ["Weibull", "RFM"]
        cac_vals   = [cac["CAC_retention"], cac["rfm_CAC_retention"]]
        colors_cac = ["#3498db", "#e74c3c"]
        bars = ax.bar(policies, cac_vals, color=colors_cac, alpha=0.85)
        for bar, val in zip(bars, cac_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(cac_vals) * 0.01,
                    f"{val:.2f} MU", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("CAC_retention (MU per customer retained)", fontsize=10)
        ax.set_title(f"Customer Retention Cost\n(Weibull: {cac['cac_reduction_pct']:+.1f}% vs RFM)",
                     fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

        # ── [0,1] ROI ───────────────────────────────────────────────────────────
        ax2 = axes[0, 1]
        roi_items = [
            ("Revenue\nRetained",   roi["revenue_retained"],  "#2ecc71"),
            ("Campaign\nCost",     -roi["campaign_cost"],    "#e74c3c"),
            ("Net Profit\n(Weibull MC)", roi["MC_weibull_profit"], "#3498db"),
            ("Net Profit\n(RFM MC)",    roi["MC_rfm_profit"],     "#e67e22"),
        ]
        x2 = np.arange(len(roi_items))
        bar_colors = [c for _, _, c in roi_items]
        bar_vals   = [v for _, v, _ in roi_items]
        bar_lbls   = [l for l, _, _ in roi_items]
        bars2 = ax2.bar(x2, bar_vals, color=bar_colors, alpha=0.85)
        for bar, val in zip(bars2, bar_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     (bar.get_height() if val >= 0 else 0) + max(abs(v) for v in bar_vals) * 0.01,
                     f"{val:,.0f}", ha="center", fontsize=9, fontweight="bold")
        ax2.axhline(0, color="#333", lw=0.8)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(bar_lbls, fontsize=9)
        ax2.set_ylabel("MU", fontsize=10)
        ax2.set_title(f"ROI Components | Weibull ROI: {roi['ROI_pct']:+.1f}%",
                      fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # ── [1,0] Payback period (from simulation or estimate) ─────────────────
        ax3 = axes[1, 0]
        payback_m = pay.get("payback_period_median_months")
        rfm_payback_m = pay.get("rfm_payback_period")
        final_profit  = pay.get("final_cumulative_profit", 0)

        if self.advsim_results is not None:
            sim_results = self.advsim_results.get("results", {})
            realistic   = sim_results.get("realistic", [])
            by_policy   = {r.policy: r for r in realistic}
            w_res = by_policy.get("Weibull")
            r_res = by_policy.get("RFM")
            if w_res is not None:
                periods = list(range(1, len(w_res.cumulative_profit_median) + 1))
                ax3.plot(periods, w_res.cumulative_profit_median, "o-",
                         color="#3498db", lw=2.5, ms=7, label="Weibull")
                ax3.fill_between(periods,
                                 w_res.cumulative_profit_lo,
                                 w_res.cumulative_profit_hi,
                                 alpha=0.15, color="#3498db")
                if payback_m:
                    ax3.axvline(payback_m, color="#3498db", lw=1.5, ls="--",
                                alpha=0.7, label=f"Weibull payback: M{payback_m}")
            if r_res is not None:
                ax3.plot(periods, r_res.cumulative_profit_median, "s--",
                         color="#e74c3c", lw=2.0, ms=6, label="RFM")
                if rfm_payback_m:
                    ax3.axvline(rfm_payback_m, color="#e74c3c", lw=1.5, ls=":",
                                alpha=0.7, label=f"RFM payback: M{rfm_payback_m}")
            ax3.axhline(0, color="#ccc", lw=0.8)
            ax3.set_xlabel("Month", fontsize=10)
            ax3.set_ylabel("Cumulative Profit (MU)", fontsize=10)
            ax3.set_title("Payback Period (Cumulative Profit Trajectory)",
                          fontsize=11, fontweight="bold")
            ax3.legend(fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        else:
            payback_label = f"Month {payback_m}" if payback_m else "Not reached"
            ax3.text(0.5, 0.5,
                     f"Payback Period:\n{payback_label}\n\n"
                     f"(Run advanced simulation\nfor trajectory plot)",
                     ha="center", va="center", fontsize=12,
                     transform=ax3.transAxes)
            ax3.set_title("Payback Period", fontsize=11, fontweight="bold")

        # ── [1,1] Retention metrics ─────────────────────────────────────────────
        ax4 = axes[1, 1]
        ret_items = [
            ("Natural\nRetention", ret["natural_retention_rate"], "#95a5a6"),
            ("Campaign\nRetention", ret["campaign_retention_rate"], "#2ecc71"),
        ]
        x4 = np.arange(len(ret_items))
        for i, (label, val, color) in enumerate(ret_items):
            ax4.bar(i, val, color=color, alpha=0.85)
            ax4.text(i, val + 0.5, f"{val:.1f}%", ha="center",
                     fontsize=12, fontweight="bold")
        ax4.set_xticks(x4)
        ax4.set_xticklabels([l for l, _, _ in ret_items], fontsize=10)
        ax4.set_ylim(0, 110)
        ax4.set_ylabel("Retention Rate (%)", fontsize=10)
        ax4.set_title(
            f"Retention Rate Uplift: +{ret['retention_lift_pp']:.1f}pp\n"
            f"Break-even at K={bev['break_even_k']} contacts (avg EVI={bev['avg_evi_intervene']:.2f} MU)",
            fontsize=11, fontweight="bold",
        )
        ax4.grid(True, alpha=0.3, axis="y")
        ax4.axhline(100, color="#ccc", lw=0.5, ls="--")

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[BusinessMetrics] Summary dashboard saved -> %s", save_path)
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Flatten all metrics into a single-row DataFrame for easy reporting."""
        all_m = self.compute_all()
        flat  = {"dataset": self.dataset_label}
        for section, vals in all_m.items():
            for k, v in vals.items():
                flat[f"{section}_{k}"] = v
        return pd.DataFrame([flat])


# =============================================================================
# Convenience wrapper
# =============================================================================

def compute_business_metrics(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
    mc_results: dict,
    advsim_results: Optional[dict] = None,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    cost_per_contact: float = 1.0,
    p_response: float = 0.15,
) -> dict:
    """
    One-call: compute all business metrics, generate plots, save outputs.

    Returns
    -------
    dict with keys:
        calculator : BusinessMetricsCalculator instance
        metrics    : dict of all metrics
        metrics_df : pd.DataFrame (single row, all metrics)
        figs       : dict of matplotlib Figures
    """
    calc = BusinessMetricsCalculator(
        weibull_decisions=weibull_decisions,
        customer_df=customer_df,
        mc_results=mc_results,
        advsim_results=advsim_results,
        cost_per_contact=cost_per_contact,
        p_response=p_response,
        dataset_label=dataset_label,
    )

    metrics    = calc.compute_all()
    metrics_df = calc.to_dataframe()
    figs       = {}

    cohort_path  = os.path.join(save_dir, "business_cohort_retention.png") if save_dir else None
    summary_path = os.path.join(save_dir, "business_summary_dashboard.png")if save_dir else None

    figs["cohort_heatmap"]     = calc.plot_cohort_retention_heatmap(save_path=cohort_path)
    figs["business_dashboard"] = calc.plot_business_summary(save_path=summary_path)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(save_dir, "business_metrics.csv"), index=False)

    logger.info(
        "[BusinessMetrics] CAC=%.2f | ROI=%.1f%% | Payback=M%s | Retention_lift=+%.1f pp",
        metrics["CAC"]["CAC_retention"],
        metrics["ROI"]["ROI_pct"],
        metrics["Payback"].get("payback_period_median_months", "N/A"),
        metrics["Retention"]["retention_lift_pp"],
    )
    return {
        "calculator": calc,
        "metrics":    metrics,
        "metrics_df": metrics_df,
        "figs":       figs,
    }
