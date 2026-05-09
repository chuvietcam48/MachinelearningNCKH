"""
src/analysis/sensitivity_analysis.py
======================================
Comprehensive Multi-Parameter Sensitivity Analysis.

Addresses the core limitation: "how robust are the business conclusions
to changes in key modelling assumptions?"

Parameters swept
-----------------
response_rate        : P(customer responds to campaign) — captures marketing
                       effectiveness uncertainty
sleeping_dog_penalty : brand-damage cost fraction for mis-targeting low-risk
                       customers — captures the cost of mass-marketing errors
marketing_budget     : total spend per campaign period — captures operational
                       budget constraints
hazard_threshold     : h(t) threshold for INTERVENE decision — policy aggressiveness

For each parameter value the existing budget-constrained Monte Carlo is
re-run (n_mc iterations) and three outcome metrics are recorded:
  - Weibull median profit
  - RFM median profit
  - Efficiency gain (Weibull vs RFM)
  - n_intervene (policy selectivity)

Outputs
-------
  - 4 line charts (one per parameter)
  - Tornado chart: which parameter drives the most outcome variance
  - 2D heatmap: response_rate x sleeping_dog_penalty interaction
  - Stability table: CV (std/mean) for each scenario
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Default parameter grids
# =============================================================================

_DEFAULT_RESPONSE_RATES   = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
_DEFAULT_PENALTIES        = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
_DEFAULT_BUDGETS          = [100, 200, 300, 500, 750, 1000, 1500, 2000]
_DEFAULT_HAZARD_THRESHOLDS= [0.002, 0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050]
_N_MC_SENSITIVITY         = 300     # MC iterations per parameter value


# =============================================================================
# ComprehensiveSensitivityAnalyzer
# =============================================================================

class ComprehensiveSensitivityAnalyzer:
    """
    Sweeps four key parameters and measures impact on Weibull vs RFM profit gap.

    Parameters
    ----------
    df_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions(). Required cols:
        decision, Monetary, survival, evi, hazard_now.
    base_response_rate : float
        Nominal response rate (paper baseline).
    base_penalty : float
        Nominal sleeping-dog penalty fraction.
    base_budget : float
        Nominal marketing budget (MU).
    base_hazard_threshold : float
        Nominal hazard threshold theta_h.
    rfm_top_pct : float
        Fraction of customers targeted by RFM baseline.
    n_mc : int
        Monte Carlo iterations per parameter value.
    seed : int
    """

    def __init__(
        self,
        df_decisions: pd.DataFrame,
        base_response_rate: float = 0.15,
        base_penalty: float = 0.20,
        base_budget: float = 500.0,
        base_hazard_threshold: float = 0.01,
        rfm_top_pct: float = 0.40,
        n_mc: int = _N_MC_SENSITIVITY,
        seed: int = 42,
    ):
        self.decisions           = df_decisions.copy()
        self.base_rr             = base_response_rate
        self.base_penalty        = base_penalty
        self.base_budget         = base_budget
        self.base_theta_h        = base_hazard_threshold
        self.rfm_top_pct         = rfm_top_pct
        self.n_mc                = n_mc
        self.rng                 = np.random.default_rng(seed)

        # Pre-extract arrays for speed
        self._monetary   = df_decisions["Monetary"].fillna(0).values.astype(float)
        self._churn_prob = np.clip(1.0 - df_decisions["survival"].fillna(0.5).values, 0, 1)
        self._evi        = df_decisions["evi"].fillna(0).values.astype(float)
        self._hazard     = df_decisions["hazard_now"].fillna(0).values.astype(float)
        self._is_intervene = (df_decisions["decision"] == "INTERVENE").values

        # RFM pool (static across sweeps)
        rfm_thresh = np.quantile(self._monetary, 1.0 - rfm_top_pct)
        self._rfm_idx = np.where(self._monetary >= rfm_thresh)[0]
        self._rfm_sorted = self._rfm_idx[np.argsort(self._monetary[self._rfm_idx])[::-1]]

        logger.info(
            "[Sensitivity] Ready | n=%d | weibull_pool=%d | rfm_pool=%d",
            len(df_decisions), self._is_intervene.sum(), len(self._rfm_sorted),
        )

    # =========================================================================
    # Core MC engine (fast, param-agnostic)
    # =========================================================================

    def _run_mc(
        self,
        response_rate: float,
        penalty: float,
        budget: float,
        hazard_threshold: float,
    ) -> Tuple[float, float, float, int]:
        """
        Run n_mc iterations and return (weibull_median, rfm_median, eff_gain, n_intervene).
        Re-applies hazard_threshold to rebuild the INTERVENE pool dynamically.
        """
        # Rebuild INTERVENE pool for this theta_h
        w_mask   = (self._hazard > hazard_threshold) & (self._evi > 0) & \
                   (1 - self._churn_prob > 0.05)        # survival floor guard
        w_sorted = np.where(w_mask)[0]
        w_sorted = w_sorted[np.argsort(self._evi[w_sorted])[::-1]]
        n_w_pool = len(w_sorted)

        w_profits = np.empty(self.n_mc)
        r_profits = np.empty(self.n_mc)

        for i in range(self.n_mc):
            p_resp  = float(np.clip(self.rng.normal(response_rate, 0.03), 0, 1))
            cost_i  = float(max(self.rng.normal(1.0, 0.10), 0.1))
            max_c   = max(int(np.floor(budget / cost_i)), 1)

            # Weibull
            n_w = min(n_w_pool, max_c)
            tgt = w_sorted[:n_w]
            w_profits[i] = float(np.sum(
                self._monetary[tgt] * p_resp * self._churn_prob[tgt] - cost_i
            ))

            # RFM with sleeping dog penalty
            n_r   = min(len(self._rfm_sorted), max_c)
            r_tgt = self._rfm_sorted[:n_r]
            persuadable = self._hazard[r_tgt] > hazard_threshold
            r_pos = np.where(persuadable,
                             self._monetary[r_tgt] * p_resp - cost_i, 0.0)
            r_neg = np.where(~persuadable,
                             -self._monetary[r_tgt] * penalty - cost_i, 0.0)
            r_profits[i] = float(np.sum(r_pos + r_neg))

        w_med    = float(np.median(w_profits))
        r_med    = float(np.median(r_profits))
        eff_gain = (w_med - r_med) / max(abs(r_med), 1.0)
        return w_med, r_med, eff_gain, n_w_pool

    # =========================================================================
    # Individual parameter sweeps
    # =========================================================================

    def sweep_response_rate(
        self, values: Optional[List[float]] = None
    ) -> pd.DataFrame:
        values = values or _DEFAULT_RESPONSE_RATES
        rows = []
        for v in values:
            w, r, eg, n = self._run_mc(v, self.base_penalty,
                                        self.base_budget, self.base_theta_h)
            rows.append({"response_rate": v, "weibull": w, "rfm": r,
                         "efficiency_gain": eg, "n_intervene": n})
            logger.info("  response_rate=%.2f -> Weibull=%+.0f | RFM=%+.0f | eff=%+.3f",
                        v, w, r, eg)
        return pd.DataFrame(rows)

    def sweep_sleeping_dog_penalty(
        self, values: Optional[List[float]] = None
    ) -> pd.DataFrame:
        values = values or _DEFAULT_PENALTIES
        rows = []
        for v in values:
            w, r, eg, n = self._run_mc(self.base_rr, v,
                                        self.base_budget, self.base_theta_h)
            rows.append({"sleeping_dog_penalty": v, "weibull": w, "rfm": r,
                         "efficiency_gain": eg, "n_intervene": n})
            logger.info("  penalty=%.2f -> Weibull=%+.0f | RFM=%+.0f | eff=%+.3f",
                        v, w, r, eg)
        return pd.DataFrame(rows)

    def sweep_budget(
        self, values: Optional[List[float]] = None
    ) -> pd.DataFrame:
        values = values or _DEFAULT_BUDGETS
        rows = []
        for v in values:
            w, r, eg, n = self._run_mc(self.base_rr, self.base_penalty,
                                        float(v), self.base_theta_h)
            rows.append({"budget": v, "weibull": w, "rfm": r,
                         "efficiency_gain": eg, "n_intervene": n})
            logger.info("  budget=%.0f -> Weibull=%+.0f | RFM=%+.0f | eff=%+.3f",
                        v, w, r, eg)
        return pd.DataFrame(rows)

    def sweep_hazard_threshold(
        self, values: Optional[List[float]] = None
    ) -> pd.DataFrame:
        values = values or _DEFAULT_HAZARD_THRESHOLDS
        rows = []
        for v in values:
            w, r, eg, n = self._run_mc(self.base_rr, self.base_penalty,
                                        self.base_budget, v)
            rows.append({"hazard_threshold": v, "weibull": w, "rfm": r,
                         "efficiency_gain": eg, "n_intervene": n})
            logger.info("  theta_h=%.4f -> Weibull=%+.0f | RFM=%+.0f | eff=%+.3f | pool=%d",
                        v, w, r, eg, n)
        return pd.DataFrame(rows)

    def sweep_2d(
        self,
        rr_values: Optional[List[float]] = None,
        penalty_values: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        """2D sweep: response_rate x sleeping_dog_penalty."""
        rr_values      = rr_values      or [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        penalty_values = penalty_values or [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        rows = []
        for rr in rr_values:
            for pen in penalty_values:
                w, r, eg, _ = self._run_mc(rr, pen, self.base_budget, self.base_theta_h)
                rows.append({"response_rate": rr, "penalty": pen,
                             "efficiency_gain": eg, "weibull": w, "rfm": r})
        return pd.DataFrame(rows)

    # =========================================================================
    # Full sweep + tornado
    # =========================================================================

    def run_full_sweep(self) -> Dict[str, pd.DataFrame]:
        """Run all four individual sweeps and 2D interaction. Return dict of DataFrames."""
        logger.info("[Sensitivity] Starting full parameter sweep...")
        results = {
            "response_rate":        self.sweep_response_rate(),
            "sleeping_dog_penalty": self.sweep_sleeping_dog_penalty(),
            "budget":               self.sweep_budget(),
            "hazard_threshold":     self.sweep_hazard_threshold(),
            "2d_interaction":       self.sweep_2d(),
        }
        logger.info("[Sensitivity] Full sweep complete.")
        return results

    def _compute_tornado_range(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        For each parameter, compute the range (max - min) of efficiency_gain
        across its sweep values. Larger range = more sensitive parameter.
        """
        param_names = {
            "response_rate":        "Response Rate",
            "sleeping_dog_penalty": "Sleeping Dog Penalty",
            "budget":               "Marketing Budget",
            "hazard_threshold":     "Hazard Threshold (theta_h)",
        }
        rows = []
        for key, label in param_names.items():
            df = results.get(key)
            if df is None or df.empty:
                continue
            eg = df["efficiency_gain"]
            w  = df["weibull"]
            rows.append({
                "parameter":           label,
                "param_key":           key,
                "eff_gain_min":        eg.min(),
                "eff_gain_max":        eg.max(),
                "eff_gain_range":      eg.max() - eg.min(),
                "eff_gain_cv":         eg.std() / (abs(eg.mean()) + 1e-9),
                "weibull_profit_min":  w.min(),
                "weibull_profit_max":  w.max(),
                "weibull_profit_range":w.max() - w.min(),
                "weibull_always_positive": bool((w > 0).all()),
                "weibull_always_beats_rfm": bool((df["weibull"] > df["rfm"]).all()),
            })
        return pd.DataFrame(rows).sort_values("eff_gain_range", ascending=False)

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_sensitivity_lines(
        self,
        results: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        4-panel line chart: one per parameter.
        Each panel shows Weibull profit, RFM profit, and efficiency gain (right axis).
        """
        param_specs = [
            ("response_rate",        "response_rate",        "Response Rate",
             lambda x: f"{x:.0%}"),
            ("sleeping_dog_penalty", "sleeping_dog_penalty", "Sleeping Dog Penalty",
             lambda x: f"{x:.0%}"),
            ("budget",               "budget",               "Marketing Budget (MU)",
             lambda x: f"{x:,.0f}"),
            ("hazard_threshold",     "hazard_threshold",     "Hazard Threshold θ_h",
             lambda x: f"{x:.3f}"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        fig.suptitle(
            f"Sensitivity Analysis — {dataset_label}\n"
            f"(baseline: response_rate={self.base_rr:.0%}, "
            f"penalty={self.base_penalty:.0%}, budget={self.base_budget:.0f} MU, "
            f"theta_h={self.base_theta_h:.3f})",
            fontsize=12, fontweight="bold",
        )

        for ax, (key, x_col, x_label, x_fmt) in zip(axes, param_specs):
            df = results.get(key)
            if df is None or df.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")
                continue

            x = df[x_col].values
            x_ticks = [x_fmt(v) for v in x]

            # Primary axis: profit
            ax.plot(x_ticks, df["weibull"].values / 1000, "o-",
                    color="#3498db", lw=2.5, ms=7, label="Weibull", zorder=3)
            ax.plot(x_ticks, df["rfm"].values / 1000, "s--",
                    color="#e74c3c", lw=2.0, ms=6, label="RFM", zorder=3)
            ax.axhline(0, color="#ccc", lw=0.8)
            ax.fill_between(x_ticks,
                            df["weibull"].values / 1000,
                            df["rfm"].values / 1000,
                            alpha=0.08, color="#3498db", label="_nolegend_")

            ax.set_xlabel(x_label, fontsize=10)
            ax.set_ylabel("Profit (K MU)", fontsize=10)
            ax.tick_params(axis="x", rotation=30, labelsize=8)
            ax.grid(True, alpha=0.3)

            # Secondary axis: efficiency gain
            ax2 = ax.twinx()
            ax2.plot(x_ticks, df["efficiency_gain"].values * 100, "^:",
                     color="#2ecc71", lw=1.8, ms=6, label="Eff. Gain %")
            ax2.axhline(0, color="#2ecc71", lw=0.5, alpha=0.3)
            ax2.set_ylabel("Efficiency Gain (%)", fontsize=9, color="#2ecc71")
            ax2.tick_params(axis="y", colors="#2ecc71")

            # Mark baseline
            base_vals = {
                "response_rate": self.base_rr,
                "sleeping_dog_penalty": self.base_penalty,
                "budget": self.base_budget,
                "hazard_threshold": self.base_theta_h,
            }
            base_v = base_vals.get(key)
            base_x = x_fmt(base_v)
            if base_x in x_ticks:
                base_idx = x_ticks.index(base_x)
                ax.axvline(base_idx, color="#f39c12", lw=1.5, ls="--",
                           alpha=0.7, label="Baseline")

            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2,
                      fontsize=8, loc="best")
            ax.set_title(x_label, fontsize=11, fontweight="bold")

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[Sensitivity] Lines plot saved -> %s", save_path)
        return fig

    def plot_tornado(
        self,
        results: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Tornado chart: parameter impact on efficiency gain range.
        Longer bar = more sensitive = more important assumption.
        """
        tornado_df = self._compute_tornado_range(results)
        if tornado_df.empty:
            return plt.figure()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Efficiency gain range (tornado)
        ax = axes[0]
        y_pos  = np.arange(len(tornado_df))
        colors = ["#e74c3c" if r > tornado_df["eff_gain_range"].median()
                  else "#3498db" for r in tornado_df["eff_gain_range"]]
        bars   = ax.barh(tornado_df["parameter"], tornado_df["eff_gain_range"],
                         color=colors, alpha=0.85)
        for bar, val, beats in zip(bars,
                                   tornado_df["eff_gain_range"],
                                   tornado_df["weibull_always_beats_rfm"]):
            symbol = "✓" if beats else "!"
            ax.text(bar.get_width() + tornado_df["eff_gain_range"].max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}  [{symbol}]",
                    va="center", fontsize=9)
        ax.set_xlabel("Efficiency Gain Range (max - min)", fontsize=11)
        ax.set_title(f"Parameter Sensitivity (Tornado Chart)\n{dataset_label}",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.text(1.0, -0.12, "[✓] = Weibull always beats RFM   [!] = Not always",
                transform=ax.transAxes, ha="right", fontsize=8, color="#666")

        # Right: Coefficient of Variation (stability)
        ax2 = axes[1]
        bars2 = ax2.barh(tornado_df["parameter"],
                         tornado_df["eff_gain_cv"] * 100,
                         color=["#e74c3c" if cv > 0.5 else "#3498db"
                                for cv in tornado_df["eff_gain_cv"]],
                         alpha=0.85)
        for bar, cv in zip(bars2, tornado_df["eff_gain_cv"]):
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{cv:.2f}", va="center", fontsize=9)
        ax2.axvline(50, color="#f39c12", lw=1.5, ls="--",
                    label="50% CV threshold (high instability)")
        ax2.set_xlabel("Coefficient of Variation (%)", fontsize=11)
        ax2.set_title("Result Stability\n(lower CV = more robust conclusion)",
                      fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3, axis="x")

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[Sensitivity] Tornado chart saved -> %s", save_path)
        return fig

    def plot_2d_heatmap(
        self,
        results: Dict[str, pd.DataFrame],
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        2D heatmap: response_rate × sleeping_dog_penalty -> efficiency gain.
        Shows which combinations are safe vs risky.
        """
        df2d = results.get("2d_interaction")
        if df2d is None or df2d.empty:
            return plt.figure()

        pivot = df2d.pivot(index="penalty", columns="response_rate",
                           values="efficiency_gain")

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Left: Efficiency gain heatmap
        import matplotlib.colors as mcolors
        cmap = plt.cm.RdYlGn
        im = axes[0].imshow(pivot.values, cmap=cmap, aspect="auto",
                            vmin=pivot.values.min(), vmax=pivot.values.max())
        axes[0].set_xticks(range(len(pivot.columns)))
        axes[0].set_xticklabels([f"{v:.0%}" for v in pivot.columns], rotation=30)
        axes[0].set_yticks(range(len(pivot.index)))
        axes[0].set_yticklabels([f"{v:.0%}" for v in pivot.index])
        axes[0].set_xlabel("Response Rate", fontsize=11)
        axes[0].set_ylabel("Sleeping Dog Penalty", fontsize=11)
        axes[0].set_title(f"Efficiency Gain Heatmap\n(green=robust, red=risky)",
                          fontsize=12, fontweight="bold")
        plt.colorbar(im, ax=axes[0], label="Efficiency Gain")

        # Annotate cells with values
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                color = "white" if abs(val) > 0.5 else "black"
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=7, color=color, fontweight="bold")

        # Highlight baseline
        base_rr_vals = [f"{v:.0%}" for v in pivot.columns]
        base_pen_vals = [f"{v:.0%}" for v in pivot.index]
        base_rr_str  = f"{self.base_rr:.0%}"
        base_pen_str = f"{self.base_penalty:.0%}"
        if base_rr_str in base_rr_vals and base_pen_str in base_pen_vals:
            bj = base_rr_vals.index(base_rr_str)
            bi = base_pen_vals.index(base_pen_str)
            axes[0].add_patch(plt.Rectangle(
                (bj - 0.5, bi - 0.5), 1, 1,
                fill=False, edgecolor="gold", lw=3, label="Baseline"
            ))

        # Right: Weibull vs RFM sign map (green=Weibull wins, red=RFM wins)
        pivot_sign = df2d.pivot(index="penalty", columns="response_rate",
                                values="weibull") > df2d.pivot(
                                    index="penalty", columns="response_rate",
                                    values="rfm")
        sign_matrix = pivot_sign.values.astype(float)
        axes[1].imshow(sign_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        axes[1].set_xticks(range(len(pivot.columns)))
        axes[1].set_xticklabels([f"{v:.0%}" for v in pivot.columns], rotation=30)
        axes[1].set_yticks(range(len(pivot.index)))
        axes[1].set_yticklabels([f"{v:.0%}" for v in pivot.index])
        axes[1].set_xlabel("Response Rate", fontsize=11)
        axes[1].set_ylabel("Sleeping Dog Penalty", fontsize=11)
        axes[1].set_title("Weibull Wins (green) vs RFM Wins (red)\nunder each assumption combo",
                          fontsize=12, fontweight="bold")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                axes[1].text(j, i, "W" if sign_matrix[i, j] else "R",
                             ha="center", va="center", fontsize=10,
                             color="white", fontweight="bold")

        fig.suptitle(f"2D Sensitivity: Response Rate x Sleeping Dog Penalty — {dataset_label}",
                     fontsize=12, fontweight="bold")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[Sensitivity] 2D heatmap saved -> %s", save_path)
        return fig

    def get_stability_summary(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Returns a summary table showing whether Weibull always beats RFM
        across all parameter values for each parameter dimension.
        """
        tornado = self._compute_tornado_range(results)
        summary_rows = []
        for _, row in tornado.iterrows():
            key = row["param_key"]
            df  = results.get(key, pd.DataFrame())
            if df.empty:
                continue
            always_wins   = bool((df["weibull"] > df["rfm"]).all())
            weibull_pos   = bool((df["weibull"] > 0).all())
            min_eff       = float(df["efficiency_gain"].min())
            max_eff       = float(df["efficiency_gain"].max())
            mean_eff      = float(df["efficiency_gain"].mean())
            summary_rows.append({
                "Parameter":             row["parameter"],
                "Values_tested":         len(df),
                "Weibull_always_wins":   always_wins,
                "Weibull_always_positive": weibull_pos,
                "Min_efficiency_gain":   round(min_eff, 4),
                "Max_efficiency_gain":   round(max_eff, 4),
                "Mean_efficiency_gain":  round(mean_eff, 4),
                "Sensitivity_rank":      None,  # filled below
            })
        summary_df = pd.DataFrame(summary_rows)
        ranges = tornado["eff_gain_range"].values
        summary_df["Sensitivity_rank"] = pd.Series(ranges).rank(ascending=False).astype(int).values
        return summary_df.sort_values("Sensitivity_rank")


# =============================================================================
# Convenience wrapper
# =============================================================================

def run_sensitivity_analysis(
    df_decisions: pd.DataFrame,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    base_response_rate: float = 0.15,
    base_penalty: float = 0.20,
    base_budget: float = 500.0,
    base_hazard_threshold: float = 0.01,
    n_mc: int = _N_MC_SENSITIVITY,
    seed: int = 42,
) -> Dict:
    """
    One-call entry point: runs all sweeps, generates all plots, saves outputs.

    Returns
    -------
    dict with keys:
        analyzer       : ComprehensiveSensitivityAnalyzer instance
        sweep_results  : dict of DataFrames (one per parameter)
        stability_df   : pd.DataFrame summary table
        figs           : dict of matplotlib Figures
    """
    analyzer = ComprehensiveSensitivityAnalyzer(
        df_decisions=df_decisions,
        base_response_rate=base_response_rate,
        base_penalty=base_penalty,
        base_budget=base_budget,
        base_hazard_threshold=base_hazard_threshold,
        n_mc=n_mc,
        seed=seed,
    )

    logger.info("[Sensitivity] Running full parameter sweep (%d MC/value)...", n_mc)
    sweep_results = analyzer.run_full_sweep()
    stability_df  = analyzer.get_stability_summary(sweep_results)

    figs = {}
    lines_path   = os.path.join(save_dir, "sensitivity_lines.png")   if save_dir else None
    tornado_path = os.path.join(save_dir, "sensitivity_tornado.png") if save_dir else None
    heatmap_path = os.path.join(save_dir, "sensitivity_heatmap.png") if save_dir else None

    figs["lines"]   = analyzer.plot_sensitivity_lines(
        sweep_results, save_path=lines_path, dataset_label=dataset_label)
    figs["tornado"] = analyzer.plot_tornado(
        sweep_results, save_path=tornado_path, dataset_label=dataset_label)
    figs["heatmap"] = analyzer.plot_2d_heatmap(
        sweep_results, save_path=heatmap_path, dataset_label=dataset_label)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        stability_df.to_csv(os.path.join(save_dir, "sensitivity_stability.csv"),
                            index=False)
        for key, df in sweep_results.items():
            df.to_csv(os.path.join(save_dir, f"sensitivity_{key}.csv"), index=False)
        logger.info("[Sensitivity] All outputs saved to %s", save_dir)

    logger.info("[Sensitivity] Complete.\n%s", stability_df.to_string(index=False))

    return {
        "analyzer":      analyzer,
        "sweep_results": sweep_results,
        "stability_df":  stability_df,
        "figs":          figs,
    }
