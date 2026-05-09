"""
src/simulation/production_simulator.py
=========================================
Production-Style Rolling Simulation.

Simulates real-time deployment of the Weibull AFT retention policy over a
long horizon (12–24 months) with:

  1. **Rolling decision cycles** — At each cycle (weekly/monthly), the scorer
     evaluates ALL active customers. Only newly high-risk customers (hazard
     just crossed theta_h) are added to the campaign queue, avoiding
     re-contacting customers who were recently reached.

  2. **Cool-down period** — After a customer is contacted, they enter a
     cool-down window (e.g., 60 days) during which they are not re-targeted.
     This prevents spam and models real campaign constraints.

  3. **Response simulation** — Contacted customers respond with probability
     p_response. Responders' churn probability is reduced by the retention
     effect. Non-responders continue on their natural Weibull hazard path.

  4. **Customer state tracking** — Tracks each customer's status:
     ACTIVE, CONTACTED (in cool-down), RETAINED (responded), CHURNED.

  5. **Rolling metrics** — Per-cycle: contacts made, retentions, churn events,
     budget spent, cumulative profit.

  6. **24-month CLV horizon** — Revenue is accumulated over the full
     simulation period.

Usage
-----
    from src.simulation.production_simulator import ProductionSimulator

    sim = ProductionSimulator(
        waf=waf,
        df_scaled=df_scaled_waf,
        customer_df=customer_df,
        predicted_clv=predicted_clv_all,
        n_months=12,
        cooldown_days=60,
    )
    history_df = sim.run()
    fig = sim.plot_rolling_metrics(history_df)
    fig2 = sim.plot_state_evolution(history_df)
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
from typing import Optional, Dict, List
from enum import IntEnum

logger = logging.getLogger(__name__)


# ── Customer states ───────────────────────────────────────────────────────────
class State(IntEnum):
    ACTIVE    = 0
    CONTACTED = 1   # in cool-down
    RETAINED  = 2   # responded to intervention
    CHURNED   = 3


# =============================================================================
# ProductionSimulator
# =============================================================================

class ProductionSimulator:
    """
    Rolling production simulation with cool-down and state tracking.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Pre-processed customer features (same index as customer_df).
    customer_df : pd.DataFrame
        Original customer features indexed by CustomerID.
    predicted_clv : pd.Series, optional
        Forward-looking CLV per customer.
    n_months : int
        Simulation horizon in months (default: 12).
    cycle_days : int
        Days between decision cycles (default: 30 = monthly).
    cooldown_days : int
        Days a customer cannot be re-contacted after intervention (default: 60).
    theta_h : float
        Hazard threshold for INTERVENE decision.
    theta_s : float
        Survival floor — customers below this are LOST.
    p_response : float
        Campaign response rate.
    churn_reduction : float
        Fraction by which intervention reduces churn probability for responders.
    budget_per_cycle : float
        Marketing budget per decision cycle (MU).
    cost_per_contact : float
        Cost per outreach (MU).
    n_mc : int
        Monte Carlo paths to run (for uncertainty).
    seed : int
    """

    def __init__(
        self,
        waf,
        df_scaled: pd.DataFrame,
        customer_df: pd.DataFrame,
        predicted_clv: Optional[pd.Series] = None,
        n_months: int = 12,
        cycle_days: int = 30,
        cooldown_days: int = 60,
        theta_h: float = 0.01,
        theta_s: float = 0.05,
        p_response: float = 0.15,
        churn_reduction: float = 0.65,
        budget_per_cycle: float = 500.0,
        cost_per_contact: float = 1.0,
        n_mc: int = 100,
        seed: int = 42,
    ):
        self.waf              = waf
        self.df_scaled        = df_scaled.copy()
        self.customer_df      = customer_df.copy()
        self.n_months         = n_months
        self.cycle_days       = cycle_days
        self.cooldown_days    = cooldown_days
        self.theta_h          = theta_h
        self.theta_s          = theta_s
        self.p_response       = p_response
        self.churn_reduction  = churn_reduction
        self.budget_per_cycle = budget_per_cycle
        self.cost             = cost_per_contact
        self.n_mc             = n_mc
        self.rng              = np.random.default_rng(seed)

        self.n_customers = len(df_scaled)
        self.n_cycles    = int(np.ceil(n_months * 30 / cycle_days))

        # CLV per customer
        if predicted_clv is not None:
            self.clv = predicted_clv.reindex(df_scaled.index).fillna(0).values.astype(float)
        else:
            self.clv = customer_df["Monetary"].fillna(0).values.astype(float)

        # Base observation time
        self.base_T = float(df_scaled["T"].median())

        # Pre-compute survival at all cycle time points
        logger.info(
            "[ProductionSim] Pre-computing Weibull survival at %d cycles "
            "(%d days each, %d month horizon)...",
            self.n_cycles, cycle_days, n_months,
        )
        self._survival_matrix = self._precompute_survival()
        self._hazard_matrix   = self._compute_hazard_matrix()

        logger.info(
            "[ProductionSim] Ready | n=%d customers | %d cycles | "
            "cooldown=%d days | budget=%.0f/cycle",
            self.n_customers, self.n_cycles, cooldown_days, budget_per_cycle,
        )

    def _precompute_survival(self) -> np.ndarray:
        """S(base_T + k*cycle_days) for k=0..n_cycles. Shape: (n_cycles+1, n_customers)."""
        times = [self.base_T + k * self.cycle_days for k in range(self.n_cycles + 1)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surv_df = self.waf.predict_survival_function(self.df_scaled, times=times)
        return np.clip(surv_df.values, 0, 1)

    def _compute_hazard_matrix(self) -> np.ndarray:
        """Numerical hazard at each cycle. Shape: (n_cycles, n_customers)."""
        S  = self._survival_matrix
        dS = S[:-1] - S[1:]
        h  = dS / (S[:-1] + 1e-9) / self.cycle_days
        return np.clip(h, 0, None)

    # =========================================================================
    # Single Monte Carlo path
    # =========================================================================

    def _run_single_path(self, rng: np.random.Generator) -> List[dict]:
        """
        Simulate one MC path over n_cycles. Returns list of per-cycle dicts.
        """
        N = self.n_customers

        # Per-customer state variables
        state           = np.full(N, State.ACTIVE, dtype=int)
        cooldown_ends   = np.full(N, -1, dtype=int)   # cycle when cool-down ends
        cumulative_clv  = np.zeros(N)                 # CLV accumulated so far
        cycles_retained = np.zeros(N, dtype=int)      # how many cycles retained

        cycle_history = []

        for k in range(self.n_cycles):
            S_k  = self._survival_matrix[k + 1]   # survival at end of cycle k
            h_k  = self._hazard_matrix[k]          # hazard during cycle k
            p_churn_k = np.clip(
                (self._survival_matrix[k] - S_k) / (self._survival_matrix[k] + 1e-9),
                0, 1
            )

            # Stochastic parameters for this cycle
            p_resp  = float(np.clip(rng.normal(self.p_response, 0.03), 0, 1))
            cost_k  = float(max(rng.normal(self.cost, 0.10), 0.1))
            max_contacts = max(int(np.floor(self.budget_per_cycle / cost_k)), 1)

            # ── 1. Who is eligible for intervention? ──────────────────────────
            # Must be ACTIVE (not cooldown, not churned, not already retained)
            # AND hazard > theta_h AND survival above floor
            cooldown_active = (state == State.CONTACTED) & (k < cooldown_ends)
            eligible = (
                (state == State.ACTIVE)
                & (~cooldown_active)
                & (h_k > self.theta_h)
                & (S_k >= self.theta_s)
            )

            # Compute EVI for eligible customers
            evi_k = p_resp * self.clv * (1 - S_k) - cost_k
            eligible &= (evi_k > 0)

            # ── 2. Select top-EVI within budget ────────────────────────────────
            eligible_idx = np.where(eligible)[0]
            if len(eligible_idx) > 0:
                best = eligible_idx[np.argsort(evi_k[eligible_idx])[::-1]]
                funded = best[:max_contacts]
            else:
                funded = np.array([], dtype=int)

            n_contacts = len(funded)

            # ── 3. Simulate responses ──────────────────────────────────────────
            responded_mask = np.zeros(N, dtype=bool)
            if n_contacts > 0:
                responded = rng.random(n_contacts) < p_resp
                responders = funded[responded]
                responded_mask[responders] = True
                # Update state: contacted → cool-down
                state[funded] = State.CONTACTED
                cooldown_ends[funded] = k + int(np.ceil(self.cooldown_days / self.cycle_days))
                # Responders get retained status
                state[responders] = State.RETAINED

            # ── 4. Simulate churn ──────────────────────────────────────────────
            # Adjusted churn probability: responders get reduction
            p_adj = p_churn_k.copy()
            p_adj[responded_mask] *= (1 - self.churn_reduction)
            p_adj = np.clip(p_adj, 0, 1)

            # Only active/contacted/retained customers can churn
            active_mask = state != State.CHURNED
            u = rng.random(N)
            churned_this_cycle = active_mask & (u < p_adj) & (S_k < 0.50)
            state[churned_this_cycle] = State.CHURNED

            # ── 5. CLV accumulation for still-active customers ─────────────────
            still_active = state != State.CHURNED
            cumulative_clv[still_active] += self.clv[still_active] / self.n_months
            cycles_retained[still_active] += 1

            # ── 6. Release cool-down ───────────────────────────────────────────
            released = (state == State.CONTACTED) & (k >= cooldown_ends)
            state[released] = State.ACTIVE

            # ── 7. Cycle metrics ───────────────────────────────────────────────
            n_active    = int((state == State.ACTIVE).sum())
            n_contacted = int((state == State.CONTACTED).sum())
            n_retained  = int((state == State.RETAINED).sum())
            n_churned   = int((state == State.CHURNED).sum())
            n_churn_event = int(churned_this_cycle.sum())

            campaign_cost   = n_contacts * cost_k
            period_revenue  = float(cumulative_clv[still_active].sum()) / max(self.n_months, 1)
            cycle_profit    = period_revenue - campaign_cost

            cycle_history.append({
                "cycle":           k + 1,
                "month":           round((k + 1) * self.cycle_days / 30, 1),
                "n_active":        n_active,
                "n_contacted":     n_contacted,
                "n_retained":      n_retained,
                "n_churned":       n_churned,
                "n_churn_event":   n_churn_event,
                "n_contacts":      n_contacts,
                "n_responders":    int(responded_mask.sum()),
                "budget_used":     round(campaign_cost, 2),
                "period_revenue":  round(period_revenue, 2),
                "cycle_profit":    round(cycle_profit, 2),
                "churn_rate_pct":  round(n_churn_event / max(N, 1) * 100, 3),
                "retention_rate_pct": round(n_retained / max(N, 1) * 100, 2),
                "active_rate_pct": round(n_active / max(N, 1) * 100, 2),
            })

        return cycle_history

    # =========================================================================
    # Main run
    # =========================================================================

    def run(self) -> pd.DataFrame:
        """
        Run n_mc Monte Carlo paths and aggregate into median + CI per cycle.

        Returns
        -------
        pd.DataFrame
            Per-cycle statistics: median, 2.5th, 97.5th percentile
            for profit, churn rate, contacts, retention rate.
        """
        logger.info(
            "[ProductionSim] Running %d MC paths × %d cycles...",
            self.n_mc, self.n_cycles,
        )
        all_paths = []
        for mc_i in range(self.n_mc):
            path = self._run_single_path(self.rng)
            for row in path:
                row["mc_path"] = mc_i
            all_paths.extend(path)

        raw = pd.DataFrame(all_paths)
        agg_cols = ["cycle_profit", "period_revenue", "budget_used",
                    "n_contacts", "n_responders", "n_churn_event",
                    "n_retained", "n_active", "churn_rate_pct",
                    "retention_rate_pct", "active_rate_pct"]

        groups = raw.groupby("cycle")
        rows   = []
        for cycle, grp in groups:
            row = {
                "cycle": int(cycle),
                "month": grp["month"].median(),
            }
            for col in agg_cols:
                if col in grp:
                    row[f"{col}_median"] = grp[col].median()
                    row[f"{col}_lo"]     = grp[col].quantile(0.025)
                    row[f"{col}_hi"]     = grp[col].quantile(0.975)
            # Cumulative profit
            row["cumulative_profit_median"] = raw[raw["cycle"] <= cycle]["cycle_profit"].groupby(
                raw[raw["cycle"] <= cycle]["mc_path"]
            ).sum().median()
            rows.append(row)

        history_df = pd.DataFrame(rows)
        logger.info(
            "[ProductionSim] Done | final active_rate=%.1f%% | "
            "cumulative_profit=%.0f MU",
            history_df["active_rate_pct_median"].iloc[-1],
            history_df["cumulative_profit_median"].iloc[-1],
        )
        return history_df

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_rolling_metrics(
        self,
        history_df: pd.DataFrame,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        4-panel rolling metrics dashboard:
          [0,0] Cumulative profit with CI band
          [0,1] Churn rate per cycle
          [1,0] Contacts per cycle + responders
          [1,1] Customer state evolution (active, retained, churned %)
        """
        months = history_df["month"].values
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f"Production Simulation — Rolling Metrics ({self.n_months}-Month Horizon)\n"
            f"Dataset: {dataset_label} | cool-down={self.cooldown_days}d | "
            f"p_response={self.p_response:.0%} | budget={self.budget_per_cycle:.0f}/cycle",
            fontsize=12, fontweight="bold",
        )

        # ── [0,0] Cumulative profit ──────────────────────────────────────────
        ax = axes[0, 0]
        if "cumulative_profit_median" in history_df.columns:
            ax.plot(months, history_df["cumulative_profit_median"],
                    color="#3498db", lw=2.5, label="Cumulative Profit")
            if "cycle_profit_lo" in history_df.columns:
                cum_lo = history_df["cycle_profit_lo"].cumsum()
                cum_hi = history_df["cycle_profit_hi"].cumsum()
                ax.fill_between(months, cum_lo, cum_hi, alpha=0.15, color="#3498db")
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_ylabel("Cumulative Profit (MU)", fontsize=10)
        ax.set_title("Cumulative Profit Trajectory", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # ── [0,1] Churn rate ─────────────────────────────────────────────────
        ax2 = axes[0, 1]
        if "churn_rate_pct_median" in history_df.columns:
            ax2.plot(months, history_df["churn_rate_pct_median"],
                     color="#e74c3c", lw=2.0, label="Churn Rate %")
            ax2.fill_between(months,
                             history_df.get("churn_rate_pct_lo", history_df["churn_rate_pct_median"]),
                             history_df.get("churn_rate_pct_hi", history_df["churn_rate_pct_median"]),
                             alpha=0.15, color="#e74c3c")
        ax2.set_xlabel("Month", fontsize=10)
        ax2.set_ylabel("Churn Rate per Cycle (%)", fontsize=10)
        ax2.set_title("Monthly Churn Rate (Rolling)", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # ── [1,0] Contacts + responders ─────────────────────────────────────
        ax3 = axes[1, 0]
        if "n_contacts_median" in history_df.columns:
            ax3.fill_between(months,
                             history_df["n_contacts_median"], alpha=0.4,
                             color="#f39c12", label="Contacts")
            ax3.plot(months, history_df["n_contacts_median"],
                     color="#f39c12", lw=1.5)
        if "n_responders_median" in history_df.columns:
            ax3.fill_between(months,
                             history_df["n_responders_median"], alpha=0.6,
                             color="#2ecc71", label="Responders")
            ax3.plot(months, history_df["n_responders_median"],
                     color="#2ecc71", lw=1.5)
        ax3.set_xlabel("Month", fontsize=10)
        ax3.set_ylabel("# Customers", fontsize=10)
        ax3.set_title("Contacts vs Responders per Cycle\n(cool-down prevents re-contacting)",
                      fontsize=11, fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

        # ── [1,1] Customer state evolution ───────────────────────────────────
        ax4 = axes[1, 1]
        state_map = {
            "active_rate_pct_median":    ("Active",    "#3498db"),
            "retention_rate_pct_median": ("Retained",  "#2ecc71"),
        }
        for col, (label, color) in state_map.items():
            if col in history_df.columns:
                ax4.plot(months, history_df[col], lw=2.0, color=color, label=label)

        # Churned rate = 100 - active - retained - contacted
        if ("active_rate_pct_median" in history_df.columns and
                "retention_rate_pct_median" in history_df.columns):
            churned_rate = (
                100
                - history_df["active_rate_pct_median"]
                - history_df["retention_rate_pct_median"]
            ).clip(lower=0)
            ax4.fill_between(months, churned_rate, alpha=0.25, color="#e74c3c",
                             label="Churned %")

        ax4.set_xlabel("Month", fontsize=10)
        ax4.set_ylabel("% of Customer Base", fontsize=10)
        ax4.set_title("Customer State Evolution Over Time",
                      fontsize=11, fontweight="bold")
        ax4.legend(fontsize=9)
        ax4.set_ylim(0, 110)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[ProductionSim] Rolling metrics plot saved -> %s", save_path)
        return fig

    def get_summary(self, history_df: pd.DataFrame) -> dict:
        """Return key aggregate statistics from the simulation."""
        last = history_df.iloc[-1]
        return {
            "n_months":                self.n_months,
            "n_cycles":                self.n_cycles,
            "final_active_rate_pct":   round(float(last.get("active_rate_pct_median", 0)), 2),
            "final_retained_rate_pct": round(float(last.get("retention_rate_pct_median", 0)), 2),
            "total_cumulative_profit": round(float(last.get("cumulative_profit_median", 0)), 2),
            "avg_contacts_per_cycle":  round(float(history_df["n_contacts_median"].mean()), 1),
            "avg_responders_per_cycle":round(float(history_df["n_responders_median"].mean()), 1),
            "avg_churn_rate_pct":      round(float(history_df["churn_rate_pct_median"].mean()), 3),
            "total_budget_spent":      round(float(history_df["budget_used_median"].sum()), 2),
        }
