"""
src/simulation/advanced_policy_simulator.py
============================================
Advanced Multi-Period Business Policy Simulator.

Extends the baseline single-snapshot Monte Carlo (src/simulator.py) with
a full temporal campaign simulation spanning 6–12 monthly periods.

Key enhancements over the base simulator
-----------------------------------------
1. **Multi-period dynamics** — Customers churn stochastically each period
   according to their Weibull survival curve.  The active pool shrinks over
   time; retained customers contribute ongoing revenue.

2. **Scenario analysis** — Three response-rate / budget scenarios:
     • Optimistic   : p_response=0.22, budget × 1.5
     • Realistic    : p_response=0.15, budget × 1.0  (baseline)
     • Pessimistic  : p_response=0.08, budget × 0.7

3. **Dynamic budget allocation** — Each period's intervention budget is
   computed from the scenario multiplier.  Unspent budget does NOT roll
   over (conservative — avoids under-spending distortion).

4. **Cumulative CLV tracking** — Revenue contribution of active customers
   accumulated period-by-period, distinguishing Weibull-retained vs.
   naturally-surviving vs. churned cohorts.

5. **Churn reduction curves** — Compares three policies:
     • No Intervention  (baseline churn)
     • RFM Heuristic    (top-40% Monetary re-targeted every period)
     • Weibull Policy   (precision h(t) + EVI targeting)

Usage
-----
    from src.simulation import MultiPeriodSimulator, run_advanced_simulation

    sim = MultiPeriodSimulator(
        waf=waf,
        df_scaled=df_scaled_waf,
        customer_df=customer_df,
        predicted_clv=predicted_clv_all,
    )
    results = sim.run()
    fig_profit = sim.plot_profit_trajectory(results, save_path="profit.png")
    fig_churn  = sim.plot_churn_reduction(results, save_path="churn.png")
    fig_dash   = sim.plot_combined_dashboard(results, save_path="dashboard.png")
    summary_df = sim.get_summary_table(results)
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
# Default Parameters
# =============================================================================

_SCENARIOS: Dict[str, dict] = {
    "optimistic": {
        "label":         "Optimistic",
        "color":         "#2ecc71",
        "response_rate": 0.22,
        "budget_mult":   1.5,
        "linestyle":     "--",
    },
    "realistic": {
        "label":         "Realistic",
        "color":         "#3498db",
        "response_rate": 0.15,
        "budget_mult":   1.0,
        "linestyle":     "-",
    },
    "pessimistic": {
        "label":         "Pessimistic",
        "color":         "#e74c3c",
        "response_rate": 0.08,
        "budget_mult":   0.7,
        "linestyle":     ":",
    },
}

_POLICY_COLORS = {
    "Weibull": "#3498db",
    "RFM":     "#e74c3c",
    "None":    "#95a5a6",
}

_DEFAULT_PERIOD_DAYS    = 30      # one month per period
_DEFAULT_N_PERIODS      = 6       # 6 months
_DEFAULT_BASE_BUDGET    = 500.0   # MU per period
_DEFAULT_COST_PER_CONTACT = 1.0
_DEFAULT_HAZARD_THRESHOLD = 0.01
_DEFAULT_SURVIVAL_FLOOR   = 0.05
_DEFAULT_RFM_TOP_PCT      = 0.40
_DEFAULT_SLEEPING_DOG_PENALTY = 0.20
_DEFAULT_CHURN_REDUCTION  = 0.65  # fraction by which intervention reduces churn prob
_DEFAULT_N_MC             = 300   # MC iterations per scenario


# =============================================================================
# SimulationResult (lightweight named container)
# =============================================================================

class SimulationResult:
    """Holds per-period statistics for one scenario × one policy."""

    __slots__ = [
        "scenario", "policy", "periods",
        "profit_median", "profit_lo", "profit_hi",
        "cumulative_profit_median", "cumulative_profit_lo", "cumulative_profit_hi",
        "churn_rate_median", "churn_rate_lo", "churn_rate_hi",
        "cumulative_churn_median",
        "active_pct_median",
        "contacts_median", "budget_used_median",
        "clv_retained_median",
        "n_mc",
    ]

    def __init__(self, scenario: str, policy: str, n_periods: int, n_mc: int):
        self.scenario = scenario
        self.policy   = policy
        self.periods  = list(range(1, n_periods + 1))
        self.n_mc     = n_mc
        # Per-period arrays (n_periods,)
        z = np.zeros(n_periods)
        self.profit_median              = z.copy()
        self.profit_lo                  = z.copy()
        self.profit_hi                  = z.copy()
        self.cumulative_profit_median   = z.copy()
        self.cumulative_profit_lo       = z.copy()
        self.cumulative_profit_hi       = z.copy()
        self.churn_rate_median          = z.copy()
        self.churn_rate_lo              = z.copy()
        self.churn_rate_hi              = z.copy()
        self.cumulative_churn_median    = z.copy()
        self.active_pct_median          = z.copy()
        self.contacts_median            = z.copy()
        self.budget_used_median         = z.copy()
        self.clv_retained_median        = z.copy()


# =============================================================================
# MultiPeriodSimulator
# =============================================================================

class MultiPeriodSimulator:
    """
    Multi-period retention campaign simulator using the Weibull AFT model.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model (from src.models.train_weibull_aft).
    df_scaled : pd.DataFrame
        Preprocessed customer feature DataFrame (same as used for policy engine).
        Must contain columns: T, E, and active feature columns.
    customer_df : pd.DataFrame
        Original (unscaled) customer DataFrame with Monetary column.
    predicted_clv : pd.Series, optional
        Forward-looking CLV per customer indexed by CustomerID.
        Falls back to Monetary when None.
    period_days : int
        Length of each simulation period in days (default: 30).
    n_periods : int
        Number of periods to simulate (default: 6).
    base_marketing_budget : float
        Budget per period at realistic scenario (default: 500 MU).
    cost_per_contact : float
        Cost per marketing contact in MU (default: 1.0).
    theta_h : float
        Hazard threshold for INTERVENE decision (default: 0.01).
    theta_s : float
        Survival floor; customers below this are LOST (default: 0.05).
    rfm_top_pct : float
        Fraction of customers targeted by RFM baseline (default: 0.40).
    sleeping_dog_penalty : float
        Revenue fraction lost when RFM contacts low-hazard customers (default: 0.20).
    churn_reduction_effect : float
        Fractional reduction in churn probability for intervention responders
        (default: 0.65, i.e., 65% reduction).
    n_mc : int
        Monte Carlo iterations per scenario (default: 300).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        waf,
        df_scaled: pd.DataFrame,
        customer_df: pd.DataFrame,
        predicted_clv: Optional[pd.Series] = None,
        period_days: int = _DEFAULT_PERIOD_DAYS,
        n_periods: int = _DEFAULT_N_PERIODS,
        base_marketing_budget: float = _DEFAULT_BASE_BUDGET,
        cost_per_contact: float = _DEFAULT_COST_PER_CONTACT,
        theta_h: float = _DEFAULT_HAZARD_THRESHOLD,
        theta_s: float = _DEFAULT_SURVIVAL_FLOOR,
        rfm_top_pct: float = _DEFAULT_RFM_TOP_PCT,
        sleeping_dog_penalty: float = _DEFAULT_SLEEPING_DOG_PENALTY,
        churn_reduction_effect: float = _DEFAULT_CHURN_REDUCTION,
        n_mc: int = _DEFAULT_N_MC,
        seed: int = 42,
    ):
        self.waf                   = waf
        self.df_scaled             = df_scaled.copy()
        self.customer_df           = customer_df.copy()
        self.period_days           = period_days
        self.n_periods             = n_periods
        self.base_budget           = base_marketing_budget
        self.cost_per_contact      = cost_per_contact
        self.theta_h               = theta_h
        self.theta_s               = theta_s
        self.rfm_top_pct           = rfm_top_pct
        self.sleeping_dog_penalty  = sleeping_dog_penalty
        self.churn_reduction       = churn_reduction_effect
        self.n_mc                  = n_mc
        self.rng                   = np.random.default_rng(seed)

        self.n_customers = len(df_scaled)

        # CLV values (aligned to df_scaled index)
        if predicted_clv is not None:
            self.clv = predicted_clv.reindex(df_scaled.index).fillna(0.0).values.astype(float)
        else:
            self.clv = customer_df["Monetary"].values.astype(float)

        self.monetary = customer_df["Monetary"].values.astype(float)

        # Base time offset: median observation window
        self.base_T = float(df_scaled["T"].median())

        # Pre-compute survival at all period boundaries (t_0, t_1, ..., t_K)
        logger.info(
            "[AdvancedSim] Pre-computing Weibull survival at %d time points "
            "(periods 0..%d, period_days=%d)...",
            n_periods + 1, n_periods, period_days,
        )
        self._survival_matrix = self._precompute_survival()
        # shape: (n_periods+1, n_customers)
        # survival_matrix[k, i] = S(base_T + k * period_days | x_i)

        # Pre-compute period-k hazard and churn probabilities
        self._hazard_matrix = self._compute_hazard_matrix()
        # shape: (n_periods, n_customers)  — hazard at midpoint of each period

        self._churn_prob_matrix = self._compute_churn_prob_matrix()
        # shape: (n_periods, n_customers)  — P(churn in period k | active at k-1)

        # RFM pool: fixed top-rfm_top_pct by Monetary
        rfm_thresh = np.quantile(self.monetary, 1.0 - rfm_top_pct)
        self._rfm_mask = self.monetary >= rfm_thresh
        self._rfm_sorted_idx = np.argsort(self.monetary)[::-1]
        self._rfm_pool_idx = self._rfm_sorted_idx[self._rfm_mask[self._rfm_sorted_idx]]

        logger.info(
            "[AdvancedSim] Ready | %d customers | %d periods × %d days | "
            "budget=%.0f | RFM pool=%d",
            self.n_customers, n_periods, period_days,
            base_marketing_budget, len(self._rfm_pool_idx),
        )

    # =========================================================================
    # Pre-computation helpers
    # =========================================================================

    def _precompute_survival(self) -> np.ndarray:
        """
        Compute S(t_k | x_i) for k=0..K and all customers.

        Returns ndarray of shape (n_periods+1, n_customers).
        """
        times = [
            self.base_T + k * self.period_days
            for k in range(self.n_periods + 1)
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            surv_df = self.waf.predict_survival_function(self.df_scaled, times=times)
        # surv_df: index=times, columns=customer indices → shape (K+1, N)
        return np.clip(surv_df.values, 0.0, 1.0)  # (K+1, N)

    def _compute_hazard_matrix(self) -> np.ndarray:
        """
        Numerical hazard at midpoint of each period.

        h_k ≈ (S_{k-1} - S_k) / (S_{k-1} * period_days)

        Returns ndarray of shape (n_periods, n_customers).
        """
        S = self._survival_matrix  # (K+1, N)
        dS = S[:-1] - S[1:]        # (K, N)  positive because S is decreasing
        h = dS / (S[:-1] + 1e-9) / self.period_days
        return np.clip(h, 0.0, None)

    def _compute_churn_prob_matrix(self) -> np.ndarray:
        """
        P(churn in period k | active at period k-1).

        p_k = (S_{k-1} - S_k) / S_{k-1}

        Returns ndarray of shape (n_periods, n_customers), values in [0, 1].
        """
        S = self._survival_matrix  # (K+1, N)
        dS = S[:-1] - S[1:]
        p_churn = dS / (S[:-1] + 1e-9)
        return np.clip(p_churn, 0.0, 1.0)

    # =========================================================================
    # Main simulation entry point
    # =========================================================================

    def run(self, scenarios: Optional[Dict[str, dict]] = None) -> Dict[str, List[SimulationResult]]:
        """
        Run the multi-period simulation across scenarios.

        Parameters
        ----------
        scenarios : dict, optional
            Override scenario definitions. Default = _SCENARIOS (3 scenarios).

        Returns
        -------
        dict
            Keys: scenario name → list of SimulationResult objects,
            one per policy ("Weibull", "RFM", "None").
        """
        if scenarios is None:
            scenarios = _SCENARIOS

        all_results: Dict[str, List[SimulationResult]] = {}

        for sc_name, sc_cfg in scenarios.items():
            response_rate  = sc_cfg["response_rate"]
            budget_period  = self.base_budget * sc_cfg["budget_mult"]

            logger.info(
                "[AdvancedSim] Scenario '%s' | response_rate=%.2f | budget/period=%.0f | n_mc=%d",
                sc_name, response_rate, budget_period, self.n_mc,
            )

            results_sc = self._run_scenario(sc_name, response_rate, budget_period)
            all_results[sc_name] = results_sc

        return all_results

    # =========================================================================
    # Per-scenario simulation
    # =========================================================================

    def _run_scenario(
        self,
        scenario_name: str,
        response_rate: float,
        budget_per_period: float,
    ) -> List[SimulationResult]:
        """
        Run n_mc Monte Carlo paths for one scenario; return SimulationResult
        objects for Weibull policy, RFM policy, and No Intervention.
        """
        n_mc = self.n_mc
        K    = self.n_periods
        N    = self.n_customers

        # Per-iteration, per-period accumulators: shape (n_mc, K)
        w_profits  = np.zeros((n_mc, K))
        r_profits  = np.zeros((n_mc, K))
        n_profits  = np.zeros((n_mc, K))  # no-intervention baseline

        w_churns   = np.zeros((n_mc, K))
        r_churns   = np.zeros((n_mc, K))
        n_churns   = np.zeros((n_mc, K))

        w_active   = np.zeros((n_mc, K))
        r_active   = np.zeros((n_mc, K))
        n_active   = np.zeros((n_mc, K))

        w_contacts = np.zeros((n_mc, K))
        r_contacts = np.zeros((n_mc, K))
        w_clv_ret  = np.zeros((n_mc, K))

        rng = self.rng  # seeded in __init__

        for mc_i in range(n_mc):
            # Stochastic parameters for this iteration
            p_resp  = float(np.clip(rng.normal(response_rate, 0.03), 0.0, 1.0))
            cost_i  = float(max(rng.normal(self.cost_per_contact, 0.10), 0.1))
            max_contacts_per_period = max(int(np.floor(budget_per_period / cost_i)), 1)

            # Independent active states for each policy arm
            w_alive = np.ones(N, dtype=bool)   # Weibull policy
            r_alive = np.ones(N, dtype=bool)   # RFM policy
            n_alive = np.ones(N, dtype=bool)   # No intervention

            # Stochastic churn matrix for this MC path: shape (K, N)
            # Churn in period k iff U_ki < p_churn_k (and was active)
            # Use a fixed random draw so all three policies see same base churn
            U = rng.random((K, N))  # uniform random, same for all policies

            for k in range(K):
                p_churn_k = self._churn_prob_matrix[k]   # (N,)
                h_k       = self._hazard_matrix[k]        # (N,)
                S_k       = self._survival_matrix[k + 1]  # S at end of period k

                # ── WEIBULL POLICY ─────────────────────────────────────────────
                # Step 1: make intervention decisions for active customers
                evi_k = p_resp * self.clv * (1.0 - S_k) - cost_i
                is_intervene = (
                    w_alive
                    & (S_k >= self.theta_s)     # not LOST
                    & (h_k > self.theta_h)       # hazard above threshold
                    & (evi_k > 0.0)             # positive expected value
                )
                # Sort intervene candidates by EVI descending, fund top-k within budget
                w_intervene_idx = np.where(is_intervene)[0]
                if len(w_intervene_idx) > 0:
                    best_order = w_intervene_idx[np.argsort(evi_k[w_intervene_idx])[::-1]]
                    funded_idx = best_order[:max_contacts_per_period]
                else:
                    funded_idx = np.array([], dtype=int)

                n_w_contacts = len(funded_idx)

                # Step 2: simulate responses (stochastic)
                resp_mask = np.zeros(N, dtype=bool)
                if n_w_contacts > 0:
                    responded = rng.random(n_w_contacts) < p_resp
                    resp_mask[funded_idx[responded]] = True

                # Step 3: adjusted churn probability (responders get reduction)
                p_churn_adj = p_churn_k.copy()
                p_churn_adj[resp_mask] *= (1.0 - self.churn_reduction)
                p_churn_adj = np.clip(p_churn_adj, 0.0, 1.0)

                # Step 4: simulate churns (using pre-drawn U)
                w_churn_event = w_alive & (U[k] < p_churn_adj)
                w_alive[w_churn_event] = False

                # Step 5: period revenue = active CLV / n_periods + intervention benefit
                period_revenue = self.clv[w_alive].sum() / self.n_periods
                campaign_cost  = n_w_contacts * cost_i
                clv_retained_this_period = self.clv[resp_mask & (~w_churn_event)].sum()

                w_profits[mc_i, k]  = period_revenue - campaign_cost
                w_churns[mc_i, k]   = w_churn_event.sum()
                w_active[mc_i, k]   = w_alive.sum()
                w_contacts[mc_i, k] = n_w_contacts
                w_clv_ret[mc_i, k]  = clv_retained_this_period

                # ── RFM POLICY ────────────────────────────────────────────────
                # RFM contacts top-rfm_top_pct% by Monetary every period
                rfm_funded_alive = self._rfm_pool_idx[r_alive[self._rfm_pool_idx]]
                rfm_funded = rfm_funded_alive[:max_contacts_per_period]
                n_r_contacts = len(rfm_funded)

                # Sleeping dog: RFM contacts customers with low hazard → penalty
                if n_r_contacts > 0:
                    rfm_h = h_k[rfm_funded]
                    rfm_persuadable = rfm_h > self.theta_h
                    rfm_sleeping    = ~rfm_persuadable

                    # Revenue from persuadables
                    r_rev_persuadable = np.sum(
                        self.monetary[rfm_funded[rfm_persuadable]] * p_resp
                    )
                    # Brand damage from sleeping dogs
                    r_rev_sleeping = np.sum(
                        -self.monetary[rfm_funded[rfm_sleeping]] * self.sleeping_dog_penalty
                    )
                    rfm_campaign_profit = r_rev_persuadable + r_rev_sleeping - n_r_contacts * cost_i
                else:
                    rfm_campaign_profit = 0.0

                # Simulate RFM churns (same U draws, no intervention benefit)
                r_churn_event = r_alive & (U[k] < p_churn_k)
                r_alive[r_churn_event] = False

                r_period_revenue = self.clv[r_alive].sum() / self.n_periods
                r_profits[mc_i, k] = r_period_revenue + rfm_campaign_profit
                r_churns[mc_i, k]  = r_churn_event.sum()
                r_active[mc_i, k]  = r_alive.sum()
                r_contacts[mc_i, k] = n_r_contacts

                # ── NO INTERVENTION ───────────────────────────────────────────
                n_churn_event = n_alive & (U[k] < p_churn_k)
                n_alive[n_churn_event] = False
                n_period_revenue = self.clv[n_alive].sum() / self.n_periods
                n_profits[mc_i, k] = n_period_revenue
                n_churns[mc_i, k]  = n_churn_event.sum()
                n_active[mc_i, k]  = n_alive.sum()

        # ── Aggregate MC results into SimulationResult objects ────────────────
        sc_results = []
        for policy_name, profit_arr, churn_arr, active_arr, contacts_arr, clv_arr in [
            ("Weibull", w_profits, w_churns, w_active, w_contacts, w_clv_ret),
            ("RFM",     r_profits, r_churns, r_active, r_contacts, np.zeros((n_mc, K))),
            ("None",    n_profits, n_churns, n_active, np.zeros((n_mc, K)), np.zeros((n_mc, K))),
        ]:
            res = SimulationResult(scenario_name, policy_name, K, n_mc)

            # Per-period stats (percentiles across MC iterations)
            res.profit_median = np.percentile(profit_arr, 50, axis=0)
            res.profit_lo     = np.percentile(profit_arr, 2.5, axis=0)
            res.profit_hi     = np.percentile(profit_arr, 97.5, axis=0)

            # Cumulative profit
            cum = np.cumsum(profit_arr, axis=1)
            res.cumulative_profit_median = np.percentile(cum, 50, axis=0)
            res.cumulative_profit_lo     = np.percentile(cum, 2.5, axis=0)
            res.cumulative_profit_hi     = np.percentile(cum, 97.5, axis=0)

            # Churn rate per period
            churn_rate_arr = churn_arr / max(self.n_customers, 1)
            res.churn_rate_median = np.percentile(churn_rate_arr, 50, axis=0)
            res.churn_rate_lo     = np.percentile(churn_rate_arr, 2.5, axis=0)
            res.churn_rate_hi     = np.percentile(churn_rate_arr, 97.5, axis=0)

            # Cumulative churn %
            cum_churn = np.cumsum(churn_arr, axis=1) / max(self.n_customers, 1) * 100
            res.cumulative_churn_median = np.percentile(cum_churn, 50, axis=0)

            # Active % remaining
            res.active_pct_median = np.percentile(active_arr / max(self.n_customers, 1) * 100, 50, axis=0)

            # Contacts and CLV
            res.contacts_median    = np.percentile(contacts_arr, 50, axis=0)
            res.clv_retained_median = np.percentile(clv_arr, 50, axis=0)
            res.budget_used_median = res.contacts_median * self.cost_per_contact

            sc_results.append(res)

        return sc_results

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_profit_trajectory(
        self,
        results: Dict[str, List[SimulationResult]],
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Line chart: cumulative profit over periods for each scenario × policy.

        Parameters
        ----------
        results : dict
            Output of run().
        save_path : str, optional
            If provided, saves figure to this path.
        dataset_label : str
            Dataset name to show in title.

        Returns
        -------
        matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle(
            f"Cumulative Profit Trajectory by Scenario — {dataset_label}",
            fontsize=14, fontweight="bold", y=1.02
        )

        for ax, (sc_name, sc_results) in zip(axes, results.items()):
            sc_cfg = _SCENARIOS.get(sc_name, {})
            period_labels = [f"M{k}" for k in range(1, self.n_periods + 1)]

            for res in sc_results:
                if res.policy == "None":
                    continue
                color  = _POLICY_COLORS.get(res.policy, "#888")
                lw     = 2.5 if res.policy == "Weibull" else 1.8
                lstyle = "-" if res.policy == "Weibull" else "--"

                ax.plot(
                    period_labels,
                    res.cumulative_profit_median,
                    color=color, lw=lw, ls=lstyle,
                    label=f"{res.policy}",
                    zorder=3,
                )
                ax.fill_between(
                    period_labels,
                    res.cumulative_profit_lo,
                    res.cumulative_profit_hi,
                    alpha=0.15, color=color,
                )

            ax.axhline(0, color="#ccc", lw=0.8, zorder=1)
            ax.set_title(
                f"{sc_cfg.get('label', sc_name)}\n(p_response={sc_cfg.get('response_rate', '?'):.0%})",
                fontsize=11, fontweight="bold"
            )
            ax.set_xlabel("Period", fontsize=10)
            ax.set_ylabel("Cumulative Profit (MU)" if ax == axes[0] else "", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:,.0f}")
            )

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[AdvancedSim] Profit trajectory saved → %s", save_path)
        return fig

    def plot_churn_reduction(
        self,
        results: Dict[str, List[SimulationResult]],
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Cumulative churn reduction curves: No Intervention vs RFM vs Weibull
        using the Realistic scenario.

        Parameters
        ----------
        results : dict
            Output of run().
        save_path : str, optional
        dataset_label : str

        Returns
        -------
        matplotlib Figure
        """
        realistic = results.get("realistic", list(results.values())[0])
        by_policy = {r.policy: r for r in realistic}

        fig, ax = plt.subplots(figsize=(9, 6))

        period_labels = list(range(1, self.n_periods + 1))
        policy_order = ["None", "RFM", "Weibull"]
        policy_meta  = {
            "None":    {"label": "No Intervention (Baseline)", "color": "#95a5a6", "lw": 1.8, "ls": "-."},
            "RFM":     {"label": "RFM Policy",                 "color": "#e74c3c", "lw": 2.0, "ls": "--"},
            "Weibull": {"label": "Weibull Decision Policy",    "color": "#3498db", "lw": 2.5, "ls": "-"},
        }

        for pol in policy_order:
            if pol not in by_policy:
                continue
            res  = by_policy[pol]
            meta = policy_meta[pol]
            ax.plot(
                period_labels,
                res.cumulative_churn_median,
                color=meta["color"], lw=meta["lw"], ls=meta["ls"],
                label=meta["label"], zorder=3,
            )

        # Churn reduction annotation
        if "None" in by_policy and "Weibull" in by_policy:
            none_final   = by_policy["None"].cumulative_churn_median[-1]
            weibull_final = by_policy["Weibull"].cumulative_churn_median[-1]
            reduction_pct = max(none_final - weibull_final, 0)
            ax.annotate(
                f"↓ {reduction_pct:.1f}pp churn\nreduction",
                xy=(self.n_periods, weibull_final),
                xytext=(self.n_periods - 1.2, weibull_final + reduction_pct * 0.4),
                arrowprops=dict(arrowstyle="->", color="#2c3e50"),
                fontsize=10, color="#2c3e50", fontweight="bold",
            )

        ax.set_xlabel("Month", fontsize=12)
        ax.set_ylabel("Cumulative Customers Churned (%)", fontsize=12)
        ax.set_title(
            f"Churn Reduction Curve — Weibull vs RFM vs No Intervention\n"
            f"Dataset: {dataset_label} | Realistic Scenario",
            fontsize=13, fontweight="bold"
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(period_labels)
        ax.set_xticklabels([f"Month {k}" for k in period_labels], rotation=20)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[AdvancedSim] Churn reduction plot saved → %s", save_path)
        return fig

    def plot_combined_dashboard(
        self,
        results: Dict[str, List[SimulationResult]],
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        2×2 publication-quality dashboard:
          [0,0] Cumulative profit — Realistic scenario
          [0,1] Churn reduction curve — Realistic scenario
          [1,0] Per-period contacts — Weibull vs RFM
          [1,1] Scenario comparison — final cumulative profit bar chart
        """
        fig = plt.figure(figsize=(16, 11))
        gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)
        ax_profit  = fig.add_subplot(gs[0, 0])
        ax_churn   = fig.add_subplot(gs[0, 1])
        ax_contact = fig.add_subplot(gs[1, 0])
        ax_scenario= fig.add_subplot(gs[1, 1])

        period_labels = list(range(1, self.n_periods + 1))
        realistic = results.get("realistic", list(results.values())[0])
        by_policy = {r.policy: r for r in realistic}

        # ── [0,0] Profit trajectory (Realistic) ──────────────────────────────
        for pol, color, lw, ls in [
            ("Weibull", "#3498db", 2.5, "-"),
            ("RFM",     "#e74c3c", 2.0, "--"),
        ]:
            if pol not in by_policy:
                continue
            res = by_policy[pol]
            ax_profit.plot(period_labels, res.cumulative_profit_median,
                           color=color, lw=lw, ls=ls, label=pol)
            ax_profit.fill_between(period_labels,
                                   res.cumulative_profit_lo,
                                   res.cumulative_profit_hi,
                                   alpha=0.12, color=color)
        ax_profit.axhline(0, color="#ccc", lw=0.8)
        ax_profit.set_title("Cumulative Profit (Realistic)", fontsize=11, fontweight="bold")
        ax_profit.set_xlabel("Month", fontsize=10)
        ax_profit.set_ylabel("Cumulative Profit (MU)", fontsize=10)
        ax_profit.legend(fontsize=9)
        ax_profit.grid(True, alpha=0.3)
        ax_profit.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # ── [0,1] Churn reduction ──────────────────────────────────────────────
        for pol, color, lw, ls, label in [
            ("None",    "#95a5a6", 1.8, "-.", "No Intervention"),
            ("RFM",     "#e74c3c", 2.0, "--", "RFM Policy"),
            ("Weibull", "#3498db", 2.5, "-",  "Weibull Policy"),
        ]:
            if pol not in by_policy:
                continue
            res = by_policy[pol]
            ax_churn.plot(period_labels, res.cumulative_churn_median,
                          color=color, lw=lw, ls=ls, label=label)
        ax_churn.set_title("Cumulative Churn Reduction", fontsize=11, fontweight="bold")
        ax_churn.set_xlabel("Month", fontsize=10)
        ax_churn.set_ylabel("Customers Churned (%)", fontsize=10)
        ax_churn.legend(fontsize=9)
        ax_churn.grid(True, alpha=0.3)

        # ── [1,0] Contacts per period ─────────────────────────────────────────
        x = np.arange(self.n_periods)
        bw = 0.35
        for pol, color, offset in [("Weibull", "#3498db", -bw/2), ("RFM", "#e74c3c", bw/2)]:
            if pol not in by_policy:
                continue
            res = by_policy[pol]
            ax_contact.bar(x + offset, res.contacts_median, bw,
                           color=color, alpha=0.8, label=pol)
        ax_contact.set_title("Contacts per Period (Realistic)", fontsize=11, fontweight="bold")
        ax_contact.set_xlabel("Month", fontsize=10)
        ax_contact.set_ylabel("# Customers Contacted", fontsize=10)
        ax_contact.set_xticks(x)
        ax_contact.set_xticklabels([f"M{k}" for k in period_labels])
        ax_contact.legend(fontsize=9)
        ax_contact.grid(True, alpha=0.3, axis="y")

        # ── [1,1] Scenario comparison — final period cumulative profit ─────────
        sc_names  = list(results.keys())
        w_finals  = []
        r_finals  = []
        for sc in sc_names:
            sc_res = {r.policy: r for r in results[sc]}
            w_finals.append(sc_res.get("Weibull", SimulationResult(sc, "Weibull", self.n_periods, 1)).cumulative_profit_median[-1])
            r_finals.append(sc_res.get("RFM",     SimulationResult(sc, "RFM",     self.n_periods, 1)).cumulative_profit_median[-1])

        x2 = np.arange(len(sc_names))
        bw2 = 0.35
        sc_labels = [_SCENARIOS.get(s, {}).get("label", s) for s in sc_names]
        ax_scenario.bar(x2 - bw2/2, w_finals, bw2, color="#3498db", alpha=0.85, label="Weibull")
        ax_scenario.bar(x2 + bw2/2, r_finals, bw2, color="#e74c3c", alpha=0.85, label="RFM")
        ax_scenario.axhline(0, color="#ccc", lw=0.8)
        ax_scenario.set_title("Final Cumulative Profit by Scenario", fontsize=11, fontweight="bold")
        ax_scenario.set_xlabel("Scenario", fontsize=10)
        ax_scenario.set_ylabel("Cumulative Profit (MU)", fontsize=10)
        ax_scenario.set_xticks(x2)
        ax_scenario.set_xticklabels(sc_labels, fontsize=9)
        ax_scenario.legend(fontsize=9)
        ax_scenario.grid(True, alpha=0.3, axis="y")
        ax_scenario.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.suptitle(
            f"Advanced Policy Simulation Dashboard — {dataset_label}",
            fontsize=14, fontweight="bold"
        )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[AdvancedSim] Dashboard saved → %s", save_path)
        return fig

    # =========================================================================
    # Summary table
    # =========================================================================

    def get_summary_table(
        self, results: Dict[str, List[SimulationResult]]
    ) -> pd.DataFrame:
        """
        Returns a DataFrame summarising final-period results across
        scenarios and policies.

        Columns: Scenario, Policy, FinalCumulativeProfit, TotalChurn%,
                 AvgContactsPerPeriod, CLVRetained, EfficiencyVsRFM%
        """
        rows = []
        for sc_name, sc_results in results.items():
            sc_label = _SCENARIOS.get(sc_name, {}).get("label", sc_name)
            by_pol   = {r.policy: r for r in sc_results}

            rfm_final = by_pol.get("RFM", None)
            rfm_cum   = rfm_final.cumulative_profit_median[-1] if rfm_final else 0

            for res in sc_results:
                weibull_cum = by_pol.get("Weibull", None)
                if weibull_cum:
                    weibull_cum_val = weibull_cum.cumulative_profit_median[-1]
                    eff_vs_rfm = (
                        (weibull_cum_val - rfm_cum) / max(abs(rfm_cum), 1) * 100
                        if res.policy == "Weibull" else np.nan
                    )
                else:
                    eff_vs_rfm = np.nan

                rows.append({
                    "Scenario":               sc_label,
                    "Policy":                 res.policy,
                    "FinalCumulativeProfit":  round(res.cumulative_profit_median[-1], 2),
                    "TotalChurnPct":          round(res.cumulative_churn_median[-1], 2),
                    "AvgContactsPerPeriod":   round(res.contacts_median.mean(), 1),
                    "CLVRetained":            round(res.clv_retained_median.sum(), 2),
                    "EfficiencyVsRFM_pct":    round(eff_vs_rfm, 1) if not np.isnan(eff_vs_rfm) else None,
                })

        return pd.DataFrame(rows)


# =============================================================================
# Convenience wrapper
# =============================================================================

def run_advanced_simulation(
    waf,
    df_scaled: pd.DataFrame,
    customer_df: pd.DataFrame,
    predicted_clv: Optional[pd.Series] = None,
    period_days: int = _DEFAULT_PERIOD_DAYS,
    n_periods: int = _DEFAULT_N_PERIODS,
    base_budget: float = _DEFAULT_BASE_BUDGET,
    n_mc: int = _DEFAULT_N_MC,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    seed: int = 42,
) -> Dict:
    """
    One-call wrapper: initialises MultiPeriodSimulator, runs all scenarios,
    saves figures and summary CSV, returns results dict.

    Parameters
    ----------
    waf, df_scaled, customer_df, predicted_clv
        Same as MultiPeriodSimulator constructor.
    save_dir : str, optional
        Directory to save figures and CSV.  Created if needed.
    dataset_label : str
        Dataset name shown in plot titles.

    Returns
    -------
    dict
        {
          'results':   raw simulation results dict (scenario → SimulationResult list)
          'summary_df': pd.DataFrame summary table
          'figs': dict of matplotlib Figures
        }
    """
    sim = MultiPeriodSimulator(
        waf=waf,
        df_scaled=df_scaled,
        customer_df=customer_df,
        predicted_clv=predicted_clv,
        period_days=period_days,
        n_periods=n_periods,
        base_marketing_budget=base_budget,
        n_mc=n_mc,
        seed=seed,
    )

    results = sim.run()
    summary_df = sim.get_summary_table(results)

    figs = {}
    profit_path  = os.path.join(save_dir, "advsim_profit_trajectory.png")  if save_dir else None
    churn_path   = os.path.join(save_dir, "advsim_churn_reduction.png")    if save_dir else None
    dash_path    = os.path.join(save_dir, "advsim_dashboard.png")          if save_dir else None
    csv_path     = os.path.join(save_dir, "advsim_summary.csv")            if save_dir else None

    figs["profit"]    = sim.plot_profit_trajectory(results, save_path=profit_path, dataset_label=dataset_label)
    figs["churn"]     = sim.plot_churn_reduction(results,   save_path=churn_path,  dataset_label=dataset_label)
    figs["dashboard"] = sim.plot_combined_dashboard(results, save_path=dash_path,  dataset_label=dataset_label)

    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        summary_df.to_csv(csv_path, index=False)
        logger.info("[AdvancedSim] Summary table saved → %s", csv_path)

    logger.info(
        "[AdvancedSim] Complete | %d scenarios | %d periods | %d MC iterations",
        len(results), n_periods, n_mc,
    )

    return {
        "results":    results,
        "summary_df": summary_df,
        "figs":       figs,
        "simulator":  sim,
    }
