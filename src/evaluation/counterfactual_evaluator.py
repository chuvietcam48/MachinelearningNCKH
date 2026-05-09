"""
src/evaluation/counterfactual_evaluator.py
============================================
Offline Counterfactual / Observational Policy Evaluation.

Replaces the need for a live A/B test by estimating the causal value of
different intervention policies from observational data.

Three estimators are implemented, ordered by assumption strength:

  1. **Direct Method (DM)**
     Policy value ≈ average predicted outcome under the policy.
     Relies entirely on the outcome model (mu_1 / mu_0 from T-Learner).
     Biased if the outcome model is misspecified.

  2. **Inverse Propensity Scoring (IPS)**
     Re-weights observed outcomes by the inverse probability of treatment.
     Unbiased when propensity model is correct; high variance with extreme weights.
     V̂_IPS(π) = (1/n) Σ_i  [1(aᵢ = π(xᵢ)) / e_i] * rᵢ

  3. **Doubly Robust (DR)**
     Combines DM and IPS: consistent if EITHER the outcome model OR the
     propensity model is correctly specified.
     V̂_DR(π) = (1/n) Σ_i  [μ̂(xᵢ, π(xᵢ)) + 1(aᵢ=π(xᵢ))/e_i * (rᵢ - μ̂(xᵢ,aᵢ))]

Policies compared
------------------
  • Weibull    — precision policy (INTERVENE where h(t) > θ_h AND EVI > 0)
  • RFM        — heuristic (INTERVENE if RFM_Segment == "At Risk")
  • Random     — random treatment with P(T=1) = observed treatment rate
  • AlwaysTreat — treat everyone (100% intervention)
  • NeverTreat  — never intervene (control baseline)

Usage
-----
    from src.evaluation.counterfactual_evaluator import PolicyEvaluator

    evaluator = PolicyEvaluator(
        weibull_decisions=weibull_decisions,
        customer_df=customer_df,
        uplift_results=uplift_results,     # from run_uplift_analysis()
        rfm_decisions=rfm_decisions,
    )
    comparison_df = evaluator.compare_all_policies()
    fig = evaluator.plot_policy_comparison(comparison_df)
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Utility functions
# =============================================================================

def _policy_vector(decisions_df: pd.DataFrame, decision_col: str = "decision") -> np.ndarray:
    """Convert decision column ('INTERVENE'/'WAIT'/'LOST') to binary 0/1."""
    return (decisions_df[decision_col] == "INTERVENE").astype(int).values


def _align_series(
    arr1: np.ndarray,
    arr2: np.ndarray,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure two arrays have the same length, truncating the longer one."""
    n = min(len(arr1), len(arr2))
    if len(arr1) != len(arr2):
        logger.warning(
            "[CounterfactualEval] %s length mismatch: %d vs %d — truncating to %d",
            label, len(arr1), len(arr2), n,
        )
    return arr1[:n], arr2[:n]


# =============================================================================
# PolicyEvaluator
# =============================================================================

class PolicyEvaluator:
    """
    Offline counterfactual policy evaluator.

    Computes DM, IPS, and DR estimates of policy value for multiple policies,
    with bootstrap confidence intervals.

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions().
        Required columns: CustomerID, decision, hazard_now, survival, evi, Monetary.
    customer_df : pd.DataFrame
        Original customer-level DataFrame (indexed by CustomerID).
    uplift_results : dict
        Output of run_uplift_analysis(). Required keys:
          'uplift_df'  — contains treatment, Monetary (outcome), tau_hat, mu_1, mu_0,
                         propensity, iptw, uplift_segment.
    rfm_decisions : pd.DataFrame, optional
        Output of policy.rfm_intervention_decisions().
        Required columns: CustomerID, decision.
    outcome_col : str
        Column in uplift_df to use as observed outcome (default: 'Monetary').
    n_bootstrap : int
        Bootstrap iterations for CI estimation (default: 500).
    seed : int
    """

    def __init__(
        self,
        weibull_decisions: pd.DataFrame,
        customer_df: pd.DataFrame,
        uplift_results: dict,
        rfm_decisions: Optional[pd.DataFrame] = None,
        outcome_col: str = "Monetary",
        n_bootstrap: int = 500,
        seed: int = 42,
    ):
        self.weibull_decisions = weibull_decisions.copy()
        self.customer_df       = customer_df.copy()
        self.rfm_decisions     = rfm_decisions
        self.outcome_col       = outcome_col
        self.n_bootstrap       = n_bootstrap
        self.rng               = np.random.default_rng(seed)

        # ── Extract uplift components ─────────────────────────────────────────
        uplift_df = uplift_results.get("uplift_df", pd.DataFrame())
        if uplift_df.empty:
            raise ValueError(
                "[PolicyEvaluator] uplift_results must contain non-empty 'uplift_df'."
            )

        # Merge weibull_decisions with uplift_df on CustomerID
        merged = weibull_decisions.merge(
            uplift_df.reset_index(drop=True)
            if "CustomerID" not in uplift_df.columns
            else uplift_df,
            on="CustomerID" if "CustomerID" in uplift_df.columns else None,
            how="inner",
            suffixes=("_waf", "_uplift"),
        )

        # Prefer 'Monetary_waf' or 'Monetary' as outcome
        if f"{outcome_col}_waf" in merged.columns:
            merged[outcome_col] = merged[f"{outcome_col}_waf"]

        if merged.empty:
            logger.warning(
                "[PolicyEvaluator] Merge between weibull_decisions and uplift_df "
                "produced 0 rows.  Falling back to index-based alignment."
            )
            # Fallback: align by position
            n_align = min(len(weibull_decisions), len(uplift_df))
            merged  = weibull_decisions.iloc[:n_align].copy()
            for col in ["treatment", "mu_1", "mu_0", "propensity", "iptw", outcome_col]:
                if col in uplift_df.columns:
                    merged[col] = uplift_df[col].values[:n_align]

        self.merged = merged

        # Core arrays
        self.treatment   = merged["treatment"].fillna(0).astype(int).values
        self.outcome     = merged[outcome_col].fillna(0).astype(float).values
        self.propensity  = merged["propensity"].clip(0.01, 0.99).values \
                           if "propensity" in merged.columns \
                           else np.full(len(merged), 0.5)
        self.mu_1        = merged["mu_1"].fillna(merged[outcome_col].mean()).values \
                           if "mu_1" in merged.columns else np.zeros(len(merged))
        self.mu_0        = merged["mu_0"].fillna(0).values \
                           if "mu_0" in merged.columns else np.zeros(len(merged))
        self.n           = len(merged)

        logger.info(
            "[PolicyEvaluator] Ready | n=%d | treated=%d (%.1f%%) | "
            "outcome mean=%.2f",
            self.n,
            int(self.treatment.sum()),
            self.treatment.mean() * 100,
            self.outcome.mean(),
        )

    # =========================================================================
    # Core Estimators
    # =========================================================================

    def dm_policy_value(self, pi: np.ndarray) -> float:
        """
        Direct Method (DM) policy value estimate.

        V̂_DM(π) = (1/n) Σ_i  π(xᵢ)·μ̂₁(xᵢ) + (1−π(xᵢ))·μ̂₀(xᵢ)

        Parameters
        ----------
        pi : np.ndarray, shape (n,)
            Binary policy vector (1=INTERVENE, 0=WAIT).

        Returns
        -------
        float : DM policy value estimate.
        """
        pi = pi[:self.n].astype(float)
        return float(np.mean(pi * self.mu_1 + (1 - pi) * self.mu_0))

    def ips_policy_value(self, pi: np.ndarray, clip_weights: float = 20.0) -> float:
        """
        Inverse Propensity Scoring (IPS) policy value estimate.

        V̂_IPS(π) = (1/n) Σ_i  [1(aᵢ=π(xᵢ)) / ê(xᵢ)] · rᵢ

        where ê(xᵢ) = P(aᵢ=1|xᵢ) when aᵢ=1, else 1−P(aᵢ=1|xᵢ).

        Parameters
        ----------
        pi : np.ndarray, shape (n,)
            Binary policy vector.
        clip_weights : float
            Maximum importance weight (clipping reduces variance; default: 20).

        Returns
        -------
        float : IPS estimate.
        """
        pi = pi[:self.n].astype(float)
        e  = np.where(self.treatment == 1, self.propensity, 1.0 - self.propensity)
        e  = np.clip(e, 0.01, 1.0)

        # Indicator: did the observed action match the policy?
        match = (self.treatment == pi.astype(int)).astype(float)

        # Importance weights
        w = match / e
        w = np.clip(w, 0.0, clip_weights)

        return float(np.mean(w * self.outcome))

    def dr_policy_value(self, pi: np.ndarray, clip_weights: float = 20.0) -> float:
        """
        Doubly Robust (DR) policy value estimate.

        V̂_DR(π) = (1/n) Σ_i [μ̂(xᵢ, π(xᵢ)) + 1(aᵢ=π(xᵢ))/ê(xᵢ) · (rᵢ−μ̂(xᵢ,aᵢ))]

        Parameters
        ----------
        pi : np.ndarray, shape (n,)
            Binary policy vector.
        clip_weights : float
            IPS weight clipping threshold.

        Returns
        -------
        float : DR estimate.
        """
        pi = pi[:self.n].astype(float)
        e  = np.where(self.treatment == 1, self.propensity, 1.0 - self.propensity)
        e  = np.clip(e, 0.01, 1.0)

        # Direct term: model prediction under the policy
        dm_term = pi * self.mu_1 + (1 - pi) * self.mu_0

        # Residual correction: observed - predicted for the actual action taken
        mu_actual = np.where(self.treatment == 1, self.mu_1, self.mu_0)
        residual  = self.outcome - mu_actual

        # IPS correction
        match = (self.treatment == pi.astype(int)).astype(float)
        w     = np.clip(match / e, 0.0, clip_weights)

        dr = dm_term + w * residual
        return float(np.mean(dr))

    # =========================================================================
    # Bootstrap CI
    # =========================================================================

    def bootstrap_policy_ci(
        self,
        pi: np.ndarray,
        estimator: str = "dr",
        n_boot: Optional[int] = None,
        alpha: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for a policy value estimator.

        Parameters
        ----------
        pi : np.ndarray, shape (n,)
            Binary policy vector.
        estimator : str
            'dm', 'ips', or 'dr' (default: 'dr').
        n_boot : int, optional
            Bootstrap iterations (default: self.n_bootstrap).
        alpha : float
            Significance level (default: 0.05 → 95% CI).

        Returns
        -------
        tuple (lower, median, upper)
        """
        n_boot = n_boot or self.n_bootstrap
        n      = self.n

        est_fn = {
            "dm":  self.dm_policy_value,
            "ips": self.ips_policy_value,
            "dr":  self.dr_policy_value,
        }.get(estimator, self.dr_policy_value)

        boot_vals = np.empty(n_boot)
        for b in range(n_boot):
            idx = self.rng.integers(0, n, size=n)
            # Resample all arrays
            _orig_treatment  = self.treatment
            _orig_outcome    = self.outcome
            _orig_propensity = self.propensity
            _orig_mu1        = self.mu_1
            _orig_mu0        = self.mu_0

            self.treatment   = _orig_treatment[idx]
            self.outcome     = _orig_outcome[idx]
            self.propensity  = _orig_propensity[idx]
            self.mu_1        = _orig_mu1[idx]
            self.mu_0        = _orig_mu0[idx]

            boot_vals[b] = est_fn(pi[idx])

            # Restore
            self.treatment   = _orig_treatment
            self.outcome     = _orig_outcome
            self.propensity  = _orig_propensity
            self.mu_1        = _orig_mu1
            self.mu_0        = _orig_mu0

        lo  = float(np.percentile(boot_vals, alpha / 2 * 100))
        med = float(np.percentile(boot_vals, 50))
        hi  = float(np.percentile(boot_vals, (1 - alpha / 2) * 100))
        return lo, med, hi

    # =========================================================================
    # Policy definitions
    # =========================================================================

    def _get_policy_vectors(self) -> Dict[str, np.ndarray]:
        """
        Build binary policy vectors for core + extended comparison policies.

        Core policies
        -------------
        Weibull      : h(t) > theta_h AND EVI > 0
        RFM          : RFM_Segment == "At Risk"
        Random       : Bernoulli(observed_rate)
        AlwaysTreat  : treat everyone
        NeverTreat   : never intervene

        Extended policies (Level 1 enhancement)
        ----------------------------------------
        TopK_50      : Top-50 customers by EVI  (budget-constrained precision)
        TopK_100     : Top-100 customers by EVI
        TopK_200     : Top-200 customers by EVI
        Threshold_03 : intervene if churn_prob > 0.30  (aggressive)
        Threshold_05 : intervene if churn_prob > 0.50  (moderate — LR baseline)
        Threshold_07 : intervene if churn_prob > 0.70  (conservative)
        CostSensitive: high-CLV customers get lower threshold; low-CLV higher
        """
        n = self.n

        # ── Core policies ──────────────────────────────────────────────────────
        w_pi = _policy_vector(self.merged.rename(columns={"decision_waf": "decision"})
                              if "decision_waf" in self.merged.columns
                              else self.merged)[:n]

        rfm_pi = np.zeros(n, dtype=int)
        if self.rfm_decisions is not None:
            rfm_map = dict(zip(
                self.rfm_decisions["CustomerID"],
                (self.rfm_decisions["decision"] == "INTERVENE").astype(int),
            ))
            rfm_pi = np.array([
                rfm_map.get(cid, 0)
                for cid in self.merged["CustomerID"].values[:n]
            ])

        obs_rate  = self.treatment.mean()
        rand_pi   = self.rng.binomial(1, obs_rate, size=n)
        always_pi = np.ones(n, dtype=int)
        never_pi  = np.zeros(n, dtype=int)

        policies = {
            "Weibull":     w_pi,
            "RFM":         rfm_pi,
            "Random":      rand_pi,
            "AlwaysTreat": always_pi,
            "NeverTreat":  never_pi,
        }

        # ── Extended: Top-K by EVI ─────────────────────────────────────────────
        # Look for EVI column (may be suffixed after merge)
        evi_col = None
        for col in ["evi", "EVI", "evi_waf", "lr_evi"]:
            if col in self.merged.columns:
                evi_col = col
                break

        if evi_col is not None:
            evi_vals  = self.merged[evi_col].fillna(-999).values[:n]
            evi_order = np.argsort(evi_vals)[::-1]     # best EVI first
            for k in [50, 100, 200, 300]:
                if k >= n:
                    continue
                pi_k = np.zeros(n, dtype=int)
                pi_k[evi_order[:k]] = 1
                policies[f"TopK_{k}"] = pi_k

        # ── Extended: Probability Thresholding ────────────────────────────────
        # Look for survival column (may be suffixed after merge)
        surv_col = None
        for col in ["survival", "survival_waf", "survival_now", "SurvivalProb"]:
            if col in self.merged.columns:
                surv_col = col
                break

        if surv_col is not None:
            churn_prob = np.clip(1.0 - self.merged[surv_col].fillna(0.5).values[:n], 0, 1)
            for thresh in [0.30, 0.50, 0.70]:
                pi_t = (churn_prob > thresh).astype(int)
                key  = f"Threshold_{int(thresh*100):02d}"
                policies[key] = pi_t

        # ── Extended: Cost-Sensitive Policy ────────────────────────────────────
        clv_col = None
        for col in ["predicted_clv", "predicted_clv_waf", "CLV", "Monetary", "Monetary_waf"]:
            if col in self.merged.columns:
                clv_col = col
                break

        if clv_col is not None and surv_col is not None:
            clv_vals   = self.merged[clv_col].fillna(0).values[:n]
            q33        = np.quantile(clv_vals, 0.33)
            q67        = np.quantile(clv_vals, 0.67)
            churn_prob = np.clip(1.0 - self.merged[surv_col].fillna(0.5).values[:n], 0, 1)

            pi_cs = np.zeros(n, dtype=int)
            pi_cs[clv_vals > q67]                        = (churn_prob[clv_vals > q67]   > 0.30).astype(int)
            pi_cs[(clv_vals > q33) & (clv_vals <= q67)] = (churn_prob[(clv_vals > q33) & (clv_vals <= q67)] > 0.50).astype(int)
            pi_cs[clv_vals <= q33]                       = (churn_prob[clv_vals <= q33]  > 0.70).astype(int)
            policies["CostSensitive"] = pi_cs

        return policies

    # =========================================================================
    # Main comparison
    # =========================================================================

    def compare_all_policies(
        self, bootstrap: bool = True, n_boot: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Compare all policies using DM, IPS, and DR estimators.
        Optionally computes 95% bootstrap CIs.

        Parameters
        ----------
        bootstrap : bool
            Whether to compute bootstrap CIs (slower). Default: True.
        n_boot : int, optional
            Override bootstrap iterations.

        Returns
        -------
        pd.DataFrame
            Columns: Policy, DM, IPS, DR,
                     [DR_lo, DR_med, DR_hi] (if bootstrap=True),
                     TreatmentRate, N.
        """
        policies = self._get_policy_vectors()
        rows     = []

        for pol_name, pi in policies.items():
            dm_val  = self.dm_policy_value(pi)
            ips_val = self.ips_policy_value(pi)
            dr_val  = self.dr_policy_value(pi)

            row = {
                "Policy":        pol_name,
                "TreatmentRate": float(pi.mean()),
                "N_treated":     int(pi.sum()),
                "DM":            round(dm_val,  4),
                "IPS":           round(ips_val, 4),
                "DR":            round(dr_val,  4),
            }

            if bootstrap:
                lo, med, hi = self.bootstrap_policy_ci(
                    pi, estimator="dr", n_boot=n_boot
                )
                row["DR_lo"]  = round(lo,  4)
                row["DR_med"] = round(med, 4)
                row["DR_hi"]  = round(hi,  4)
                row["DR_CI_width"] = round(hi - lo, 4)

            rows.append(row)
            logger.info(
                "[PolicyEval] %s | treat_rate=%.1f%% | DM=%.3f | IPS=%.3f | DR=%.3f",
                pol_name, pi.mean() * 100, dm_val, ips_val, dr_val,
            )

        df = pd.DataFrame(rows)

        # Relative lift vs NeverTreat baseline
        if "NeverTreat" in df["Policy"].values:
            baseline_dr = df.loc[df["Policy"] == "NeverTreat", "DR"].values[0]
            df["DR_lift_vs_never"] = df["DR"] - baseline_dr
        else:
            df["DR_lift_vs_never"] = np.nan

        # Policy group for clean plotting
        def _policy_group(name):
            if name in ("Weibull", "RFM", "Random", "AlwaysTreat", "NeverTreat"):
                return "Core"
            if name.startswith("TopK"):
                return "Top-K EVI"
            if name.startswith("Threshold"):
                return "Prob. Threshold"
            if name == "CostSensitive":
                return "Cost-Sensitive"
            return "Other"

        df["PolicyGroup"] = df["Policy"].map(_policy_group)
        return df

    def plot_threshold_curve(
        self,
        comparison_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Policy-Value vs Treatment-Rate frontier curve.
        Shows the trade-off between how many customers are treated and the
        estimated value per customer (DR estimator).
        Points above the Random baseline curve indicate super-random policies.
        """
        if comparison_df is None:
            comparison_df = self.compare_all_policies(bootstrap=False)

        df = comparison_df.copy()
        fig, ax = plt.subplots(figsize=(10, 6))

        group_styles = {
            "Core":             {"marker": "D", "ms": 10, "lw": 0},
            "Top-K EVI":        {"marker": "o", "ms":  8, "lw": 1.5},
            "Prob. Threshold":  {"marker": "s", "ms":  8, "lw": 1.5},
            "Cost-Sensitive":   {"marker": "^", "ms": 10, "lw": 0},
        }
        group_colors = {
            "Core":             "#3498db",
            "Top-K EVI":        "#2ecc71",
            "Prob. Threshold":  "#e74c3c",
            "Cost-Sensitive":   "#9b59b6",
        }
        core_order = ["NeverTreat", "Random", "RFM", "Weibull",
                      "AlwaysTreat", "CostSensitive"]

        # Plot each group
        for group, style in group_styles.items():
            sub = df[df.get("PolicyGroup", pd.Series("Core", index=df.index)) == group]
            if sub.empty:
                continue
            color = group_colors.get(group, "#888")

            if group in ("Top-K EVI", "Prob. Threshold"):
                # Connect with a line (frontier)
                sub_s = sub.sort_values("TreatmentRate")
                ax.plot(sub_s["TreatmentRate"] * 100, sub_s["DR"],
                        "-", color=color, lw=style["lw"], alpha=0.7, zorder=2)
                ax.scatter(sub_s["TreatmentRate"] * 100, sub_s["DR"],
                           marker=style["marker"], s=style["ms"]**2,
                           color=color, label=group, zorder=3)
            else:
                ax.scatter(sub["TreatmentRate"] * 100, sub["DR"],
                           marker=style["marker"], s=style["ms"]**2,
                           color=color, label=group, zorder=4)
                for _, row in sub.iterrows():
                    ax.annotate(
                        row["Policy"],
                        (row["TreatmentRate"] * 100, row["DR"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=color,
                    )

        # Random baseline reference
        never_dr = df.loc[df["Policy"] == "NeverTreat", "DR"].values
        if len(never_dr):
            ax.axhline(never_dr[0], color="#95a5a6", lw=1.5, ls="--",
                       alpha=0.6, label=f"NeverTreat DR={never_dr[0]:.2f}")

        ax.set_xlabel("Treatment Rate (% customers contacted)", fontsize=12)
        ax.set_ylabel("DR Policy Value (MU/customer)", fontsize=12)
        ax.set_title(
            f"Policy Value vs Treatment Rate Frontier — {dataset_label}\n"
            f"(higher-left is better: high value, fewer contacts)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10, loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[PolicyEval] Threshold curve saved -> %s", save_path)
        return fig

    # =========================================================================
    # Uplift segment profit breakdown
    # =========================================================================

    def segment_profit_breakdown(self) -> pd.DataFrame:
        """
        Compute average DR-estimated profit per uplift segment.

        Returns a DataFrame showing which segments contribute most to
        policy value under the Weibull policy.
        """
        if "uplift_segment" not in self.merged.columns:
            logger.warning("[PolicyEval] 'uplift_segment' column not found in merged data.")
            return pd.DataFrame()

        policies = self._get_policy_vectors()
        w_pi     = policies["Weibull"]

        dm_term  = w_pi * self.mu_1 + (1 - w_pi) * self.mu_0
        e_adj    = np.where(self.treatment == 1, self.propensity, 1.0 - self.propensity)
        e_adj    = np.clip(e_adj, 0.01, 1.0)
        match    = (self.treatment == w_pi).astype(float)
        w        = np.clip(match / e_adj, 0.0, 20.0)
        mu_act   = np.where(self.treatment == 1, self.mu_1, self.mu_0)
        dr_vals  = dm_term + w * (self.outcome - mu_act)

        segments  = self.merged["uplift_segment"].values
        rows = []
        for seg in ["Persuadables", "Sure Things", "Sleeping Dogs", "Lost Causes"]:
            mask = segments == seg
            if mask.sum() == 0:
                continue
            rows.append({
                "Segment":         seg,
                "N":               int(mask.sum()),
                "TreatmentRate":   float(w_pi[mask].mean()),
                "AvgOutcome":      float(self.outcome[mask].mean()),
                "AvgTauHat":       float(self.merged.get("tau_hat", pd.Series(0)).values[mask].mean())
                                   if "tau_hat" in self.merged.columns else np.nan,
                "AvgDR_profit":    float(dr_vals[mask].mean()),
                "TotalDR_profit":  float(dr_vals[mask].sum()),
            })

        return pd.DataFrame(rows)

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_policy_comparison(
        self,
        comparison_df: Optional[pd.DataFrame] = None,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Publication-quality comparison chart: DR policy value with 95% CI.

        Parameters
        ----------
        comparison_df : pd.DataFrame, optional
            Output of compare_all_policies(). If None, runs the comparison.
        save_path : str, optional
        dataset_label : str

        Returns
        -------
        matplotlib Figure
        """
        if comparison_df is None:
            comparison_df = self.compare_all_policies()

        df = comparison_df.copy()

        policy_colors = {
            "Weibull":     "#3498db",
            "RFM":         "#e74c3c",
            "Random":      "#f39c12",
            "AlwaysTreat": "#9b59b6",
            "NeverTreat":  "#95a5a6",
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ── [0] DR policy value with CI ───────────────────────────────────────
        ax = axes[0]
        y_pos = np.arange(len(df))
        colors = [policy_colors.get(p, "#888") for p in df["Policy"]]
        bars = ax.barh(df["Policy"], df["DR"], color=colors, alpha=0.85, height=0.6)

        if "DR_lo" in df.columns:
            ax.errorbar(
                df["DR"], y_pos,
                xerr=[df["DR"] - df["DR_lo"], df["DR_hi"] - df["DR"]],
                fmt="none", color="#333", capsize=5, lw=2,
            )

        ax.axvline(0, color="#ccc", lw=0.8)
        ax.set_xlabel("Estimated Policy Value — DR Estimator (MU/customer)", fontsize=11)
        ax.set_title(f"Policy Value Comparison\n{dataset_label}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")

        # Annotate bars with DR value
        for bar, val in zip(bars, df["DR"]):
            ax.text(
                val + abs(df["DR"]).max() * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}",
                va="center", ha="left", fontsize=9,
            )

        # ── [1] DR lift vs NeverTreat + treatment rate ────────────────────────
        ax2 = axes[1]
        if "DR_lift_vs_never" in df.columns:
            pivot = df[df["Policy"] != "NeverTreat"].copy()
            x2    = np.arange(len(pivot))
            bw2   = 0.35
            colors2 = [policy_colors.get(p, "#888") for p in pivot["Policy"]]

            ax2.bar(x2 - bw2/2, pivot["DR_lift_vs_never"], bw2,
                    color=colors2, alpha=0.85, label="DR Lift")
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x2, pivot["TreatmentRate"] * 100,
                          "D--", color="#2c3e50", ms=7, lw=1.5, label="Treatment Rate %")
            ax2_twin.set_ylabel("Treatment Rate (%)", fontsize=10)
            ax2_twin.legend(loc="upper right", fontsize=8)

            ax2.set_xticks(x2)
            ax2.set_xticklabels(pivot["Policy"], rotation=15, fontsize=9)
            ax2.axhline(0, color="#ccc", lw=0.8)
            ax2.set_ylabel("DR Lift over NeverTreat (MU/customer)", fontsize=11)
            ax2.set_title("DR Lift vs No-Treatment Baseline", fontsize=12, fontweight="bold")
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[PolicyEval] Comparison plot saved → %s", save_path)
        return fig

    def plot_segment_breakdown(
        self,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Stacked bar: DR profit contribution by uplift segment.
        """
        seg_df = self.segment_profit_breakdown()
        if seg_df.empty:
            return plt.figure()

        seg_colors = {
            "Persuadables": "#2ecc71",
            "Sure Things":  "#3498db",
            "Sleeping Dogs": "#e74c3c",
            "Lost Causes":  "#95a5a6",
        }

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Bar 1: Average DR profit per segment
        ax = axes[0]
        colors = [seg_colors.get(s, "#888") for s in seg_df["Segment"]]
        ax.bar(seg_df["Segment"], seg_df["AvgDR_profit"], color=colors, alpha=0.85)
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_title(f"Avg DR Profit per Customer\nby Segment — {dataset_label}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Avg DR Profit (MU/customer)", fontsize=10)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(True, alpha=0.3, axis="y")

        # Bar 2: Total DR profit contribution
        ax2 = axes[1]
        ax2.bar(seg_df["Segment"], seg_df["TotalDR_profit"], color=colors, alpha=0.85)
        ax2.axhline(0, color="#ccc", lw=0.8)
        ax2.set_title("Total DR Profit Contribution\nby Segment",
                      fontsize=11, fontweight="bold")
        ax2.set_ylabel("Total DR Profit (MU)", fontsize=10)
        ax2.tick_params(axis="x", rotation=15)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[PolicyEval] Segment breakdown saved → %s", save_path)
        return fig


# =============================================================================
# Convenience wrapper
# =============================================================================

def run_counterfactual_evaluation(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
    uplift_results: dict,
    rfm_decisions: Optional[pd.DataFrame] = None,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict:
    """
    One-call wrapper: creates PolicyEvaluator, runs full comparison, saves outputs.

    Returns
    -------
    dict
        {
          'evaluator':       PolicyEvaluator,
          'comparison_df':   pd.DataFrame,
          'segment_df':      pd.DataFrame,
          'figs':            dict of matplotlib Figures,
        }
    """
    evaluator = PolicyEvaluator(
        weibull_decisions=weibull_decisions,
        customer_df=customer_df,
        uplift_results=uplift_results,
        rfm_decisions=rfm_decisions,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    comparison_df = evaluator.compare_all_policies()
    segment_df    = evaluator.segment_profit_breakdown()

    figs = {}
    comp_path = os.path.join(save_dir, "counterfactual_policy_comparison.png") if save_dir else None
    seg_path  = os.path.join(save_dir, "counterfactual_segment_breakdown.png")  if save_dir else None

    figs["policy_comparison"]   = evaluator.plot_policy_comparison(
        comparison_df, save_path=comp_path, dataset_label=dataset_label
    )
    figs["segment_breakdown"]   = evaluator.plot_segment_breakdown(
        save_path=seg_path, dataset_label=dataset_label
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        comparison_df.to_csv(os.path.join(save_dir, "counterfactual_comparison.csv"), index=False)
        if not segment_df.empty:
            segment_df.to_csv(os.path.join(save_dir, "counterfactual_segments.csv"), index=False)
        logger.info("[CounterfactualEval] Results saved to %s", save_dir)

    return {
        "evaluator":     evaluator,
        "comparison_df": comparison_df,
        "segment_df":    segment_df,
        "figs":          figs,
    }
