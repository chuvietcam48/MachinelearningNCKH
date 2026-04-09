"""
src/uplift.py
=============
Uplift Modeling Proxy for the Decision-Centric Customer Re-Engagement framework.

Background
----------
True uplift modeling requires a randomised controlled trial (A/B test) with
treatment and control groups.  Most datasets (UCI, TaFeng, CDNOW) contain no
such experiment, so this module implements a *proxy* approach that is
scientifically sound and common in academic literature.

When the customer DataFrame contains ground-truth RCT labels
(`treatment_flg` and `target_flag` — as in the X5 RetailHero dataset),
the module automatically switches to **real-treatment mode**: these labels
replace the Weibull intervention proxy, enabling an unbiased Qini
coefficient (potentially positive for the first time).

  1. **IPTW-Corrected T-Learner**
     The Weibull intervention signal (EVI > 0 AND h > theta_h) acts as the
     "treatment assignment" proxy.  To mitigate *selection bias* inherent in
     observational treatment assignment, we:
       a) Estimate propensity scores e(X) = P(T=1 | X) via Logistic Regression.
       b) Compute Inverse Probability of Treatment Weights (IPTW):
            w_i = 1/e(X_i)     if treated (T=1)
            w_i = 1/(1-e(X_i)) if control  (T=0)
       c) Pass IPTW weights as sample_weight into each T-Learner branch.
     This reweights each sample to emulate a balanced pseudo-population,
     producing a debiased estimate of the Average Treatment Effect (ATE).
     See: Rosenbaum & Rubin (1983), Hirano & Imbens (2001).

  2. **T-Learner CATE Estimation (tau_hat)**
     Given IPTW-corrected training, we estimate per-customer CATE:
       tau_hat(x) = mu_1(x) - mu_0(x)
     where mu_1, mu_0 are GradientBoosting regressors on treated/control sets.

  3. **Persuadables Segmentation**
     Following Radcliffe & Surry (1999), customers are split into 4 quadrants:
       - Persuadables  : uplift > 0 and would NOT respond without intervention
       - Sure Things   : respond regardless of intervention
       - Lost Causes   : do not respond regardless
       - Sleeping Dogs : respond better WITHOUT intervention (negative uplift)

  4. **Qini Curve** (Radcliffe, 2007)
     Measures cumulative incremental gain over a random targeting baseline.

Usage
-----
    from src.uplift import run_uplift_analysis
    uplift_df, qini_fig = run_uplift_analysis(weibull_decisions, customer_df)
"""

import logging
import warnings
import numpy as np
from scipy.integrate import trapezoid as _trapz
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.dataset_registry import get_currency_code

logger = logging.getLogger(__name__)

# ── Segmentation thresholds ───────────────────────────────────────────────────
_UPLIFT_HIGH_THR = 0.0   # tau_hat > this → responds positively to intervention
_RESPONSE_THR    = 0.5   # predicted response prob threshold for "Sure Things"


# =============================================================================
# 1. Feature Matrix Assembly
# =============================================================================

def _build_feature_matrix(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge decision signals with customer RFM features.

    Returns a DataFrame with:
      treatment : 1 = treated, 0 = control
                  Source depends on data availability:
                    Real-treatment mode  (X5): uses `treatment_flg` from customer_df
                    Proxy mode (UCI/TaFeng/CDNOW): uses Weibull INTERVENE decision
      RFM features for T-Learner / PSM
      Monetary : outcome proxy
                  Real-treatment mode: uses `target_flag` (binary purchase outcome)
                  Proxy mode: uses historical Monetary spend
    """
    # Defensively reset index on customer_df in case CustomerID is the index
    cdf = customer_df.reset_index() if "CustomerID" not in customer_df.columns else customer_df.copy()

    # ── Detect real-treatment mode (X5 RetailHero) ────────────────────────────
    has_real_labels = (
        "treatment_flg" in cdf.columns and "target_flag" in cdf.columns
    )

    if has_real_labels:
        logger.info(
            "[Uplift] Ground-truth labels detected (treatment_flg + target_flag). "
            "Switching to REAL-TREATMENT mode — Qini may be positive."
        )
        # Use ALL customers (no LOST filter — we have real treatment assignment)
        df = weibull_decisions.copy()
        # Map real treatment labels from customer_df
        label_cols = ["CustomerID", "treatment_flg", "target_flag"]
        avail_labels = [c for c in label_cols if c in cdf.columns]
        df = df.merge(
            cdf[avail_labels],
            on="CustomerID",
            how="left",
        )
        df["treatment"] = df["treatment_flg"].fillna(0).astype(int)
        # Use target_flag (0/1 purchase outcome) as the monetary outcome proxy
        df["Monetary"] = df["target_flag"].fillna(0).astype(float)
        logger.info(
            "[Uplift] Real treatment rate: %.1f%% | Conversion rate: %.1f%%",
            df["treatment"].mean() * 100,
            df["Monetary"].mean() * 100,
        )
    else:
        logger.info(
            "[Uplift] No ground-truth labels found. "
            "Using Weibull INTERVENE proxy as treatment assignment."
        )
        # Exclude LOST — they are not actionable in proxy mode
        df = weibull_decisions[weibull_decisions["decision"] != "LOST"].copy()
        df["treatment"] = (df["decision"] == "INTERVENE").astype(int)

    # ── Merge RFM covariates ──────────────────────────────────────────────────
    rfm_cols = ["CustomerID", "Recency", "Frequency", "Monetary",
                "InterPurchaseTime", "GapDeviation", "SinglePurchase"]
    # In real-treatment mode Monetary was already set from target_flag;
    # only pull it from cdf in proxy mode.
    if has_real_labels:
        rfm_cols = [c for c in rfm_cols if c != "Monetary"]

    available = [c for c in rfm_cols if c in cdf.columns]
    merged = df.merge(
        cdf[available].set_index("CustomerID"),
        left_on="CustomerID",
        right_index=True,
        how="left",
        suffixes=("_decision", ""),
    )

    # Drop any Monetary duplication from the decisions table itself
    if "Monetary_decision" in merged.columns:
        merged = merged.drop(columns=["Monetary_decision"])

    return merged


# =============================================================================
# 2. IPTW-Corrected T-Learner CATE Estimation
# =============================================================================

def _estimate_propensity_scores(
    X: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    """
    Estimate propensity scores e(X) = P(T=1 | X) via Logistic Regression.

    Propensity scores are clipped to [0.01, 0.99] to prevent extreme weights
    (positivity assumption enforcement).

    Parameters
    ----------
    X : np.ndarray, shape (n, p)
        Covariate matrix (already imputed and scaled).
    treatment : np.ndarray, shape (n,)
        Binary treatment indicator (1=INTERVENE, 0=WAIT).

    Returns
    -------
    np.ndarray, shape (n,)
        Clipped propensity scores in [0.01, 0.99].
    """
    from sklearn.linear_model import LogisticRegression

    ps_model = LogisticRegression(
        max_iter=1000, penalty="l2", C=1.0, solver="lbfgs", random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps_model.fit(X, treatment)
    propensity = ps_model.predict_proba(X)[:, 1]          # P(T=1 | X)
    propensity = np.clip(propensity, 0.01, 0.99)          # positivity
    logger.info(
        "[Uplift][IPTW] Propensity scores — min=%.4f | mean=%.4f | max=%.4f",
        propensity.min(), propensity.mean(), propensity.max(),
    )
    return propensity


def _compute_iptw_weights(
    propensity: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    """
    Compute Inverse Probability of Treatment Weights (IPTW).

    Stabilised IPTW formula:
      treated  : w = P(T=1) / e(X)
      control  : w = P(T=0) / (1 - e(X))

    Stabilisation by mean treatment probability reduces variance of weights.
    Weights are clipped at 99th-percentile to limit extreme influence.
    """
    p_t = treatment.mean()                                # marginal P(T=1)
    p_c = 1.0 - p_t                                      # marginal P(T=0)

    weights = np.where(
        treatment == 1,
        p_t / propensity,          # stabilised treated weight
        p_c / (1.0 - propensity),  # stabilised control weight
    )
    # Winsorise at 99th percentile to remove extreme leverage
    clip_val = np.percentile(weights, 99)
    weights  = np.clip(weights, 0.0, clip_val)

    eff_n = (weights.sum() ** 2) / (weights ** 2).sum()  # effective sample size
    logger.info(
        "[Uplift][IPTW] Weights — mean=%.4f | max(clipped)=%.4f | "
        "effective N=%.0f (raw N=%d)",
        weights.mean(), weights.max(), eff_n, len(weights),
    )
    return weights


def _fit_t_learner(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Fit an IPTW-corrected T-Learner to estimate causal uplift tau_hat(x).

    Step 1 — Propensity Estimation:
      Logistic Regression → e(X) = P(T=1|X), clipped to [0.01, 0.99]

    Step 2 — IPTW Computation:
      w_i = P(T=1)/e(X_i) for treated | w_i = P(T=0)/(1-e(X_i)) for control
      Stabilised + winsorised at 99th percentile.

    Step 3 — Weighted T-Learner:
      mu_1 fitted on treated subset   with sample_weight=w[treated]
      mu_0 fitted on control subset   with sample_weight=w[control]

    Step 4 — CATE:
      tau_hat(x) = mu_1(x) - mu_0(x)   (evaluated on ALL customers)
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise ImportError("scikit-learn is required for uplift modeling.")

    treatment = df["treatment"].values.astype(int)
    y         = df["Monetary"].values.astype(float)

    # ------------------------------------------------------------------
    # Pre-process X for propensity model (impute + scale)
    # ------------------------------------------------------------------
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_raw   = df[feature_cols].values
    X_imp   = imputer.fit_transform(X_raw)
    X_sc    = scaler.fit_transform(X_imp)

    treated_mask = treatment == 1
    control_mask = ~treated_mask

    if treated_mask.sum() < 10:
        logger.warning(
            "[Uplift] Too few treated customers (%d) for T-Learner. "
            "Lower the hazard_threshold to increase INTERVENE count.",
            treated_mask.sum(),
        )

    # ------------------------------------------------------------------
    # STEP 1+2 : Propensity scores → IPTW weights
    # ------------------------------------------------------------------
    propensity   = _estimate_propensity_scores(X_sc, treatment)
    iptw_weights = _compute_iptw_weights(propensity, treatment)

    # ------------------------------------------------------------------
    # STEP 3 : Weighted T-Learner  (GradientBoosting, raw X to avoid
    #           double-scaling inside the pipeline)
    # ------------------------------------------------------------------
    def _make_gbr():
        return GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    mu_1_model = _make_gbr()
    mu_0_model = _make_gbr()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Pass IPTW weights as sample_weight to debias each branch
        mu_1_model.fit(
            X_sc[treated_mask], y[treated_mask],
            sample_weight=iptw_weights[treated_mask],
        )
        mu_0_model.fit(
            X_sc[control_mask], y[control_mask],
            sample_weight=iptw_weights[control_mask],
        )

    # ------------------------------------------------------------------
    # STEP 4 : Predict on full population
    # ------------------------------------------------------------------
    tau_hat = mu_1_model.predict(X_sc) - mu_0_model.predict(X_sc)
    df = df.copy()
    df["tau_hat"]    = tau_hat
    df["mu_1"]       = mu_1_model.predict(X_sc)  # predicted outcome IF treated
    df["mu_0"]       = mu_0_model.predict(X_sc)  # predicted outcome IF control
    df["propensity"] = propensity                # for diagnostics
    df["iptw"]       = iptw_weights              # for diagnostics
    return df


# =============================================================================
# 3. Persuadables Segmentation
# =============================================================================

def _assign_uplift_segment(row: pd.Series) -> str:
    """
    Assign Radcliffe & Surry (1999) uplift quadrant based on:
      tau_hat  : estimated uplift from T-Learner
      mu_1     : predicted outcome IF treated

    Quadrant definitions (simplified for revenue proxy):
      Persuadables  : uplift > 0 AND mu_1 > response_threshold
      Sure Things   : uplift <= 0 AND mu_1 > response_threshold
      Lost Causes   : uplift <= 0 AND mu_1 <= response_threshold
      Sleeping Dogs : uplift > 0 AND mu_1 <= response_threshold
    """
    is_uplift    = row["tau_hat"] > _UPLIFT_HIGH_THR
    is_responder = row["mu_1"] > _RESPONSE_THR

    if is_uplift and is_responder:
        return "Persuadables"
    elif not is_uplift and is_responder:
        return "Sure Things"
    elif is_uplift and not is_responder:
        return "Sleeping Dogs"
    else:
        return "Lost Causes"


# =============================================================================
# 4. Qini Curve
# =============================================================================

def _compute_qini(df: pd.DataFrame, outcome_col: str = "Monetary") -> pd.DataFrame:
    """
    Compute Qini curve for incremental gain assessment (vectorized O(n)).

    Qini(k) = Y_t_top_k - (n_t_k / n_t) * Y_t_all
    where k = top-k percentile targeted by uplift score.

    Returns
    -------
    pd.DataFrame
        Columns: ['pct_targeted', 'qini_gain', 'random_baseline']
    """
    df_sorted = df.sort_values("tau_hat", ascending=False).reset_index(drop=True)
    n   = len(df_sorted)
    n_t = (df_sorted["treatment"] == 1).sum()
    n_c = n - n_t

    if n_t == 0 or n_c == 0:
        logger.warning("[Uplift] Qini curve requires both treated and control groups.")
        return pd.DataFrame({"pct_targeted": [0, 1], "qini_gain": [0, 0], "random_baseline": [0, 0]})

    treat_flag = (df_sorted["treatment"] == 1).values
    outcome    = df_sorted[outcome_col].values

    Y_t_all = outcome[treat_flag].sum()
    Y_c_all = outcome[~treat_flag].sum()

    # Vectorized cumulative sums
    cum_Y_t  = np.cumsum(outcome * treat_flag)           # cumulative treated revenue
    cum_n_c  = np.cumsum(~treat_flag).astype(float)     # cumulative control count

    qini_gain        = cum_Y_t - (Y_t_all * cum_n_c / n_c)
    random_baseline  = Y_t_all * np.arange(1, n + 1) / n

    return pd.DataFrame({
        "pct_targeted":    np.linspace(0, 1, n + 1)[1:],
        "qini_gain":       qini_gain,
        "random_baseline": random_baseline,
    })


# =============================================================================
# 5. Qini Plot
# =============================================================================

def _plot_qini(qini_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
    """Render and optionally save the Qini curve comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(qini_df["pct_targeted"] * 100, qini_df["qini_gain"],
            color="#00b4d8", lw=2, label="T-Learner Uplift")
    ax.plot(qini_df["pct_targeted"] * 100, qini_df["random_baseline"],
            color="#888", lw=1.5, ls="--", label="Random Targeting")
    ax.fill_between(qini_df["pct_targeted"] * 100,
                    qini_df["qini_gain"], qini_df["random_baseline"],
                    alpha=0.15, color="#00b4d8")
    ax.set_xlabel("% Population Targeted", fontsize=11)
    ax.set_ylabel(f"Incremental Revenue ({get_currency_code()})", fontsize=11)
    ax.set_title("Qini Curve — Uplift vs Random Targeting", fontsize=13, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved Qini curve → %s", save_path)
    return fig


# =============================================================================
# 5b. Cumulative Gain Chart (E5)
# =============================================================================

def _plot_cumulative_gain(
    df: pd.DataFrame,
    outcome_col: str = "Monetary",
    save_path: str = None,
) -> plt.Figure:
    """
    E5: Cumulative Gain Chart for uplift analysis.

    X-axis : % customers targeted (sorted by uplift score descending)
    Y-axis : cumulative % of total revenue captured
    Compare : Model-based targeting vs Random targeting

    A good uplift model captures most revenue with few contacts.
    """
    if "tau_hat" not in df.columns or outcome_col not in df.columns:
        logger.warning("[CumulativeGain] Missing tau_hat or outcome column — skipping.")
        return None

    sorted_df = df.sort_values("tau_hat", ascending=False).reset_index(drop=True)
    n = len(sorted_df)
    total_rev = sorted_df[outcome_col].sum()

    if total_rev <= 0:
        logger.warning("[CumulativeGain] Total revenue is 0 — skipping.")
        return None

    cum_rev_model = sorted_df[outcome_col].cumsum() / total_rev * 100
    pct_targeted = np.arange(1, n + 1) / n * 100
    cum_rev_random = pct_targeted  # random targeting: captured % = targeted %

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(pct_targeted, cum_rev_model, color="#e74c3c", lw=2.5,
            label="Uplift-Ranked Targeting")
    ax.plot(pct_targeted, cum_rev_random, color="#888", lw=1.5, ls="--",
            label="Random Targeting")
    ax.fill_between(pct_targeted, cum_rev_model, cum_rev_random,
                    where=cum_rev_model >= cum_rev_random,
                    alpha=0.15, color="#e74c3c")

    # Mark the point where 80% of revenue is captured
    idx_80 = np.searchsorted(cum_rev_model.values, 80.0)
    if idx_80 < n:
        pct_at_80 = pct_targeted[idx_80]
        ax.axhline(80, color="#3498db", ls=":", alpha=0.5)
        ax.axvline(pct_at_80, color="#3498db", ls=":", alpha=0.5)
        ax.annotate(
            f"80% revenue at {pct_at_80:.0f}% targeted",
            xy=(pct_at_80, 80), xytext=(pct_at_80 + 10, 70),
            arrowprops=dict(arrowstyle="->", color="#3498db"),
            fontsize=10, color="#3498db", fontweight="bold",
        )

    ax.set_xlabel("% Customers Targeted", fontsize=12)
    ax.set_ylabel("% Cumulative Revenue Captured", fontsize=12)
    ax.set_title("Cumulative Gain Chart — Uplift Model vs Random",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Cumulative Gain chart → {save_path}")
    return fig


# =============================================================================
# 6. Main Entry Point
# =============================================================================

_UPLIFT_FEATURE_COLS = [
    "Recency", "Frequency", "InterPurchaseTime", "GapDeviation", "SinglePurchase",
    "survival", "evi",
]


def run_uplift_analysis(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
    save_path: str = None,
) -> dict:
    """
    Run the full uplift modeling pipeline.

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions() — must include columns:
        'CustomerID', 'decision', 'survival', 'evi'.
    customer_df : pd.DataFrame
        Original customer-level DataFrame for RFM covariates.
    save_path : str, optional
        If provided, saves Qini curve plot to this path.

    Returns
    -------
    dict
        Keys:
          'uplift_df'         : pd.DataFrame with tau_hat and segment per customer
          'segment_counts'    : dict of segment → count
          'qini_df'           : Qini curve DataFrame
          'persuadable_pct'   : float (fraction of INTERVENE that are Persuadables)
          'qini_auc_ratio'    : float (model Qini AUC / random Qini AUC — Qini coefficient)
    """
    logger.info("[Uplift] Starting T-Learner uplift analysis...")

    # 1. Build feature matrix
    merged = _build_feature_matrix(weibull_decisions, customer_df)
    logger.info(
        "[Uplift] %d customers in analysis | treated=%d | control=%d",
        len(merged),
        (merged["treatment"] == 1).sum(),
        (merged["treatment"] == 0).sum(),
    )

    # 2. Determine available features
    available_features = [c for c in _UPLIFT_FEATURE_COLS if c in merged.columns]

    # 3. Fit T-Learner
    uplift_df = _fit_t_learner(merged, available_features)

    # 4. Segment
    uplift_df["uplift_segment"] = uplift_df.apply(_assign_uplift_segment, axis=1)

    # 5. Log segment distribution
    counts = uplift_df["uplift_segment"].value_counts().to_dict()
    logger.info("[Uplift] Segment distribution: %s", counts)

    intervene_df = uplift_df[uplift_df["treatment"] == 1]
    persuadable_pct = (
        (intervene_df["uplift_segment"] == "Persuadables").mean()
        if len(intervene_df) > 0 else 0.0
    )
    logger.info(
        "[Uplift] Of INTERVENE customers, %.1f%% are Persuadables "
        "(positive uplift + predicted responder).",
        persuadable_pct * 100,
    )

    # 6. Qini curve
    qini_df = _compute_qini(uplift_df)

    # 7. Qini coefficient (AUC ratio)
    qini_auc  = _trapz(qini_df["qini_gain"],      qini_df["pct_targeted"])
    rand_auc  = _trapz(qini_df["random_baseline"], qini_df["pct_targeted"])
    qini_coef = (qini_auc / rand_auc) if rand_auc != 0 else 0.0
    logger.info("[Uplift] Qini coefficient (model AUC / random AUC): %.4f", qini_coef)

    # 8. Plot Qini + Cumulative Gain
    _plot_qini(qini_df, save_path=save_path)

    # E5: Cumulative Gain Chart
    cum_gain_path = None
    if save_path:
        import os
        cum_gain_path = save_path.replace("qini_curve", "cumulative_gain")
        if cum_gain_path == save_path:
            cum_gain_path = os.path.join(os.path.dirname(save_path), "cumulative_gain.png")
    _plot_cumulative_gain(uplift_df, outcome_col="Monetary", save_path=cum_gain_path)

    return {
        "uplift_df":          uplift_df,
        "segment_counts":     counts,
        "qini_df":            qini_df,
        "persuadable_pct":    persuadable_pct,
        "qini_auc_ratio":     qini_coef,
    }
