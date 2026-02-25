"""
src/policy.py
=============
Decision-Centric Intervention Policy Engine.

For each customer, computes:
  1. Instantaneous hazard h(t_now | x)  — risk of churning right now
  2. Survival probability S(t_now | x)  — probability still active
  3. Expected Value of Intervention (EVI) — economic signal

Decision Rule:
  IF h(t_now) > θ_h  AND  EVI > 0  → INTERVENE
  ELIF S(t_now) < θ_s               → LOST (do not contact)
  ELSE                               → WAIT

EVI Formula:
  EVI(t*, i) = p_response * Monetary_i * [1 - S(t* | x_i)] - C_contact

where:
  p_response  = campaign response rate (default: 0.15)
  C_contact   = cost per outreach in GBP (default: 1.0)
  t*          = current observation time (days since first purchase)
"""

import logging
import os
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter

logger = logging.getLogger(__name__)

# ── Load policy defaults from config/simulation_params.yaml (if available) ────
def _load_policy_config() -> dict:
    """Load policy section from YAML config. Falls back to hard-coded defaults."""
    _cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "simulation_params.yaml"
    )
    try:
        import yaml  # pyyaml
        with open(_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        policy = cfg.get("policy", {})
        logger.debug("[Policy] Loaded defaults from config/simulation_params.yaml")
        return policy
    except Exception:
        return {}

_policy_cfg = _load_policy_config()

# ── Default Policy Thresholds (YAML overrides hard-coded fallbacks) ───────────
DEFAULT_HAZARD_THRESHOLD = _policy_cfg.get("hazard_threshold",   0.01)
DEFAULT_SURVIVAL_FLOOR   = _policy_cfg.get("survival_floor",     0.05)
DEFAULT_RESPONSE_RATE    = _policy_cfg.get("response_rate",      0.15)
DEFAULT_COST_PER_CONTACT = _policy_cfg.get("cost_per_contact",   1.0)
DEFAULT_MIN_EVI_THRESHOLD = _policy_cfg.get("min_evi_threshold", 0.0)


def _compute_hazard_from_survival(
    survival_fn: pd.DataFrame,
    t_grid: np.ndarray,
) -> pd.DataFrame:
    """
    Numerically differentiate survival function to obtain hazard rates.

    h(t) ≈ -ΔS(t) / (S(t) * Δt)

    Parameters
    ----------
    survival_fn : pd.DataFrame
        Survival function matrix, shape (len(t_grid), n_customers).
        Columns = customer indices, rows = time points.
    t_grid : np.ndarray
        Time points corresponding to survival_fn rows.

    Returns
    -------
    pd.DataFrame
        Hazard rate matrix, same shape as survival_fn.
    """
    S = survival_fn.values  # shape: (T, N)
    dt = np.diff(t_grid, prepend=t_grid[0])[:, None]  # (T, 1)

    # Numerical derivative: h(t) = -dS/dt / S(t)
    dS = np.diff(S, axis=0, prepend=S[[0], :])
    hazard = -dS / (S + 1e-10) / (dt + 1e-10)
    hazard = np.clip(hazard, 0, None)  # hazard must be non-negative

    return pd.DataFrame(hazard, index=survival_fn.index, columns=survival_fn.columns)


def compute_intervention_signals(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    customer_df: pd.DataFrame,
    t_now: float = None,
    t_grid_steps: int = 200,
    predicted_clv: pd.Series = None,
) -> pd.DataFrame:
    """
    Compute hazard, survival, and EVI signals for all customers.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame (same as used for fitting), with T and E.
    customer_df : pd.DataFrame
        Original (unscaled) customer DataFrame — used for Monetary values.
    t_now : float, optional
        Current time point in days. Defaults to median T in dataset.
    t_grid_steps : int
        Number of time steps for survival function evaluation (default: 200).
    predicted_clv : pd.Series, optional
        Predicted future CLV per customer (indexed by CustomerID).
        When provided, replaces historical Monetary in the EVI CLV term.
        Falls back to ``customer_df["Monetary"]`` when None.

    Returns
    -------
    pd.DataFrame
        Per-customer signals with columns:
        [hazard_now, survival_now, evi, optimal_window_days, clv_used]
    """
    t_max = df_scaled["T"].max()
    t_grid = np.linspace(1, t_max, t_grid_steps)

    if t_now is None:
        t_now = float(df_scaled["T"].median())
        logger.info(f"t_now not specified — using median T = {t_now:.1f} days")

    # ── Survival function S(t | x) for all customers ─────────────────────────
    logger.info(f"Computing survival functions over {t_grid_steps} time steps...")
    survival_fn = waf.predict_survival_function(df_scaled, times=t_grid)
    # survival_fn: shape (t_grid_steps, n_customers), columns = customer index

    # ── Hazard function h(t | x) ──────────────────────────────────────────────
    hazard_fn = _compute_hazard_from_survival(survival_fn, t_grid)

    # ── Extract values at t_now ───────────────────────────────────────────────
    t_idx = np.argmin(np.abs(t_grid - t_now))

    hazard_now  = hazard_fn.iloc[t_idx]    # Series, index = customer index
    survival_now = survival_fn.iloc[t_idx]  # Series, index = customer index

    # ── Optimal intervention window: time at which h(t) is maximized ─────────
    optimal_t_idx = hazard_fn.idxmax(axis=0)  # index label of max hazard row
    # Map index label back to t_grid value
    idx_to_t = dict(zip(range(len(t_grid)), t_grid))
    optimal_window_days = optimal_t_idx.map(
        lambda row_label: t_grid[list(hazard_fn.index).index(row_label)]
        if row_label in hazard_fn.index else np.nan
    )

    # ── CLV for EVI calculation ───────────────────────────────────────────────
    # Prefer forward-looking predicted_clv when available (anti-leakage design).
    # Fall back to historical Monetary if no CLV model has been trained.
    if predicted_clv is not None:
        # Align by index (CustomerID); fill missing with 0
        clv_values = predicted_clv.reindex(df_scaled.index).fillna(0.0).values
        clv_source = "predicted_clv"
    else:
        clv_values = customer_df["Monetary"].values
        clv_source = "historical_Monetary"

    # ── Assemble signals DataFrame ────────────────────────────────────────────
    signals = pd.DataFrame({
        "hazard_now":          hazard_now.values,
        "survival_now":        survival_now.values,
        "optimal_window_days": t_grid[hazard_fn.values.argmax(axis=0)],
        "clv_used":            clv_values,
    }, index=df_scaled.index)

    # Keep historical Monetary for reference / dashboard artifact
    signals["Monetary"] = customer_df["Monetary"].values
    logger.info(f"[Policy] EVI CLV source: {clv_source}")

    return signals


def make_intervention_decisions(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    customer_df: pd.DataFrame,
    t_now: float = None,
    theta_h: float = DEFAULT_HAZARD_THRESHOLD,
    theta_s: float = DEFAULT_SURVIVAL_FLOOR,
    p_response: float = DEFAULT_RESPONSE_RATE,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
    vip_pct: float = None,  # E6: override VIP guard percentile (1.0 = disable)
    predicted_clv: pd.Series = None,
    min_evi_threshold: float = DEFAULT_MIN_EVI_THRESHOLD,
) -> pd.DataFrame:
    """
    Apply the full decision policy to all customers.

    Decision Rule:
      IF h(t_now) > θ_h  AND  EVI > min_evi  → INTERVENE
      ELIF S(t_now) < θ_s                    → LOST
      ELSE                                   → WAIT
    """
    clv_label = "predictive" if predicted_clv is not None else "historical Monetary"
    logger.info(
        f"Running intervention policy | θ_h={theta_h} | θ_s={theta_s} | "
        f"p_response={p_response} | cost_per_contact={cost_per_contact:.2f} MU | "
        f"min_evi={min_evi_threshold:.2f} | CLV source={clv_label}"
    )

    signals = compute_intervention_signals(
        waf, df_scaled, customer_df, t_now,
        predicted_clv=predicted_clv,
    )

    # ── Expected Value of Intervention ────────────────────────────────────────
    # EVI(t*, i) = p_response * CLV_i * [1 - S(t* | x_i)] - C_contact
    # CLV_i is predicted_clv when available, else historical Monetary
    signals["evi"] = (
        p_response * signals["clv_used"] * (1 - signals["survival_now"])
        - cost_per_contact
    )

    # ── Apply decision rule (vectorized) ─────────────────────────────────────
    # LOST   : S(t_now) < theta_s
    # INTERVENE : h(t_now) > theta_h AND EVI > min_evi_threshold
    # WAIT   : everything else
    is_lost      = signals["survival_now"] < theta_s
    is_intervene = (~is_lost) & (signals["hazard_now"] > theta_h) & (signals["evi"] > min_evi_threshold)

    # E4: VIP Sleeping Dog Guard — prevent spamming high-value happy customers.
    # If a customer has very high Monetary (top percentile) but very LOW hazard,
    # they are a VIP who is NOT at risk → force WAIT even if EVI > 0.
    vip_pct_val = vip_pct if vip_pct is not None else _policy_cfg.get("vip_threshold_percentile", 0.90)
    monetary_threshold = signals["Monetary"].quantile(vip_pct_val)
    is_vip_sleeping_dog = (
        (~is_lost)
        & (signals["hazard_now"] < theta_h * 0.5)
        & (signals["Monetary"] > monetary_threshold)
    )
    n_vip_guarded = is_vip_sleeping_dog.sum()
    if n_vip_guarded > 0:
        logger.info(
            f"[VIP Guard] Protected {n_vip_guarded} VIP Sleeping Dogs "
            f"(Monetary > P{vip_pct_val*100:.0f}={monetary_threshold:.1f}, "
            f"hazard < {theta_h*0.5:.4f})"
        )

    signals["decision"] = np.select(
        [is_lost, is_vip_sleeping_dog, is_intervene],
        ["LOST",  "WAIT",              "INTERVENE"],
        default="WAIT",
    )

    # ── Add CustomerID ────────────────────────────────────────────────────────
    signals.index.name = "CustomerID"
    signals = signals.reset_index()

    # Expose predicted_clv as its own column (or copy Monetary if no CLV model)
    if predicted_clv is not None:
        signals["predicted_clv"] = predicted_clv.reindex(
            signals["CustomerID"]
        ).values
    else:
        signals["predicted_clv"] = signals["Monetary"]

    # ── Log decision distribution ─────────────────────────────────────────────
    dist = signals["decision"].value_counts()
    logger.info(f"Decision distribution:\n{dist.to_string()}")
    logger.info(
        f"Intervention rate: {(signals['decision'] == 'INTERVENE').mean() * 100:.1f}% | "
        f"Lost rate: {(signals['decision'] == 'LOST').mean() * 100:.1f}% | "
        f"Wait rate: {(signals['decision'] == 'WAIT').mean() * 100:.1f}%"
    )

    return signals[
        ["CustomerID", "hazard_now", "survival_now", "evi", "decision",
         "optimal_window_days", "Monetary", "predicted_clv"]
    ].rename(columns={"survival_now": "survival"})


def rfm_intervention_decisions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate intervention decisions from RFM segmentation (baseline policy).
    Used for comparison against the Weibull AFT policy.

    Rule:
      - At Risk  → INTERVENE
      - Lost     → LOST
      - Loyal    → WAIT
      - Champions → WAIT

    Parameters
    ----------
    rfm_df : pd.DataFrame
        Output of models.rfm_segment() with RFM_Segment column.

    Returns
    -------
    pd.DataFrame
        Decision table with columns: [CustomerID, RFM_Segment, decision]
    """
    segment_to_decision = {
        "At Risk":   "INTERVENE",
        "Lost":      "LOST",
        "Loyal":     "WAIT",
        "Champions": "WAIT",
    }
    df = rfm_df.copy().reset_index()
    df["decision"] = df["RFM_Segment"].map(segment_to_decision)
    return df[["CustomerID", "RFM_Segment", "decision"]]


def lr_intervention_decisions(
    lr_pipeline,
    customer_df: pd.DataFrame,
    uplift_scores: pd.Series,
    predicted_clv: pd.Series = None,
    p_response: float = DEFAULT_RESPONSE_RATE,
    cost_per_contact: float = DEFAULT_COST_PER_CONTACT,
    churn_prob_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Generate intervention decisions from the Logistic Regression baseline model
    combined with EVI, creating an apples-to-apples comparison with Weibull EVI.

    EVI Formula (mirrors Weibull policy for fair comparison):
      lr_evi = uplift_scores * predicted_clv - cost_per_contact

    Decision Rule:
      INTERVENE if lr_churn_prob > churn_prob_threshold AND lr_evi > 0
      WAIT      otherwise

    Parameters
    ----------
    lr_pipeline : sklearn Pipeline
        Fitted LR pipeline (imputer -> scaler -> LogisticRegression).
    customer_df : pd.DataFrame
        Customer-level DataFrame indexed by CustomerID.
        Must contain LOGISTIC_FEATURES columns (Frequency, Monetary,
        InterPurchaseTime, GapDeviation, SinglePurchase).
    uplift_scores : pd.Series
        Per-customer uplift scores (tau_hat) from the T-Learner, indexed by
        CustomerID.  Used as the ``p_response``-equivalent weight in EVI.
        If uplift is not available, pass a constant Series (e.g. all=p_response).
    predicted_clv : pd.Series, optional
        Forward-looking CLV per customer (indexed by CustomerID).
        Falls back to customer_df["Monetary"] when None.
    p_response : float
        Campaign response rate — used when uplift_scores is constant.
    cost_per_contact : float
        Cost per marketing contact (MU).
    churn_prob_threshold : float
        Minimum LR churn probability to qualify for intervention (default: 0.5).

    Returns
    -------
    pd.DataFrame
        Decision table with columns:
        [CustomerID, lr_churn_prob, lr_evi, decision, predicted_clv]
    """
    from src.models import LOGISTIC_FEATURES  # avoid circular at module level

    logger.info(
        f"[LR Policy] Running LR+EVI intervention policy | "
        f"threshold={churn_prob_threshold} | cost={cost_per_contact:.2f} MU"
    )

    # ── Features (same as train_logistic — no Recency) ────────────────────────
    available_lr_feats = [f for f in LOGISTIC_FEATURES if f in customer_df.columns]
    X = customer_df[available_lr_feats].values

    # ── LR churn probability ───────────────────────────────────────────────────
    lr_churn_prob = lr_pipeline.predict_proba(X)[:, 1]  # P(E=1 | x)

    # ── CLV for EVI ───────────────────────────────────────────────────────────
    if predicted_clv is not None:
        clv_values = predicted_clv.reindex(customer_df.index).fillna(0.0).values
        clv_label  = "predicted_clv"
    else:
        clv_values = customer_df["Monetary"].values
        clv_label  = "historical_Monetary"
    logger.info(f"[LR Policy] CLV source: {clv_label}")

    # ── Uplift scores (T-Learner tau_hat) aligned to customer_df index ────────
    uplift_vals = uplift_scores.reindex(customer_df.index).fillna(p_response).values

    # ── EVI: uplift * CLV - cost (same formula as Weibull for fair comparison) ─
    lr_evi = uplift_vals * clv_values - cost_per_contact

    # ── Decision rule ─────────────────────────────────────────────────────────
    is_intervene = (lr_churn_prob > churn_prob_threshold) & (lr_evi > 0)
    decision = np.where(is_intervene, "INTERVENE", "WAIT")

    result = pd.DataFrame({
        "CustomerID":    customer_df.index,
        "lr_churn_prob": lr_churn_prob,
        "lr_evi":        lr_evi,
        "decision":      decision,
        "predicted_clv": clv_values,
    })

    n_intervene = is_intervene.sum()
    n_total     = len(result)
    logger.info(
        f"[LR Policy] Decisions — INTERVENE: {n_intervene:,} ({n_intervene/n_total*100:.1f}%) | "
        f"WAIT: {n_total - n_intervene:,} ({(n_total - n_intervene)/n_total*100:.1f}%)"
    )

    return result

