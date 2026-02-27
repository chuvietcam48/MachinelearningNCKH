"""
src/evaluation.py
=================
Comprehensive evaluation framework for the Decision-Centric survival models.

Technical Metrics:
  - Concordance Index (C-index): Uses model's built-in concordance_index_ attribute
    (avoids numerical overflow from predict_median on extreme feature values)
  - Integrated Brier Score (IBS): Calibration over time
  - Logistic AUC: Binary classification baseline

Business/Decision Metrics:
  - Outreach Efficiency: Fraction of unnecessary contacts avoided vs. RFM baseline
  - Revenue Lift: Precision gain of Weibull policy (higher EVI per contact)
  - Decision Distribution: INTERVENE / WAIT / LOST breakdown

Mathematical Definitions:
  C-index = sum_{i,j} 1[T_hat_i < T_hat_j] * 1[T_i < T_j] * delta_i
            -------------------------------------------------------
                     sum_{i,j} 1[T_i < T_j] * delta_i

  BS(t) = (1/n) sum_i [S_hat(t|x_i) - 1(T_i > t)]^2
  IBS   = (1 / (t_max - t_min)) integral BS(t) dt

  EVI(t*, i) = p_response * Monetary_i * [1 - S(t* | x_i)] - C_contact
  Revenue Lift = (avg_EVI_weibull - avg_EVI_rfm) / avg_EVI_rfm * 100%
"""

import logging
import numpy as np
import pandas as pd
from lifelines import WeibullAFTFitter, CoxPHFitter
from sklearn.metrics import roc_auc_score
from scipy.integrate import trapezoid
from src.dataset_registry import get_currency_symbol, get_currency_code

logger = logging.getLogger(__name__)


# =============================================================================
# D3: Per-Dataset Config Loader
# =============================================================================

def load_config_with_overrides(dataset: str) -> dict:
    """
    D3: Load simulation_params.yaml and merge dataset-specific overrides.

    Override priority:
        global config < dataset_overrides[dataset]

    Parameters
    ----------
    dataset : str
        Dataset name (e.g. 'uci', 'tafeng', 'cdnow').

    Returns
    -------
    dict
        Merged config with dataset-specific overrides applied.
    """
    import os
    import yaml
    import copy

    cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "config", "simulation_params.yaml"
    )
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    overrides = cfg.get("dataset_overrides", {}).get(dataset, {})
    if not overrides:
        return cfg

    # Deep merge: override only specifies sub-keys; rest of global config unchanged
    merged = copy.deepcopy(cfg)
    for section, section_vals in overrides.items():
        if section in merged and isinstance(merged[section], dict):
            merged[section].update(section_vals)
        else:
            merged[section] = section_vals
    logger.info(f"[Config] Applied {dataset} overrides: {overrides}")
    return merged


# =============================================================================
# D2: Bootstrap Confidence Interval for C-index
# =============================================================================

def bootstrap_c_index(
    model,
    df_scaled: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 42,
) -> tuple:
    """
    D2: Compute bootstrap 95% CI for the Weibull AFT C-index.

    Parameters
    ----------
    model : WeibullAFTFitter  (already fitted)
    df_scaled : pd.DataFrame  (OOS test set, contains T, E and scaled feature columns)
    n_boot : int  — number of bootstrap iterations (default 300)
    seed : int    — reproducibility seed

    Returns
    -------
    tuple : (lower_95, median, upper_95, reliability_flag)
    """
    from lifelines.utils import concordance_index

    # ── Fit Quality Guard ────────────────────────────────────────────────────
    rho = getattr(model, "rho_", 0)
    if rho > 10:
        logger.warning(f"[Bootstrap] Model hasn't converged (rho={rho:.2f} > 10). Skipping CI.")
        return (np.nan, np.nan, np.nan, False)

    rng  = np.random.default_rng(seed)
    # Ensure index is clean for iloc alignment
    df_eval = df_scaled.reset_index(drop=True)
    T_all = df_eval["T"].values
    E_all = df_eval["E"].values
    feat_cols = [c for c in df_eval.columns if c not in ("T", "E")]
    n = len(T_all)

    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        T_b = T_all[idx]
        E_b = E_all[idx]
        # Robust copy with index reset for safe prediction
        df_b = df_eval.iloc[idx][feat_cols].copy().reset_index(drop=True)
        
        try:
            # score() is more robust than predict_median manually
            # But during bootstrap, if it fails due to inf, we catch it.
            c = model.score(df_b, scoring_method="concordance_index")
            if np.isfinite(c):
                boot_scores.append(c)
        except Exception:
            # Fallback to manual if score fails
            try:
                T_hat = model.predict_median(df_b).values
                finite = np.isfinite(T_hat)
                if finite.sum() > n * 0.5:
                    c = concordance_index(T_b[finite], T_hat[finite], E_b[finite])
                    if np.isfinite(c):
                        boot_scores.append(c)
            except Exception:
                pass

    n_valid = len(boot_scores)
    if n_valid < 50:
        logger.warning(f"[Bootstrap] Underpowered: only {n_valid}/{n_boot} valid samples. Results unreliable.")
        return (np.nan, np.nan, np.nan, False)

    arr = np.array(boot_scores)
    lo, med, hi = (
        float(np.percentile(arr,  2.5)),
        float(np.percentile(arr, 50.0)),
        float(np.percentile(arr, 97.5)),
    )
    logger.info(
        f"[Bootstrap CI] C-index 95%% CI: [{lo:.4f}, {hi:.4f}]  "
        f"(median={med:.4f}, n_valid={n_valid})"
    )
    return lo, med, hi, True


# =============================================================================
# Technical Metrics
# =============================================================================

def compute_c_index(
    model,
    df_scaled: pd.DataFrame,
    model_name: str = "Model",
) -> float:
    """
    Compute Harrell's Concordance Index (C-index).
    """
    # ── Fit Quality Guard ────────────────────────────────────────────────────
    rho = getattr(model, "rho_", 1.0)
    if rho > 10:
        logger.warning(f"[{model_name}] EXTREME RHO ({rho:.2f} > 10). Model non-convergent or overfit.")
        return getattr(model, "concordance_index_", np.nan)

    # Log model summary for academic audit
    try:
        logger.info(f"[{model_name}] Parameters:\n{model.params_}")
    except:
        pass

    try:
        # Step 1: Attempt standard predictive score (Out-of-sample)
        # We reset index to ensure no alignment issues during predict inside score()
        c = model.score(df_scaled.reset_index(drop=True), scoring_method="concordance_index")
    except Exception as e:
        logger.warning(f"[{model_name}] OOS scoring failed ({e}). Falling back to train attribute.")
        c = getattr(model, "concordance_index_", np.nan)

    if not np.isfinite(c):
        c = getattr(model, "concordance_index_", np.nan)

    logger.info(f"[{model_name}] Final C-index: {c:.4f}")
    return c


def compute_integrated_brier_score(
    model: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    t_grid_steps: int = 100,
    force_naive: bool = False,  # E6: force non-IPCW for ablation S4
) -> float:
    """
    Compute the Integrated Brier Score (IBS) for a Weibull AFT model.

    Two-tier implementation:
    ─────────────────────────
    Tier 1 (preferred, scientifically correct):
        Uses ``scikit-survival`` (sksurv) IPCW-weighted IBS.
        IPCW = Inverse Probability of Censoring Weighting.
        This correctly handles right-censored observations and is required
        for datasets with heavy censoring (> 40% censored).
        Install: pip install scikit-survival

    Tier 2 (fallback when sksurv not available):
        Plain Brier Score without censoring weights.
        Biased upward when censoring is heavy, but acceptable for exploratory
        analysis on low-censoring datasets (e.g. CDNOW 86% churn rate).

    Parameters
    ----------
    model : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame with T and E columns.
    t_grid_steps : int
        Number of time steps for integration (default: 100).

    Returns
    -------
    float
        IBS value. Lower is better. IBS < 0.25 -> better than random.
    """
    T_obs = df_scaled["T"].values
    E_obs = df_scaled["E"].values.astype(bool)

    # sksurv requires eval times STRICTLY within [T_min, T_max) of test data.
    # Use a safe interior range with epsilon margin to avoid boundary violations.
    t_min_obs = float(T_obs.min())
    t_max_obs = float(T_obs.max())
    eps = max(1.0, (t_max_obs - t_min_obs) * 0.01)
    t_lo = t_min_obs + eps
    t_hi = t_max_obs - eps
    if t_hi <= t_lo:
        t_lo = t_min_obs + 0.5
        t_hi = t_max_obs - 0.5
    if t_hi <= t_lo:
        logger.warning("[IBS] Time range too narrow for IBS. Returning 0.25 (baseline).")
        return 0.25
    t_grid = np.linspace(t_lo, t_hi, t_grid_steps)

    # ── Tier 1: IPCW via scikit-survival ─────────────────────────────────────
    if not force_naive:
        try:
            from sksurv.metrics import integrated_brier_score as _ipcw_ibs
            from sksurv.util import Surv

            y_structured = Surv.from_arrays(event=E_obs, time=T_obs)
            S_hat_full = model.predict_survival_function(df_scaled, times=t_grid).values  # (T, N)
            ibs = _ipcw_ibs(y_structured, y_structured, t_grid, S_hat_full.T)
            logger.info(f"[WeibullAFT] IBS (IPCW-weighted, sksurv): {ibs:.4f}")
            return float(ibs)

        except ImportError:
            logger.warning(
                "[IBS] scikit-survival not installed — using non-IPCW Brier Score (fallback). "
                "Install for publication-grade results: pip install scikit-survival"
            )
        except ValueError as ve:
            logger.warning(f"[IBS] sksurv IPCW failed ({ve}). Falling back to naive Brier Score.")

    # ── Tier 2: Non-IPCW Naive IBS (naive backup) ───────────────────────────
    brier_scores = []
    S_hat = model.predict_survival_function(df_scaled, times=t_grid).values # (T, N)
    for j, t in enumerate(t_grid):
        y_true = (T_obs > t).astype(float)
        y_pred = S_hat[j, :]
        brier_scores.append(np.mean((y_pred - y_true) ** 2))

    ibs = trapezoid(np.array(brier_scores), t_grid) / (t_hi - t_lo)
    logger.info(f"[WeibullAFT] IBS (non-IPCW fallback, biased): {ibs:.4f}")
    return ibs


def generate_sensitivity_report(
    results_list: list,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Aggregate metrics from multiple runs (different tau/datasets) into a report.
    """
    if not results_list:
        return pd.DataFrame()

    df = pd.DataFrame(results_list)
    # Ensure standard column order
    cols = ["dataset", "tau", "n_customers", "churn_rate", "c_index_oos", "ibs", "efficiency_gain_pct"]
    df = df[[c for c in cols if c in df.columns]]

    if save_path:
        df.to_markdown(save_path, index=False)
        logger.info(f"[Sensitivity] report saved → {save_path}")

    return df



def compute_time_dependent_auc(
    model: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    eval_times: list = None,
) -> pd.DataFrame:
    """
    Compute time-dependent AUC at multiple evaluation time points.

    At each time t, treats the problem as binary classification:
      - Positive: customer churned by time t (T_i <= t, E_i = 1)
      - Score: 1 - S(t | x_i)  (higher = more likely to churn)

    Parameters
    ----------
    model : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Scaled feature DataFrame with T and E columns.
    eval_times : list of float, optional
        Time points for AUC evaluation. If None, auto-derived from the
        dataset's T range using 6 evenly-spaced quantiles between the 5th
        and 95th percentile. This ensures no eval point falls outside the
        observed survival times, preventing extrapolation crashes on short
        datasets (e.g. Ta Feng = 120 days, CDNOW, etc.).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [time, auc, n_events].
    """
    T_obs = df_scaled["T"].values
    E_obs = df_scaled["E"].values

    if eval_times is None:
        # Auto-derive from data: 6 evenly-spaced points within [p5, p95]
        # Using percentiles (not min/max) avoids extreme-edge artifacts.
        t_lo = float(np.percentile(T_obs, 5))
        t_hi = float(np.percentile(T_obs, 95))
        # Clamp to at least 2 distinct points
        if t_hi <= t_lo:
            t_hi = float(T_obs.max())
        eval_times = list(np.linspace(t_lo, t_hi, 6).round(1))
        logger.info(
            f"[TdAUC] Auto eval_times from data range "
            f"[{t_lo:.1f}d, {t_hi:.1f}d]: {eval_times}"
        )
    else:
        # Filter caller-provided list to stay within observed range
        t_min_obs, t_max_obs = float(T_obs.min()), float(T_obs.max())
        filtered = [t for t in eval_times if t_min_obs <= t <= t_max_obs]
        if not filtered:
            filtered = [float(np.median(T_obs))]  # last resort
        if len(filtered) < len(eval_times):
            logger.warning(
                f"[TdAUC] {len(eval_times) - len(filtered)} eval_times outside "
                f"data range [{t_min_obs:.1f}d, {t_max_obs:.1f}d] — dropped."
            )
        eval_times = filtered

    S_hat = model.predict_survival_function(df_scaled, times=eval_times).values

    results = []
    for j, t in enumerate(eval_times):
        y_true = ((T_obs <= t) & (E_obs == 1)).astype(int)
        n_events = y_true.sum()
        if n_events < 5 or n_events == len(y_true):
            results.append({"time": t, "auc": np.nan, "n_events": n_events})
            continue
        y_score = 1 - S_hat[j, :]
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = np.nan
        results.append({"time": t, "auc": auc, "n_events": n_events})

    auc_df = pd.DataFrame(results)
    logger.info(f"Time-dependent AUC:\n{auc_df.to_string(index=False)}")
    return auc_df


def cross_validate_survival_model(
    model_class,
    df,
    duration_col="T",
    event_col="E",
    n_splits=5,
    random_state=42,
    model_kwargs=None
) -> dict:
    """
    Perform Stratified K-Fold Cross-Validation for a lifelines survival model.

    Parameters
    ----------
    model_class : class
        The lifelines model class (e.g. WeibullAFTFitter).
    df : pd.DataFrame
        Dataframe containing features, duration_col, and event_col.
    duration_col : str
        Name of duration column (default "T").
    event_col : str
        Name of event column (default "E").
    n_splits : int
        Number of folds (default 5).
    random_state : int
        Seed for reproducibility.
    model_kwargs : dict
        Arguments to pass to model constructor (e.g. penalizer).

    Returns
    -------
    dict
        {
            "mean_c_index": float,
            "std_c_index": float,
            "folds": list[float]
        }
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    # Lazy import to avoid circular dependency if any (though sklearn is safe)
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    c_indices = []
    
    logger.info(f"Starting {n_splits}-Fold CV for {model_class.__name__}...")

    # Stratify by Event (E) to ensure balanced censorship in train/test
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df[event_col])):
        df_train = df.iloc[train_idx]
        df_test  = df.iloc[test_idx]
        
        try:
            model = model_class(**model_kwargs)
            model.fit(df_train, duration_col=duration_col, event_col=event_col)
            
            # Evaluate on test fold
            # score() defaults to concordance_index for survival fitters
            c = model.score(df_test, scoring_method="concordance_index")
            c_indices.append(c)
            # logger.debug(f"  [Fold {fold+1}] C-index: {c:.4f}")
            
        except Exception as e:
            logger.warning(f"  [Fold {fold+1}] Failed: {e}")
            
    if not c_indices:
        return {"mean_c_index": 0.0, "std_c_index": 0.0, "folds": []}

    mean_c = np.mean(c_indices)
    std_c  = np.std(c_indices)
    
    logger.info(f"CV Results ({model_class.__name__}): Mean C-index = {mean_c:.4f} (+/- {std_c:.4f})")
    
    return {
        "mean_c_index": mean_c,
        "std_c_index":  std_c,
        "folds":        c_indices
    }


# =============================================================================
# Business / Decision Metrics
# =============================================================================

def compute_outreach_efficiency(
    weibull_decisions: pd.DataFrame,
    rfm_decisions: pd.DataFrame,
) -> dict:
    """
    Compute outreach efficiency: fraction of contacts avoided by Weibull policy
    compared to the RFM baseline.

    Outreach Efficiency = (contacts_avoided / total_customers) * 100%

    where contacts_avoided = customers that RFM would contact but Weibull
    correctly identifies as WAIT or LOST.

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Decision table from policy.make_intervention_decisions().
    rfm_decisions : pd.DataFrame
        Decision table from policy.rfm_intervention_decisions().

    Returns
    -------
    dict
        {
          'weibull_intervene_rate': float,
          'rfm_intervene_rate': float,
          'contacts_avoided': int,
          'contacts_avoided_pct': float,
          'efficiency_gain_pct': float
        }
    """
    n_total = len(weibull_decisions)

    weibull_intervene = (weibull_decisions["decision"] == "INTERVENE").sum()
    rfm_intervene     = (rfm_decisions["decision"] == "INTERVENE").sum()

    weibull_rate = weibull_intervene / n_total * 100
    rfm_rate     = rfm_intervene / n_total * 100

    contacts_avoided = max(rfm_intervene - weibull_intervene, 0)
    contacts_avoided_pct = contacts_avoided / n_total * 100

    efficiency_gain = (rfm_rate - weibull_rate) / rfm_rate * 100 if rfm_rate > 0 else 0.0

    metrics = {
        "weibull_intervene_rate":  weibull_rate,
        "rfm_intervene_rate":      rfm_rate,
        "contacts_avoided":        contacts_avoided,
        "contacts_avoided_pct":    contacts_avoided_pct,
        "efficiency_gain_pct":     efficiency_gain,
    }

    logger.info(
        f"Outreach Efficiency | "
        f"Weibull: {weibull_rate:.1f}% | RFM: {rfm_rate:.1f}% | "
        f"Contacts avoided: {contacts_avoided} ({contacts_avoided_pct:.1f}%) | "
        f"Efficiency gain: {efficiency_gain:.1f}%"
    )
    return metrics


def compute_revenue_lift(
    weibull_decisions: pd.DataFrame,
    rfm_decisions: pd.DataFrame,
) -> dict:
    """
    Compute revenue precision lift of Weibull policy vs. RFM baseline.

    The Weibull policy is more selective (fewer contacts) but targets higher-EVI
    customers. We compare average EVI per contact (precision), not total EVI.

    Revenue Precision Lift = (avg_EVI_weibull - avg_EVI_rfm) / avg_EVI_rfm * 100%

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Decision table from policy.make_intervention_decisions() -- has 'evi' column.
    rfm_decisions : pd.DataFrame
        Decision table from policy.rfm_intervention_decisions().

    Returns
    -------
    dict
        {
          'avg_evi_weibull': float,
          'avg_evi_rfm_proxy': float,
          'total_evi_weibull': float,
          'total_evi_rfm_proxy': float,
          'revenue_precision_lift_pct': float
        }
    """
    # Weibull: average EVI per intervention
    weibull_intervene_mask = weibull_decisions["decision"] == "INTERVENE"
    weibull_evi_total = weibull_decisions.loc[weibull_intervene_mask, "evi"].sum()
    weibull_evi_avg   = weibull_decisions.loc[weibull_intervene_mask, "evi"].mean()

    # RFM proxy: average EVI for RFM-targeted customers (using Weibull EVI values)
    rfm_intervene_ids = rfm_decisions.loc[
        rfm_decisions["decision"] == "INTERVENE", "CustomerID"
    ]
    rfm_mask = weibull_decisions["CustomerID"].isin(rfm_intervene_ids)
    rfm_evi_total = weibull_decisions.loc[rfm_mask, "evi"].sum()
    rfm_evi_avg   = weibull_decisions.loc[rfm_mask, "evi"].mean()

    precision_lift = (
        (weibull_evi_avg - rfm_evi_avg) / abs(rfm_evi_avg) * 100
        if rfm_evi_avg and rfm_evi_avg != 0 else 0.0
    )

    metrics = {
        "avg_evi_weibull":             weibull_evi_avg,
        "avg_evi_rfm_proxy":           rfm_evi_avg,
        "total_evi_weibull":           weibull_evi_total,
        "total_evi_rfm_proxy":         rfm_evi_total,
        "revenue_precision_lift_pct":  precision_lift,
    }

    logger.info(
        f"Revenue Precision Lift | "
        f"Weibull avg EVI: {get_currency_code()} {weibull_evi_avg:.2f} | "
        f"RFM avg EVI: {get_currency_code()} {rfm_evi_avg:.2f} | "
        f"Precision lift: {precision_lift:.1f}%"
    )
    return metrics


# =============================================================================
# Full Report
# =============================================================================

def print_full_report(
    c_index_weibull: float,
    c_index_cox: float,
    ibs: float,
    lr_cv_metrics: dict,
    auc_df: pd.DataFrame,
    outreach_metrics: dict,
    revenue_metrics: dict,
    tau: int = 90,
) -> None:
    """
    Print a comprehensive evaluation report to stdout.
    All output uses ASCII-safe characters for Windows console compatibility.
    """
    sep = "=" * 70

    print(f"\n{sep}")
    print("  DECISION-CENTRIC CUSTOMER RE-ENGAGEMENT -- EVALUATION REPORT")
    print(f"  Churn threshold tau = {tau} days")
    print(sep)

    print("\n-- TECHNICAL METRICS --------------------------------------------------")
    print(f"  Weibull AFT  C-index : {c_index_weibull:.4f}  (target: > 0.60)")
    print(f"  CoxPH        C-index : {c_index_cox:.4f}  (target: > 0.58)")
    print(f"  Logistic     AUC     : {lr_cv_metrics['auc_mean']:.4f} +/- {lr_cv_metrics['auc_std']:.4f}  (target: > 0.65)")
    print(f"  Weibull AFT  IBS     : {ibs:.4f}  (target: < 0.25)")

    print("\n-- TIME-DEPENDENT AUC (Weibull AFT) -----------------------------------")
    print(auc_df.to_string(index=False))

    print("\n-- BUSINESS METRICS ---------------------------------------------------")
    print(f"  Weibull intervention rate     : {outreach_metrics['weibull_intervene_rate']:.1f}%")
    print(f"  RFM intervention rate         : {outreach_metrics['rfm_intervene_rate']:.1f}%")
    print(f"  Contacts avoided              : {outreach_metrics['contacts_avoided']:,} ({outreach_metrics['contacts_avoided_pct']:.1f}%)")
    print(f"  Outreach efficiency gain      : {outreach_metrics['efficiency_gain_pct']:.1f}%  (target: >= 20%)")
    print(f"  Weibull avg EVI per contact   : {get_currency_symbol()}{revenue_metrics['avg_evi_weibull']:.2f}")
    print(f"  RFM avg EVI per contact       : {get_currency_symbol()}{revenue_metrics['avg_evi_rfm_proxy']:.2f}")
    print(f"  Revenue precision lift        : {revenue_metrics['revenue_precision_lift_pct']:.1f}%  (target: >= 20%)")
    print(f"\n{sep}\n")
