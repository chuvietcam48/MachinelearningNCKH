"""
src/models.py
=============
Implements and trains all four models in the Decision-Centric framework:

  1. WeibullAFTFitter  (lifelines) -- Primary survival model
     * Grid search penalizer via k-fold cross-validation (PHASE 4)
  2. CoxPHFitter       (lifelines) -- Semi-parametric baseline
  3. LogisticRegression (sklearn)  -- Binary classification baseline
  4. RFM Segmentation  (custom)    -- Heuristic quintile-based baseline

Scientific safeguards (PHASE 4 additions):
  _check_vif()     : VIF > 5 -> drop the worst collinear feature (non-destructive).
  train_weibull_aft: Grid search over penalizer in [0.0, 0.01, 0.1, 1.0] using
                     lifelines 3-fold k_fold_cross_validation; selects by C-index.

Mathematical background:
  Weibull AFT: log(T) = beta'x + sigma*epsilon,  epsilon ~ Gumbel(0,1)
    S(t|x) = exp(-(t/lambda(x))^rho),  lambda(x) = exp(beta'x),  rho = 1/sigma
    h(t|x) = (rho/lambda(x)) * (t/lambda(x))^(rho-1)

  CoxPH: h(t|x) = h0(t) * exp(beta'x)   [partial likelihood, no dist. assumption]

  Logistic: P(E=1 | x) = sigma(beta'x)   [binary horizon-based label]
    NOTE: Recency is EXCLUDED to prevent data leakage (E = Recency > tau)

  RFM Score: Quintile rank on Recency (inverted), Frequency, Monetary
"""

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lifelines import WeibullAFTFitter, CoxPHFitter
from lifelines.utils import k_fold_cross_validation
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor as _vif_func
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Feature columns used by survival models (Weibull AFT, CoxPH)
# These are used as a fallback / for save_artifacts metadata.
# Actual model training uses get_survival_features(customer_df) which
# auto-discovers numeric columns from the data — adding a new feature to
# feature_engine.py will automatically include it in models without any
# manual list update.
SURVIVAL_FEATURES = [
    "Recency", "Frequency", "Monetary",
    "InterPurchaseTime", "GapDeviation", "SinglePurchase",
]

# Columns that are never input features regardless of what's in the DataFrame
# BUG-6: future_spend is the CLV regression TARGET — must never leak into survival features
_NON_FEATURE_COLS = {"T", "E", "CustomerID", "future_spend"}


def get_survival_features(customer_df: pd.DataFrame) -> list:
    """
    Auto-discover survival model feature columns from a customer DataFrame.

    Rules:
      - Must be numeric (int or float)
      - Must NOT be a survival target column: T, E
      - Must NOT be an identifier: CustomerID
      - Must NOT contain all-NaN values

    Falls back to the static ``SURVIVAL_FEATURES`` list if no numeric columns
    are found (e.g. in tests with minimal synthetic data).

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level feature DataFrame (output of build_customer_features).

    Returns
    -------
    list of str
        Feature column names to use for model training.
    """
    numeric_cols = customer_df.select_dtypes(include=[np.number]).columns.tolist()
    features = [
        c for c in numeric_cols
        if c not in _NON_FEATURE_COLS
        and customer_df[c].notna().any()
    ]
    if not features:
        logger.warning(
            "[get_survival_features] No numeric features discovered — "
            "falling back to static SURVIVAL_FEATURES list."
        )
        return [f for f in SURVIVAL_FEATURES if f in customer_df.columns]
    logger.info(f"[get_survival_features] Auto-discovered {len(features)} features: {features}")
    return features


# Feature columns for Logistic Regression (Recency EXCLUDED to prevent leakage)
# E = (Recency > tau) => including Recency gives AUC = 1.0 trivially
LOGISTIC_FEATURES = [
    "Frequency", "Monetary",
    "InterPurchaseTime", "GapDeviation", "SinglePurchase",
]



def _get_preprocessor() -> Pipeline:
    """Return a sklearn preprocessing pipeline (impute -> scale)."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


# =============================================================================
# VIF Multicollinearity Guard (Phase 4)
# =============================================================================

def _check_vif(
    df_scaled: pd.DataFrame,
    feature_cols: list,
    vif_threshold: float = 5.0,
) -> list:
    """
    Recursively compute Variance Inflation Factor (VIF) and drop features
    until all remaining VIFs are <= *vif_threshold*.

    Rationale
    ---------
    VIF > 5 signals severe multicollinearity, which inflates coefficient
    variance, causes numerical instability in the Weibull log-likelihood, and
    leads to misleadingly high concordance on in-sample predictions.

    Strategy: recursive drop-one.  Each iteration drops only the single worst
    offender, then recomputes VIF on the remaining features.  This is more
    conservative than dropping all violators at once, keeps changes auditable,
    and avoids over-pruning induced by masking effects.
    The global SURVIVAL_FEATURES list is *never* mutated — the pruned list is
    returned and used for this run only.

    Parameters
    ----------
    df_scaled : pd.DataFrame
        Pre-scaled feature matrix (columns = feature_cols).
    feature_cols : list of str
        Features to evaluate. Must all be columns in df_scaled.
    vif_threshold : float
        VIF cutoff above which a feature is flagged (default: 5.0).

    Returns
    -------
    list of str
        Pruned feature list (same as input if no VIF violation found).
    """
    if not _STATSMODELS_AVAILABLE:
        logger.warning(
            "[VIF] statsmodels not installed — skipping multicollinearity check. "
            "Install with: pip install statsmodels"
        )
        return list(feature_cols)

    cols = list(feature_cols)
    iteration = 0

    while True:
        if len(cols) < 2:
            logger.info("[VIF] Only one feature remaining — stopping VIF loop.")
            break

        X = df_scaled[cols].values.astype(float)

        # Compute VIF for all current features
        vif_scores = {}
        for i, col in enumerate(cols):
            try:
                score = _vif_func(X, i)
            except Exception:
                score = np.nan
            vif_scores[col] = score

        logger.info(f"[VIF] Iteration {iteration} — Variance Inflation Factors:")
        for col, score in vif_scores.items():
            flag = " <-- HIGH" if (not np.isnan(score) and score > vif_threshold) else ""
            logger.info(f"  {col:25s}: {score:8.3f}{flag}")

        # Find the worst offender among valid (non-NaN) scores
        valid_scores = {k: v for k, v in vif_scores.items() if not np.isnan(v)}
        if not valid_scores:
            logger.info("[VIF] All VIF scores are NaN — stopping loop.")
            break

        max_col = max(valid_scores, key=lambda k: valid_scores[k])
        max_vif  = valid_scores[max_col]

        if max_vif <= vif_threshold:
            logger.info(
                f"[VIF] All VIFs <= {vif_threshold} (max={max_vif:.2f} for '{max_col}'). "
                f"Final feature set ({len(cols)}): {cols}"
            )
            break

        # Drop the worst offender and loop again
        logger.warning(
            f"[VIF] Iteration {iteration}: '{max_col}' has VIF={max_vif:.2f} "
            f"(threshold={vif_threshold}). Dropping and re-checking."
        )
        cols.remove(max_col)
        logger.warning(f"[VIF] Remaining features ({len(cols)}): {cols}")
        iteration += 1

    return cols


# =============================================================================
# 1. Weibull AFT Model  (Phase 4: VIF guard + penalizer grid search)
# =============================================================================

# Default penalizer candidates for cross-validated grid search
_WEIBULL_PENALIZER_GRID = [0.0, 0.01, 0.1, 1.0]
_WEIBULL_CV_FOLDS       = 3


def train_weibull_aft(
    customer_df: pd.DataFrame,
    penalizer: float = 0.01,
    penalizer_grid: list = None,
    vif_threshold: float = 5.0,  # E6: pass inf to disable VIF guard
) -> tuple:
    """
    Fit a Weibull Accelerated Failure Time model with scientific safeguards.

    Phase 4 additions
    -----------------
    1. VIF check  : Before training, computes VIF for all SURVIVAL_FEATURES.
                    If any feature has VIF > 5.0, the single worst offender is
                    dropped for this run (SURVIVAL_FEATURES is not mutated).

    2. Grid search: Iterates over *penalizer_grid* (default [0.0, 0.01, 0.1, 1.0])
                    using lifelines 3-fold k_fold_cross_validation. Selects the
                    penalizer with the highest mean concordance and retrains the
                    final model on the full dataset.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with features + T + E columns.
    penalizer : float
        Fallback penalizer used if grid search fails (default: 0.01).
    penalizer_grid : list of float, optional
        Candidate penalizer values. Defaults to [0.0, 0.01, 0.1, 1.0].

    Returns
    -------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model (retrained on full data with optimal penalizer).
    df_scaled : pd.DataFrame
        Scaled feature DataFrame used for fitting (preserves T, E).
    preprocessor : Pipeline
        Fitted sklearn preprocessing pipeline.
    """
    logger.info("Training Weibull AFT model (Phase 4: VIF + grid search)...")

    if penalizer_grid is None:
        penalizer_grid = _WEIBULL_PENALIZER_GRID

    # ── B2: Auto-discover features from data ─────────────────────────────────
    # This replaces the hardcoded SURVIVAL_FEATURES reference so that any new
    # feature added to feature_engine.py is automatically picked up here.
    input_features = get_survival_features(customer_df)

    # ── Scale features ───────────────────────────────────────────────────────
    preprocessor = _get_preprocessor()
    X_scaled = preprocessor.fit_transform(customer_df[input_features])
    df_scaled_full = pd.DataFrame(
        X_scaled, columns=input_features, index=customer_df.index
    )
    df_scaled_full["T"] = customer_df["T"].values
    df_scaled_full["E"] = customer_df["E"].values

    # ── Phase 4 Task 2: VIF multicollinearity check ───────────────────────────
    active_features = _check_vif(df_scaled_full, input_features, vif_threshold=vif_threshold)

    # Rebuild df_scaled with only active features + T, E (drop pruned cols)
    df_scaled = df_scaled_full[active_features + ["T", "E"]].copy()

    # ── Phase 4 Task 3: Grid search via k-fold cross-validation ──────────────
    logger.info(
        f"[GridSearch] Testing penalizers {penalizer_grid} "
        f"with {_WEIBULL_CV_FOLDS}-fold CV..."
    )

    best_penalizer  = penalizer  # fallback
    best_cv_score   = -np.inf
    grid_results    = {}

    for p in penalizer_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = k_fold_cross_validation(
                    WeibullAFTFitter(penalizer=p),
                    df_scaled,
                    duration_col="T",
                    event_col="E",
                    k=_WEIBULL_CV_FOLDS,
                    scoring_method="concordance_index",
                )
            mean_score = np.mean(scores)
            grid_results[p] = mean_score
            logger.info(f"  penalizer={p:<6} -> CV C-index: {mean_score:.4f}")

            if mean_score > best_cv_score:
                best_cv_score  = mean_score
                best_penalizer = p

        except Exception as exc:
            logger.warning(f"  penalizer={p} -> CV failed: {exc}")
            grid_results[p] = np.nan

    logger.info(
        f"[GridSearch] *** Optimal penalizer: {best_penalizer} "
        f"| CV C-index: {best_cv_score:.4f} ***"
    )

    # ── Final model: retrain on full dataset with optimal penalizer ───────────
    waf = WeibullAFTFitter(penalizer=best_penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waf.fit(df_scaled, duration_col="T", event_col="E")

    rho_val = waf.params_["rho_"]["Intercept"]
    logger.info(f"Weibull AFT fitted | rho = {rho_val:.4f}")
    logger.info(f"  Shape param rho > 1 -> increasing hazard over time: {rho_val > 1}")
    logger.info(f"  Active features ({len(active_features)}): {active_features}")

    return waf, df_scaled, preprocessor, active_features


# =============================================================================
# 2. Cox Proportional Hazards Model
# =============================================================================

def train_coxph(
    customer_df: pd.DataFrame,
    penalizer: float = 0.1,
    penalizer_grid: list = None,
    check_assumptions: bool = True,
) -> tuple:
    """
    Fit a Cox Proportional Hazards model with Phase 4/5 scientific safeguards.

    Additions (mirroring Weibull AFT)
    ----------------------------------
    1. VIF check   : Drops the worst collinear feature if VIF > 5.0.
    2. Grid search : Tests penalizer in [0.01, 0.1, 0.5, 1.0] via 3-fold
                     k_fold_cross_validation; selects by concordance index.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with features + T + E columns.
    penalizer : float
        Fallback penalizer if grid search fails (default: 0.1).
    penalizer_grid : list of float, optional
        Candidate penalizers. Defaults to [0.01, 0.1, 0.5, 1.0].
        (Higher values than Weibull grid -- CoxPH log-partial likelihood
        benefits from stronger regularization when features are correlated.)
    check_assumptions : bool
        If True, run Schoenfeld residuals test for PH assumption.

    Returns
    -------
    cph : CoxPHFitter
        Fitted CoxPH model (retrained on full data with optimal penalizer).
    df_scaled : pd.DataFrame
        Scaled feature DataFrame (VIF-pruned if applicable) with T, E columns.
    preprocessor : Pipeline
        Fitted sklearn preprocessing pipeline.
    active_features : list of str
        Features used for training (may be shorter than SURVIVAL_FEATURES if
        VIF pruned one).
    """
    logger.info("Training Cox Proportional Hazards model (Phase 5: VIF + grid search)...")

    if penalizer_grid is None:
        penalizer_grid = [0.01, 0.1, 0.5, 1.0]

    # ── B2: Auto-discover features from data ─────────────────────────────────
    input_features = get_survival_features(customer_df)

    # ── Scale features ────────────────────────────────────────────────────────
    preprocessor = _get_preprocessor()
    X_scaled = preprocessor.fit_transform(customer_df[input_features])
    df_scaled_full = pd.DataFrame(X_scaled, columns=input_features, index=customer_df.index)
    df_scaled_full["T"] = customer_df["T"].values
    df_scaled_full["E"] = customer_df["E"].values

    # ── VIF multicollinearity check ───────────────────────────────────────────
    active_features = _check_vif(df_scaled_full, input_features)
    df_scaled = df_scaled_full[active_features + ["T", "E"]].copy()

    # ── Grid search via k-fold cross-validation ───────────────────────────────
    logger.info(
        f"[CoxPH GridSearch] Testing penalizers {penalizer_grid} "
        f"with {_WEIBULL_CV_FOLDS}-fold CV..."
    )

    best_penalizer = penalizer
    best_cv_score  = -np.inf

    for p in penalizer_grid:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = k_fold_cross_validation(
                    CoxPHFitter(penalizer=p),
                    df_scaled,
                    duration_col="T",
                    event_col="E",
                    k=_WEIBULL_CV_FOLDS,
                    scoring_method="concordance_index",
                )
            mean_score = np.mean(scores)
            logger.info(f"  penalizer={p:<6} -> CV C-index: {mean_score:.4f}")
            if mean_score > best_cv_score:
                best_cv_score  = mean_score
                best_penalizer = p
        except Exception as exc:
            logger.warning(f"  penalizer={p} -> CV failed: {exc}")

    logger.info(
        f"[CoxPH GridSearch] *** Optimal penalizer: {best_penalizer} "
        f"| CV C-index: {best_cv_score:.4f} ***"
    )

    # ── Final model: retrain on full dataset with optimal penalizer ───────────
    cph = CoxPHFitter(penalizer=best_penalizer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cph.fit(df_scaled, duration_col="T", event_col="E", show_progress=False)

    logger.info("CoxPH fitted.")
    logger.info(f"\n{cph.summary[['coef', 'exp(coef)', 'p']].to_string()}")
    logger.info(f"  Active features ({len(active_features)}): {active_features}")

    if check_assumptions:
        logger.info("Running Schoenfeld residuals test (PH assumption check)...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cph.check_assumptions(df_scaled, p_value_threshold=0.05, show_plots=False)
        except Exception as e:
            logger.warning(f"Assumption check raised: {e}")

    return cph, df_scaled, preprocessor, active_features


# =============================================================================
# 3. Logistic Regression Baseline (Data-Leakage-Free)
# =============================================================================

def train_logistic(
    customer_df: pd.DataFrame,
    cv_folds: int = 5,
) -> tuple:
    """
    Train a Logistic Regression binary classifier as a baseline.
    Target: E (1 = churned within tau days, 0 = still active).

    DATA LEAKAGE FIX:
    -----------------
    Recency is EXCLUDED from the Logistic Regression feature set.
    Rationale: The churn label E is defined as (Recency > tau), making
    Recency a tautological predictor that inflates AUC to 1.0.

    Survival models (Weibull AFT, CoxPH) are exempt from this rule because
    they model the full time-to-event distribution T, not a binary label
    derived from Recency.

    Logistic features: Frequency, Monetary, InterPurchaseTime,
                       GapStability, SinglePurchase
    (Recency deliberately excluded to prevent data leakage)

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with features + E column.
    cv_folds : int
        Number of stratified cross-validation folds (default: 5).

    Returns
    -------
    lr : LogisticRegression
        Fitted logistic regression model (on full data).
    pipeline : Pipeline
        Full preprocessing + model pipeline.
    cv_metrics : dict
        Cross-validated AUC and accuracy scores.
    """
    logger.info("Training Logistic Regression baseline (Recency excluded to prevent leakage)...")
    logger.info(f"  Logistic features: {LOGISTIC_FEATURES}")

    X = customer_df[LOGISTIC_FEATURES].values
    y = customer_df["E"].values

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    cv_acc = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    cv_metrics = {
        "auc_mean":  cv_auc.mean(),
        "auc_std":   cv_auc.std(),
        "acc_mean":  cv_acc.mean(),
        "acc_std":   cv_acc.std(),
        "features":  LOGISTIC_FEATURES,
    }

    logger.info(
        f"Logistic CV AUC (no Recency): {cv_metrics['auc_mean']:.4f} +/- {cv_metrics['auc_std']:.4f} | "
        f"Accuracy: {cv_metrics['acc_mean']:.4f} +/- {cv_metrics['acc_std']:.4f}"
    )

    # Fit on full dataset for downstream use
    pipeline.fit(X, y)
    lr = pipeline.named_steps["lr"]

    return lr, pipeline, cv_metrics


# =============================================================================
# 5.  CLV Regressor (Predictive Customer Lifetime Value)
# =============================================================================

# Features for CLV regression — mirrors LOGISTIC_FEATURES (no Recency to avoid
# leakage, since E = Recency > tau defines the churn label).
CLV_FEATURES = LOGISTIC_FEATURES  # ["Frequency", "Monetary", "InterPurchaseTime", "GapDeviation", "SinglePurchase"]


def train_clv_regressor(
    X_train: pd.DataFrame,
    y_train_spend: pd.Series,
    n_estimators: int = 100,
    max_depth: int = 5,
    random_state: int = 42,
) -> tuple:
    """
    Train a lightweight RandomForestRegressor to predict future customer spend.

    Leakage-free design
    -------------------
    The function receives pre-split **training-only** data.  The caller is
    responsible for computing predictions on the test / full set using
    ``rf_clv_pipeline.predict(X_test)`` — this function never touches test data.

    Target
    ------
    ``y_train_spend`` is the actual total spend per customer in the tau-day
    look-forward window (i.e. ``customer_df["future_spend"]`` on the train split).
    This column is computed by ``build_customer_features(..., df_raw=df_clean)``
    in feature_engine.py.

    Features
    --------
    Same as LOGISTIC_FEATURES (Frequency, Monetary, InterPurchaseTime,
    GapDeviation, SinglePurchase) — Recency is excluded to prevent leakage
    because the churn label E is derived from Recency.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training customer feature DataFrame indexed by CustomerID.
        Must contain all columns in CLV_FEATURES.
    y_train_spend : pd.Series
        Target: actual future spend per customer in the prediction window,
        indexed identically to X_train.  Zero for customers with no future
        transactions.
    n_estimators : int
        Number of trees in the forest (default: 100).
    max_depth : int
        Maximum tree depth (default: 5) — limits overfitting on sparse CLV
        distributions.
    random_state : int
        Random seed for reproducibility (default: 42).

    Returns
    -------
    rf_clv_pipeline : sklearn Pipeline
        Fitted impute -> scale -> RandomForestRegressor pipeline.
        Call ``.predict(X)`` on any feature matrix to get predicted CLV.
    predicted_clv_train : pd.Series
        In-sample CLV predictions on X_train, indexed by CustomerID.
        Useful for diagnostics; do NOT use for out-of-sample evaluation.
    """
    logger.info(
        f"[CLV] Training CLV RandomForestRegressor | "
        f"n_est={n_estimators} | max_depth={max_depth} | "
        f"n_train={len(X_train):,}"
    )

    # ── Use available CLV features from the DataFrame ─────────────────────────
    available_clv_feats = [f for f in CLV_FEATURES if f in X_train.columns]
    if not available_clv_feats:
        raise ValueError(
            f"[CLV] None of CLV_FEATURES {CLV_FEATURES} found in X_train.columns "
            f"({list(X_train.columns)}). Cannot train CLV regressor."
        )

    X = X_train[available_clv_feats].values
    y = y_train_spend.values.astype(float)

    rf_clv_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("rf",      RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])
    rf_clv_pipeline.fit(X, y)

    # ── In-sample predictions (diagnostic only) ───────────────────────────────
    y_pred_train = rf_clv_pipeline.predict(X)
    predicted_clv_train = pd.Series(y_pred_train, index=X_train.index, name="predicted_clv")

    # ── Log diagnostics ───────────────────────────────────────────────────────
    rf_model = rf_clv_pipeline.named_steps["rf"]
    feat_imp  = dict(zip(available_clv_feats, rf_model.feature_importances_))
    top_feat  = max(feat_imp, key=feat_imp.get)

    logger.info(
        f"[CLV] Training complete | "
        f"mean predicted CLV={predicted_clv_train.mean():.2f} | "
        f"mean actual spend={y.mean():.2f} | "
        f"top feature='{top_feat}' ({feat_imp[top_feat]:.3f})"
    )
    logger.info(
        f"[CLV] Feature importances: "
        + " | ".join(f"{k}={v:.3f}" for k, v in sorted(feat_imp.items(), key=lambda x: -x[1]))
    )

    # Pearson correlation between prediction and target (in-sample diagnostic)
    if len(y) > 2:
        corr = float(np.corrcoef(y, y_pred_train)[0, 1])
        logger.info(f"[CLV] In-sample correlation (predicted vs actual): r={corr:.4f}")

    return rf_clv_pipeline, predicted_clv_train


# =============================================================================
# 4. RFM Segmentation Baseline
# =============================================================================

def rfm_segment(customer_df: pd.DataFrame, n_quintiles: int = 5) -> pd.DataFrame:
    """
    Compute RFM quintile scores and assign customer segments.

    Scoring logic:
      - Recency:   lower is better -> inverted quintile (5 = most recent)
      - Frequency: higher is better -> normal quintile (5 = most frequent)
      - Monetary:  higher is better -> normal quintile (5 = highest spend)
      - RFM_Score = R_score + F_score + M_score  (range: 3-15)

    Segments (based on RFM_Score):
      - Champions : 13-15
      - Loyal     : 10-12
      - At Risk   : 7-9
      - Lost      : 3-6

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level DataFrame with Recency, Frequency, Monetary columns.
    n_quintiles : int
        Number of quantile bins (default: 5).

    Returns
    -------
    pd.DataFrame
        customer_df with added columns: R_score, F_score, M_score,
        RFM_Score, RFM_Segment, intervention_priority (1=highest).
    """
    logger.info("Computing RFM segmentation...")

    df = customer_df.copy()
    labels = list(range(1, n_quintiles + 1))

    # Recency: lower recency -> higher score (more recent = better)
    df["R_score"] = pd.qcut(
        df["Recency"], q=n_quintiles, labels=labels[::-1], duplicates="drop"
    ).astype(int)

    # Frequency: higher frequency -> higher score
    df["F_score"] = pd.qcut(
        df["Frequency"].rank(method="first"), q=n_quintiles, labels=labels, duplicates="drop"
    ).astype(int)

    # Monetary: higher spend -> higher score
    df["M_score"] = pd.qcut(
        df["Monetary"].rank(method="first"), q=n_quintiles, labels=labels, duplicates="drop"
    ).astype(int)

    df["RFM_Score"] = df["R_score"] + df["F_score"] + df["M_score"]

    def _assign_segment(score: int) -> str:
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal"
        elif score >= 7:
            return "At Risk"
        else:
            return "Lost"

    df["RFM_Segment"] = df["RFM_Score"].apply(_assign_segment)

    # Intervention priority: "At Risk" customers are the primary targets
    priority_map = {"At Risk": 1, "Lost": 2, "Loyal": 3, "Champions": 4}
    df["intervention_priority"] = df["RFM_Segment"].map(priority_map)

    segment_counts = df["RFM_Segment"].value_counts()
    logger.info(f"RFM Segments:\n{segment_counts.to_string()}")

    return df


# =============================================================================
# 5. Model Selection Justification (E3: AIC / BIC Comparison)
# =============================================================================

def compare_survival_distributions(
    customer_df: pd.DataFrame,
    penalizer: float = 0.01,
) -> dict:
    """
    E3: Compare Weibull, Log-Normal, and Log-Logistic AFT models via AIC/BIC.

    Provides a scientific justification for choosing Weibull over alternatives.

    AIC = 2k - 2 ln(L_hat)
    BIC = k ln(n) - 2 ln(L_hat)

    Where k = number of parameters, L_hat = maximum likelihood, n = sample size.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer features (output of build_customer_features).
    penalizer : float
        L2 penalizer for all models (same penalty for fair comparison).

    Returns
    -------
    dict
        Keys: model_name -> {"aic": float, "bic": float, "c_index": float, "ll": float}
        Plus "winner_aic" and "winner_bic" with the best model name.
    """
    from lifelines import LogNormalAFTFitter, LogLogisticAFTFitter

    # Prepare data (same preprocessing for fair comparison)
    prep = _get_preprocessor()
    available = [f for f in SURVIVAL_FEATURES if f in customer_df.columns]
    X = prep.fit_transform(customer_df[available])
    df_s = pd.DataFrame(X, columns=available, index=customer_df.index)

    # VIF check
    active_feats = _check_vif(df_s, available)
    df_s = df_s[active_feats].copy()
    df_s["T"] = customer_df["T"].values
    df_s["E"] = customer_df["E"].values
    df_s = df_s[(df_s["T"] > 0) & df_s["T"].notna()]

    n = len(df_s)
    results = {}

    models_to_try = [
        ("Weibull AFT",     WeibullAFTFitter),
        ("Log-Normal AFT",  LogNormalAFTFitter),
        ("Log-Logistic AFT", LogLogisticAFTFitter),
    ]

    for name, ModelClass in models_to_try:
        try:
            m = ModelClass(penalizer=penalizer)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(df_s, duration_col="T", event_col="E")

            ll = m.log_likelihood_
            k = m.params_.shape[0]
            aic = 2 * k - 2 * ll
            bic = k * np.log(n) - 2 * ll
            c = m.concordance_index_

            results[name] = {
                "aic": round(aic, 2),
                "bic": round(bic, 2),
                "c_index": round(c, 4),
                "log_likelihood": round(ll, 2),
                "n_params": k,
            }
            logger.info(
                f"[ModelSelection] {name:20s}  AIC={aic:10.2f}  BIC={bic:10.2f}  "
                f"C-index={c:.4f}  LL={ll:.2f}  k={k}"
            )
        except Exception as exc:
            logger.warning(f"[ModelSelection] {name} failed: {exc}")
            results[name] = {"aic": float("inf"), "bic": float("inf"), "error": str(exc)}

    # Determine winners
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        winner_aic = min(valid, key=lambda k: valid[k]["aic"])
        winner_bic = min(valid, key=lambda k: valid[k]["bic"])
        results["winner_aic"] = winner_aic
        results["winner_bic"] = winner_bic
        logger.info(f"[ModelSelection] ★ Winner (AIC): {winner_aic}")
        logger.info(f"[ModelSelection] ★ Winner (BIC): {winner_bic}")
        if winner_aic != winner_bic:
            logger.info(
                f"[ModelSelection] AIC and BIC disagree — BIC penalizes complexity harder. "
                f"Prefer {winner_bic} if parsimony is priority."
            )
    else:
        results["winner_aic"] = "N/A"
        results["winner_bic"] = "N/A"

    return results

