"""
main.py
=======
Pipeline Orchestrator -- Decision-Centric Customer Re-Engagement
================================================================
Runs the full end-to-end pipeline:

  1. Load & clean data  (UCI Online Retail  OR  Ta Feng Grocery)
  2. Engineer customer-level RFM + Survival features
  3. Train Weibull AFT, CoxPH, Logistic Regression, RFM models
  4. Apply intervention policy (Weibull + RFM baseline)
  5. Evaluate: C-index, IBS, AUC, Outreach Efficiency, Revenue Lift
  6. Serialize models and processed data for dashboard
  7. Generate all publication-quality figures
  8. Export decision table to CSV

Usage:
  python main.py                          # UCI dataset (default)
  python main.py --dataset uci
  python main.py --dataset tafeng
  python main.py --dataset uci   --tau 60
  python main.py --dataset tafeng --tau 120 --no-shap
  python main.py --sensitivity
"""

import os
import sys
import argparse
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Suppress verbose third-party warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.dataset_registry   import get_dataset, list_datasets
from src.feature_engine import (
    build_customer_features,
    calculate_dynamic_tau,
    sensitivity_analysis_tau,
)
from src.models         import (
    train_weibull_aft, train_coxph, train_logistic, rfm_segment,
    train_clv_regressor, CLV_FEATURES,
    SURVIVAL_FEATURES,
    compare_survival_distributions,  # E3
)
from src.simulator       import run_monte_carlo_simulation
from src.policy         import (
    make_intervention_decisions,
    rfm_intervention_decisions,
    lr_intervention_decisions,
)
from src.uplift         import run_uplift_analysis
from src.evaluation     import (
    compute_c_index,
    compute_integrated_brier_score,
    compute_time_dependent_auc,
    compute_outreach_efficiency,
    compute_revenue_lift,
    print_full_report,
    cross_validate_survival_model,
    bootstrap_c_index,          # D2
    load_config_with_overrides, # D3
)
from src.visualization  import (
    plot_kaplan_meier_by_segment,
    plot_weibull_survival_curves,
    plot_hazard_trajectories,
    plot_shap_summary,
    plot_decision_distribution,
    plot_brier_score_over_time,
    plot_calibration,           # D1
)
from src.reporter import generate_report  # D5
from src.sensitivity import sleeping_dog_sensitivity  # E1
from src.benchmark import run_benchmark              # E2
import time as _time                                 # E7

# ── New Framework Modules ─────────────────────────────────────────────────────
try:
    from src.simulation import run_advanced_simulation
    _HAS_ADVANCED_SIM = True
except Exception:
    _HAS_ADVANCED_SIM = False

try:
    from src.explainability import explain_decisions as _explain_decisions
    _HAS_EXPLAINABILITY = True
except Exception:
    _HAS_EXPLAINABILITY = False

try:
    from src.evaluation.counterfactual_evaluator import run_counterfactual_evaluation
    _HAS_COUNTERFACTUAL = True
except Exception:
    _HAS_COUNTERFACTUAL = False

try:
    from src.inference import CustomerScorer
    _HAS_SCORER = True
except Exception:
    _HAS_SCORER = False

# ── Level 1: Analysis & Extended Counterfactual ───────────────────────────────
try:
    from src.analysis.sensitivity_analysis import run_sensitivity_analysis
    _HAS_SENSITIVITY = True
except Exception:
    _HAS_SENSITIVITY = False

try:
    from src.analysis.temporal_cv import temporal_cross_validate
    _HAS_TEMPORAL_CV = True
except Exception:
    _HAS_TEMPORAL_CV = False

# ── Level 2: Business Metrics & Production Sim ───────────────────────────────
try:
    from src.metrics.business_metrics import compute_business_metrics
    _HAS_BIZ_METRICS = True
except Exception:
    _HAS_BIZ_METRICS = False

try:
    from src.simulation.production_simulator import ProductionSimulator
    _HAS_PROD_SIM = True
except Exception:
    _HAS_PROD_SIM = False

# Timestamped log so every run preserves its own file
import datetime as _dt
_RUN_TS = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_logging():
    """Configure console-only logging. File handler added in main() after parse_args."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decision-Centric Customer Re-Engagement Pipeline"
    )
    parser.add_argument(
        "--tau", type=int, default=0,
        help="Inactivity threshold (days). Set to 0 for data-driven dynamic tau (default: 0)"
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP computation (faster run for testing)"
    )
    parser.add_argument(
        "--sensitivity", action="store_true",
        help="Run sensitivity analysis across tau in {60, 90, 120}"
    )
    parser.add_argument(
        "--dataset",
        choices=[name for name, _ in list_datasets()],
        default="uci",
        help=(
            "Dataset to run. Choices auto-derived from DatasetRegistry: "
            + ", ".join(f"'{n}' ({d})" for n, d in list_datasets())
        ),
    )
    parser.add_argument(
        "--uplift", action="store_true", default=True,
        help="Run uplift modeling step (T-Learner + Qini curve). Enabled by default."
    )
    parser.add_argument(
        "--no-uplift", dest="uplift", action="store_false",
        help="Skip uplift modeling step"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow experiment tracking even if mlflow is installed"
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold cross-validation for survival models (slower)"
    )
    parser.add_argument(
        "--sensitivity-penalty", action="store_true",
        help="E1: Sweep sleeping_dog_penalty 5→50%% and plot robustness"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="E2: Run pipeline on ALL registered datasets and produce comparison table"
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help="E6: Run ablation study (5 scenarios) to prove each module's value"
    )
    return parser.parse_args()


def save_artifacts(
    waf,
    preprocessor,
    customer_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    weibull_decisions: pd.DataFrame,
    df_scaled_waf: pd.DataFrame,
    tau: int,
    models_dir: str,
    active_features_waf: list = None,
    monte_carlo_results: dict = None,
    metrics: dict = None,
) -> None:
    """
    Serialize trained models and processed data for the Streamlit dashboard.

    Saves to outputs/{DATASET}_tau{N}/models/:
      - weibull_model.pkl
      - preprocessor.pkl
      - processed_data.pkl
      - decisions.pkl
      - df_scaled.pkl
      - pipeline_meta.pkl
    """
    os.makedirs(models_dir, exist_ok=True)
    logger = logging.getLogger("main")

    # Reset index so CustomerID becomes a column (it's the index in customer_df)
    merged = customer_df.reset_index()  # CustomerID index -> column
    for col in ["R_score", "F_score", "M_score", "RFM_Score", "RFM_Segment", "intervention_priority"]:
        if col in rfm_df.columns:
            merged[col] = rfm_df[col].values

    # Add decision and EVI columns from weibull_decisions
    if "CustomerID" in weibull_decisions.columns:
        decision_map = weibull_decisions.set_index("CustomerID")["decision"].to_dict()
        evi_map      = weibull_decisions.set_index("CustomerID")["evi"].to_dict()
        merged["decision"] = merged["CustomerID"].map(decision_map)
        merged["evi"]      = merged["CustomerID"].map(evi_map)

    meta = {
        "tau":                tau,
        "survival_features":  SURVIVAL_FEATURES,
        "active_features_waf": active_features_waf,
        "n_customers":        len(customer_df),
        "churn_rate":         customer_df["E"].mean(),
        "monte_carlo_results": monte_carlo_results,
        "metrics":            metrics,
    }

    artifacts = {
        "weibull_model.pkl":  waf,
        "preprocessor.pkl":   preprocessor,
        "processed_data.pkl": merged,
        "decisions.pkl":      weibull_decisions,
        "df_scaled.pkl":      df_scaled_waf,
        "pipeline_meta.pkl":  meta,
    }

    for filename, obj in artifacts.items():
        path = os.path.join(models_dir, filename)
        joblib.dump(obj, path)
        logger.info(f"  Saved -> {path}")

    logger.info(f"All artifacts saved to {models_dir}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    setup_logging()
    logger = logging.getLogger("main")
    args = parse_args()

    # ── STEP 1: Load Data & Resolve Tau ───────────────────────────────────────
    logger.info(f"\n[STEP 1] Loading and cleaning dataset ({args.dataset.upper()})...")

    ds = get_dataset(args.dataset)
    logger.info(f"  Dataset : {ds.display}")
    logger.info(f"  Path    : {ds.data_path}")
    df_clean = ds.loader_fn(ds.data_path)
    snapshot = ds.snapshot_fn(df_clean)

    # Resolve Tau (Dynamic or Fixed)
    dataset_duration = (df_clean["InvoiceDate"].max() - df_clean["InvoiceDate"].min()).days
    tau = args.tau

    if tau == 0:
        logger.info("\n[STEP 1b] tau=0 requested. Calculating dynamic threshold...")
        tau = calculate_dynamic_tau(df_clean)
        logger.info(f"  Dynamic tau resolved to {tau} days.")
    else:
        logger.info(f"\n[STEP 1b] Using fixed threshold tau={tau} days.")

    # Auto-Sanity Check
    if tau > dataset_duration * 0.5:
        corrected_tau = max(dataset_duration // 3, 1)
        logger.warning("!" * 70)
        logger.warning(f"  CRITICAL: tau ({tau}d) exceeds 50% of dataset duration ({dataset_duration}d).")
        logger.warning(f"  AUTO-CORRECTING tau: {tau}d -> {corrected_tau}d")
        logger.warning("!" * 70)
        tau = corrected_tau
    else:
        logger.info(f"[AutoTau] tau={tau}d is safe (duration={dataset_duration}d).")

    logger.info(f"  Effective tau = {tau} days")

    # ── Dynamic Output Paths (Phase 11) ───────────────────────────────────────
    # Create isolated output directories for this specific dataset & tau configuration
    # Use resolved tau in the path name so it doesn't stay 'tau0'
    RUN_DIR     = os.path.join("outputs", f"{args.dataset.upper()}_tau{tau}")
    FIGURES_DIR = os.path.join(RUN_DIR, "figures")
    REPORTS_DIR = os.path.join(RUN_DIR, "reports")
    MODELS_DIR  = os.path.join(RUN_DIR, "models")
    LOGS_DIR    = os.path.join(RUN_DIR, "logs")
    CACHE_DIR   = os.path.join("data", "processed")

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(CACHE_DIR,   exist_ok=True)

    # Add run-specific file handler now that we know the output directory
    LOG_PATH = os.path.join(LOGS_DIR, f"pipeline_{_RUN_TS}.log")
    file_handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)

    logger.info("=" * 70)
    logger.info("  DECISION-CENTRIC CUSTOMER RE-ENGAGEMENT PIPELINE")
    logger.info(f"  Dataset = {args.dataset.upper()} | tau = {tau}d | SHAP = {not args.no_shap}")
    logger.info(f"  Output Dir = {RUN_DIR}")
    logger.info("=" * 70)

    # ── MLflow Experiment Tracking (optional) ─────────────────────────────────
    _use_mlflow = False
    try:
        if not args.no_mlflow:
            import mlflow
            mlflow.set_experiment("customer_retention_survival")
            mlflow.start_run(run_name=f"{args.dataset.upper()}_tau{tau}_{_RUN_TS}")
            mlflow.log_params({
                "dataset":   args.dataset,
                "tau":       tau,
                "shap":      not args.no_shap,
                "uplift":    args.uplift,
            })
            _use_mlflow = True
            logger.info("[MLflow] Experiment tracking started.")
    except Exception as _mlf_exc:
        pass # Silently fail if mlflow not installed or configured, handled by log below
        logger.info(f"[MLflow] Tracking disabled: {_mlf_exc}")

    # ── STEP 2: Feature Engineering (with Caching) ────────────────────────────
    logger.info(f"\n[STEP 2] Engineering customer features (tau={tau}d)...")
    
    cache_path = os.path.join(CACHE_DIR, f"{args.dataset}_tau{tau}_features.parquet")
    
    if os.path.exists(cache_path):
        logger.info(f"  [Cache Hit] Loading features from {cache_path}...")
        customer_df = pd.read_parquet(cache_path)
        logger.info(f"  Loaded {len(customer_df):,} customers from cache.")
    else:
        logger.info("  [Cache Miss] Computing features from scratch...")
        customer_df = build_customer_features(df_clean, snapshot, tau=tau, df_raw=df_clean)
        customer_df.to_parquet(cache_path)
        logger.info(f"  Saved features to cache -> {cache_path}")

    if args.sensitivity:
        logger.info("\n[STEP 2b] Running sensitivity analysis across tau in {60, 90, 120}...")
        sensitivity_results = sensitivity_analysis_tau(df_clean, snapshot)
        for tau_val, cdf in sensitivity_results.items():
            churn_rate = cdf["E"].mean() * 100
            logger.info(f"  tau={tau_val}d -> churn rate: {churn_rate:.1f}%")

    # ── STEP 2c: Stratified 80/20 Customer-Level Holdout ──────────────────────
    logger.info("\n[STEP 2c] Stratified 80/20 customer-level holdout split (stratify=E)...")
    
    # For degenerate datasets (e.g. X5 with τ=14, only 1 churn event),
    # skip stratification to avoid sklearn error
    n_churn = customer_df["E"].sum()
    if n_churn < 2:
        logger.warning(
            f"[X5 DEGENERATE] Only {n_churn} churn event detected. "
            f"Skipping stratification (would fail StratifiedKFold). "
            f"Using random 80/20 split instead."
        )
        customer_df_train, customer_df_test = train_test_split(
            customer_df,
            test_size=0.2,
            random_state=42,
            stratify=None,
        )
    else:
        customer_df_train, customer_df_test = train_test_split(
            customer_df,
            test_size=0.2,
            random_state=42,
            stratify=customer_df["E"],
        )
    logger.info(
        f"  Train: {len(customer_df_train):,} customers | "
        f"churn rate: {customer_df_train['E'].mean()*100:.1f}%"
    )
    logger.info(
        f"  Test : {len(customer_df_test):,} customers | "
        f"churn rate: {customer_df_test['E'].mean()*100:.1f}%"
    )

    # ── STEP 3: RFM Segmentation (Baseline) ──────────────────────────────────
    logger.info("\n[STEP 3] Computing RFM segmentation...")
    rfm_df = rfm_segment(customer_df)

    # ── STEP 4: Train Models (on df_train only) ───────────────────────────────
    # Models are fit exclusively on the 80% training set.
    # Preprocessors (imputer + scaler) are fit on df_train and then applied
    # (transform-only) to df_test to prevent any leakage across the split.
    logger.info("\n[STEP 4] Training survival models on 80% train split...")
    _t4_start = _time.perf_counter()  # E7: runtime benchmark

    waf, df_scaled_train_waf, preprocessor_waf, active_features_waf = train_weibull_aft(customer_df_train)
    
    # ── CoxPH: Try to fit, but skip if dataset is too degenerate (X5 RCT) ──────
    try:
        cph, df_scaled_train_cph, preprocessor_cph, active_features_cph = train_coxph(customer_df_train)
    except Exception as e:
        logger.warning(
            f"[CoxPH] Failed to converge (likely degenerate dataset like X5 RCT): {e}. "
            f"Skipping CoxPH — will use Weibull AFT only."
        )
        cph = None
        df_scaled_train_cph = None
        preprocessor_cph = None
        active_features_cph = []
    
    lr, lr_pipeline, lr_cv_metrics = train_logistic(customer_df_train)

    # ── STEP 4b-CLV: Train CLV Regressor on train split ───────────────────────
    # future_spend is only available if cache was bypassed (df_raw was passed).
    # If loaded from cache, it may already be present too.
    predicted_clv_all = None  # default: fall back to historical Monetary
    if "future_spend" in customer_df_train.columns:
        logger.info("[STEP 4b-CLV] Training CLV regressor (RandomForest) on train split...")
        avail_clv = [f for f in CLV_FEATURES if f in customer_df_train.columns]
        X_train_clv  = customer_df_train[avail_clv]
        y_train_spend = customer_df_train["future_spend"]
        rf_clv_pipeline, _ = train_clv_regressor(X_train_clv, y_train_spend)

        # Predict on ALL customers (test set is included — no leakage since
        # model was fit on train only; predict is just transform)
        avail_clv_all = [f for f in CLV_FEATURES if f in customer_df.columns]
        X_all_clv = customer_df[avail_clv_all]
        raw_predictions = rf_clv_pipeline.predict(X_all_clv.values)
        predicted_clv_all = pd.Series(
            np.clip(raw_predictions, 0, None),
            index=customer_df.index,
            name="predicted_clv",
        )
        logger.info(
            f"[CLV] Predicted CLV on all {len(predicted_clv_all):,} customers | "
            f"mean={predicted_clv_all.mean():.2f} | median={predicted_clv_all.median():.2f}"
        )
    else:
        rf_clv_pipeline = None
        logger.warning(
            "[STEP 4b-CLV] 'future_spend' column not found in customer_df_train. "
            "CLV regressor skipped — EVI will use historical Monetary."
        )

    _t4_end = _time.perf_counter()
    logger.info(f"  [Runtime] Model training: {_t4_end - _t4_start:.2f}s")

    # ── E3: Model Selection Justification (AIC/BIC) ───────────────────────────
    logger.info("\n[STEP 4-E3] Model Selection: Weibull vs Log-Normal vs Log-Logistic (AIC/BIC)...")
    model_selection_results = {}
    try:
        model_selection_results = compare_survival_distributions(customer_df_train)
    except Exception as e:
        logger.warning(f"  Model selection comparison failed: {e}")

    # ── STEP 4b: Apply train-fitted preprocessors to remaining splits ──────────
    #
    # df_scaled_test_*   : OOS evaluation (20% held-out)
    # df_scaled_waf      : All customers   (policy, dashboard, artifact saving)
    #
    # Only .transform() is called here — no fitting on test/all data.

    def _apply_preprocessor(prep, source_df, active_feats, input_feats):
        """Transform source_df with a fitted preprocessor, keep only active features + T, E."""
        X = prep.transform(source_df[input_feats])
        df_out = pd.DataFrame(X, columns=input_feats, index=source_df.index)
        df_out = df_out[active_feats].copy()
        df_out["T"] = source_df["T"].values
        df_out["E"] = source_df["E"].values
        return df_out

    # Recover the full input_features used by each model's preprocessor
    from src.models import get_survival_features
    input_features_waf = get_survival_features(customer_df_train)
    input_features_cph = get_survival_features(customer_df_train)

    # OOS (test) scaled frames
    df_scaled_test_waf = _apply_preprocessor(preprocessor_waf, customer_df_test, active_features_waf, input_features_waf)
    df_scaled_test_cph = None
    if cph is not None:
        df_scaled_test_cph = _apply_preprocessor(preprocessor_cph, customer_df_test, active_features_cph, input_features_cph)

    # All-customer scaled frame (for intervention policy + dashboard artifacts)
    df_scaled_waf = _apply_preprocessor(preprocessor_waf, customer_df, active_features_waf, input_features_waf)

    # ── STEP 4a: Cross-Validation (optional) ──────────────────────────────────
    cv_mean_c = None
    cv_std_c  = None
    if args.cv:
        logger.info("\n[STEP 4a] Running 5-Fold Stratified Cross-Validation on WeibullAFT...")
        # Use the training set active features + T, E
        # Note: Ideally we would redo feature selection in each fold, but for
        # stability estimation of the final featured set, this is acceptable.
        try:
            from lifelines import WeibullAFTFitter
            cv_results = cross_validate_survival_model(
                WeibullAFTFitter,
                df_scaled_train_waf,
                duration_col="T", event_col="E",
                n_splits=5,
                random_state=42,
                model_kwargs={"penalizer": 0.01} # Match train_weibull_aft default
            )
            cv_mean_c = cv_results["mean_c_index"]
            cv_std_c  = cv_results["std_c_index"]
            # Log results handled inside function, but we can access them here if needed
        except Exception as e:
             logger.warning(f"CV process failed: {e}")

    # ── STEP 5: Intervention Policy ───────────────────────────────────────────
    logger.info("\n[STEP 5] Applying intervention policy...")
    _t5_start = _time.perf_counter()  # E7
    t_now = float(df_scaled_waf["T"].median())

    # EVI safety margin — use dataset override or global default
    policy_cfg = load_config_with_overrides(args.dataset).get("policy", {})
    min_evi = policy_cfg.get("min_evi_threshold", 0.0)

    weibull_decisions = make_intervention_decisions(
        waf, df_scaled_waf, customer_df, t_now=t_now,
        min_evi_threshold=min_evi,
        predicted_clv=predicted_clv_all,     # None → falls back to historical Monetary
    )
    rfm_decisions = rfm_intervention_decisions(rfm_df)

    # ── LR+EVI Baseline Policy ────────────────────────────────────────────────
    lr_decisions_df = None
    if rf_clv_pipeline is not None:
        logger.info("[STEP 5] Computing LR+EVI baseline policy decisions...")
        try:
            # Build a constant uplift Series (p_response) — real uplift from T-Learner
            # will be used if available (see STEP 5c), but for the baseline we use a
            # constant so LR EVI mirrors the Weibull EVI formula exactly.
            from src.policy import DEFAULT_RESPONSE_RATE
            constant_uplift = pd.Series(
                DEFAULT_RESPONSE_RATE,
                index=customer_df.index,
                name="tau_hat",
            )
            lr_decisions_df = lr_intervention_decisions(
                lr_pipeline=lr_pipeline,
                customer_df=customer_df,
                uplift_scores=constant_uplift,
                predicted_clv=predicted_clv_all,
            )
            lr_csv = os.path.join(REPORTS_DIR, "lr_decisions.csv")
            lr_decisions_df.to_csv(lr_csv, index=False)
            logger.info(f"  LR decisions saved -> {lr_csv}")
        except Exception as lr_exc:
            logger.warning(f"  LR+EVI policy failed: {lr_exc}")

    _t5_end = _time.perf_counter()
    logger.info(f"  [Runtime] Policy decisions (vectorized): {_t5_end - _t5_start:.3f}s")

    decision_path = os.path.join(REPORTS_DIR, "intervention_decisions.csv")
    weibull_decisions.to_csv(decision_path, index=False)
    logger.info(f"Decision table saved -> {decision_path}")

    # ── STEP 5b: Monte Carlo Policy Simulation ────────────────────────────────
    logger.info("\n[STEP 5b] Running Monte Carlo Policy Simulation (n=1,000)...")
    monte_carlo_results = {}
    try:
        monte_carlo_results = run_monte_carlo_simulation(
            df_decisions=weibull_decisions,
            n_iterations=1000,
            lr_decisions=lr_decisions_df,   # None if CLV regressor skipped
        )
        w_ci  = monte_carlo_results.get("weibull_profit_ci",  (0, 0, 0))
        r_ci  = monte_carlo_results.get("rfm_profit_ci",      (0, 0, 0))
        lr_ci = monte_carlo_results.get("lr_profit_ci",       None)
        eg_ci = monte_carlo_results.get("efficiency_gain_ci", (0, 0, 0))
        
        # DYNAMIC CURRENCY
        from src.dataset_registry import get_currency_symbol
        _sym = get_currency_symbol()
        
        logger.info(
            f"  Weibull Policy  Profit  95% CI: "
            f"[{_sym}{w_ci[0]:,.0f}, {_sym}{w_ci[1]:,.0f}, {_sym}{w_ci[2]:,.0f}]"
        )
        logger.info(
            f"  RFM Baseline    Profit  95% CI: "
            f"[{_sym}{r_ci[0]:,.0f}, {_sym}{r_ci[1]:,.0f}, {_sym}{r_ci[2]:,.0f}]"
        )
        if lr_ci:
            logger.info(
                f"  LR+EVI Baseline Profit  95% CI: "
                f"[{_sym}{lr_ci[0]:,.0f}, {_sym}{lr_ci[1]:,.0f}, {_sym}{lr_ci[2]:,.0f}]"
            )
        logger.info(
            f"  Efficiency Gain         95% CI: "
            f"[{eg_ci[0]:+.1%}, {eg_ci[1]:+.1%}, {eg_ci[2]:+.1%}]"
        )
    except Exception as exc:
        logger.warning(f"  Monte Carlo simulation failed: {exc}")

    # ── STEP 5c: Uplift Modeling (optional — pass --uplift) ───────────────────
    uplift_results = {}
    if args.uplift:
        logger.info("\n[STEP 5c] Running Uplift Modeling (T-Learner proxy)...")
        try:
            qini_path = os.path.join(FIGURES_DIR, "07_qini_curve.png")
            uplift_results = run_uplift_analysis(
                weibull_decisions=weibull_decisions,
                customer_df=customer_df,
                save_path=qini_path,
            )
            logger.info(
                f"  Persuadables: {uplift_results['persuadable_pct']:.1%} of INTERVENE | "
                f"Qini coefficient: {uplift_results['qini_auc_ratio']:.4f}"
            )
            # Save uplift segment table
            uplift_csv = os.path.join(REPORTS_DIR, "uplift_segments.csv")
            uplift_results["uplift_df"][["CustomerID", "tau_hat", "uplift_segment"]].to_csv(
                uplift_csv, index=False
            )
            logger.info(f"  Uplift segments saved -> {uplift_csv}")
        except Exception as upexc:
            logger.warning(f"  Uplift analysis failed: {upexc}")
    logger.info("\n[STEP 6] Evaluating models...")

    c_index_weibull  = compute_c_index(waf, df_scaled_train_waf, model_name="WeibullAFT")
    c_index_cox      = compute_c_index(cph, df_scaled_train_cph, model_name="CoxPH") if cph is not None else np.nan
    ibs              = compute_integrated_brier_score(waf, df_scaled_train_waf)
    auc_df           = compute_time_dependent_auc(waf, df_scaled_train_waf)
    outreach_metrics = compute_outreach_efficiency(weibull_decisions, rfm_decisions)
    revenue_metrics  = compute_revenue_lift(weibull_decisions, rfm_decisions)

    print_full_report(
        c_index_weibull=c_index_weibull,
        c_index_cox=c_index_cox,
        ibs=ibs,
        lr_cv_metrics=lr_cv_metrics,
        auc_df=auc_df,
        outreach_metrics=outreach_metrics,
        revenue_metrics=revenue_metrics,
        tau=tau,
    )

    # ── STEP 6b: Out-of-Sample (Stratified) C-index + IBS Validation ─────────
    #
    # Method: Stratified 80/20 random holdout (fit in STEP 2c / STEP 4).
    # The test set was NEVER seen during training or preprocessor fitting.
    #
    # Why NOT temporal cohort splitting:
    #   Late-cohort customers are systematically right-censored by the study
    #   end date (administrative censoring), NOT by random dropout.  This
    #   violates the Independent Censoring Assumption, shifts the T distribution
    #   leftward, and inverts the concordance ranking (C-index 0.34 is an
    #   artifact, not a real signal). Stratified random splitting preserves
    #   the joint (T, E) distribution in both splits and gives unbiased OOS.
    logger.info("\n[STEP 6b] Out-of-Sample (Stratified Holdout) Evaluation...")
    logger.info(
        f"  Test set: {len(df_scaled_test_waf):,} customers | "
        f"churn rate: {df_scaled_test_waf['E'].mean()*100:.1f}%"
    )

    try:
        # ── Weibull AFT OOS C-index ───────────────────────────────────────────
        # model.score() uses the built-in concordance computation on df_test
        oos_c_waf = waf.score(df_scaled_test_waf, scoring_method="concordance_index")

        logger.info("  [WeibullAFT]")
        logger.info(f"    C-index In-Sample (train) : {c_index_weibull:.4f}")
        logger.info(f"    C-index OOS       (test)  : {oos_c_waf:.4f}")
        gap_waf = c_index_weibull - oos_c_waf
        if gap_waf > 0.05:
            logger.warning(
                f"    [OOS WARNING] Gap = {gap_waf:.4f} > 0.05 "
                f"-- possible overfitting."
            )
        else:
            logger.info(
                f"    [OOS OK] Gap = {gap_waf:.4f} <= 0.05 "
                f"-- generalisation is healthy."
            )

        # ── CoxPH OOS C-index ─────────────────────────────────────────────────
        if cph is not None:
            oos_c_cph = cph.score(df_scaled_test_cph, scoring_method="concordance_index")

            logger.info("  [CoxPH]")
            logger.info(f"    C-index In-Sample (train) : {c_index_cox:.4f}")
            logger.info(f"    C-index OOS       (test)  : {oos_c_cph:.4f}")
            gap_cph = c_index_cox - oos_c_cph
            if gap_cph > 0.05:
                logger.warning(
                    f"    [OOS WARNING] Gap = {gap_cph:.4f} > 0.05 "
                    f"-- possible overfitting."
                )
            else:
                logger.info(
                    f"    [OOS OK] Gap = {gap_cph:.4f} <= 0.05 "
                    f"-- generalisation is healthy."
                )
        else:
            logger.warning("  [CoxPH] Skipped (model not fitted due to convergence failure).")
            oos_c_cph = np.nan

        # ── OOS Integrated Brier Score (Weibull AFT on test set) ──────────────
        logger.info("  Computing OOS Integrated Brier Score (WeibullAFT on test set)...")
        ibs_oos = compute_integrated_brier_score(waf, df_scaled_test_waf)
        logger.info(f"    IBS In-Sample : {ibs:.4f}")
        logger.info(f"    IBS OOS       : {ibs_oos:.4f}  (target: < 0.25)")

        # ── Summary table to stdout ───────────────────────────────────────────
        print("\n" + "=" * 70)
        print("  OOS GENERALIZATION REPORT (Stratified 80/20 Holdout)")
        print("=" * 70)
        print(f"  {'Model':<15} {'In-Sample C':>14} {'OOS C':>10} {'Gap':>8}")
        print(f"  {'-'*15} {'-'*14} {'-'*10} {'-'*8}")
        print(f"  {'WeibullAFT':<15} {c_index_weibull:>14.4f} {oos_c_waf:>10.4f} {gap_waf:>+8.4f}")
        print(f"  {'CoxPH':<15} {c_index_cox:>14.4f} {oos_c_cph:>10.4f} {gap_cph:>+8.4f}")
        print(f"  {'IBS (Weibull)':<15} {ibs:>14.4f} {ibs_oos:>10.4f} {'N/A':>8}")
        print("=" * 70 + "\n")

    except Exception as exc:
        logger.warning(f"  OOS evaluation failed: {exc}")
        oos_c_waf = None
        ibs_oos = None

    # Bundle metrics for saving
    run_metrics = {
        "dataset":               args.dataset,
        "tau":                   tau,
        "n_customers":           len(customer_df),
        "churn_rate":            customer_df["E"].mean(),
        "c_index_weibull_train": c_index_weibull,
        "c_index_oos":           oos_c_waf, # Match reporter key
        "c_index_cox_train":     c_index_cox,
        "c_index_cox":           c_index_cox,  # BUG-2: reporter reads this key
        "ibs":                   ibs_oos if ibs_oos is not None else ibs, # Prioritize OOS
        "lr_auc":                lr_cv_metrics.get("auc_mean"),  # BUG-3: reporter reads this key
        "cv_mean_c_index":       cv_mean_c,
        "cv_std_c_index":        cv_std_c,
        "outreach_efficiency":   outreach_metrics.get("efficiency_gain_pct", 0.0),
        "efficiency_gain_pct":   outreach_metrics.get("efficiency_gain_pct", 0.0), # Duplicate for report
        "revenue_lift":          revenue_metrics.get("revenue_precision_lift_pct", 0.0),
        "uplift_qini":           uplift_results.get("qini_auc_ratio", None),
        "uplift_persuadables":   uplift_results.get("persuadable_pct", None),
        "n_test_customers":      len(df_scaled_test_waf),
        "test_churn_rate":       df_scaled_test_waf['E'].mean(),
    }

    # ── MLflow: Log all metrics ───────────────────────────────────────────────
    if _use_mlflow:
        try:
            # Filter None values
            mlflow.log_metrics({k: v for k, v in run_metrics.items() if v is not None})
            mlflow.log_artifacts(FIGURES_DIR, artifact_path="figures")
            mlflow.end_run()
            logger.info("[MLflow] Run logged and closed.")
        except Exception as _mlf_exc:
            logger.warning(f"[MLflow] Failed to log metrics: {_mlf_exc}")


    # ── STEP 7: Serialize Models & Data for Dashboard ─────────────────────────
    logger.info("\n[STEP 7] Serializing models and processed data...")
    save_artifacts(
        waf=waf,
        preprocessor=preprocessor_waf,
        customer_df=customer_df,
        rfm_df=rfm_df,
        weibull_decisions=weibull_decisions,
        df_scaled_waf=df_scaled_waf,
        tau=tau,
        models_dir=MODELS_DIR,
        active_features_waf=active_features_waf,
        monte_carlo_results=monte_carlo_results,
        metrics=run_metrics,
    )

    # ── STEP 8: Visualizations ────────────────────────────────────────────────
    logger.info("\n[STEP 8] Generating figures...")

    try:
        plot_kaplan_meier_by_segment(
            customer_df, rfm_df,
            save_path=os.path.join(FIGURES_DIR, "01_kaplan_meier_by_segment.png")
        )
    except Exception as e:
        logger.warning(f"KM plot failed: {e}")

    try:
        plot_weibull_survival_curves(
            waf, df_scaled_waf,
            save_path=os.path.join(FIGURES_DIR, "02_weibull_survival_curves.png")
        )
    except Exception as e:
        logger.warning(f"Weibull survival plot failed: {e}")

    try:
        plot_hazard_trajectories(
            waf, df_scaled_waf, rfm_df,
            save_path=os.path.join(FIGURES_DIR, "03_hazard_trajectories.png")
        )
    except Exception as e:
        logger.warning(f"Hazard trajectory plot failed: {e}")

    try:
        plot_decision_distribution(
            weibull_decisions, rfm_decisions,
            save_path=os.path.join(FIGURES_DIR, "04_decision_distribution.png")
        )
    except Exception as e:
        logger.warning(f"Decision distribution plot failed: {e}")

    try:
        plot_brier_score_over_time(
            waf, df_scaled_waf,
            save_path=os.path.join(FIGURES_DIR, "05_brier_score_over_time.png")
        )
    except Exception as e:
        logger.warning(f"Brier score plot failed: {e}")

    if not args.no_shap:
        try:
            plot_shap_summary(
                waf, df_scaled_waf,
                feature_cols=active_features_waf,  # BUG-7: use VIF-pruned features, not static list
                save_path=os.path.join(FIGURES_DIR, "06_shap_summary.png"),
                save_csv_path=os.path.join(REPORTS_DIR, "shap_feature_importance.csv"),
            )
        except Exception as e:
            logger.warning(f"SHAP plot failed: {e}")

    # ── D1: Calibration Plot for LR ───────────────────────────────────────────
    try:
        plot_calibration(
            lr_pipeline, customer_df, tau=args.tau,
            save_path=os.path.join(FIGURES_DIR, "07_lr_calibration.png")
        )
    except Exception as e:
        logger.warning(f"Calibration plot failed: {e}")

    # ── D2: Bootstrap CI for C-index ──────────────────────────────────────────
    boot_ci = None
    try:
        logger.info("[D2] Computing bootstrap 95% CI for C-index (n_boot=300)...")
        # Now returns (lo, med, hi, reliable)
        boot_ci = bootstrap_c_index(waf, df_scaled_test_waf, n_boot=300)
        lo, med, hi, reliable = boot_ci
        if reliable:
            logger.info(f"  C-index 95%% CI: [{lo:.4f}, {hi:.4f}]  (median={med:.4f})")
            run_metrics["c_index_ci"] = boot_ci
        else:
            logger.warning("  Bootstrap CI is unreliable (insufficient valid samples).")
    except Exception as e:
        logger.warning(f"Bootstrap CI failed: {e}")

    # ── D5: Markdown Auto-Report ──────────────────────────────────────────────
    try:
        ds_display = get_dataset(args.dataset).display
        rpt_path = generate_report(
            meta={
                "n_customers":        len(customer_df),
                "churn_rate":         customer_df["E"].mean(),
                "active_features_waf": active_features_waf,
                "tau":                tau,
            },
            metrics=run_metrics,
            outreach=outreach_metrics,
            revenue=revenue_metrics,
            run_dir=RUN_DIR,
            dataset_name=ds_display,
            tau=tau,
            c_index_boot_ci=boot_ci,
            monte_carlo_results=monte_carlo_results if monte_carlo_results else None,
        )
        logger.info(f"[D5] Report saved → {rpt_path}")
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 9: Advanced Framework Modules
    # ═══════════════════════════════════════════════════════════════════════════

    # ── STEP 9a: Advanced Multi-Period Policy Simulation ──────────────────────
    if _HAS_ADVANCED_SIM:
        logger.info("\n[STEP 9a] Running Advanced Multi-Period Policy Simulation...")
        try:
            advsim_out = run_advanced_simulation(
                waf=waf,
                df_scaled=df_scaled_waf,
                customer_df=customer_df,
                predicted_clv=predicted_clv_all,
                n_periods=6,
                base_budget=monte_carlo_results.get("marketing_budget", 500.0),
                n_mc=300,
                save_dir=FIGURES_DIR,
                dataset_label=get_dataset(args.dataset).display,
            )
            advsim_summary = advsim_out["summary_df"]
            logger.info(
                "[STEP 9a] Multi-period simulation complete.\n%s",
                advsim_summary.to_string(index=False),
            )
            advsim_summary.to_csv(
                os.path.join(REPORTS_DIR, "advsim_summary.csv"), index=False
            )
        except Exception as exc:
            logger.warning(f"[STEP 9a] Advanced simulation failed: {exc}")
    else:
        logger.info("[STEP 9a] Advanced simulation module not available — skipping.")

    # ── STEP 9b: SHAP Explainability ──────────────────────────────────────────
    if _HAS_EXPLAINABILITY and not args.no_shap:
        logger.info("\n[STEP 9b] Running SHAP Explainability Analysis...")
        try:
            explain_out = _explain_decisions(
                waf=waf,
                df_scaled=df_scaled_waf,
                feature_cols=active_features_waf,
                weibull_decisions=weibull_decisions,
                uplift_results=uplift_results if uplift_results else None,
                save_dir=FIGURES_DIR,
                dataset_label=get_dataset(args.dataset).display,
                n_background=80,
                n_explain=200,
                n_local_examples=2,
            )
            if explain_out.get("importance_survival") is not None:
                explain_out["importance_survival"].to_csv(
                    os.path.join(REPORTS_DIR, "shap_importance_extended.csv"),
                    index=False,
                )
            logger.info("[STEP 9b] Explainability complete — %d figures generated.",
                        len(explain_out.get("figs", {})))
        except Exception as exc:
            logger.warning(f"[STEP 9b] Explainability failed: {exc}")
    else:
        logger.info("[STEP 9b] Explainability skipped (--no-shap or module unavailable).")

    # ── STEP 9c: Counterfactual Policy Evaluation ──────────────────────────────
    if _HAS_COUNTERFACTUAL and uplift_results:
        logger.info("\n[STEP 9c] Running Counterfactual Policy Evaluation (DM / IPS / DR)...")
        try:
            cf_out = run_counterfactual_evaluation(
                weibull_decisions=weibull_decisions,
                customer_df=customer_df,
                uplift_results=uplift_results,
                rfm_decisions=rfm_decisions,
                save_dir=REPORTS_DIR,
                dataset_label=get_dataset(args.dataset).display,
                n_bootstrap=300,
            )
            logger.info(
                "[STEP 9c] Policy comparison:\n%s",
                cf_out["comparison_df"][
                    ["Policy", "TreatmentRate", "DM", "IPS", "DR"]
                ].to_string(index=False),
            )
        except Exception as exc:
            logger.warning(f"[STEP 9c] Counterfactual evaluation failed: {exc}")
    else:
        logger.info("[STEP 9c] Counterfactual evaluation skipped (no uplift results).")

    # ── STEP 9d: Save Production Inference Pipeline ────────────────────────────
    if _HAS_SCORER:
        logger.info("\n[STEP 9d] Saving production CustomerScorer...")
        try:
            scorer = CustomerScorer.from_artifacts(
                waf=waf,
                preprocessor=preprocessor_waf,
                active_features=active_features_waf,
                tau=tau,
                clv_pipeline=rf_clv_pipeline,
                uplift_results=uplift_results if uplift_results else None,
                dataset_name=args.dataset,
            )
            scorer.save(MODELS_DIR)
            logger.info("[STEP 9d] CustomerScorer meta saved → %s/scorer_meta.pkl", MODELS_DIR)

            # Quick self-test: score a sample of 5 customers
            sample_df = customer_df.sample(min(5, len(customer_df)), random_state=42)
            sample_scores = scorer.score_from_features(sample_df)
            logger.info("[STEP 9d] Self-test on 5 customers:\n%s",
                        sample_scores[["CustomerID", "RecommendedAction", "HazardScore",
                                       "EVI", "Priority"]].to_string(index=False))
        except Exception as exc:
            logger.warning(f"[STEP 9d] Inference pipeline save failed: {exc}")
    else:
        logger.info("[STEP 9d] Inference module not available — skipping.")

    # =============================================================================
    # STEP 10: Level 1 + Level 2 Enhancement Modules
    # =============================================================================

    # ── STEP 10a: Comprehensive Sensitivity Analysis (Level 1) ────────────────
    if _HAS_SENSITIVITY:
        logger.info("\n[STEP 10a] Running Comprehensive Sensitivity Analysis...")
        try:
            _policy_cfg_sa = load_config_with_overrides(args.dataset).get("policy", {})
            sens_out = run_sensitivity_analysis(
                df_decisions=weibull_decisions,
                save_dir=os.path.join(FIGURES_DIR, "sensitivity"),
                dataset_label=get_dataset(args.dataset).display,
                base_response_rate=_policy_cfg_sa.get("response_rate", 0.15),
                base_penalty=monte_carlo_results.get("sleeping_dog_penalty", 0.20),
                base_budget=monte_carlo_results.get("marketing_budget", 500.0),
                base_hazard_threshold=_policy_cfg_sa.get("hazard_threshold", 0.01),
                n_mc=300,
            )
            stability = sens_out["stability_df"]
            most_sensitive = stability.iloc[0]["Parameter"] if not stability.empty else "N/A"
            logger.info(
                "[STEP 10a] Sensitivity analysis complete.\n"
                "  Most sensitive parameter: %s\n%s",
                most_sensitive,
                stability[["Parameter", "Weibull_always_wins",
                            "Min_efficiency_gain", "Max_efficiency_gain"]].to_string(index=False),
            )
            sens_out["stability_df"].to_csv(
                os.path.join(REPORTS_DIR, "sensitivity_stability.csv"), index=False
            )
        except Exception as exc:
            logger.warning(f"[STEP 10a] Sensitivity analysis failed: {exc}")
    else:
        logger.info("[STEP 10a] Sensitivity module not available — skipping.")

    # ── STEP 10b: Temporal Cross-Validation (Level 1) ─────────────────────────
    if _HAS_TEMPORAL_CV:
        logger.info("\n[STEP 10b] Running Temporal Cross-Validation (4 expanding folds)...")
        try:
            tcv_out = temporal_cross_validate(
                customer_df=customer_df,
                active_features=active_features_waf,
                n_folds=4,
                random_cv_cindex=run_metrics.get("c_index_oos"),
                random_cv_std=None,
                save_dir=os.path.join(FIGURES_DIR, "temporal_cv"),
                dataset_label=get_dataset(args.dataset).display,
            )
            tcv_summary = tcv_out["summary"]
            logger.info(
                "[STEP 10b] Temporal CV complete | "
                "mean_C=%.4f +/- %.4f | all>=0.70: %s",
                tcv_summary.get("mean_c_index_test", 0),
                tcv_summary.get("std_c_index_test", 0),
                tcv_summary.get("all_above_07", False),
            )
            tcv_out["fold_df"].to_csv(
                os.path.join(REPORTS_DIR, "temporal_cv_folds.csv"), index=False
            )
        except Exception as exc:
            logger.warning(f"[STEP 10b] Temporal CV failed: {exc}")
    else:
        logger.info("[STEP 10b] Temporal CV module not available — skipping.")

    # ── STEP 10c: Business Metrics (Level 2) ──────────────────────────────────
    if _HAS_BIZ_METRICS and monte_carlo_results:
        logger.info("\n[STEP 10c] Computing Business Metrics (CAC, ROI, Payback, Cohort)...")
        try:
            advsim_ref = None
            if _HAS_ADVANCED_SIM:
                try:
                    advsim_ref = run_advanced_simulation(
                        waf=waf, df_scaled=df_scaled_waf,
                        customer_df=customer_df,
                        predicted_clv=predicted_clv_all,
                        n_periods=6, n_mc=100,
                        base_budget=monte_carlo_results.get("marketing_budget", 500.0),
                        dataset_label=get_dataset(args.dataset).display,
                    )
                except Exception:
                    pass

            biz_out = compute_business_metrics(
                weibull_decisions=weibull_decisions,
                customer_df=customer_df,
                mc_results=monte_carlo_results,
                advsim_results=advsim_ref,
                save_dir=os.path.join(FIGURES_DIR, "business"),
                dataset_label=get_dataset(args.dataset).display,
                p_response=_policy_cfg_sa.get("response_rate", 0.15) if '_policy_cfg_sa' in dir() else 0.15,
            )
            m = biz_out["metrics"]
            logger.info(
                "[STEP 10c] Business Metrics:\n"
                "  CAC_retention=%.2f MU | ROI=%.1f%% | "
                "Payback=M%s | Retention_lift=+%.1f pp",
                m["CAC"]["CAC_retention"],
                m["ROI"]["ROI_pct"],
                m["Payback"].get("payback_period_median_months", "N/A"),
                m["Retention"]["retention_lift_pp"],
            )
            biz_out["metrics_df"].to_csv(
                os.path.join(REPORTS_DIR, "business_metrics.csv"), index=False
            )
        except Exception as exc:
            logger.warning(f"[STEP 10c] Business metrics failed: {exc}")
    else:
        logger.info("[STEP 10c] Business metrics skipped.")

    # ── STEP 10d: Production Simulation 12-month (Level 2) ────────────────────
    if _HAS_PROD_SIM:
        logger.info("\n[STEP 10d] Running Production Simulation (12-month rolling)...")
        try:
            prod_sim = ProductionSimulator(
                waf=waf,
                df_scaled=df_scaled_waf,
                customer_df=customer_df,
                predicted_clv=predicted_clv_all,
                n_months=12,
                cycle_days=30,
                cooldown_days=60,
                budget_per_cycle=monte_carlo_results.get("marketing_budget", 500.0),
                n_mc=100,
            )
            prod_history = prod_sim.run()
            prod_summary = prod_sim.get_summary(prod_history)
            logger.info(
                "[STEP 10d] Production sim complete | "
                "final_active=%.1f%% | cumulative_profit=%.0f MU | "
                "avg_contacts/cycle=%.1f",
                prod_summary["final_active_rate_pct"],
                prod_summary["total_cumulative_profit"],
                prod_summary["avg_contacts_per_cycle"],
            )
            prod_sim.plot_rolling_metrics(
                prod_history,
                save_path=os.path.join(FIGURES_DIR, "production_rolling_metrics.png"),
                dataset_label=get_dataset(args.dataset).display,
            )
            prod_history.to_csv(
                os.path.join(REPORTS_DIR, "production_sim_history.csv"), index=False
            )
        except Exception as exc:
            logger.warning(f"[STEP 10d] Production simulation failed: {exc}")
    else:
        logger.info("[STEP 10d] Production simulation module not available — skipping.")

    # ── E1: Sleeping Dog Penalty Sensitivity Analysis ────────────────────────────
    if getattr(args, 'sensitivity_penalty', False):
        logger.info("\n[E1] Running Sleeping Dog Penalty Sensitivity Analysis...")
        try:
            sens_df = sleeping_dog_sensitivity(
                df_decisions=weibull_decisions,
                penalties=[0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
                n_mc=500,
                save_dir=REPORTS_DIR,
            )
            logger.info(f"[E1] Sensitivity analysis complete — {len(sens_df)} penalty levels tested.")
        except Exception as e:
            logger.warning(f"[E1] Sensitivity analysis failed: {e}")

    # ── E2: Multi-Dataset Benchmark (standalone mode) ───────────────────────────
    if getattr(args, 'benchmark', False):
        logger.info("\n[E2] Running Multi-Dataset Benchmark...")
        try:
            bench_df = run_benchmark(tau=args.tau)
            logger.info(f"[E2] Benchmark complete — {len(bench_df)} datasets compared.")
        except Exception as e:
            logger.warning(f"[E2] Benchmark failed: {e}")

    # ── E6: Ablation Study ───────────────────────────────────────────────────
    if getattr(args, 'ablation', False):
        logger.info("\n[E6] Running Ablation Study (5 scenarios)...")
        try:
            from src.ablation import AblationRunner
            runner = AblationRunner()
            ablation_df = runner.run_suite(args.dataset, base_tau=args.tau)
            logger.info(f"[E6] Ablation complete — {len(ablation_df)} scenarios tested.")
        except Exception as e:
            logger.warning(f"[E6] Ablation study failed: {e}")

    logger.info("\n[DONE] Pipeline completed successfully.")
    logger.info(f"  Figures  -> {FIGURES_DIR}")
    logger.info(f"  Reports  -> {REPORTS_DIR}")
    logger.info(f"  Models   -> {MODELS_DIR}")
    logger.info(f"  Log      -> {LOG_PATH}")
    logger.info("\nTo launch the dashboard, run:")
    logger.info("  streamlit run app.py")


if __name__ == "__main__":
    main()
