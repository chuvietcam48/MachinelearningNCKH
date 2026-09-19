"""
src/inference/customer_scorer.py
==================================
Production-style Inference Pipeline for Customer Retention Scoring.

Loads a fully-trained pipeline from disk (pickled in main.py via save_artifacts)
and scores new customers — returning their intervention decision, uplift segment,
risk score, and expected profit per contact.

Input  : DataFrame of raw transactions  OR  pre-built customer feature DataFrame
Output : Scored decisions DataFrame with columns:
           CustomerID, Segment, HazardScore, SurvivalProb, UpliftScore,
           RecommendedAction, ExpectedProfit, CLV, Priority

Usage
-----
    # Load from a saved run directory
    from src.inference import CustomerScorer, load_scorer

    scorer = load_scorer("outputs/UCI_tau124")
    scored_df = scorer.score_from_transactions(df_new_transactions, snapshot_date)
    scored_df = scorer.score_from_features(customer_feature_df)
    scorer.export(scored_df, "new_customer_scores.csv")

    # Or programmatically (in-process, no disk I/O)
    scorer = CustomerScorer.from_artifacts(
        waf=waf,
        preprocessor=preprocessor_waf,
        active_features=active_features_waf,
        tau=tau,
        clv_pipeline=rf_clv_pipeline,      # optional
        uplift_results=uplift_results,      # optional
    )
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CustomerScorer
# =============================================================================

class CustomerScorer:
    """
    Production inference scorer that wraps the full trained pipeline.

    Attributes set after construction:
    ------------------------------------
    waf                : WeibullAFTFitter (survival model)
    preprocessor       : sklearn Pipeline (imputer + scaler, fitted on train)
    active_features    : list of str (VIF-pruned feature names)
    tau                : int (inactivity threshold in days)
    theta_h            : float (hazard threshold for INTERVENE)
    theta_s            : float (survival floor for LOST)
    p_response         : float (campaign response rate)
    cost_per_contact   : float (cost in MU)
    clv_pipeline       : sklearn Pipeline or None
    uplift_results     : dict or None (for segment assignment)
    """

    def __init__(
        self,
        waf,
        preprocessor,
        active_features: List[str],
        tau: int,
        theta_h: float = 0.01,
        theta_s: float = 0.05,
        p_response: float = 0.15,
        cost_per_contact: float = 1.0,
        clv_pipeline=None,
        uplift_results: Optional[dict] = None,
        dataset_name: str = "unknown",
    ):
        self.waf             = waf
        self.preprocessor    = preprocessor
        self.active_features = active_features
        self.tau             = tau
        self.theta_h         = theta_h
        self.theta_s         = theta_s
        self.p_response      = p_response
        self.cost_per_contact = cost_per_contact
        self.clv_pipeline    = clv_pipeline
        self.uplift_results  = uplift_results
        self.dataset_name    = dataset_name

        logger.info(
            "[CustomerScorer] Initialised | dataset=%s | tau=%d | "
            "theta_h=%.4f | theta_s=%.4f | features=%s",
            dataset_name, tau, theta_h, theta_s, active_features,
        )

    # =========================================================================
    # Class-method constructors
    # =========================================================================

    @classmethod
    def from_run_dir(
        cls,
        run_dir: str,
        uplift_results: Optional[dict] = None,
    ) -> "CustomerScorer":
        """
        Load a CustomerScorer from a saved run directory.

        The run directory must have been created by main.py's save_artifacts(),
        i.e., it must contain:
            models/weibull_model.pkl
            models/preprocessor.pkl
            models/pipeline_meta.pkl

        Parameters
        ----------
        run_dir : str
            Path to the dataset run directory, e.g. 'outputs/UCI_tau124'.
        uplift_results : dict, optional
            Previously computed uplift results to attach to the scorer.

        Returns
        -------
        CustomerScorer
        """
        import joblib

        models_dir = os.path.join(run_dir, "models")

        def _load(filename):
            path = os.path.join(models_dir, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[CustomerScorer] Required file not found: {path}\n"
                    f"Run 'python main.py --dataset <name>' first to create it."
                )
            return joblib.load(path)

        waf          = _load("weibull_model.pkl")
        preprocessor = _load("preprocessor.pkl")
        meta         = _load("pipeline_meta.pkl")

        clv_pipeline = None
        clv_path     = os.path.join(models_dir, "clv_pipeline.pkl")
        if os.path.exists(clv_path):
            clv_pipeline = joblib.load(clv_path)
            logger.info("[CustomerScorer] CLV pipeline loaded from %s", clv_path)

        active_features = meta.get("active_features_waf") or meta.get("survival_features", [])
        tau             = meta.get("tau", 90)
        dataset_name    = meta.get("dataset", run_dir)

        # Load policy config overrides if available
        try:
            from src.evaluation import load_config_with_overrides
            policy_cfg = load_config_with_overrides(dataset_name).get("policy", {})
        except Exception:
            policy_cfg = {}

        logger.info(
            "[CustomerScorer] Loaded from %s | tau=%d | features=%s",
            run_dir, tau, active_features,
        )

        return cls(
            waf=waf,
            preprocessor=preprocessor,
            active_features=active_features,
            tau=tau,
            theta_h=policy_cfg.get("hazard_threshold", 0.01),
            theta_s=policy_cfg.get("survival_floor", 0.05),
            p_response=policy_cfg.get("response_rate", 0.15),
            cost_per_contact=policy_cfg.get("cost_per_contact", 1.0),
            clv_pipeline=clv_pipeline,
            uplift_results=uplift_results,
            dataset_name=dataset_name,
        )

    @classmethod
    def from_artifacts(
        cls,
        waf,
        preprocessor,
        active_features: List[str],
        tau: int,
        clv_pipeline=None,
        uplift_results: Optional[dict] = None,
        theta_h: float = 0.01,
        theta_s: float = 0.05,
        p_response: float = 0.15,
        cost_per_contact: float = 1.0,
        dataset_name: str = "unknown",
    ) -> "CustomerScorer":
        """
        Construct scorer directly from in-memory model objects.
        Use this when the pipeline is already running (no disk I/O needed).
        """
        return cls(
            waf=waf,
            preprocessor=preprocessor,
            active_features=active_features,
            tau=tau,
            theta_h=theta_h,
            theta_s=theta_s,
            p_response=p_response,
            cost_per_contact=cost_per_contact,
            clv_pipeline=clv_pipeline,
            uplift_results=uplift_results,
            dataset_name=dataset_name,
        )

    # =========================================================================
    # Feature preprocessing
    # =========================================================================

    def _preprocess_features(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted preprocessor to new customer features.

        The preprocessor was fitted on SURVIVAL_FEATURES (all 6 standard RFM +
        temporal features). Only those columns are passed to transform(), then
        we select the VIF-pruned active_features subset.

        Parameters
        ----------
        customer_df : pd.DataFrame
            Customer-level DataFrame with RFM + temporal features.
            Index should be CustomerID.

        Returns
        -------
        pd.DataFrame
            Scaled features + T + E columns ready for survival model.
        """
        from src.models import SURVIVAL_FEATURES

        # Use the exact feature set the preprocessor was fitted on,
        # intersected with what is available in this DataFrame.
        input_features = [f for f in SURVIVAL_FEATURES if f in customer_df.columns]

        if not input_features:
            raise ValueError(
                f"[CustomerScorer] No SURVIVAL_FEATURES found in customer_df. "
                f"Expected at least one of: {SURVIVAL_FEATURES}. "
                f"Got columns: {list(customer_df.columns)}"
            )

        # Apply transform (NOT fit_transform — uses train-fitted scaler)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_scaled = self.preprocessor.transform(customer_df[input_features])

        df_scaled = pd.DataFrame(X_scaled, columns=input_features, index=customer_df.index)

        # Keep only VIF-pruned active features that exist after scaling
        avail_active = [f for f in self.active_features if f in df_scaled.columns]
        df_scaled = df_scaled[avail_active].copy()

        # Add survival columns
        df_scaled["T"] = customer_df["T"].values if "T" in customer_df.columns else float(self.tau)
        df_scaled["E"] = customer_df["E"].values if "E" in customer_df.columns else 0

        return df_scaled

    # =========================================================================
    # Core scoring function
    # =========================================================================

    def score_from_features(
        self,
        customer_df: pd.DataFrame,
        t_eval: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Score customers from a pre-built feature DataFrame.

        Parameters
        ----------
        customer_df : pd.DataFrame
            Customer-level features indexed by CustomerID.
            Required: Recency, Frequency, Monetary, InterPurchaseTime,
                      GapDeviation, SinglePurchase, T, E.
        t_eval : float, optional
            Evaluation time in days. Defaults to median(T).

        Returns
        -------
        pd.DataFrame
            Scored results with columns:
              CustomerID, HazardScore, SurvivalProb, EVI,
              RecommendedAction, Priority, CLV, UpliftScore, UpliftSegment,
              ExpectedProfit, OptimalInterventionWindow
        """
        from src.policy import compute_intervention_signals

        logger.info(
            "[CustomerScorer] Scoring %d customers...", len(customer_df)
        )

        # ── Preprocess ────────────────────────────────────────────────────────
        df_scaled = self._preprocess_features(customer_df)
        t_eval    = t_eval or float(df_scaled["T"].median())

        # ── CLV prediction ────────────────────────────────────────────────────
        predicted_clv = None
        if self.clv_pipeline is not None:
            from src.models import CLV_FEATURES
            clv_feats = [f for f in CLV_FEATURES if f in customer_df.columns]
            if clv_feats:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_clv = self.clv_pipeline.predict(customer_df[clv_feats].values)
                predicted_clv = pd.Series(
                    np.clip(raw_clv, 0, None),
                    index=customer_df.index,
                    name="predicted_clv",
                )

        # ── Survival signals ──────────────────────────────────────────────────
        signals = compute_intervention_signals(
            waf=self.waf,
            df_scaled=df_scaled,
            customer_df=customer_df,
            t_now=t_eval,
            predicted_clv=predicted_clv,
        )

        # ── EVI ───────────────────────────────────────────────────────────────
        signals["evi"] = (
            self.p_response * signals["clv_used"] * (1 - signals["survival_now"])
            - self.cost_per_contact
        )

        # ── Decision rule ─────────────────────────────────────────────────────
        is_lost      = signals["survival_now"] < self.theta_s
        is_intervene = (
            (~is_lost)
            & (signals["hazard_now"] > self.theta_h)
            & (signals["evi"] > 0)
        )
        signals["decision"] = np.select(
            [is_lost, is_intervene],
            ["LOST", "INTERVENE"],
            default="WAIT",
        )

        # ── Priority score (0-100): higher = more urgent ───────────────────────
        # Composite score: hazard rank * evi rank
        h_rank   = signals["hazard_now"].rank(pct=True)
        evi_rank = signals["evi"].rank(pct=True).clip(lower=0)
        signals["priority_score"] = (h_rank * 0.5 + evi_rank * 0.5 * (1 - signals["survival_now"])) * 100

        # ── Expected profit ───────────────────────────────────────────────────
        signals["expected_profit"] = np.where(
            signals["decision"] == "INTERVENE",
            signals["evi"],
            0.0,
        )

        # ── Uplift segment (from T-Learner, if available) ─────────────────────
        signals["uplift_score"]   = np.nan
        signals["uplift_segment"] = "Unknown"

        if self.uplift_results and "uplift_df" in self.uplift_results:
            uplift_df = self.uplift_results["uplift_df"]
            tau_map   = {}
            seg_map   = {}
            id_col    = "CustomerID" if "CustomerID" in uplift_df.columns else None
            if id_col:
                tau_map = uplift_df.set_index(id_col)["tau_hat"].to_dict() \
                          if "tau_hat" in uplift_df.columns else {}
                seg_map = uplift_df.set_index(id_col)["uplift_segment"].to_dict() \
                          if "uplift_segment" in uplift_df.columns else {}

            signals["uplift_score"] = customer_df.index.map(tau_map).values
            signals["uplift_segment"] = (
                customer_df.index.map(seg_map).fillna("Unknown").values
            )

        # ── Assemble output ───────────────────────────────────────────────────
        output = pd.DataFrame({
            "CustomerID":             customer_df.index,
            "HazardScore":            signals["hazard_now"].values.round(6),
            "SurvivalProb":           signals["survival_now"].values.round(4),
            "EVI":                    signals["evi"].values.round(4),
            "RecommendedAction":      signals["decision"].values,
            "Priority":               signals["priority_score"].values.round(1),
            "CLV":                    (predicted_clv.values if predicted_clv is not None
                                       else customer_df["Monetary"].values).round(2),
            "UpliftScore":            signals["uplift_score"],
            "UpliftSegment":          signals["uplift_segment"],
            "ExpectedProfit":         signals["expected_profit"].values.round(4),
            "OptimalInterventionDay": signals["optimal_window_days"].values.round(1),
        })

        # Sort: INTERVENE first (by priority), then WAIT, then LOST
        priority_order = {"INTERVENE": 0, "WAIT": 1, "LOST": 2}
        output["_sort"] = output["RecommendedAction"].map(priority_order)
        output = output.sort_values(["_sort", "Priority"], ascending=[True, False])
        output = output.drop(columns=["_sort"]).reset_index(drop=True)

        n_intervene = (output["RecommendedAction"] == "INTERVENE").sum()
        n_wait      = (output["RecommendedAction"] == "WAIT").sum()
        n_lost      = (output["RecommendedAction"] == "LOST").sum()
        logger.info(
            "[CustomerScorer] Scoring complete | INTERVENE=%d (%.1f%%) | "
            "WAIT=%d | LOST=%d | AvgEVI=%.3f",
            n_intervene, n_intervene / len(output) * 100,
            n_wait, n_lost,
            output.loc[output["RecommendedAction"] == "INTERVENE", "EVI"].mean()
            if n_intervene > 0 else 0.0,
        )

        return output

    def score_from_transactions(
        self,
        df_transactions: pd.DataFrame,
        snapshot_date=None,
        tau: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Score customers from raw transaction data.

        Parameters
        ----------
        df_transactions : pd.DataFrame
            Raw transaction log with columns:
            CustomerID, InvoiceNo, InvoiceDate, TotalSpend.
        snapshot_date : pd.Timestamp or str, optional
            Observation date. Defaults to max(InvoiceDate).
        tau : int, optional
            Inactivity threshold. Defaults to self.tau.

        Returns
        -------
        pd.DataFrame
            Same as score_from_features().
        """
        from src.feature_engine import build_customer_features

        if snapshot_date is None:
            snapshot_date = pd.to_datetime(df_transactions["InvoiceDate"]).max()
        else:
            snapshot_date = pd.to_datetime(snapshot_date)

        tau_use = tau or self.tau
        logger.info(
            "[CustomerScorer] Building features | n_transactions=%d | "
            "snapshot=%s | tau=%d",
            len(df_transactions), snapshot_date.date(), tau_use,
        )

        customer_df = build_customer_features(
            df_transactions, snapshot_date, tau=tau_use
        )
        return self.score_from_features(customer_df)

    # =========================================================================
    # Batch scoring
    # =========================================================================

    def score_batch(
        self,
        batches: List[pd.DataFrame],
        is_transaction_data: bool = False,
        snapshot_date=None,
    ) -> pd.DataFrame:
        """
        Score multiple batches and concatenate results.

        Parameters
        ----------
        batches : list of pd.DataFrame
            List of feature DataFrames (or transaction DataFrames if
            is_transaction_data=True).
        is_transaction_data : bool
            If True, each batch is a raw transaction log.
        snapshot_date : optional
            Shared snapshot date (only used when is_transaction_data=True).

        Returns
        -------
        pd.DataFrame
            Concatenated scored results.
        """
        results = []
        for i, batch in enumerate(batches):
            logger.info("[CustomerScorer] Scoring batch %d/%d...", i + 1, len(batches))
            if is_transaction_data:
                scored = self.score_from_transactions(batch, snapshot_date=snapshot_date)
            else:
                scored = self.score_from_features(batch)
            results.append(scored)

        return pd.concat(results, ignore_index=True)

    # =========================================================================
    # Export
    # =========================================================================

    def export(
        self,
        scored_df: pd.DataFrame,
        output_path: str,
        top_n: Optional[int] = None,
        only_intervene: bool = False,
    ) -> None:
        """
        Save scored decisions to CSV.

        Parameters
        ----------
        scored_df : pd.DataFrame
            Output of score_from_features() or score_from_transactions().
        output_path : str
            Destination CSV path.
        top_n : int, optional
            Export only the top-N customers by priority.
        only_intervene : bool
            If True, export only INTERVENE decisions.
        """
        df = scored_df.copy()
        if only_intervene:
            df = df[df["RecommendedAction"] == "INTERVENE"]
        if top_n:
            df = df.head(top_n)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(
            "[CustomerScorer] Exported %d rows → %s", len(df), output_path
        )

    # =========================================================================
    # Summary report
    # =========================================================================

    def summary_report(self, scored_df: pd.DataFrame) -> dict:
        """
        Generate a summary statistics dict from scored decisions.

        Returns
        -------
        dict with keys:
          n_total, n_intervene, n_wait, n_lost,
          intervene_rate, avg_evi_intervene, total_expected_profit,
          avg_clv_intervene, top_3_features (if SHAP available)
        """
        n       = len(scored_df)
        n_i     = int((scored_df["RecommendedAction"] == "INTERVENE").sum())
        n_w     = int((scored_df["RecommendedAction"] == "WAIT").sum())
        n_l     = int((scored_df["RecommendedAction"] == "LOST").sum())
        top_int = scored_df[scored_df["RecommendedAction"] == "INTERVENE"]

        seg_counts = {}
        if "UpliftSegment" in scored_df.columns:
            seg_counts = scored_df["UpliftSegment"].value_counts().to_dict()

        return {
            "n_total":                 n,
            "n_intervene":             n_i,
            "n_wait":                  n_w,
            "n_lost":                  n_l,
            "intervene_rate_pct":      round(n_i / max(n, 1) * 100, 2),
            "avg_evi_intervene":       round(top_int["EVI"].mean(), 4)     if n_i > 0 else 0.0,
            "total_expected_profit":   round(top_int["ExpectedProfit"].sum(), 2) if n_i > 0 else 0.0,
            "avg_clv_intervene":       round(top_int["CLV"].mean(), 2)     if n_i > 0 else 0.0,
            "median_priority_intervene": round(top_int["Priority"].median(), 1) if n_i > 0 else 0.0,
            "uplift_segments":         seg_counts,
            "dataset":                 self.dataset_name,
            "tau_days":                self.tau,
        }

    # =========================================================================
    # Save scorer pipeline to disk
    # =========================================================================

    def save(self, save_dir: str) -> None:
        """
        Persist the CustomerScorer to disk so it can be reloaded with load_scorer().

        Saves:
          scorer_meta.pkl  — all non-model attributes
          (model artifacts are already saved by main.py's save_artifacts)

        Parameters
        ----------
        save_dir : str
            Directory to save the scorer meta file.
        """
        import joblib

        os.makedirs(save_dir, exist_ok=True)
        meta = {
            "active_features": self.active_features,
            "tau":             self.tau,
            "theta_h":         self.theta_h,
            "theta_s":         self.theta_s,
            "p_response":      self.p_response,
            "cost_per_contact": self.cost_per_contact,
            "dataset_name":    self.dataset_name,
        }
        joblib.dump(meta, os.path.join(save_dir, "scorer_meta.pkl"))
        logger.info("[CustomerScorer] Meta saved → %s/scorer_meta.pkl", save_dir)


# =============================================================================
# Convenience loader
# =============================================================================

def load_scorer(
    run_dir: str,
    uplift_results: Optional[dict] = None,
) -> CustomerScorer:
    """
    Load a CustomerScorer from a saved run directory.

    Equivalent to CustomerScorer.from_run_dir(run_dir, uplift_results).

    Parameters
    ----------
    run_dir : str
        Path to the output directory created by main.py, e.g. 'outputs/UCI_tau124'.
    uplift_results : dict, optional
        Previously-computed uplift results to attach.

    Returns
    -------
    CustomerScorer
    """
    return CustomerScorer.from_run_dir(run_dir, uplift_results=uplift_results)
