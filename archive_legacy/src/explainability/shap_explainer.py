"""
src/explainability/shap_explainer.py
=====================================
SHAP-based explainability for both the Survival model and the Uplift model.

Answers two key research questions:
  1. "What drives the Weibull AFT churn timing prediction?" (SurvivalShapExplainer)
  2. "Why is customer X classified as a Persuadable / Sleeping Dog?" (UpliftShapExplainer)

Module structure
-----------------
SurvivalShapExplainer
  ├── global_summary()         — global feature importance bar chart
  ├── explain_customer()       — waterfall plot for one customer
  └── compare_segments()       — SHAP profiles: INTERVENE vs WAIT vs LOST

UpliftShapExplainer
  ├── global_summary()         — tau_hat feature importance
  ├── explain_customer()       — local explanation: why Persuadable?
  └── compare_uplift_segments()— SHAP profiles: Persuadables vs Sleeping Dogs

explain_decisions()            — convenience wrapper: runs both explainers
                                 and saves all figures to a directory

Requirements
------------
  pip install shap matplotlib pandas numpy
  (lifelines, sklearn already required by the main pipeline)
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# SHAP import with friendly error
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False
    logger.warning(
        "[Explainability] 'shap' package not found. "
        "Install with: pip install shap"
    )


def _check_shap():
    if not _SHAP_AVAILABLE:
        raise ImportError(
            "shap is required for explainability. "
            "Install with: pip install shap"
        )


# =============================================================================
# 1. Survival Model Explainer (Weibull AFT)
# =============================================================================

class SurvivalShapExplainer:
    """
    SHAP explanations for the Weibull AFT survival model.

    Uses KernelExplainer because WeibullAFTFitter is not a tree-based model.
    The target function is the predicted log-median survival time log(T_50),
    which is a monotone transformation of the underlying linear predictor.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted Weibull AFT model.
    df_scaled : pd.DataFrame
        Preprocessed customer feature DataFrame (index = CustomerID).
        Must contain T, E, and the active feature columns.
    feature_cols : list of str
        Names of the active features (after VIF pruning).
    n_background : int
        Number of background samples for KernelExplainer (default: 100).
    n_explain : int
        Number of customers to explain (default: 300).
    seed : int
        Random seed.
    """

    def __init__(
        self,
        waf,
        df_scaled: pd.DataFrame,
        feature_cols: List[str],
        n_background: int = 100,
        n_explain: int = 300,
        seed: int = 42,
    ):
        _check_shap()
        self.waf          = waf
        self.df_scaled    = df_scaled.copy()
        self.feature_cols = feature_cols
        self.seed         = seed

        X = df_scaled[feature_cols].values.astype(float)
        rng = np.random.default_rng(seed)

        # Sample background and explanation sets
        n_bg  = min(n_background, len(X))
        n_exp = min(n_explain,    len(X))

        bg_idx  = rng.choice(len(X), n_bg,  replace=False)
        exp_idx = rng.choice(len(X), n_exp, replace=False)

        self.X_background = X[bg_idx]
        self.X_explain    = X[exp_idx]
        self.idx_explain  = df_scaled.index[exp_idx]

        # Prediction function: predict median survival time (days until churn).
        # Higher value = customer survives longer = less at-risk.
        # Uses waf.predict_median() — same as visualization.py's SHAP implementation.
        _t_median = float(df_scaled["T"].median())

        def _predict_median_survival(X_array: np.ndarray) -> np.ndarray:
            df_tmp = pd.DataFrame(X_array, columns=feature_cols)
            df_tmp["T"] = _t_median
            df_tmp["E"] = 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vals = waf.predict_median(df_tmp).values.astype(float)
            # Guard against inf / nan (customers with very low hazard get inf median)
            vals = np.nan_to_num(vals, nan=_t_median * 2, posinf=_t_median * 10)
            return vals

        self._predict_fn = _predict_median_survival

        logger.info(
            "[SurvivalSHAP] Fitting KernelExplainer | "
            "background=%d | explain=%d | features=%s",
            n_bg, n_exp, feature_cols,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Use shap.sample for background to match visualization.py pattern
            bg_data = shap.sample(X[bg_idx], n_bg) if hasattr(shap, "sample") else self.X_background
            self.explainer = shap.KernelExplainer(
                self._predict_fn,
                bg_data,
                silent=True,
            )

        logger.info("[SurvivalSHAP] Computing SHAP values...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.shap_values = self.explainer.shap_values(
                self.X_explain, nsamples=200, silent=True
            )
        # shap_values: (n_explain, n_features)
        logger.info("[SurvivalSHAP] SHAP values computed.")

    def global_summary(
        self,
        save_path: Optional[str] = None,
        title: str = "Survival Model — Global SHAP Feature Importance",
        plot_type: str = "bar",
    ) -> plt.Figure:
        """
        Global SHAP feature importance plot.

        Parameters
        ----------
        save_path : str, optional
        title : str
        plot_type : str
            'bar' (mean |SHAP|) or 'beeswarm' (dot summary).

        Returns
        -------
        matplotlib Figure
        """
        _check_shap()
        fig, ax = plt.subplots(figsize=(9, max(4, len(self.feature_cols) * 0.7)))

        if plot_type == "bar":
            mean_abs = np.abs(self.shap_values).mean(axis=0)
            order    = np.argsort(mean_abs)
            colors   = [
                "#3498db" if mean_abs[i] >= np.median(mean_abs) else "#85c1e9"
                for i in order
            ]
            ax.barh(
                [self.feature_cols[i] for i in order],
                mean_abs[order],
                color=colors, edgecolor="white",
            )
            ax.set_xlabel("Mean |SHAP value|", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3, axis="x")
        else:
            # Beeswarm via shap's built-in
            plt.close(fig)
            shap_exp = shap.Explanation(
                values=self.shap_values,
                data=self.X_explain,
                feature_names=self.feature_cols,
            )
            shap.plots.beeswarm(shap_exp, show=False, max_display=len(self.feature_cols))
            fig = plt.gcf()
            fig.suptitle(title, fontsize=12, fontweight="bold")

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[SurvivalSHAP] Global summary saved → %s", save_path)
        return fig

    def explain_customer(
        self,
        customer_idx,
        customer_label: str = "",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Local SHAP waterfall plot for a single customer.

        Parameters
        ----------
        customer_idx : int or index label
            Position (int) in X_explain, or CustomerID label.
        customer_label : str
            Label for plot title.
        save_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        _check_shap()

        # Resolve integer position
        if isinstance(customer_idx, int) and customer_idx < len(self.X_explain):
            pos = customer_idx
        else:
            pos_arr = np.where(self.idx_explain == customer_idx)[0]
            pos = int(pos_arr[0]) if len(pos_arr) > 0 else 0

        shap_vals  = self.shap_values[pos]
        feat_vals  = self.X_explain[pos]
        base_val   = float(self.explainer.expected_value)

        label = customer_label or f"CustomerID={self.idx_explain[pos]}"

        fig, ax = plt.subplots(figsize=(10, max(4, len(self.feature_cols) * 0.65)))

        # Waterfall bar chart
        cumulative = base_val
        bar_rights = []
        bar_lefts  = []
        for i, (feat, sv) in enumerate(zip(self.feature_cols, shap_vals)):
            left   = min(cumulative, cumulative + sv)
            right  = max(cumulative, cumulative + sv)
            bar_lefts.append(left)
            bar_rights.append(right)
            cumulative += sv

        colors = ["#e74c3c" if sv >= 0 else "#3498db" for sv in shap_vals]
        widths = np.array([r - l for r, l in zip(bar_rights, bar_lefts)])
        ax.barh(self.feature_cols, widths, left=bar_lefts, color=colors, alpha=0.85)
        ax.axvline(base_val, color="#333", lw=1.5, ls="--", alpha=0.5, label=f"E[f(x)]={base_val:.3f}")
        ax.axvline(cumulative, color="#f39c12", lw=2, label=f"f(x)={cumulative:.3f}")

        # Feature value annotations
        for i, (feat, sv, fv) in enumerate(zip(self.feature_cols, shap_vals, feat_vals)):
            ax.text(
                bar_lefts[i] + widths[i] / 2, i,
                f"{fv:.2f}  ({sv:+.3f})",
                va="center", ha="center", fontsize=8,
                color="white" if abs(sv) > 0.01 else "#333",
            )

        ax.set_title(
            f"Local SHAP Explanation — {label}\n"
            f"(Survival Model: log partial hazard)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("SHAP value contribution", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[SurvivalSHAP] Local explanation saved → %s", save_path)
        return fig

    def compare_segments(
        self,
        weibull_decisions: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Side-by-side mean SHAP profiles for INTERVENE / WAIT / LOST segments.

        Parameters
        ----------
        weibull_decisions : pd.DataFrame
            Output of policy.make_intervention_decisions() with CustomerID + decision.
        save_path : str, optional

        Returns
        -------
        matplotlib Figure
        """
        _check_shap()

        # Map explain indices to decisions
        decision_map = weibull_decisions.set_index("CustomerID")["decision"].to_dict()
        decisions_for_explain = np.array([
            decision_map.get(cid, "WAIT") for cid in self.idx_explain
        ])

        segment_shap: Dict[str, np.ndarray] = {}
        for seg in ["INTERVENE", "WAIT", "LOST"]:
            mask = decisions_for_explain == seg
            if mask.sum() < 5:
                continue
            segment_shap[seg] = np.abs(self.shap_values[mask]).mean(axis=0)

        if not segment_shap:
            logger.warning("[SurvivalSHAP] Not enough samples per segment for comparison.")
            return plt.figure()

        n_segs = len(segment_shap)
        fig, axes = plt.subplots(1, n_segs, figsize=(5 * n_segs, max(4, len(self.feature_cols) * 0.7)),
                                  sharey=True)
        if n_segs == 1:
            axes = [axes]

        seg_colors = {"INTERVENE": "#e74c3c", "WAIT": "#3498db", "LOST": "#95a5a6"}

        for ax, (seg, mean_abs) in zip(axes, segment_shap.items()):
            order = np.argsort(mean_abs)
            ax.barh(
                [self.feature_cols[i] for i in order],
                mean_abs[order],
                color=seg_colors.get(seg, "#888"), alpha=0.85,
            )
            ax.set_title(f"{seg}\n(n={int((decisions_for_explain==seg).sum())})",
                         fontsize=11, fontweight="bold")
            ax.set_xlabel("Mean |SHAP|", fontsize=10)
            ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle(
            "SHAP Feature Importance by Decision Segment",
            fontsize=13, fontweight="bold"
        )
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[SurvivalSHAP] Segment comparison saved → %s", save_path)
        return fig

    def get_importance_df(self) -> pd.DataFrame:
        """
        Returns a DataFrame with mean |SHAP| per feature, sorted descending.
        """
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_cols, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


# =============================================================================
# 2. Uplift Model Explainer (T-Learner)
# =============================================================================

class UpliftShapExplainer:
    """
    SHAP explanations for the T-Learner uplift model.

    Uses TreeExplainer on the GradientBoosting mu_1 and mu_0 models
    (if available) for fast, exact SHAP values.  Falls back to
    KernelExplainer if model objects are not provided.

    The SHAP target is tau_hat(x) = mu_1(x) - mu_0(x).
    We approximate via: SHAP(tau_hat) ≈ SHAP(mu_1) - SHAP(mu_0).

    Parameters
    ----------
    uplift_df : pd.DataFrame
        Output of run_uplift_analysis()['uplift_df'].
        Must have columns: tau_hat, mu_1, mu_0, uplift_segment + features.
    feature_cols : list of str
        Feature columns used by the T-Learner.
    mu_1_model : GradientBoostingRegressor, optional
        Trained treated-branch model. When provided, uses TreeExplainer.
    mu_0_model : GradientBoostingRegressor, optional
        Trained control-branch model.
    scaler : sklearn transformer, optional
        StandardScaler used to preprocess features before T-Learner.
    imputer : sklearn transformer, optional
        SimpleImputer used before scaler.
    n_background : int
        Background samples for KernelExplainer fallback.
    n_explain : int
        Customers to explain.
    seed : int
    """

    def __init__(
        self,
        uplift_df: pd.DataFrame,
        feature_cols: List[str],
        mu_1_model=None,
        mu_0_model=None,
        scaler=None,
        imputer=None,
        n_background: int = 100,
        n_explain: int = 300,
        seed: int = 42,
    ):
        _check_shap()
        self.uplift_df    = uplift_df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seed         = seed

        rng = np.random.default_rng(seed)
        n   = len(uplift_df)

        # Prepare feature matrix
        available = [c for c in feature_cols if c in uplift_df.columns]
        self.feature_cols_used = available
        X_raw = uplift_df[available].fillna(uplift_df[available].median()).values.astype(float)

        # Apply same preprocessing as T-Learner
        if imputer is not None:
            X_raw = imputer.transform(X_raw)
        if scaler is not None:
            X_raw = scaler.transform(X_raw)

        n_exp = min(n_explain, n)
        exp_idx = rng.choice(n, n_exp, replace=False)
        self.X_explain      = X_raw[exp_idx]
        self.idx_explain    = exp_idx
        self.segment_explain = uplift_df["uplift_segment"].values[exp_idx]

        # ── Build SHAP explainers ────────────────────────────────────────────
        self.shap_values_mu1 = None
        self.shap_values_mu0 = None
        self.shap_tau        = None

        if mu_1_model is not None and mu_0_model is not None:
            # TreeExplainer (fast, exact)
            logger.info("[UpliftSHAP] Using TreeExplainer (GradientBoosting detected).")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                exp_mu1 = shap.TreeExplainer(mu_1_model, feature_names=available)
                exp_mu0 = shap.TreeExplainer(mu_0_model, feature_names=available)
                self.shap_values_mu1 = exp_mu1.shap_values(self.X_explain)
                self.shap_values_mu0 = exp_mu0.shap_values(self.X_explain)
            self.shap_tau = self.shap_values_mu1 - self.shap_values_mu0
            logger.info("[UpliftSHAP] Tree SHAP computed for mu_1 and mu_0.")
        else:
            # KernelExplainer fallback using tau_hat directly
            logger.info("[UpliftSHAP] mu_1/mu_0 models not provided. "
                        "Using KernelExplainer on tau_hat approximation.")
            n_bg = min(n_background, n)
            bg_idx = rng.choice(n, n_bg, replace=False)
            X_bg   = X_raw[bg_idx]

            # Use tau_hat from uplift_df as ground truth
            tau_hat_vals = uplift_df["tau_hat"].values

            def _predict_tau(X_arr):
                # Approximate via kNN lookup in the original X space
                dists = np.linalg.norm(X_raw[:, None] - X_arr[None, :], axis=2)  # (N, n_query)
                knn_idx = dists.argmin(axis=0)
                return tau_hat_vals[knn_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kernel_exp = shap.KernelExplainer(_predict_tau, X_bg, silent=True)
                self.shap_tau = kernel_exp.shap_values(self.X_explain, nsamples=100, silent=True)
            logger.info("[UpliftSHAP] Kernel SHAP computed for tau_hat.")

    def global_summary(
        self,
        save_path: Optional[str] = None,
        title: str = "Uplift Model — SHAP Feature Importance (τ̂)",
    ) -> plt.Figure:
        """
        Global bar chart: mean |SHAP| contribution to tau_hat prediction.
        """
        _check_shap()
        mean_abs = np.abs(self.shap_tau).mean(axis=0)
        order    = np.argsort(mean_abs)

        fig, ax = plt.subplots(figsize=(9, max(4, len(self.feature_cols_used) * 0.7)))
        colors = [
            "#2ecc71" if mean_abs[i] >= np.median(mean_abs) else "#82e0aa"
            for i in order
        ]
        ax.barh(
            [self.feature_cols_used[i] for i in order],
            mean_abs[order],
            color=colors, edgecolor="white",
        )
        ax.set_xlabel("Mean |SHAP(τ̂)| value", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[UpliftSHAP] Global summary saved → %s", save_path)
        return fig

    def explain_customer(
        self,
        customer_pos: int = 0,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, dict]:
        """
        Local explanation: why is this customer a Persuadable / Sleeping Dog?

        Parameters
        ----------
        customer_pos : int
            Position in the explain set (0..n_explain-1).
        save_path : str, optional

        Returns
        -------
        tuple (Figure, explanation_dict)
        """
        _check_shap()
        pos    = min(customer_pos, len(self.X_explain) - 1)
        sv     = self.shap_tau[pos]
        fv     = self.X_explain[pos]
        seg    = self.segment_explain[pos]
        tau    = self.uplift_df["tau_hat"].values[self.idx_explain[pos]]

        fig, ax = plt.subplots(figsize=(10, max(4, len(self.feature_cols_used) * 0.65)))

        # Waterfall
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in sv]
        order  = np.argsort(np.abs(sv))[::-1]
        ax.barh(
            [self.feature_cols_used[i] for i in order],
            [sv[i] for i in order],
            color=[colors[i] for i in order],
            alpha=0.85,
        )
        ax.axvline(0, color="#333", lw=1.2)

        # Annotations
        for rank, i in enumerate(order):
            ax.text(
                sv[i] + (0.001 if sv[i] >= 0 else -0.001),
                rank,
                f"val={fv[i]:.2f}  ({sv[i]:+.3f})",
                va="center",
                ha="left" if sv[i] >= 0 else "right",
                fontsize=8,
            )

        seg_color = {
            "Persuadables": "#2ecc71", "Sleeping Dogs": "#e74c3c",
            "Sure Things": "#3498db", "Lost Causes": "#95a5a6",
        }.get(seg, "#888")

        ax.set_title(
            f"Uplift SHAP — Customer #{self.idx_explain[pos]}\n"
            f"Segment: {seg}  |  τ̂ = {tau:+.4f}",
            fontsize=12, fontweight="bold", color=seg_color,
        )
        ax.set_xlabel("SHAP contribution to τ̂", fontsize=10)
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()

        explanation = {
            "customer_pos":    pos,
            "segment":         seg,
            "tau_hat":         float(tau),
            "top_drivers":     [
                {"feature": self.feature_cols_used[i], "shap": float(sv[i]), "value": float(fv[i])}
                for i in order[:3]
            ],
        }

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[UpliftSHAP] Local explanation saved → %s", save_path)
        return fig, explanation

    def compare_uplift_segments(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Compare mean SHAP profiles for Persuadables vs Sleeping Dogs.
        """
        _check_shap()
        target_segs = ["Persuadables", "Sleeping Dogs", "Sure Things", "Lost Causes"]
        seg_shap    = {}
        for seg in target_segs:
            mask = self.segment_explain == seg
            if mask.sum() < 3:
                continue
            seg_shap[seg] = self.shap_tau[mask].mean(axis=0)

        if len(seg_shap) < 2:
            logger.warning("[UpliftSHAP] Not enough segments for comparison plot.")
            return plt.figure()

        n_segs = len(seg_shap)
        fig, axes = plt.subplots(
            1, n_segs, figsize=(5 * n_segs, max(4, len(self.feature_cols_used) * 0.7)),
            sharey=True,
        )
        if n_segs == 1:
            axes = [axes]

        seg_colors = {
            "Persuadables": "#2ecc71", "Sleeping Dogs": "#e74c3c",
            "Sure Things":  "#3498db", "Lost Causes":   "#95a5a6",
        }

        for ax, (seg, mean_sv) in zip(axes, seg_shap.items()):
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in mean_sv]
            order  = np.argsort(np.abs(mean_sv))
            ax.barh(
                [self.feature_cols_used[i] for i in order],
                [mean_sv[i] for i in order],
                color=[colors[i] for i in order],
                alpha=0.85,
            )
            mask  = self.segment_explain == seg
            ax.set_title(
                f"{seg}\n(n={mask.sum()})",
                fontsize=11, fontweight="bold",
                color=seg_colors.get(seg, "#333"),
            )
            ax.axvline(0, color="#333", lw=0.8)
            ax.set_xlabel("Mean SHAP(τ̂)", fontsize=10)
            ax.grid(True, alpha=0.3, axis="x")

        fig.suptitle(
            "Uplift SHAP by Customer Segment\n"
            "(positive SHAP = pushes toward Persuadable)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[UpliftSHAP] Segment comparison saved → %s", save_path)
        return fig

    def get_importance_df(self) -> pd.DataFrame:
        """
        Returns DataFrame: feature, mean_abs_shap_tau, sorted descending.
        """
        mean_abs = np.abs(self.shap_tau).mean(axis=0)
        return (
            pd.DataFrame({
                "feature":      self.feature_cols_used,
                "mean_shap_tau": self.shap_tau.mean(axis=0),
                "mean_abs_shap": mean_abs,
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


# =============================================================================
# 3. Convenience wrapper
# =============================================================================

def explain_decisions(
    waf,
    df_scaled: pd.DataFrame,
    feature_cols: List[str],
    weibull_decisions: pd.DataFrame,
    uplift_results: Optional[dict] = None,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    n_background: int = 100,
    n_explain: int = 300,
    n_local_examples: int = 3,
) -> dict:
    """
    Run both SurvivalShapExplainer and UpliftShapExplainer, save all figures.

    Parameters
    ----------
    waf : WeibullAFTFitter
        Fitted survival model.
    df_scaled : pd.DataFrame
        Preprocessed customer features.
    feature_cols : list of str
        Active feature names.
    weibull_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions().
    uplift_results : dict, optional
        Output of run_uplift_analysis() — must include 'uplift_df',
        and optionally 'mu_1_model', 'mu_0_model', 'scaler', 'imputer'.
    save_dir : str, optional
        Directory to save all SHAP figures. Created if needed.
    dataset_label : str
        Dataset name for plot titles.
    n_background : int
        Background samples for KernelExplainer.
    n_explain : int
        Customers to explain.
    n_local_examples : int
        Number of individual customer local explanations to generate.

    Returns
    -------
    dict
        {
          'survival_explainer': SurvivalShapExplainer,
          'uplift_explainer':   UpliftShapExplainer or None,
          'importance_survival': pd.DataFrame,
          'importance_uplift':   pd.DataFrame or None,
          'figs':               dict of matplotlib Figures,
        }
    """
    if not _SHAP_AVAILABLE:
        logger.warning("[Explainability] SHAP not available — skipping explain_decisions.")
        return {}

    figs = {}

    # ── 1. Survival SHAP ───────────────────────────────────────────────────────
    logger.info("[Explainability] Initialising SurvivalShapExplainer...")
    surv_exp = SurvivalShapExplainer(
        waf=waf,
        df_scaled=df_scaled,
        feature_cols=feature_cols,
        n_background=n_background,
        n_explain=n_explain,
    )

    figs["survival_global"] = surv_exp.global_summary(
        save_path=os.path.join(save_dir, "shap_survival_global.png") if save_dir else None,
        title=f"Survival Model SHAP — {dataset_label}",
    )
    figs["survival_segments"] = surv_exp.compare_segments(
        weibull_decisions=weibull_decisions,
        save_path=os.path.join(save_dir, "shap_survival_segments.png") if save_dir else None,
    )

    # Local explanations for sample customers
    for i in range(min(n_local_examples, len(surv_exp.X_explain))):
        figs[f"survival_local_{i}"] = surv_exp.explain_customer(
            customer_idx=i,
            save_path=os.path.join(save_dir, f"shap_survival_local_{i}.png") if save_dir else None,
        )

    importance_survival = surv_exp.get_importance_df()
    if save_dir:
        importance_survival.to_csv(os.path.join(save_dir, "shap_importance_survival.csv"), index=False)

    # ── 2. Uplift SHAP ─────────────────────────────────────────────────────────
    uplift_exp = None
    importance_uplift = None

    if uplift_results and "uplift_df" in uplift_results:
        uplift_df = uplift_results["uplift_df"]
        from src.uplift import _UPLIFT_FEATURE_COLS
        uplift_feature_cols = [c for c in _UPLIFT_FEATURE_COLS if c in uplift_df.columns]

        logger.info("[Explainability] Initialising UpliftShapExplainer...")
        uplift_exp = UpliftShapExplainer(
            uplift_df=uplift_df,
            feature_cols=uplift_feature_cols,
            mu_1_model=uplift_results.get("mu_1_model"),
            mu_0_model=uplift_results.get("mu_0_model"),
            scaler=uplift_results.get("scaler"),
            imputer=uplift_results.get("imputer"),
            n_background=n_background,
            n_explain=n_explain,
        )

        figs["uplift_global"] = uplift_exp.global_summary(
            save_path=os.path.join(save_dir, "shap_uplift_global.png") if save_dir else None,
            title=f"Uplift Model SHAP (τ̂) — {dataset_label}",
        )
        figs["uplift_segments"] = uplift_exp.compare_uplift_segments(
            save_path=os.path.join(save_dir, "shap_uplift_segments.png") if save_dir else None,
        )

        # Local examples — pick one Persuadable and one Sleeping Dog
        for seg, label in [("Persuadables", "persuadable"), ("Sleeping Dogs", "sleeping_dog")]:
            seg_positions = np.where(uplift_exp.segment_explain == seg)[0]
            if len(seg_positions) > 0:
                pos = int(seg_positions[0])
                fig, _ = uplift_exp.explain_customer(
                    customer_pos=pos,
                    save_path=os.path.join(save_dir, f"shap_uplift_local_{label}.png") if save_dir else None,
                )
                figs[f"uplift_local_{label}"] = fig

        importance_uplift = uplift_exp.get_importance_df()
        if save_dir:
            importance_uplift.to_csv(os.path.join(save_dir, "shap_importance_uplift.csv"), index=False)
    else:
        logger.info("[Explainability] No uplift results provided — skipping UpliftShapExplainer.")

    logger.info("[Explainability] All SHAP figures generated (%d total).", len(figs))

    return {
        "survival_explainer": surv_exp,
        "uplift_explainer":   uplift_exp,
        "importance_survival": importance_survival,
        "importance_uplift":   importance_uplift,
        "figs":               figs,
    }
