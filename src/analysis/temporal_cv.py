"""
src/analysis/temporal_cv.py
=============================
Time-Based (Temporal) Cross-Validation for Survival Models.

Motivation
----------
Random k-fold CV can leak future information into training because customers
who churned late may have their features correlated with earlier customers.
Temporal CV ensures the model is always trained on past data and evaluated
on future data, mimicking real deployment conditions.

Method: Expanding Window (Walk-Forward Validation)
---------------------------------------------------
Sort customers by estimated first-purchase date:
  first_purchase_proxy = snapshot - Recency - T  (approximate reconstruction)

Fold k (of K):
  Train : customers whose first purchase is in the earliest (k/K) × N quantile
  Test  : customers whose first purchase is in the NEXT (1/K) × N quantile

For each fold:
  1. Fit Weibull AFT on train set (with VIF check + penalizer grid search)
  2. Evaluate C-index on test set
  3. Record IBS on test set

Key outputs
-----------
  - C-index per fold (array of K values)
  - Mean ± std C-index across folds
  - Stability plot: C-index vs fold number
  - Comparison with random-split C-index (bias analysis)

Notes on Administrative Censoring
----------------------------------
Late-cohort customers (test folds) have shorter observation windows due to
study end-date censoring. This is a known limitation: late buyers have LESS
time to re-purchase before the study ends, so their E=1 (churn) rate is
artificially lower. We document this explicitly in fold statistics.
"""

import logging
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict

logger = logging.getLogger(__name__)


# =============================================================================
# TemporalCrossValidator
# =============================================================================

class TemporalCrossValidator:
    """
    Expanding-window temporal cross-validator for Weibull AFT survival models.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level features indexed by CustomerID.
        Must contain: Recency, T, E, plus active feature columns.
    active_features : list of str
        Feature columns to use (after VIF pruning from the main pipeline).
    n_folds : int
        Number of temporal folds (default: 4).
    min_train_fraction : float
        Minimum fraction of data used as training in fold 1 (default: 0.40).
    n_boot_per_fold : int
        Bootstrap samples for C-index CI per fold (default: 100).
    seed : int
    """

    def __init__(
        self,
        customer_df: pd.DataFrame,
        active_features: List[str],
        n_folds: int = 4,
        min_train_fraction: float = 0.40,
        n_boot_per_fold: int = 100,
        seed: int = 42,
    ):
        self.customer_df          = customer_df.copy()
        self.active_features      = active_features
        self.n_folds              = n_folds
        self.min_train_fraction   = min_train_fraction
        self.n_boot_per_fold      = n_boot_per_fold
        self.seed                 = seed

        # Estimate first-purchase ordering:
        # first_purchase_rank ∝ Recency + T  (higher = bought earlier relative to snapshot)
        # Customers who bought a long time ago AND have long observation windows are oldest.
        self._order_key = (
            customer_df["Recency"].fillna(0) + customer_df["T"].fillna(0)
        )
        # Sort descending: highest value = earliest buyer
        self._sorted_idx = np.argsort(self._order_key.values)[::-1]
        self._n          = len(customer_df)

        logger.info(
            "[TemporalCV] Ready | n=%d | n_folds=%d | features=%s",
            self._n, n_folds, active_features,
        )

    # =========================================================================
    # Fold generator
    # =========================================================================

    def _get_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate (train_idx, test_idx) pairs using expanding window strategy.

        Fold k covers:
          train : sorted_idx[ : round(n * split_points[k]) ]
          test  : sorted_idx[ round(n * split_points[k]) :
                              round(n * split_points[k+1]) ]
        """
        # Expanding split points: from min_train_fraction to (K/K+1)
        step = (1.0 - self.min_train_fraction) / self.n_folds
        split_points = [
            self.min_train_fraction + k * step
            for k in range(self.n_folds + 1)
        ]

        folds = []
        for k in range(self.n_folds):
            train_end = int(round(self._n * split_points[k]))
            test_end  = int(round(self._n * split_points[k + 1]))
            train_idx = self._sorted_idx[:train_end]
            test_idx  = self._sorted_idx[train_end:test_end]
            if len(train_idx) < 30 or len(test_idx) < 10:
                logger.warning(
                    "[TemporalCV] Fold %d: train=%d / test=%d too small — skipping",
                    k + 1, len(train_idx), len(test_idx),
                )
                continue
            folds.append((train_idx, test_idx))
        return folds

    # =========================================================================
    # Single fold evaluation
    # =========================================================================

    def _evaluate_fold(
        self,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold_num: int,
    ) -> dict:
        """
        Train Weibull AFT on train split, evaluate on test split.
        Returns dict with C-index, IBS, fold metadata.
        """
        from src.models import train_weibull_aft, SURVIVAL_FEATURES
        from src.evaluation import compute_c_index, compute_integrated_brier_score

        # Restrict to standard survival features + T + E to avoid contamination
        # from extra columns (RFM scores, evi, etc.) that may be in processed_data
        safe_cols = [c for c in SURVIVAL_FEATURES + ["T", "E"]
                     if c in self.customer_df.columns]
        df_train = self.customer_df.iloc[train_idx][safe_cols]
        df_test  = self.customer_df.iloc[test_idx][safe_cols]

        n_train_events = int(df_train["E"].sum())
        n_test_events  = int(df_test["E"].sum())

        if n_train_events < 10:
            logger.warning(
                "[TemporalCV] Fold %d: only %d events in train — skipping.",
                fold_num, n_train_events,
            )
            return {"fold": fold_num, "skipped": True}

        logger.info(
            "[TemporalCV] Fold %d | train: n=%d (events=%d, churn=%.1f%%) | "
            "test: n=%d (events=%d, churn=%.1f%%)",
            fold_num,
            len(df_train), n_train_events, df_train["E"].mean() * 100,
            len(df_test),  n_test_events,  df_test["E"].mean() * 100,
        )

        # Train Weibull AFT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waf, df_scaled_train, preprocessor, fold_active_feats = \
                train_weibull_aft(df_train)

        # Transform test set (no re-fitting)
        from src.models import get_survival_features, SURVIVAL_FEATURES

        def _transform_test(df_src):
            input_feats = [f for f in SURVIVAL_FEATURES if f in df_src.columns]
            X = preprocessor.transform(df_src[input_feats])
            df_sc = pd.DataFrame(X, columns=input_feats, index=df_src.index)
            df_sc = df_sc[[f for f in fold_active_feats if f in df_sc.columns]].copy()
            df_sc["T"] = df_src["T"].values
            df_sc["E"] = df_src["E"].values
            return df_sc

        df_scaled_test = _transform_test(df_test)

        # Evaluate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                c_index_train = float(waf.score(df_scaled_train,
                                                  scoring_method="concordance_index"))
                c_index_test  = float(waf.score(df_scaled_test,
                                                  scoring_method="concordance_index"))
            except Exception:
                c_index_train = compute_c_index(waf, df_scaled_train)
                c_index_test  = compute_c_index(waf, df_scaled_test)

            try:
                ibs_test = compute_integrated_brier_score(waf, df_scaled_test)
            except Exception:
                ibs_test = float("nan")

        # Bootstrap CI for test C-index
        rng = np.random.default_rng(self.seed + fold_num)
        boot_cindex = []
        for _ in range(self.n_boot_per_fold):
            boot_idx   = rng.integers(0, len(df_scaled_test), size=len(df_scaled_test))
            df_boot    = df_scaled_test.iloc[boot_idx]
            if df_boot["E"].sum() < 2:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bc = float(waf.score(df_boot, scoring_method="concordance_index"))
                boot_cindex.append(bc)
            except Exception:
                pass

        ci_lo = float(np.percentile(boot_cindex, 2.5))  if boot_cindex else float("nan")
        ci_hi = float(np.percentile(boot_cindex, 97.5)) if boot_cindex else float("nan")

        result = {
            "fold":               fold_num,
            "skipped":            False,
            "n_train":            len(df_train),
            "n_test":             len(df_test),
            "n_train_events":     n_train_events,
            "n_test_events":      n_test_events,
            "train_churn_rate":   float(df_train["E"].mean()),
            "test_churn_rate":    float(df_test["E"].mean()),
            "c_index_train":      round(c_index_train, 4),
            "c_index_test":       round(c_index_test, 4),
            "c_index_ci_lo":      round(ci_lo, 4),
            "c_index_ci_hi":      round(ci_hi, 4),
            "ibs_test":           round(ibs_test, 4) if not np.isnan(ibs_test) else None,
            "overfitting_gap":    round(c_index_train - c_index_test, 4),
            "active_features":    fold_active_feats,
        }
        logger.info(
            "[TemporalCV] Fold %d result: C-index train=%.4f | test=%.4f "
            "[%.4f, %.4f] | IBS=%.4f | gap=%.4f",
            fold_num, c_index_train, c_index_test, ci_lo, ci_hi,
            ibs_test if not np.isnan(ibs_test) else -1,
            c_index_train - c_index_test,
        )
        return result

    # =========================================================================
    # Full CV run
    # =========================================================================

    def run(self) -> pd.DataFrame:
        """
        Run all temporal folds and return a results DataFrame.
        """
        folds = self._get_folds()
        if not folds:
            logger.error("[TemporalCV] No valid folds generated.")
            return pd.DataFrame()

        rows = []
        for k, (train_idx, test_idx) in enumerate(folds, start=1):
            result = self._evaluate_fold(train_idx, test_idx, fold_num=k)
            rows.append(result)

        df = pd.DataFrame(rows)
        valid = df[~df.get("skipped", pd.Series(False, index=df.index))]

        if not valid.empty:
            mean_c = valid["c_index_test"].mean()
            std_c  = valid["c_index_test"].std()
            logger.info(
                "[TemporalCV] Summary | Mean C-index (test): %.4f ± %.4f "
                "| Folds: %d/%d valid",
                mean_c, std_c, len(valid), len(df),
            )

        return df

    # =========================================================================
    # Plotting
    # =========================================================================

    def plot_fold_stability(
        self,
        fold_df: pd.DataFrame,
        random_cv_cindex: Optional[float] = None,
        random_cv_std: Optional[float] = None,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Plot C-index across temporal folds with 95% CI bands.
        Optionally overlay the random-split CV result for comparison.
        """
        valid = fold_df[~fold_df.get("skipped", pd.Series(False, index=fold_df.index))]
        if valid.empty:
            return plt.figure()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Left: C-index per fold ────────────────────────────────────────────
        ax = axes[0]
        folds = valid["fold"].values
        c_test  = valid["c_index_test"].values
        c_train = valid["c_index_train"].values
        ci_lo   = valid["c_index_ci_lo"].values
        ci_hi   = valid["c_index_ci_hi"].values

        ax.plot(folds, c_train, "s--", color="#f39c12", lw=1.8, ms=7,
                label="C-index (train)", zorder=3)
        ax.plot(folds, c_test,  "o-",  color="#3498db", lw=2.5, ms=8,
                label="C-index (test)", zorder=4)
        ax.fill_between(folds, ci_lo, ci_hi, alpha=0.2, color="#3498db",
                        label="95% Bootstrap CI")

        # Reference lines
        ax.axhline(0.70, color="#e74c3c", lw=1.5, ls="--", alpha=0.7,
                   label="0.70 threshold")
        ax.axhline(c_test.mean(), color="#3498db", lw=1.0, ls=":",
                   alpha=0.6, label=f"Mean={c_test.mean():.4f}")

        if random_cv_cindex is not None:
            ax.axhline(random_cv_cindex, color="#2ecc71", lw=2.0, ls="-.",
                       label=f"Random CV={random_cv_cindex:.4f}")
            if random_cv_std:
                ax.fill_between(
                    [folds[0] - 0.3, folds[-1] + 0.3],
                    [random_cv_cindex - random_cv_std] * 2,
                    [random_cv_cindex + random_cv_std] * 2,
                    alpha=0.1, color="#2ecc71",
                )

        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {k}" for k in folds])
        ax.set_ylim(max(0, c_test.min() - 0.1), min(1.0, c_test.max() + 0.1))
        ax.set_xlabel("Temporal Fold (early -> late cohorts)", fontsize=11)
        ax.set_ylabel("Concordance Index (C-index)", fontsize=11)
        ax.set_title(f"Temporal Cross-Validation — {dataset_label}\n"
                     f"(expanding window, test C-index per fold)",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(True, alpha=0.3)

        # Overfitting gap annotation
        gap_mean = valid["overfitting_gap"].mean()
        gap_color = "#e74c3c" if gap_mean > 0.05 else "#2ecc71"
        ax.annotate(
            f"Avg train-test gap: {gap_mean:+.4f}\n"
            f"({'Possible overfit' if gap_mean > 0.05 else 'Healthy generalisation'})",
            xy=(0.98, 0.05), xycoords="axes fraction",
            ha="right", fontsize=9, color=gap_color,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        # ── Right: Fold metadata (churn rate trend) ───────────────────────────
        ax2 = axes[1]
        width = 0.35
        x = np.arange(len(folds))
        ax2.bar(x - width/2, valid["train_churn_rate"] * 100, width,
                color="#f39c12", alpha=0.8, label="Train Churn %")
        ax2.bar(x + width/2, valid["test_churn_rate"] * 100, width,
                color="#3498db", alpha=0.8, label="Test Churn %")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"Fold {k}" for k in folds])
        ax2.set_xlabel("Temporal Fold", fontsize=11)
        ax2.set_ylabel("Churn Rate (%)", fontsize=11)
        ax2.set_title("Administrative Censoring Effect\n"
                      "(later folds have lower churn rate due to shorter observation window)",
                      fontsize=11, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y")

        # Trend line for churn rate
        if len(folds) >= 3:
            z_train = np.polyfit(folds, valid["train_churn_rate"] * 100, 1)
            z_test  = np.polyfit(folds, valid["test_churn_rate"] * 100, 1)
            p_train = np.poly1d(z_train)
            p_test  = np.poly1d(z_test)
            ax2.plot(x - width/2, p_train(np.array(folds)), "o--",
                     color="#f39c12", alpha=0.6, ms=5)
            ax2.plot(x + width/2, p_test(np.array(folds)), "o--",
                     color="#3498db", alpha=0.6, ms=5)

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[TemporalCV] Stability plot saved -> %s", save_path)
        return fig

    def get_summary(self, fold_df: pd.DataFrame) -> dict:
        """
        Return a dict of key temporal CV statistics for reporting.
        """
        valid = fold_df[~fold_df.get("skipped", pd.Series(False, index=fold_df.index))]
        if valid.empty:
            return {}
        return {
            "n_folds":              len(valid),
            "mean_c_index_test":    round(valid["c_index_test"].mean(), 4),
            "std_c_index_test":     round(valid["c_index_test"].std(),  4),
            "min_c_index_test":     round(valid["c_index_test"].min(),  4),
            "max_c_index_test":     round(valid["c_index_test"].max(),  4),
            "mean_overfitting_gap": round(valid["overfitting_gap"].mean(), 4),
            "all_above_07":         bool((valid["c_index_test"] >= 0.70).all()),
            "admin_censoring_drop": round(
                valid["test_churn_rate"].iloc[0] - valid["test_churn_rate"].iloc[-1], 4
            ) if len(valid) > 1 else 0.0,
        }


# =============================================================================
# Convenience wrapper
# =============================================================================

def temporal_cross_validate(
    customer_df: pd.DataFrame,
    active_features: List[str],
    n_folds: int = 4,
    random_cv_cindex: Optional[float] = None,
    random_cv_std: Optional[float] = None,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    seed: int = 42,
) -> dict:
    """
    One-call temporal CV: run expanding-window folds, plot results, save outputs.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer-level features.
    active_features : list of str
        VIF-pruned feature names.
    n_folds : int
        Number of temporal folds (default: 4).
    random_cv_cindex : float, optional
        C-index from standard random-split CV (for comparison overlay).
    random_cv_std : float, optional
        Std of random-split C-index.
    save_dir : str, optional
        Directory to save plots and CSV.
    dataset_label : str

    Returns
    -------
    dict with keys:
        validator  : TemporalCrossValidator instance
        fold_df    : pd.DataFrame of per-fold results
        summary    : dict of aggregate statistics
        fig        : matplotlib Figure
    """
    validator = TemporalCrossValidator(
        customer_df=customer_df,
        active_features=active_features,
        n_folds=n_folds,
        seed=seed,
    )

    logger.info("[TemporalCV] Running %d temporal folds...", n_folds)
    fold_df = validator.run()
    summary = validator.get_summary(fold_df)

    save_path = os.path.join(save_dir, "temporal_cv_stability.png") if save_dir else None
    fig = validator.plot_fold_stability(
        fold_df,
        random_cv_cindex=random_cv_cindex,
        random_cv_std=random_cv_std,
        save_path=save_path,
        dataset_label=dataset_label,
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fold_df.to_csv(os.path.join(save_dir, "temporal_cv_folds.csv"), index=False)

    logger.info(
        "[TemporalCV] Summary: mean_C=%.4f +/- %.4f | all>=0.70: %s",
        summary.get("mean_c_index_test", 0),
        summary.get("std_c_index_test", 0),
        summary.get("all_above_07", False),
    )

    return {
        "validator": validator,
        "fold_df":   fold_df,
        "summary":   summary,
        "fig":       fig,
    }
