"""
src/causal/causal_forest.py
============================
Doubly-Robust Learner (DR-Learner) — Cross-fit CATE Estimation.

Addresses the core limitation of the T-Learner + IPTW approach:
  T-Learner is biased when treatment is highly imbalanced (5% INTERVENE)
  because the treated-branch model mu_1 is trained on very few samples.

DR-Learner (also called DML / R-Learner / AIPW-Learner) fixes this via:
  1. Cross-fitting: nuisance models fit on held-out folds (prevents overfitting)
  2. Doubly-robust score: unbiased if EITHER outcome OR propensity model correct
  3. Final CATE via flexible meta-learner on pseudo-outcomes

Algorithm (Robinson 1988, Chernozhukov et al. 2018, Nie & Wager 2021)
----------------------------------------------------------------------
K-fold cross-fitting (K=5):
  Fold k: fit m(x) and e(x) on folds != k, evaluate on fold k

DR pseudo-outcome per customer i:
  psi_i = [mu_1(x_i) - mu_0(x_i)]                     <-- direct model term
          + (Y_i - mu(x_i, T_i)) * (T_i - e(x_i))     <-- IPW correction
            / max(e(x_i) * (1 - e(x_i)), clip)

Final CATE:  tau_hat(x) = E[psi | X = x]   (fit GradientBoosting on psi)

Properties
----------
- Consistent if EITHER m(x,t) or e(x) is correctly specified (doubly robust)
- Root-n asymptotically normal under regularity conditions
- Cross-fitting removes finite-sample bias from high-dimensional nuisance
- Bootstrap provides valid CI for CATE distribution and ATE/ATT
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
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

_CLIP_PROPENSITY = 0.05   # clip e(x) away from 0/1 to control IPW variance


# =============================================================================
# DRLearner
# =============================================================================

class DRLearner:
    """
    Doubly-Robust Learner for Heterogeneous Treatment Effect Estimation.

    Parameters
    ----------
    n_folds : int
        Cross-fitting folds (default: 5).
    n_estimators : int
        Trees in each GradientBoosting model (default: 200).
    max_depth : int
        Tree depth for nuisance and CATE models (default: 3).
    n_bootstrap : int
        Bootstrap iterations for CI (default: 200). Set 0 to skip.
    clip_propensity : float
        Clip propensity scores to [clip, 1-clip] (default: 0.05).
    seed : int
    """

    def __init__(
        self,
        n_folds: int = 5,
        n_estimators: int = 200,
        max_depth: int = 3,
        n_bootstrap: int = 200,
        clip_propensity: float = _CLIP_PROPENSITY,
        seed: int = 42,
    ):
        self.n_folds         = n_folds
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.n_bootstrap     = n_bootstrap
        self.clip_propensity = clip_propensity
        self.seed            = seed
        self.rng             = np.random.default_rng(seed)

        self._fitted         = False
        self.feature_names_  = None
        self.ate_            = None
        self.att_            = None
        self.atc_            = None
        self.ate_ci_         = None
        self.att_ci_         = None
        self.tau_hat_        = None   # CATE per sample
        self.psi_            = None   # DR pseudo-outcomes
        self.propensity_     = None

    # =========================================================================
    # Nuisance model builder
    # =========================================================================

    def _make_outcome_model(self):
        return Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gbm",   GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.seed,
            )),
        ])

    def _make_propensity_model(self):
        return Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gbm",   GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.seed,
            )),
        ])

    def _make_cate_model(self):
        return Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gbm",   GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.seed,
            )),
        ])

    # =========================================================================
    # Fit
    # =========================================================================

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "DRLearner":
        """
        Fit the DR-Learner.

        Parameters
        ----------
        X : ndarray (n, p)
        Y : ndarray (n,)   outcome
        T : ndarray (n,)   binary treatment (0/1)
        feature_names : list of str, optional

        Returns self.
        """
        n, p = X.shape
        self.feature_names_ = feature_names or [f"x{i}" for i in range(p)]

        logger.info(
            "[DRLearner] Fitting | n=%d | p=%d | treat_rate=%.2f%% | n_folds=%d",
            n, p, T.mean() * 100, self.n_folds,
        )

        psi     = np.zeros(n)          # DR pseudo-outcomes
        e_hat   = np.zeros(n)          # propensity scores
        mu1_hat = np.zeros(n)          # E[Y|X, T=1]
        mu0_hat = np.zeros(n)          # E[Y|X, T=0]

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]
            T_tr, T_val = T[train_idx], T[val_idx]

            t1 = T_tr == 1
            t0 = T_tr == 0

            if t1.sum() < 5 or t0.sum() < 5:
                logger.warning("[DRLearner] Fold %d: too few treated/control — skipping.", fold)
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # ── Outcome models ────────────────────────────────────────────
                m1 = self._make_outcome_model()
                m0 = self._make_outcome_model()
                m1.fit(X_tr[t1], Y_tr[t1])
                m0.fit(X_tr[t0], Y_tr[t0])

                mu1_hat[val_idx] = m1.predict(X_val)
                mu0_hat[val_idx] = m0.predict(X_val)

                # ── Propensity model ──────────────────────────────────────────
                ep = self._make_propensity_model()
                ep.fit(X_tr, T_tr)
                e_hat[val_idx] = np.clip(
                    ep.predict_proba(X_val)[:, 1],
                    self.clip_propensity,
                    1.0 - self.clip_propensity,
                )

            # ── DR pseudo-outcome ─────────────────────────────────────────────
            # psi_i = (mu1 - mu0) + (Y - mu(X,T)) * (T - e) / (e*(1-e))
            mu_val = np.where(T_val == 1, mu1_hat[val_idx], mu0_hat[val_idx])
            psi[val_idx] = (
                (mu1_hat[val_idx] - mu0_hat[val_idx])
                + (Y_val - mu_val) * (T_val - e_hat[val_idx])
                / (e_hat[val_idx] * (1.0 - e_hat[val_idx]))
            )

            logger.info(
                "[DRLearner] Fold %d | mean_psi=%.4f | e range=[%.3f,%.3f]",
                fold,
                psi[val_idx].mean(),
                e_hat[val_idx].min(),
                e_hat[val_idx].max(),
            )

        # ── Fit final CATE model on pseudo-outcomes ───────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._cate_model = self._make_cate_model()
            self._cate_model.fit(X, psi)

        tau_hat = self._cate_model.predict(X)

        self.psi_        = psi
        self.propensity_ = e_hat
        self.tau_hat_    = tau_hat
        self._X          = X
        self._T          = T

        # ── Point estimates ───────────────────────────────────────────────────
        self.ate_ = float(psi.mean())
        self.att_ = float(psi[T == 1].mean()) if (T == 1).sum() > 0 else float("nan")
        self.atc_ = float(psi[T == 0].mean()) if (T == 0).sum() > 0 else float("nan")

        logger.info(
            "[DRLearner] ATE=%.4f | ATT=%.4f | ATC=%.4f",
            self.ate_, self.att_, self.atc_,
        )

        # ── Bootstrap CI ─────────────────────────────────────────────────────
        if self.n_bootstrap > 0:
            self._compute_bootstrap_ci(X, psi, T)

        self._fitted = True
        logger.info("[DRLearner] Fit complete.")
        return self

    def _compute_bootstrap_ci(self, X, psi, T, alpha=0.05):
        ate_boots = np.empty(self.n_bootstrap)
        att_boots = np.empty(self.n_bootstrap)
        atc_boots = np.empty(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            idx = self.rng.integers(0, len(X), size=len(X))
            ate_boots[b] = psi[idx].mean()
            att_boots[b] = psi[idx][T[idx] == 1].mean() if (T[idx] == 1).sum() > 0 else np.nan
            atc_boots[b] = psi[idx][T[idx] == 0].mean() if (T[idx] == 0).sum() > 0 else np.nan

        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
        self.ate_ci_ = (
            float(np.percentile(ate_boots, lo)),
            float(np.percentile(ate_boots, hi)),
        )
        self.att_ci_ = (
            float(np.nanpercentile(att_boots, lo)),
            float(np.nanpercentile(att_boots, hi)),
        )
        self.atc_ci_ = (
            float(np.nanpercentile(atc_boots, lo)),
            float(np.nanpercentile(atc_boots, hi)),
        )
        logger.info(
            "[DRLearner] Bootstrap CI (n=%d) | ATE [%.4f, %.4f] | ATT [%.4f, %.4f]",
            self.n_bootstrap,
            *self.ate_ci_, *self.att_ci_,
        )

    # =========================================================================
    # Predict
    # =========================================================================

    def effect(self, X: np.ndarray) -> np.ndarray:
        """Predict CATE for new observations."""
        return self._cate_model.predict(X)

    # =========================================================================
    # Feature importance for tau_hat
    # =========================================================================

    def cate_feature_importance(self) -> pd.DataFrame:
        """
        Feature importance of the final CATE model (GBM).
        Measures how much each feature drives heterogeneity in treatment effect.
        """
        gbm = self._cate_model.named_steps["gbm"]
        imp = gbm.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def get_summary(self) -> dict:
        pct_positive = float((self.tau_hat_ > 0).mean()) * 100
        pct_negative = float((self.tau_hat_ < 0).mean()) * 100
        return {
            "ATE":              round(self.ate_, 4),
            "ATT":              round(self.att_, 4),
            "ATC":              round(self.atc_, 4),
            "ATE_CI_lo":        round(self.ate_ci_[0], 4) if self.ate_ci_ else None,
            "ATE_CI_hi":        round(self.ate_ci_[1], 4) if self.ate_ci_ else None,
            "ATT_CI_lo":        round(self.att_ci_[0], 4) if self.att_ci_ else None,
            "ATT_CI_hi":        round(self.att_ci_[1], 4) if self.att_ci_ else None,
            "pct_positive_cate":round(pct_positive, 2),
            "pct_negative_cate":round(pct_negative, 2),
            "n_samples":        len(self.tau_hat_),
            "n_treated":        int((self._T == 1).sum()),
        }

    # =========================================================================
    # Plots
    # =========================================================================

    def plot_cate_distribution(
        self,
        uplift_segments: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Histogram of individual CATE estimates, coloured by uplift segment.
        Annotates ATE and ATT with bootstrap CI.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Left: CATE distribution ──────────────────────────────────────────
        ax = axes[0]
        tau = self.tau_hat_
        bins = min(60, max(20, len(tau) // 100))

        ax.hist(tau[self._T == 0], bins=bins, alpha=0.55,
                color="#3498db", label="Control", density=True)
        ax.hist(tau[self._T == 1], bins=bins, alpha=0.55,
                color="#e74c3c", label="Treated (INTERVENE)", density=True)

        # ATE + CI
        ax.axvline(self.ate_, color="#2c3e50", lw=2.5, ls="-", label=f"ATE={self.ate_:.4f}")
        ax.axvline(self.att_, color="#e74c3c", lw=2.0, ls="--", label=f"ATT={self.att_:.4f}")
        ax.axvline(0, color="#95a5a6", lw=1.2, ls=":")

        if self.ate_ci_:
            ax.axvspan(self.ate_ci_[0], self.ate_ci_[1], alpha=0.12,
                       color="#2c3e50", label=f"ATE 95% CI [{self.ate_ci_[0]:.3f},{self.ate_ci_[1]:.3f}]")

        pct_pos = (tau > 0).mean() * 100
        ax.set_xlabel("Individual CATE tau_hat(x)", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"CATE Distribution — DR-Learner ({dataset_label})\n"
            f"{pct_pos:.1f}% positive (Persuadables), {100-pct_pos:.1f}% negative (Sleeping Dogs)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Right: ATE / ATT / ATC with CI bars ─────────────────────────────
        ax2 = axes[1]
        labels  = ["ATE", "ATT\n(INTERVENE)", "ATC\n(WAIT)"]
        vals    = [self.ate_, self.att_, self.atc_]
        cis     = [
            self.ate_ci_ or (self.ate_, self.ate_),
            self.att_ci_ or (self.att_, self.att_),
            self.atc_ci_ or (self.atc_, self.atc_),
        ]
        colors  = ["#2c3e50", "#e74c3c", "#3498db"]
        x_pos   = np.arange(len(labels))

        bars = ax2.bar(x_pos, vals, color=colors, alpha=0.85, width=0.5)
        for bar, val, ci, color in zip(bars, vals, cis, colors):
            ax2.errorbar(
                bar.get_x() + bar.get_width() / 2, val,
                yerr=[[val - ci[0]], [ci[1] - val]],
                fmt="none", color="black", capsize=8, lw=2.5,
            )
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                max(val, 0) + abs(max(vals, key=abs)) * 0.03,
                f"{val:+.4f}\n[{ci[0]:.3f},{ci[1]:.3f}]",
                ha="center", fontsize=9, color=color, fontweight="bold",
            )

        ax2.axhline(0, color="#ccc", lw=0.8)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(labels, fontsize=11)
        ax2.set_ylabel("Treatment Effect (outcome units)", fontsize=11)
        ax2.set_title(
            "ATE / ATT / ATC — DR-Learner with 95% Bootstrap CI\n"
            "(DR-Learner is doubly robust: consistent if either m or e is correct)",
            fontsize=11, fontweight="bold",
        )
        ax2.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("[DRLearner] CATE distribution plot saved -> %s", save_path)
        return fig

    def plot_feature_importance(
        self,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Feature importance for the CATE model.
        Answers: 'What drives heterogeneity in treatment response?'
        """
        imp_df = self.cate_feature_importance()
        fig, ax = plt.subplots(figsize=(9, max(4, len(imp_df) * 0.6)))

        colors = ["#e74c3c" if v >= imp_df["importance"].median()
                  else "#3498db" for v in imp_df["importance"]]
        ax.barh(imp_df["feature"], imp_df["importance"], color=colors, alpha=0.85)
        ax.set_xlabel("Feature Importance (CATE heterogeneity)", fontsize=11)
        ax.set_title(
            f"CATE Feature Importance — DR-Learner ({dataset_label})\n"
            f"(Which features drive who is a Persuadable vs Sleeping Dog?)",
            fontsize=12, fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_cate_vs_propensity(
        self,
        save_path: Optional[str] = None,
        dataset_label: str = "",
    ) -> plt.Figure:
        """
        Scatter: CATE vs propensity score — reveals selection mechanism.
        Customers with high propensity (likely to be treated) should ideally
        have high CATE (respond to treatment) for good policy overlap.
        """
        fig, ax = plt.subplots(figsize=(9, 6))
        colors_pt = np.where(self.tau_hat_ > 0, "#2ecc71", "#e74c3c")
        scatter = ax.scatter(
            self.propensity_, self.tau_hat_,
            c=colors_pt, alpha=0.25, s=8, rasterized=True,
        )
        ax.axhline(0, color="#333", lw=1.0, ls="--")
        ax.axhline(self.ate_, color="#2c3e50", lw=1.5, ls="-",
                   label=f"ATE={self.ate_:.4f}")
        ax.set_xlabel("Propensity Score e(X) = P(T=1|X)", fontsize=11)
        ax.set_ylabel("CATE tau_hat(x)", fontsize=11)
        ax.set_title(
            f"CATE vs Propensity Score — {dataset_label}\n"
            f"Green=Persuadables (tau>0), Red=Sleeping Dogs (tau<0)",
            fontsize=12, fontweight="bold",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


# =============================================================================
# Convenience wrapper
# =============================================================================

def run_causal_forest(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    n_folds: int = 5,
    n_estimators: int = 200,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """
    One-call DR-Learner: fit, summarise, plot.

    Returns
    -------
    dict with keys: learner, summary, importance_df, figs
    """
    learner = DRLearner(
        n_folds=n_folds,
        n_estimators=n_estimators,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    learner.fit(X, Y, T, feature_names=feature_names)
    summary = learner.get_summary()
    imp_df  = learner.cate_feature_importance()

    figs = {}
    sp_dist  = os.path.join(save_dir, "dr_cate_distribution.png")  if save_dir else None
    sp_imp   = os.path.join(save_dir, "dr_cate_importance.png")    if save_dir else None
    sp_prop  = os.path.join(save_dir, "dr_cate_vs_propensity.png") if save_dir else None

    figs["cate_dist"]  = learner.plot_cate_distribution(
        save_path=sp_dist,  dataset_label=dataset_label)
    figs["cate_imp"]   = learner.plot_feature_importance(
        save_path=sp_imp,   dataset_label=dataset_label)
    figs["cate_prop"]  = learner.plot_cate_vs_propensity(
        save_path=sp_prop,  dataset_label=dataset_label)

    logger.info(
        "[DRLearner] Done | ATE=%.4f [%.4f,%.4f] | ATT=%.4f | "
        "Persuadables=%.1f%% | Sleeping Dogs=%.1f%%",
        summary["ATE"], summary.get("ATE_CI_lo",0), summary.get("ATE_CI_hi",0),
        summary["ATT"],
        summary["pct_positive_cate"],
        summary["pct_negative_cate"],
    )
    return {
        "learner":      learner,
        "summary":      summary,
        "importance_df":imp_df,
        "figs":         figs,
    }
