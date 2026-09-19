"""
src/causal/x_learner.py
========================
X-Learner — Imbalance-Aware CATE Estimation.

Motivation
----------
The T-Learner divides data into treated (INTERVENE) and control (WAIT)
and fits separate outcome models.  When treatment is rare (5%), the
treated-branch model mu_1 is trained on very few samples, leading to
high-variance CATE estimates.

The X-Learner (Kunzel et al. 2019) addresses this by:
  - Using the LARGE control group to impute what treated customers
    would have earned without intervention.
  - Using the LARGE control group model mu_0 to compute treated-side
    treatment effects: D1_i = Y_i - mu_0(X_i) for treated units.
  - The control side similarly: D0_i = mu_1(X_i) - Y_i for control.
  - Fitting CATE models on each side and blending with propensity score.

This borrowing of strength across sides is why X-Learner outperforms
T-Learner when n_treated << n_control.

Algorithm (Kunzel et al. 2019, "Meta-learners for Estimating HTEs")
--------------------------------------------------------------------
Stage 1: Outcome models (same as T-Learner)
  mu_1(x) <- fit on {X_i, Y_i : T_i=1}   (treated-branch model)
  mu_0(x) <- fit on {X_i, Y_i : T_i=0}   (control-branch model)

Stage 2: Imputed treatment effects
  D1_i = Y_i - mu_0(X_i)   for treated units    (how much more than predicted control?)
  D0_i = mu_1(X_i) - Y_i   for control units    (how much less than predicted treated?)

Stage 3: CATE models on each side
  tau_1(x) <- regress D1 on X[treated]  (large n since we use mu_0 on ALL treated)
  tau_0(x) <- regress D0 on X[control]  (large n on control side)

Stage 4: Propensity-weighted blend
  tau_hat(x) = g(x) * tau_0(x) + (1 - g(x)) * tau_1(x)
  where g(x) = e(x) = P(T=1|X)  (propensity score)

When propensity is small (e(x)~0, most customers are control):
  tau_hat(x) ≈ tau_1(x)  -- uses the treated-side model (fit on fewer but relevant samples)
When propensity is large (e(x)~1):
  tau_hat(x) ≈ tau_0(x)  -- uses the control-side model (fit on many samples)

This blend is optimal under certain conditions on the smoothness of the CATE.
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
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class XLearner:
    """
    X-Learner for heterogeneous treatment effect estimation.

    Particularly well-suited for imbalanced treatment rates (e.g., 5% INTERVENE).

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    clip_propensity : float
    n_bootstrap : int
    seed : int
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        clip_propensity: float = 0.05,
        n_bootstrap: int = 200,
        seed: int = 42,
    ):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.clip_propensity  = clip_propensity
        self.n_bootstrap      = n_bootstrap
        self.seed             = seed
        self.rng              = np.random.default_rng(seed)

        self._fitted          = False
        self.feature_names_   = None
        self.tau_hat_         = None
        self.propensity_      = None
        self.ate_             = None
        self.att_             = None
        self.atc_             = None
        self.ate_ci_          = None
        self.att_ci_          = None

    def _make_reg(self):
        return Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gbm",   GradientBoostingRegressor(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, subsample=0.8, random_state=self.seed,
            )),
        ])

    def _make_clf(self):
        return Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("gbm",   GradientBoostingClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                learning_rate=0.05, subsample=0.8, random_state=self.seed,
            )),
        ])

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        T: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "XLearner":
        """
        Fit the X-Learner.

        Parameters
        ----------
        X : ndarray (n, p)
        Y : ndarray (n,)
        T : ndarray (n,)  binary 0/1
        feature_names : list of str, optional
        """
        n = len(X)
        t1 = T == 1
        t0 = T == 0
        self.feature_names_ = feature_names or [f"x{i}" for i in range(X.shape[1])]

        logger.info(
            "[XLearner] Fitting | n=%d | n_treated=%d (%.1f%%) | n_control=%d",
            n, t1.sum(), t1.mean() * 100, t0.sum(),
        )

        if t1.sum() < 10 or t0.sum() < 10:
            raise ValueError(
                f"[XLearner] Too few treated ({t1.sum()}) or control ({t0.sum()}) samples."
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # ── Stage 1: Outcome models ───────────────────────────────────────
            self._mu1 = self._make_reg()
            self._mu0 = self._make_reg()
            self._mu1.fit(X[t1], Y[t1])
            self._mu0.fit(X[t0], Y[t0])

            mu1_all = self._mu1.predict(X)   # E[Y|X, T=1] for all
            mu0_all = self._mu0.predict(X)   # E[Y|X, T=0] for all

            # ── Stage 2: Imputed treatment effects ───────────────────────────
            # For treated: D1 = Y - mu0(X)   -- what treated gained over control prediction
            # For control: D0 = mu1(X) - Y   -- what control missed vs treated prediction
            D1 = Y[t1] - mu0_all[t1]   # shape (n_treated,)
            D0 = mu1_all[t0] - Y[t0]   # shape (n_control,)

            logger.info(
                "[XLearner] Imputed effects | D1 mean=%.4f std=%.4f | D0 mean=%.4f std=%.4f",
                D1.mean(), D1.std(), D0.mean(), D0.std(),
            )

            # ── Stage 3: CATE models ──────────────────────────────────────────
            self._tau1 = self._make_reg()   # fit on treated, target D1
            self._tau0 = self._make_reg()   # fit on control, target D0
            self._tau1.fit(X[t1], D1)
            self._tau0.fit(X[t0], D0)

            # ── Stage 4: Propensity-weighted blend ────────────────────────────
            self._ps = self._make_clf()
            self._ps.fit(X, T)
            e = np.clip(
                self._ps.predict_proba(X)[:, 1],
                self.clip_propensity,
                1.0 - self.clip_propensity,
            )
            self.propensity_ = e

        tau1_all = self._tau1.predict(X)
        tau0_all = self._tau0.predict(X)
        # tau_hat = e(x) * tau_0(x) + (1 - e(x)) * tau_1(x)
        tau_hat = e * tau0_all + (1.0 - e) * tau1_all

        self.tau_hat_ = tau_hat
        self._T       = T
        self._X       = X

        self.ate_ = float(tau_hat.mean())
        self.att_ = float(tau_hat[t1].mean())
        self.atc_ = float(tau_hat[t0].mean())

        logger.info(
            "[XLearner] ATE=%.4f | ATT=%.4f | ATC=%.4f | "
            "Persuadables=%.1f%% | Sleeping Dogs=%.1f%%",
            self.ate_, self.att_, self.atc_,
            (tau_hat > 0).mean() * 100,
            (tau_hat < 0).mean() * 100,
        )

        if self.n_bootstrap > 0:
            self._bootstrap_ci(X, T, alpha=0.05)

        self._fitted = True
        return self

    def _bootstrap_ci(self, X, T, alpha=0.05):
        ate_b = []
        att_b = []
        atc_b = []
        for _ in range(self.n_bootstrap):
            idx  = self.rng.integers(0, len(X), size=len(X))
            tau_b = self.tau_hat_[idx]
            T_b   = T[idx]
            ate_b.append(tau_b.mean())
            t1_b  = T_b == 1
            t0_b  = T_b == 0
            att_b.append(tau_b[t1_b].mean() if t1_b.sum() > 0 else np.nan)
            atc_b.append(tau_b[t0_b].mean() if t0_b.sum() > 0 else np.nan)

        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
        self.ate_ci_ = (float(np.percentile(ate_b, lo)), float(np.percentile(ate_b, hi)))
        self.att_ci_ = (float(np.nanpercentile(att_b, lo)), float(np.nanpercentile(att_b, hi)))
        self.atc_ci_ = (float(np.nanpercentile(atc_b, lo)), float(np.nanpercentile(atc_b, hi)))

    def effect(self, X: np.ndarray) -> np.ndarray:
        e = np.clip(
            self._ps.predict_proba(X)[:, 1],
            self.clip_propensity, 1.0 - self.clip_propensity,
        )
        return e * self._tau0.predict(X) + (1.0 - e) * self._tau1.predict(X)

    def get_summary(self) -> dict:
        return {
            "ATE":               round(self.ate_, 4),
            "ATT":               round(self.att_, 4),
            "ATC":               round(self.atc_, 4),
            "ATE_CI_lo":         round(self.ate_ci_[0], 4) if self.ate_ci_ else None,
            "ATE_CI_hi":         round(self.ate_ci_[1], 4) if self.ate_ci_ else None,
            "ATT_CI_lo":         round(self.att_ci_[0], 4) if self.att_ci_ else None,
            "ATT_CI_hi":         round(self.att_ci_[1], 4) if self.att_ci_ else None,
            "pct_positive_cate": round((self.tau_hat_ > 0).mean() * 100, 2),
            "pct_negative_cate": round((self.tau_hat_ < 0).mean() * 100, 2),
            "n_treated":         int((self._T == 1).sum()),
            "n_control":         int((self._T == 0).sum()),
        }

    def cate_feature_importance(self) -> pd.DataFrame:
        """Blend feature importance from tau_0 and tau_1 models."""
        imp1 = self._tau1.named_steps["gbm"].feature_importances_
        imp0 = self._tau0.named_steps["gbm"].feature_importances_
        # Weight by group sizes
        n1 = (self._T == 1).sum()
        n0 = (self._T == 0).sum()
        blended = (n1 * imp1 + n0 * imp0) / (n1 + n0)
        return (
            pd.DataFrame({"feature": self.feature_names_, "importance": blended})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def run_x_learner(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_dir: Optional[str] = None,
    dataset_label: str = "",
    n_estimators: int = 200,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """One-call X-Learner fit and summary."""
    xl = XLearner(n_estimators=n_estimators, n_bootstrap=n_bootstrap, seed=seed)
    xl.fit(X, Y, T, feature_names=feature_names)
    summary  = xl.get_summary()
    imp_df   = xl.cate_feature_importance()

    logger.info(
        "[XLearner] ATE=%.4f [%.4f,%.4f] | ATT=%.4f | Persuadables=%.1f%%",
        summary["ATE"], summary.get("ATE_CI_lo",0), summary.get("ATE_CI_hi",0),
        summary["ATT"], summary["pct_positive_cate"],
    )
    return {"learner": xl, "summary": summary, "importance_df": imp_df}


# =============================================================================
# Comparison table: T-Learner vs X-Learner vs DR-Learner
# =============================================================================

def compare_learners(
    t_learner_summary: dict,
    x_learner_summary: dict,
    dr_learner_summary: dict,
    save_path: Optional[str] = None,
    dataset_label: str = "",
) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Side-by-side comparison of three CATE estimators.

    Shows: ATE, ATT, ATC with 95% CI for each method.
    Highlights which estimators agree on the sign of ATT
    (a key question: does treatment help the INTERVENE group?).
    """
    rows = []
    for name, s in [
        ("T-Learner + IPTW", t_learner_summary),
        ("X-Learner",        x_learner_summary),
        ("DR-Learner",       dr_learner_summary),
    ]:
        rows.append({
            "Method":    name,
            "ATE":       s.get("ATE", 0),
            "ATE_lo":    s.get("ATE_CI_lo", s.get("ATE", 0)),
            "ATE_hi":    s.get("ATE_CI_hi", s.get("ATE", 0)),
            "ATT":       s.get("ATT", 0),
            "ATT_lo":    s.get("ATT_CI_lo", s.get("ATT", 0)),
            "ATT_hi":    s.get("ATT_CI_hi", s.get("ATT", 0)),
            "Pct_positive": s.get("pct_positive_cate", 50),
        })

    comp_df = pd.DataFrame(rows)

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"CATE Estimator Comparison — {dataset_label}\n"
        f"(T-Learner vs X-Learner vs DR-Learner)",
        fontsize=13, fontweight="bold",
    )

    methods = comp_df["Method"].tolist()
    x_pos   = np.arange(len(methods))
    colors  = ["#95a5a6", "#3498db", "#2ecc71"]

    for ax, metric, lo_col, hi_col, ylabel in [
        (axes[0], "ATE", "ATE_lo", "ATE_hi", "Average Treatment Effect (ATE)"),
        (axes[1], "ATT", "ATT_lo", "ATT_hi", "ATT (on INTERVENE group)"),
    ]:
        vals = comp_df[metric].values
        lo   = comp_df[lo_col].values
        hi   = comp_df[hi_col].values
        bars = ax.bar(x_pos, vals, color=colors, alpha=0.85, width=0.55)
        ax.errorbar(
            x_pos, vals,
            yerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)],
            fmt="none", color="#2c3e50", capsize=9, lw=2.5,
        )
        for bar, val, lo_v, hi_v in zip(bars, vals, lo, hi):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(val, 0) + abs(vals).max() * 0.04,
                f"{val:+.4f}\n[{lo_v:.3f},{hi_v:.3f}]",
                ha="center", fontsize=9, fontweight="bold",
            )
        ax.axhline(0, color="#ccc", lw=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=10, rotation=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(
            f"{metric} Comparison\n"
            f"(Methods above 0 = positive treatment effect)",
            fontsize=11, fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("[XLearner] Comparison plot saved -> %s", save_path)

    return comp_df, fig
