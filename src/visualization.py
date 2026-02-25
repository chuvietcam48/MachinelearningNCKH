"""
src/visualization.py
====================
Plotting utilities for the Decision-Centric Customer Re-Engagement project.

Produces:
  1. Kaplan-Meier curves by RFM segment
  2. Weibull survival curves (population + individual)
  3. Hazard trajectory plots
  4. SHAP summary plot (feature importance)
  5. Policy decision distribution chart
  6. Time-dependent AUC curve
  7. Brier Score over time
"""

import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script mode
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from lifelines import KaplanMeierFitter, WeibullAFTFitter

logger = logging.getLogger(__name__)

# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = {
    "Champions": "#2ecc71",
    "Loyal":     "#3498db",
    "At Risk":   "#e67e22",
    "Lost":      "#e74c3c",
}
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
})


def plot_kaplan_meier_by_segment(
    customer_df: pd.DataFrame,
    rfm_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot Kaplan-Meier survival curves stratified by RFM segment.
    Provides a non-parametric baseline for comparing survival patterns.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()

    for segment, color in PALETTE.items():
        mask = rfm_df["RFM_Segment"] == segment
        if mask.sum() < 5:
            continue
        T_seg = customer_df.loc[mask, "T"]
        E_seg = customer_df.loc[mask, "E"]
        kmf.fit(T_seg, E_seg, label=f"{segment} (n={mask.sum()})")
        kmf.plot_survival_function(ax=ax, color=color, ci_show=True, linewidth=2)

    ax.set_title("Kaplan-Meier Survival Curves by RFM Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("S(t) — Probability of Remaining Active", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved KM plot → {save_path}")
    return fig


def plot_weibull_survival_curves(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    n_samples: int = 50,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot individual Weibull survival curves for a random sample of customers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: Individual survival curves ─────────────────────────────────────
    sample_idx = np.random.choice(len(df_scaled), size=min(n_samples, len(df_scaled)), replace=False)
    sample_df  = df_scaled.iloc[sample_idx]
    t_max      = df_scaled["T"].max()
    t_grid     = np.linspace(1, t_max, 200)

    S_hat = waf.predict_survival_function(sample_df, times=t_grid)
    for col in S_hat.columns:
        axes[0].plot(t_grid, S_hat[col].values, alpha=0.15, color="#3498db", linewidth=0.8)

    # Population mean survival
    S_all  = waf.predict_survival_function(df_scaled, times=t_grid)
    S_mean = S_all.mean(axis=1)
    axes[0].plot(t_grid, S_mean.values, color="#e74c3c", linewidth=2.5, label="Population Mean")
    axes[0].set_title("Individual Weibull Survival Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("S(t | x)")
    axes[0].legend()

    # ── Right: Median survival time distribution ──────────────────────────────
    median_times = waf.predict_median(df_scaled)
    # Numerical Guard: Filter infs and clip to reasonable range
    median_times = np.nan_to_num(median_times, nan=0.0, posinf=t_max*2)
    
    axes[1].hist(median_times, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    med_val = np.median(median_times)
    axes[1].axvline(med_val, color="#e74c3c", linewidth=2,
                    linestyle="--", label=f"Median = {med_val:.0f}d")
    axes[1].set_title("Distribution of Predicted Median Survival Time", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Median Survival (days)")
    axes[1].set_ylabel("Number of Customers")
    axes[1].legend()

    plt.suptitle("Weibull AFT Model — Survival Analysis", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved Weibull survival plot → {save_path}")
    return fig


def plot_hazard_trajectories(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    rfm_df: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot mean hazard trajectories h(t) stratified by RFM segment.
    Demonstrates how risk evolves differently across customer groups.
    """
    t_max  = df_scaled["T"].max()
    t_grid = np.linspace(1, t_max, 300)

    S_hat = waf.predict_survival_function(df_scaled, times=t_grid).values  # (T, N)
    S_hat = np.clip(S_hat, 1e-8, 1.0) # Numerical Guard

    dS    = np.diff(S_hat, axis=0, prepend=S_hat[[0], :])
    dt    = np.diff(t_grid, prepend=t_grid[0])[:, None]
    H     = -dS / (S_hat * dt)
    H     = np.nan_to_num(H, nan=0.0, posinf=1.0) # Numerical Guard
    H     = np.clip(H, 0, 1.0) # Clip hazard to sane range for plotting

    fig, ax = plt.subplots(figsize=(11, 6))

    segments = rfm_df["RFM_Segment"].values
    for segment, color in PALETTE.items():
        mask = segments == segment
        if mask.sum() < 3:
            continue
        h_mean = H[:, mask].mean(axis=1)
        ax.plot(t_grid, h_mean, label=f"{segment} (n={mask.sum()})",
                color=color, linewidth=2.2)

    ax.set_title("Mean Hazard Trajectory h(t) by RFM Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (days)", fontsize=12)
    ax.set_ylabel("Instantaneous Hazard Rate h(t)", fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved hazard trajectory plot → {save_path}")
    return fig


def plot_shap_summary(
    waf: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    feature_cols: list,
    n_background: int = 100,
    n_explain: int = 300,
    save_path: str = None,
    save_csv_path: str = None,
) -> None:
    """
    Generate SHAP summary plot for the Weibull AFT model.
    Uses KernelExplainer with median survival time as the scalar output.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Skipping SHAP plot. Run: pip install shap")
        return

    logger.info("Computing SHAP values (this may take a few minutes)...")

    X_feat = df_scaled[feature_cols].values
    background = shap.sample(X_feat, min(n_background, len(X_feat)))

    def _predict_median(X_arr):
        df_tmp = pd.DataFrame(X_arr, columns=feature_cols)
        df_tmp["T"] = df_scaled["T"].median()
        df_tmp["E"] = 0
        return waf.predict_median(df_tmp).values

    explainer   = shap.KernelExplainer(_predict_median, background)
    X_explain   = X_feat[:min(n_explain, len(X_feat))]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_explain, nsamples=100)

    # Save feature importance to CSV
    if save_csv_path:
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": feature_cols,
            "Mean_Abs_SHAP": mean_abs_shap
        }).sort_values("Mean_Abs_SHAP", ascending=False)
        shap_df.to_csv(save_csv_path, index=False)
        logger.info(f"Saved SHAP importance table → {save_csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_explain,
        feature_names=feature_cols,
        show=False, plot_type="bar",
    )
    plt.title("SHAP Feature Importance — Weibull AFT (Median Survival Time)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved SHAP plot → {save_path}")
    plt.close()


def plot_decision_distribution(
    weibull_decisions: pd.DataFrame,
    rfm_decisions: pd.DataFrame,
    save_path: str = None,
) -> plt.Figure:
    """
    Side-by-side bar chart comparing decision distributions:
    Weibull AFT policy vs. RFM baseline policy.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    decision_colors = {
        "INTERVENE": "#e67e22",
        "WAIT":      "#3498db",
        "LOST":      "#e74c3c",
    }

    for ax, (decisions, title) in zip(axes, [
        (weibull_decisions, "Weibull AFT Policy"),
        (rfm_decisions,     "RFM Baseline Policy"),
    ]):
        counts = decisions["decision"].value_counts()
        bars = ax.bar(
            counts.index,
            counts.values,
            color=[decision_colors.get(d, "#95a5a6") for d in counts.index],
            edgecolor="white",
            linewidth=1.2,
        )
        ax.bar_label(bars, fmt="%d", fontsize=11, padding=3)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Number of Customers")
        ax.set_ylim(0, counts.max() * 1.15)

    plt.suptitle("Intervention Decision Distribution: Weibull vs. RFM",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved decision distribution plot → {save_path}")
    return fig


def plot_brier_score_over_time(
    model: WeibullAFTFitter,
    df_scaled: pd.DataFrame,
    t_grid_steps: int = 100,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot Brier Score BS(t) over time to visualize calibration dynamics.
    """
    T_obs  = df_scaled["T"].values
    t_min  = T_obs.min()
    t_max  = T_obs.max()
    t_grid = np.linspace(t_min, t_max, t_grid_steps)

    S_hat = model.predict_survival_function(df_scaled, times=t_grid).values
    S_hat = np.nan_to_num(S_hat, nan=0.5, posinf=1.0, neginf=0.0)

    brier_scores = []
    for j, t in enumerate(t_grid):
        y_true = (T_obs > t).astype(float)
        y_pred = S_hat[j, :]
        brier_scores.append(np.mean((y_pred - y_true) ** 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_grid, brier_scores, color="#3498db", linewidth=2)
    ax.axhline(0.25, color="#e74c3c", linestyle="--", linewidth=1.5, label="Random baseline (0.25)")
    ax.fill_between(t_grid, brier_scores, alpha=0.15, color="#3498db")
    ax.set_title("Brier Score Over Time — Weibull AFT Calibration", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("BS(t)")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved Brier score plot → {save_path}")
    return fig


def plot_calibration(
    lr_pipeline,
    customer_df,
    tau: int = 90,
    n_bins: int = 10,
    save_path: str = None,
):
    """
    D1: Reliability Diagram (Calibration Curve) for the Logistic Regression classifier.

    A well-calibrated model lies close to the diagonal.
    Produces: reliability diagram + predicted-probability histogram.
    """
    from sklearn.calibration import calibration_curve
    from src.models import LOGISTIC_FEATURES

    available_features = [f for f in LOGISTIC_FEATURES if f in customer_df.columns]
    if not available_features:
        logger.warning("[Calibration] No logistic features found — skipping.")
        return None

    X = customer_df[available_features]
    y = customer_df["E"].values

    try:
        prob_pos = lr_pipeline.predict_proba(X)[:, 1]
    except Exception as exc:
        logger.warning(f"[Calibration] predict_proba failed: {exc}")
        return None

    frac_pos, mean_pred = calibration_curve(y, prob_pos, n_bins=n_bins, strategy="uniform")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfect calibration")
    ax1.plot(mean_pred, frac_pos, "o-", color="#e74c3c",
             linewidth=2, markersize=6, label="Logistic Regression")
    ax1.fill_between(mean_pred, frac_pos, mean_pred, alpha=0.15, color="#e74c3c")
    ax1.set_xlabel("Mean predicted probability", fontsize=11)
    ax1.set_ylabel(f"Fraction of churners (E=1, tau={tau}d)", fontsize=11)
    ax1.set_title("Reliability Diagram", fontsize=13, fontweight="bold")
    ax1.legend()
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Probability histogram
    ax2.hist(prob_pos[y == 1], bins=20, alpha=0.7, color="#e74c3c", label="Churners (E=1)")
    ax2.hist(prob_pos[y == 0], bins=20, alpha=0.7, color="#3498db", label="Retained (E=0)")
    ax2.set_xlabel("Predicted P(churn)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Predicted Probability Distribution", fontsize=13, fontweight="bold")
    ax2.legend()

    plt.suptitle(f"LR Calibration Analysis  (tau={tau}d)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        logger.info(f"Saved calibration plot → {save_path}")
    return fig
