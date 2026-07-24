"""
Gate 9 — Survival Baseline (Behavioral Features Only)
======================================================
Purpose : Establish survival analysis baseline BEFORE semantic features.
          This script uses ONLY behavioral/rating features (B0, B1).
          Semantic features will be added in Gate 10 after Dataset Freeze.

Model   : Weibull AFT (parametric) + Cox PH (semi-parametric)
Target  : Time-to-dormancy (days from episode_endpoint to next episode,
          right-censored at H=270 days).
Split   : Temporal — same cutoffs as Phase 3A baseline for direct comparison.

DO NOT add semantic features here.
DO NOT change feature sets without updating the Gate 10 comparison scaffold.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = Path("outputs/amazon_v5_rebuild/data/verified_episode_snapshots_h270_v1.parquet")
OUT_DIR    = Path("outputs/amazon_v5_rebuild/gate9_survival_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Feature Sets (mirrors Phase 3A for clean comparison) ──────────────────
B0_COLS = [
    "prior_verified_episode_count",
    "prior_review_count",
    "days_since_prior_episode",
    "active_tenure_days",
    "historical_mean_inter_episode_gap",
    "historical_median_inter_episode_gap",
    "prior_episode_count_90d",
    "prior_episode_count_270d",
    "prior_review_count_90d",
    "prior_review_count_270d",
]

B1_COLS = B0_COLS + [
    "episode_min_rating",
    "episode_max_rating",
    "episode_mean_rating",
    "episode_review_count",
    "historical_mean_rating",
    "historical_rating_std",
    "prior_severe_low_rating_count",
    "prior_severe_low_rating_rate",
    "current_vs_historical_rating_delta",
    "severe_low_rating_feedback_flag",
    "high_engagement_verified_reviewer_flag",
]

# PLACEHOLDER — will be populated in Gate 10 after Dataset Freeze
SEMANTIC_COLS: list[str] = []

FEATURE_SETS: dict[str, list[str]] = {
    "B0":  B0_COLS,
    "B1":  B1_COLS,
    # Gate 10 will add:
    # "B1_S": B1_COLS + SEMANTIC_COLS,
    # "S_only": SEMANTIC_COLS,
    # "All": B1_COLS + SEMANTIC_COLS,
}

TARGET_EVENT   = "Y_dormant_270"
TARGET_DURATION = "duration_days"          # days from episode_endpoint to next episode (or censored)
HORIZON        = 270
RANDOM_STATE   = 42


# ── 1. Load Data ───────────────────────────────────────────────────────────
def load_and_engineer(path: Path) -> pd.DataFrame:
    """Load snapshots and add derived behavioral features + survival target."""
    df = pd.read_parquet(path)
    df = df.drop_duplicates(subset=["reviewerID", "episode_id"]).copy()
    df = df.sort_values(["reviewerID", "episode_start"]).reset_index(drop=True)

    # ── Behavioral features (identical to 05_phase3a_baseline_ml.py) ──────
    df["days_since_prior_episode"] = (
        df["episode_start"]
        - df.groupby("reviewerID")["episode_endpoint"].shift(1)
    ).dt.days

    df["active_tenure_days"] = (
        df["episode_endpoint"]
        - df.groupby("reviewerID")["episode_start"].transform("min")
    ).dt.days

    g_gap = df.groupby("reviewerID")["days_since_prior_episode"].expanding()
    df["historical_mean_inter_episode_gap"] = (
        g_gap.mean().reset_index(level=0, drop=True).groupby(df["reviewerID"]).shift(1)
    )
    df["historical_median_inter_episode_gap"] = (
        g_gap.median().reset_index(level=0, drop=True).groupby(df["reviewerID"]).shift(1)
    )

    for lag, col in [(90, "prior_episode_count_90d"), (270, "prior_episode_count_270d")]:
        df[f"start_{lag}d_ago"] = df["episode_start"] - pd.Timedelta(days=lag)
        hist = df[["reviewerID", "episode_start", "episode_review_count"]].copy()
        hist["prior_episode_count"] = hist.groupby("reviewerID").cumcount()
        hist["prior_review_count"] = (
            hist.groupby("reviewerID")["episode_review_count"].cumsum().shift(1).fillna(0)
        )
        merged = pd.merge_asof(
            df[["reviewerID", f"start_{lag}d_ago"]].sort_values(f"start_{lag}d_ago"),
            hist.sort_values("episode_start"),
            left_on=f"start_{lag}d_ago", right_on="episode_start",
            by="reviewerID", direction="backward"
        )
        merged = merged.set_index(
            df[["reviewerID", f"start_{lag}d_ago"]].sort_values(f"start_{lag}d_ago").index
        )
        df[col] = df["prior_verified_episode_count"] - merged["prior_episode_count"].fillna(0).values
        df[f"prior_review_count_{lag}d"] = (
            df["prior_review_count"] - merged["prior_review_count"].fillna(0).values
        )

    g_rat = df.groupby("reviewerID")["episode_mean_rating"].expanding()
    df["historical_rating_std"] = (
        g_rat.std(ddof=0).reset_index(level=0, drop=True).groupby(df["reviewerID"]).shift(1)
    )
    df["prior_severe_low_rating_count"] = (
        (df["episode_min_rating"] <= 2).astype(int)
        .groupby(df["reviewerID"]).cumsum()
        .groupby(df["reviewerID"]).shift(1).fillna(0).astype(int)
    )
    df["prior_severe_low_rating_rate"] = (
        df["prior_severe_low_rating_count"]
        / df["prior_verified_episode_count"].replace(0, np.nan)
    )
    df["current_vs_historical_rating_delta"] = df["episode_mean_rating"] - df["historical_mean_rating"]
    df["severe_low_rating_feedback_flag"] = (df["episode_min_rating"] <= 2).astype(int)
    df["high_engagement_verified_reviewer_flag"] = (
        df["prior_verified_episode_count"] >= 3
    ).astype(int)

    # ── Survival target construction ───────────────────────────────────────
    # duration_days: days until next episode, capped at HORIZON (right-censored)
    # Y_dormant_270: 1 = event (churned within 270d), 0 = censored
    next_start = df.groupby("reviewerID")["episode_start"].shift(-1)
    duration_raw = (next_start - df["episode_endpoint"]).dt.days
    df[TARGET_DURATION] = duration_raw.clip(upper=HORIZON).fillna(HORIZON)
    # Y_dormant_270 is already in the snapshot; keep for reference
    # event observed = 1 means churn happened (duration < HORIZON or exactly at horizon with no next)
    df["event_observed"] = df[TARGET_EVENT]

    return df


# ── 2. Temporal Split ──────────────────────────────────────────────────────
def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoints = np.sort(df["episode_endpoint"].dt.date.unique())
    n = len(endpoints)
    train_cut = endpoints[int(n * 0.6)]
    val_cut   = endpoints[int(n * 0.8)]

    df = df.copy()
    df["split"] = "eval"
    df.loc[df["episode_endpoint"].dt.date < val_cut,   "split"] = "val"
    df.loc[df["episode_endpoint"].dt.date < train_cut, "split"] = "train"

    return df[df["split"] == "train"], df[df["split"] == "val"], df[df["split"] == "eval"]


# ── 3. Imputation ──────────────────────────────────────────────────────────
def impute(train: pd.DataFrame, val: pd.DataFrame, eval_: pd.DataFrame,
           cols: list[str]) -> tuple:
    imp = SimpleImputer(strategy="median")
    imp.fit(train[cols])
    X_tr = pd.DataFrame(imp.transform(train[cols]),  columns=cols, index=train.index)
    X_va = pd.DataFrame(imp.transform(val[cols]),    columns=cols, index=val.index)
    X_ev = pd.DataFrame(imp.transform(eval_[cols]),  columns=cols, index=eval_.index)
    return X_tr, X_va, X_ev


# ── 4. Survival Models ─────────────────────────────────────────────────────
def fit_weibull_aft(X_tr, T_tr, E_tr, X_ev, T_ev, E_ev) -> dict:
    """
    Weibull Accelerated Failure Time model.
    Requires: pip install lifelines
    Returns evaluation metrics dict.
    """
    try:
        from lifelines import WeibullAFTFitter
    except ImportError:
        print("  [SKIP] lifelines not installed. Run: pip install lifelines")
        return {}

    from lifelines.utils import concordance_index

    train_df = X_tr.copy()
    train_df[TARGET_DURATION] = T_tr.values
    train_df["event_observed"] = E_tr.values

    # Standardise for AFT
    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_ev_s = pd.DataFrame(scaler.transform(X_ev),     columns=X_ev.columns)

    fit_df = X_tr_s.copy()
    fit_df[TARGET_DURATION] = T_tr.values
    fit_df["event_observed"] = E_tr.values

    aft = WeibullAFTFitter(penalizer=0.1)
    aft.fit(fit_df, duration_col=TARGET_DURATION, event_col="event_observed")

    median_surv = aft.predict_median(X_ev_s)
    c_index = concordance_index(T_ev, median_surv, E_ev)

    return {
        "model": "WeibullAFT",
        "c_index": round(c_index, 4),
        "n_train": len(X_tr),
        "n_eval": len(X_ev),
        "event_rate_eval": round(E_ev.mean(), 4),
    }


def fit_cox_ph(X_tr, T_tr, E_tr, X_ev, T_ev, E_ev) -> dict:
    """
    Cox Proportional Hazards model.
    Requires: pip install lifelines
    """
    try:
        from lifelines import CoxPHFitter
        from lifelines.utils import concordance_index
    except ImportError:
        print("  [SKIP] lifelines not installed.")
        return {}

    scaler = StandardScaler()
    X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_ev_s = pd.DataFrame(scaler.transform(X_ev),     columns=X_ev.columns)

    fit_df = X_tr_s.copy()
    fit_df[TARGET_DURATION] = T_tr.values
    fit_df["event_observed"] = E_tr.values

    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(fit_df, duration_col=TARGET_DURATION, event_col="event_observed")

    # Risk score: higher = higher hazard = more likely to churn
    risk_score = cox.predict_log_partial_hazard(X_ev_s)
    c_index = concordance_index(T_ev, -risk_score, E_ev)

    return {
        "model": "CoxPH",
        "c_index": round(c_index, 4),
        "n_train": len(X_tr),
        "n_eval": len(X_ev),
        "event_rate_eval": round(E_ev.mean(), 4),
    }


# ── 5. Main ────────────────────────────────────────────────────────────────
def run():
    print("=== Gate 9: Survival Baseline (Behavioral Only) ===\n")

    print("1. Loading data...")
    df = load_and_engineer(DATA_PATH)
    print(f"   {len(df):,} episodes loaded.")

    print("2. Temporal split...")
    train_df, val_df, eval_df = temporal_split(df)
    print(f"   Train: {len(train_df):,} | Val: {len(val_df):,} | Eval: {len(eval_df):,}")

    T_tr = train_df[TARGET_DURATION]
    E_tr = train_df["event_observed"]
    T_ev = eval_df[TARGET_DURATION]
    E_ev = eval_df["event_observed"]

    all_results = []

    for fs_name, cols in FEATURE_SETS.items():
        print(f"\n3. Feature set: {fs_name} ({len(cols)} features)")
        X_tr, _, X_ev = impute(train_df, val_df, eval_df, cols)

        print(f"   Fitting Weibull AFT...")
        w_res = fit_weibull_aft(X_tr, T_tr, E_tr, X_ev, T_ev, E_ev)
        if w_res:
            w_res["feature_set"] = fs_name
            all_results.append(w_res)
            print(f"   Weibull C-Index: {w_res.get('c_index', 'N/A')}")

        print(f"   Fitting Cox PH...")
        c_res = fit_cox_ph(X_tr, T_tr, E_tr, X_ev, T_ev, E_ev)
        if c_res:
            c_res["feature_set"] = fs_name
            all_results.append(c_res)
            print(f"   Cox C-Index: {c_res.get('c_index', 'N/A')}")

    # ── Save results ───────────────────────────────────────────────────────
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(OUT_DIR / "gate9_survival_baseline_results.csv", index=False)
        print(f"\n4. Results saved → {OUT_DIR}/gate9_survival_baseline_results.csv")
        print(results_df[["feature_set", "model", "c_index"]].to_string(index=False))

    manifest = {
        "gate": "Gate 9 — Survival Baseline",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_path": str(DATA_PATH),
        "feature_sets": list(FEATURE_SETS.keys()),
        "semantic_features_included": False,
        "note": "Semantic features intentionally excluded. Add in Gate 10 after Dataset Freeze.",
        "results": all_results,
    }
    with open(OUT_DIR / "gate9_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Gate 9 Complete ===")


if __name__ == "__main__":
    run()
