"""
Phase 1 - Coarse Sentiment Feature Extraction

Backend options:
  "vader"   [default] - Fast rule-based VADER, ~2M reviews/min on CPU.
  "roberta"           - cardiffnlp RoBERTa, more accurate but needs GPU.

Input:  reviews_df  [CustomerID, reviewText, overall, purchase_date, helpful_votes]
Output: customer_df augmented with 7 sentiment features + has_review flag
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
_LABEL_MAP     = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_BATCH_SIZE    = 128


# =============================================================================
# 1. Scoring backends
# =============================================================================

def _score_vader(texts: list) -> pd.DataFrame:
    """VADER compound score -> label [-1, 1]."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "vaderSentiment", "-q"], check=True)
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    rows = []
    for text in texts:
        c = sia.polarity_scores(str(text))["compound"]
        label = "positive" if c >= 0.05 else ("negative" if c <= -0.05 else "neutral")
        rows.append({"sentiment_label": label, "sentiment_score": c,
                     "confidence": abs(c)})
    return pd.DataFrame(rows)


def _score_roberta(texts: list) -> pd.DataFrame:
    """RoBERTa 3-class (needs GPU for practical speed)."""
    import warnings
    from transformers import pipeline as hf_pipeline
    pipe = hf_pipeline("text-classification", model=_ROBERTA_MODEL,
                       truncation=True, max_length=512,
                       batch_size=_BATCH_SIZE, device=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = pipe(texts)
    labels = [r["label"].lower() for r in results]
    scores = [_LABEL_MAP.get(l, 0.0) for l in labels]
    return pd.DataFrame({"sentiment_label": labels,
                          "sentiment_score": scores,
                          "confidence": [r["score"] for r in results]})


def score_reviews(texts: list, backend: str = "vader") -> pd.DataFrame:
    return _score_roberta(texts) if backend == "roberta" else _score_vader(texts)


# =============================================================================
# 2. Aggregate to customer-level features
# =============================================================================

def aggregate_sentiment_features(reviews_with_scores: pd.DataFrame,
                                  snapshot_date: str) -> pd.DataFrame:
    """
    Vectorized aggregation — avoids slow groupby-apply Python loop.
    Input:  [CustomerID, purchase_date, overall, sentiment_score, sentiment_label]
    Output: one row per CustomerID with comprehensive features + has_review=1
    """
    snap = pd.Timestamp(snapshot_date)
    
    # 1. Temporal Leakage Fix: STRICTLY keep only reviews before the prediction cutoff
    df = reviews_with_scores[reviews_with_scores["purchase_date"] < snap].copy()
    
    if len(df) == 0:
        return pd.DataFrame()
        
    df = df.sort_values(["CustomerID", "purchase_date"])
    df["days_ago"]  = (snap - df["purchase_date"]).dt.days.clip(lower=0)
    df["is_neg"]    = (df["sentiment_label"] == "negative").astype(int)
    
    # Recent flags
    df["is_recent_90"]  = (df["days_ago"] <= 90).astype(int)
    df["is_recent_180"] = (df["days_ago"] <= 180).astype(int)
    
    df["w"]         = np.exp(-0.01 * df["days_ago"])
    df["w_score"]   = df["w"] * df["sentiment_score"]

    logger.info(f"[Sentiment] Vectorized aggregation for "
                f"{df['CustomerID'].nunique():,} customers...")

    grp = df.groupby("CustomerID")

    # Current State Features
    latest = grp.apply(lambda g: g.iloc[-1]["sentiment_score"]).rename("sentiment_latest")
    latest_fail_severity = latest.apply(lambda x: max(0.0, -x)).rename("latest_fail_severity")
    
    # History Features
    sentiment_mean = grp["sentiment_score"].mean().rename("sentiment_mean")
    
    def calc_mean_before_latest(g):
        if len(g) > 1:
            return g.iloc[:-1]["sentiment_score"].mean()
        return g.iloc[0]["sentiment_score"]
    
    sentiment_mean_before_latest = grp.apply(calc_mean_before_latest).rename("sentiment_mean_before_latest")
    
    sentiment_min  = grp["sentiment_score"].min().rename("sentiment_min")
    sentiment_volatility = grp["sentiment_score"].std(ddof=0).fillna(0).rename("sentiment_volatility")
    negative_review_count = grp["is_neg"].sum().rename("negative_review_count")

    # sentiment_weighted
    wsum    = grp["w_score"].sum()
    wdenom  = grp["w"].sum()
    weighted = (wsum / wdenom).rename("sentiment_weighted")

    # sentiment_trend
    first_date = grp["purchase_date"].min()
    df = df.merge(first_date.rename("first_date"), on="CustomerID")
    df["elapsed"] = (df["purchase_date"] - df["first_date"]).dt.days.astype(float)
    grp2    = df.groupby("CustomerID")
    
    cov_ts  = grp2.apply(lambda g: ((g["elapsed"] - g["elapsed"].mean()) *
                                     (g["sentiment_score"] - g["sentiment_score"].mean())).mean())
    var_t   = grp2["elapsed"].var(ddof=0).replace(0, np.nan)
    trend   = (cov_ts / var_t).fillna(0).rename("sentiment_trend")

    # Temporal Decay Features
    neg_df  = df[df["is_neg"] == 1].groupby("CustomerID")["days_ago"].min()
    days_since_neg = neg_df.rename("days_since_last_neg").reindex(
        df["CustomerID"].unique(), fill_value=999
    )

    neg_90d = grp2.apply(
        lambda g: (g["is_neg"] * g["is_recent_90"]).sum()
    ).rename("recent_negative_count_90d")
    
    neg_180d = grp2.apply(
        lambda g: (g["is_neg"] * g["is_recent_180"]).sum()
    ).rename("recent_negative_count_180d")

    avg_star     = grp["overall"].mean().rename("avg_star_rating")
    review_count = grp.size().rename("review_count")

    agg_df = pd.concat([
        latest, latest_fail_severity, 
        sentiment_mean, sentiment_mean_before_latest, sentiment_min, sentiment_volatility, negative_review_count,
        weighted, trend, days_since_neg, 

        neg_90d, neg_180d, avg_star, review_count
    ], axis=1).reset_index()
    
    agg_df["has_review"] = 1
    return agg_df


# =============================================================================
# 3. Merge into full customer table
# =============================================================================

def merge_into_customer_table(customer_df: pd.DataFrame,
                               sentiment_features: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join sentiment into customer table.
    Customers without reviews: has_review=0, all sentiment cols imputed to 0/999.
    Also creates loyal_customer_flag and interaction term.
    """
    if len(sentiment_features) == 0:
        df = customer_df.copy()
        df["has_review"] = 0
    else:
        df = customer_df.merge(sentiment_features, on="CustomerID", how="left")
        df["has_review"] = df["has_review"].fillna(0).astype(int)

    zero_cols = [
        "sentiment_latest",        "latest_fail_severity", 
        "sentiment_mean", "sentiment_mean_before_latest", "sentiment_min", "sentiment_volatility", "negative_review_count",
        "sentiment_weighted", "sentiment_trend",
        "recent_negative_count_90d", "recent_negative_count_180d", "review_count"
    ]
    for col in zero_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    # Add Missing Sentiment Flag
    df["sentiment_missing_flag_latest"] = (df["has_review"] == 0).astype(int)
    assert "sentiment_missing_flag_latest" in df.columns
    assert df.loc[df["has_review"] == 0, "sentiment_missing_flag_latest"].eq(1).all()
    
    # Add Normalized Rates
    df["negative_review_rate"] = df["negative_review_count"] / df["review_count"].replace(0, 1)
            
    # Add Loyal Customer Flag and Interactions
    if "Frequency" in df.columns:
        df["loyal_customer_flag"] = (df["Frequency"] >= 5).astype(int)
        
        # Original interaction
        if "latest_fail_severity" in df.columns:
            df["loyal_x_failure"] = df["loyal_customer_flag"] * df["latest_fail_severity"]
            df["latest_fail_flag"] = (df["latest_fail_severity"] > 0).astype(int)
            df["loyal_x_fail_flag"] = df["loyal_customer_flag"] * df["latest_fail_flag"]
        else:
            df["loyal_x_failure"] = 0
            df["loyal_x_fail_flag"] = 0
            
        # New interactions
        df["loyal_x_missing_sentiment"] = df["loyal_customer_flag"] * df["sentiment_missing_flag_latest"]
        
        # Sentiment Shock interactions
        if "sentiment_latest" in df.columns and "sentiment_mean_before_latest" in df.columns:
            df["sentiment_drop"] = df["sentiment_latest"] - df["sentiment_mean_before_latest"]
            df["negative_sentiment_shock"] = np.maximum(0, -df["sentiment_drop"])
            df["loyal_x_sentiment_shock"] = df["loyal_customer_flag"] * df["negative_sentiment_shock"]
            
            df["failure_after_positive_history"] = ((df["sentiment_mean_before_latest"] > 0.30) & (df["sentiment_latest"] < -0.05)).astype(int)
            df["loyal_x_failure_after_positive_history"] = df["loyal_customer_flag"] * df["failure_after_positive_history"]
        else:
            df["loyal_x_sentiment_shock"] = 0
            df["loyal_x_failure_after_positive_history"] = 0

    if "days_since_last_neg" in df.columns:
        df["days_since_last_neg"] = df["days_since_last_neg"].fillna(999)
        
    if "avg_star_rating" in df.columns:
        df["avg_star_rating"] = df["avg_star_rating"].fillna(
            df["avg_star_rating"].median() if not df["avg_star_rating"].isna().all() else 3.0
        )
        
    logger.info(
        f"[Sentiment] Merged: {len(df):,} customers | "
        f"has_review=1: {df['has_review'].sum():,} ({df['has_review'].mean()*100:.1f}%)"
    )
    return df


# =============================================================================
# 4. Main entry point
# =============================================================================

def run_sentiment_phase1(reviews_df: pd.DataFrame,
                          customer_df: pd.DataFrame,
                          snapshot_date: str,
                          save_dir: str = None,
                          sample_n: int = None,
                          backend: str = "vader",
                          chunk_size: int = 100_000) -> pd.DataFrame:
    """
    Full Phase 1 pipeline.

    Parameters
    ----------
    reviews_df    : [CustomerID, reviewText, overall, purchase_date]
    customer_df   : [CustomerID, T, E, Recency, ...]
    snapshot_date : reference date string
    save_dir      : save outputs here (optional)
    sample_n      : sample N customers for quick test
    backend       : "vader" (fast) or "roberta" (accurate, needs GPU)
    chunk_size    : rows per scoring chunk
    """
    if sample_n:
        ids = reviews_df["CustomerID"].drop_duplicates().sample(
            min(sample_n, reviews_df["CustomerID"].nunique()), random_state=42
        )
        reviews_df  = reviews_df[reviews_df["CustomerID"].isin(ids)].copy()
        customer_df = customer_df[customer_df["CustomerID"].isin(ids)].copy()
        logger.info(f"[Sentiment] Sample mode: {sample_n:,} customers")

    reviews_df = reviews_df[
        reviews_df["reviewText"].str.split().str.len() >= 5
    ].copy().reset_index(drop=True)

    total = len(reviews_df)
    logger.info(f"[Sentiment] backend={backend} | {total:,} reviews to score")

    # Score in chunks
    parts = []
    for start in range(0, total, chunk_size):
        end   = min(start + chunk_size, total)
        batch = reviews_df["reviewText"].iloc[start:end].tolist()
        parts.append(score_reviews(batch, backend=backend))
        if start % (chunk_size * 5) == 0 and start > 0:
            logger.info(f"[Sentiment]   {end:,}/{total:,} ({end/total*100:.1f}%)")

    scored = pd.concat(parts, ignore_index=True)
    reviews_with_scores = pd.concat([reviews_df, scored], axis=1)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        reviews_with_scores.to_csv(
            str(Path(save_dir) / "phase1_reviews_scored.csv"), index=False
        )

    features  = aggregate_sentiment_features(reviews_with_scores, snapshot_date)
    augmented = merge_into_customer_table(customer_df, features)

    if save_dir:
        features.to_csv(str(Path(save_dir) / "phase1_sentiment_features.csv"), index=False)
        augmented.to_csv(str(Path(save_dir) / "phase1_customer_augmented.csv"), index=False)
        logger.info(f"[Sentiment] All outputs saved -> {save_dir}/")

    _print_phase1_summary(augmented)
    return augmented


def _print_phase1_summary(df: pd.DataFrame):
    r = df[df["has_review"] == 1]
    logger.info("\n[Phase 1 Summary]")
    logger.info(f"  has_review=1   : {len(r):,} ({len(r)/len(df)*100:.1f}%)")
    logger.info(f"  Sentiment dist : "
                f"pos={( r['sentiment_latest'] > 0).mean()*100:.1f}% | "
                f"neu={( r['sentiment_latest']==0).mean()*100:.1f}% | "
                f"neg={( r['sentiment_latest'] < 0).mean()*100:.1f}%")
    logger.info(f"  Median trend   : {r['sentiment_trend'].median():.5f}")
    if "recent_negative_count_90d" in r.columns:
        logger.info(f"  recent_negative_count_90d>0: {(r['recent_negative_count_90d']>0).mean()*100:.1f}%")
