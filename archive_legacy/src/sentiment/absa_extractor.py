"""
Phase 2 - Aspect-Based Sentiment Analysis (ABSA)

Model: PyABSA ATEPC (Aspect Term Extraction & Polarity Classification)
Checkpoint: multilingual (English supported, no fine-tune needed)

Input:  reviews_df [CustomerID, reviewText, purchase_date]
Output: customer_df augmented with 5 ABSA features per customer

Features per customer:
  neg_aspect_count   : total (aspect, Negative) pairs across all reviews
  pos_aspect_count   : total (aspect, Positive) pairs
  mixed_review_count : reviews containing BOTH pos and neg aspects
  critical_neg_flag  : 1 if "quality" or "delivery" negatively mentioned >= 1
  dominant_complaint : most-frequent negative aspect (string, or None)
"""

import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Aspect keywords for retail/CD domain
RETAIL_ASPECTS = {
    "delivery":         ["delivery", "shipping", "arrived", "package", "fast", "slow",
                         "late", "delay", "ship", "transit"],
    "quality":          ["quality", "broken", "damaged", "defective", "excellent", "poor",
                         "scratch", "crack", "condition", "pristine", "mint"],
    "price":            ["price", "expensive", "cheap", "worth", "value", "overpriced",
                         "cost", "afford", "deal", "bargain"],
    "customer_service": ["service", "support", "refund", "return", "response", "helpful",
                         "seller", "vendor", "contacted", "resolved"],
    "packaging":        ["packaging", "wrapped", "box", "sealed", "cover", "case",
                         "jacket", "sleeve", "insert"],
}

CRITICAL_ASPECTS = {"quality", "delivery"}


# =============================================================================
# 1. Aspect extraction
# =============================================================================

def load_absa_model():
    """Load PyABSA ATEPC model (downloads on first call, cached after)."""
    try:
        from pyabsa import AspectTermExtraction as ATEPC
        # english checkpoint
        model = ATEPC.AspectExtractor(
            "multilingual",
            auto_device=False,       # CPU; set True for GPU auto-detect
            cal_perplexity=False,
        )
        logger.info("[ABSA] Model loaded: PyABSA ATEPC multilingual")
        return model
    except Exception as e:
        logger.error(f"[ABSA] Failed to load PyABSA model: {e}")
        raise


def extract_aspects_pyabsa(text: str, model) -> list:
    """
    Extract (aspect_term, polarity) pairs from one review using PyABSA.
    Returns list of (str, str) tuples, polarity in ['Positive','Negative','Neutral']
    """
    try:
        result = model.predict(text, print_result=False, ignore_error=True)
        aspects   = result.get("aspect", [])
        polarities = result.get("sentiment", [])
        return list(zip(aspects, polarities))
    except Exception:
        return []


def extract_aspects_keyword(text: str) -> list:
    """
    Keyword-fallback ABSA: match aspect keywords + surrounding sentiment words.
    Much faster than model-based, used when PyABSA unavailable or for large batches.

    Returns list of (aspect_category, polarity) — polarity = Positive/Negative/Neutral
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    text_lower = text.lower()
    results    = []

    for aspect, keywords in RETAIL_ASPECTS.items():
        for kw in keywords:
            idx = text_lower.find(kw)
            if idx == -1:
                continue
            # Get local window (±50 chars) around keyword
            window = text_lower[max(0, idx-50): idx+len(kw)+50]
            score  = sia.polarity_scores(window)["compound"]
            if score >= 0.05:
                pol = "Positive"
            elif score <= -0.05:
                pol = "Negative"
            else:
                pol = "Neutral"
            results.append((aspect, pol))
            break   # one match per aspect per review

    return results


# =============================================================================
# 2. Per-customer ABSA aggregation
# =============================================================================

def aggregate_absa_per_customer(texts: list, use_model: bool = False,
                                  model=None) -> dict:
    """
    Run ABSA on all reviews of ONE customer, return 5 aggregate features.

    Parameters
    ----------
    texts      : list of review strings for this customer
    use_model  : True = PyABSA model; False = keyword fallback (faster)
    model      : loaded PyABSA model (required if use_model=True)
    """
    all_pairs   = []
    mixed_count = 0

    for text in texts:
        if use_model and model:
            pairs = extract_aspects_pyabsa(text, model)
        else:
            pairs = extract_aspects_keyword(text)

        if not pairs:
            continue

        pols = [p for _, p in pairs]
        if "Positive" in pols and "Negative" in pols:
            mixed_count += 1
        all_pairs.extend(pairs)

    neg_pairs = [(a, p) for a, p in all_pairs if p == "Negative"]
    pos_pairs = [(a, p) for a, p in all_pairs if p == "Positive"]

    dominant_complaint = None
    if neg_pairs:
        dominant_complaint = Counter(a for a, _ in neg_pairs).most_common(1)[0][0]

    critical_neg = int(
        any(a in CRITICAL_ASPECTS for a, _ in neg_pairs)
    )

    return {
        "neg_aspect_count":   len(neg_pairs),
        "pos_aspect_count":   len(pos_pairs),
        "mixed_review_count": mixed_count,
        "critical_neg_flag":  critical_neg,
        "dominant_complaint": dominant_complaint,
    }


# =============================================================================
# 3. Vectorized ABSA — fast path using pre-computed Phase 1 sentiment scores
# =============================================================================

def run_absa_vectorized(reviews_scored: pd.DataFrame,
                         customer_df: pd.DataFrame,
                         save_dir: str = None) -> pd.DataFrame:
    """
    Fast vectorized ABSA using Phase 1 VADER scores already computed.

    Strategy:
    - Detect aspect keywords via vectorized str.contains (no per-row Python loop)
    - Use review-level sentiment_score from Phase 1 as polarity signal
    - Aggregate to customer level with pure pandas operations

    Input: reviews_scored [CustomerID, reviewText, sentiment_score, sentiment_label]
    """
    import re
    df = reviews_scored.copy()
    df["text_lower"] = df["reviewText"].str.lower().fillna("")

    logger.info(f"[ABSA] Vectorized keyword detection on {len(df):,} reviews...")

    # Detect each aspect with str.contains (vectorized)
    aspect_cols = {}
    for aspect, keywords in RETAIL_ASPECTS.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        aspect_cols[aspect] = df["text_lower"].str.contains(pattern, regex=True, na=False)

    # Classify aspect polarity using review-level sentiment
    # positive = sentiment_score >= 0.05, negative = <= -0.05
    is_pos = df["sentiment_score"] >= 0.05
    is_neg = df["sentiment_score"] <= -0.05

    logger.info("[ABSA] Aggregating features per customer...")

    rows = []
    for aspect in RETAIL_ASPECTS:
        mask     = aspect_cols[aspect]
        df[f"asp_neg_{aspect}"] = (mask & is_neg).astype(int)
        df[f"asp_pos_{aspect}"] = (mask & is_pos).astype(int)

    neg_asp_cols = [f"asp_neg_{a}" for a in RETAIL_ASPECTS]
    pos_asp_cols = [f"asp_pos_{a}" for a in RETAIL_ASPECTS]

    # mixed = review mentions any aspect AND has both pos and neg signal
    df["has_any_asp"]  = df[[f"asp_neg_{a}" for a in RETAIL_ASPECTS] +
                              [f"asp_pos_{a}" for a in RETAIL_ASPECTS]].sum(axis=1) > 0
    df["mixed_review"] = df["has_any_asp"] & is_pos & is_neg  # single review pos+neg (rare with VADER)

    grp = df.groupby("CustomerID")

    neg_aspect_count   = grp[neg_asp_cols].sum().sum(axis=1).rename("neg_aspect_count")
    pos_aspect_count   = grp[pos_asp_cols].sum().sum(axis=1).rename("pos_aspect_count")
    mixed_review_count = grp["mixed_review"].sum().rename("mixed_review_count")

    # critical_neg_flag = 1 if quality or delivery was negatively mentioned
    crit_cols = [f"asp_neg_{a}" for a in CRITICAL_ASPECTS]
    critical_neg_flag = (grp[crit_cols].sum() > 0).any(axis=1).astype(int).rename("critical_neg_flag")

    # dominant_complaint = aspect with most negative mentions per customer
    neg_per_asp = grp[neg_asp_cols].sum()
    neg_per_asp.columns = list(RETAIL_ASPECTS.keys())
    dominant_complaint = neg_per_asp.idxmax(axis=1)
    dominant_complaint[neg_per_asp.max(axis=1) == 0] = "none"

    feat = pd.concat([neg_aspect_count, pos_aspect_count, mixed_review_count,
                      critical_neg_flag, dominant_complaint.rename("dominant_complaint")],
                     axis=1).reset_index()

    # Merge into customer table
    augmented = customer_df.merge(feat, on="CustomerID", how="left")
    for col in ["neg_aspect_count","pos_aspect_count","mixed_review_count","critical_neg_flag"]:
        augmented[col] = augmented[col].fillna(0).astype(int)
    augmented["dominant_complaint"] = augmented["dominant_complaint"].fillna("none")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        feat.to_csv(str(Path(save_dir) / "absa_features_v2.csv"), index=False)
        augmented.to_csv(str(Path(save_dir) / "customer_augmented_v2.csv"), index=False)
        logger.info(f"[ABSA] Saved -> {save_dir}/")

    _print_absa_summary(augmented, feat)
    return augmented


# =============================================================================
# 4. Full dataset pipeline (original per-customer loop — kept for PyABSA model use)
# =============================================================================

def run_absa_phase2(reviews_df: pd.DataFrame,
                    customer_df: pd.DataFrame,
                    save_dir: str = None,
                    sample_n: int = None,
                    use_model: bool = False) -> pd.DataFrame:
    """
    Full Phase 2 ABSA pipeline.

    Parameters
    ----------
    reviews_df  : [CustomerID, reviewText, purchase_date, ...] — from Phase 0
    customer_df : survival features per customer — from Phase 0/1
    save_dir    : save outputs here
    sample_n    : sample N customers (None = full run)
    use_model   : True = PyABSA model (slow, accurate)
                  False = keyword fallback (fast, approximate)
    """
    model = None
    if use_model:
        model = load_absa_model()

    if sample_n:
        ids         = reviews_df["CustomerID"].drop_duplicates().sample(
                          min(sample_n, reviews_df["CustomerID"].nunique()), random_state=42)
        reviews_df  = reviews_df[reviews_df["CustomerID"].isin(ids)].copy()
        customer_df = customer_df[customer_df["CustomerID"].isin(ids)].copy()
        logger.info(f"[ABSA] Sample mode: {sample_n:,} customers")

    # Filter min-length reviews
    reviews_df = reviews_df[
        reviews_df["reviewText"].str.split().str.len() >= 5
    ].copy()

    total_customers = reviews_df["CustomerID"].nunique()
    logger.info(f"[ABSA] Processing {total_customers:,} customers | "
                f"backend={'PyABSA' if use_model else 'keyword-VADER'}")

    # Group reviews by customer and run ABSA
    rows   = []
    for i, (cid, grp) in enumerate(reviews_df.groupby("CustomerID")):
        texts = grp["reviewText"].tolist()
        feat  = aggregate_absa_per_customer(texts, use_model=use_model, model=model)
        feat["CustomerID"] = cid
        rows.append(feat)

        if i % 50_000 == 0 and i > 0:
            logger.info(f"[ABSA]   {i:,}/{total_customers:,} ({i/total_customers*100:.1f}%)")

    absa_features = pd.DataFrame(rows)

    # Merge into customer table
    augmented = customer_df.merge(absa_features, on="CustomerID", how="left")
    for col in ["neg_aspect_count", "pos_aspect_count", "mixed_review_count",
                "critical_neg_flag"]:
        if col in augmented.columns:
            augmented[col] = augmented[col].fillna(0).astype(int)
    augmented["dominant_complaint"] = augmented["dominant_complaint"].fillna("none")

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        absa_features.to_csv(str(Path(save_dir) / "phase2_absa_features.csv"), index=False)
        augmented.to_csv(str(Path(save_dir) / "phase2_customer_augmented.csv"), index=False)
        logger.info(f"[ABSA] Saved -> {save_dir}/")

    _print_absa_summary(augmented, absa_features)
    return augmented


def _print_absa_summary(augmented: pd.DataFrame, features: pd.DataFrame):
    has_rev = augmented[augmented.get("has_review", pd.Series(1, index=augmented.index)) == 1] \
              if "has_review" in augmented.columns else augmented

    mixed_pct = (features["mixed_review_count"] > 0).mean() * 100
    crit_pct  = features["critical_neg_flag"].mean() * 100
    top3 = (features["dominant_complaint"]
            .replace("none", np.nan)
            .dropna()
            .value_counts()
            .head(3))

    logger.info("\n[Phase 2 Summary]")
    logger.info(f"  Customers processed     : {len(features):,}")
    logger.info(f"  Mixed sentiment reviews : {mixed_pct:.1f}%")
    logger.info(f"  critical_neg_flag=1     : {crit_pct:.1f}%")
    logger.info(f"  Top 3 complaints        : {top3.to_dict()}")

    if "E" in augmented.columns and "critical_neg_flag" in augmented.columns:
        crit_churn    = augmented[augmented["critical_neg_flag"] == 1]["E"].mean() * 100
        noncrit_churn = augmented[augmented["critical_neg_flag"] == 0]["E"].mean() * 100
        logger.info(f"  Churn critical_neg=1 : {crit_churn:.1f}%")
        logger.info(f"  Churn critical_neg=0 : {noncrit_churn:.1f}%")
