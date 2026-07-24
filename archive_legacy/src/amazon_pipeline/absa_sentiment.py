import os
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.amazon_pipeline.config import logger, TAU, SNAP_DATE, WINDOW_YEARS, MIN_FREQ, RAW_REVIEWS, RAW_META, OUT, BEHAVIORAL, SENTIMENT, _sep, _exists



def _run_absa_pyabsa_sample(reviews_df: pd.DataFrame,
                             customer_df: pd.DataFrame,
                             sample_customers: int = 200) -> pd.DataFrame:
    """
    PyABSA ATEPC model — true aspect-level mixed sentiment detection.

    Model: ATEPC multilingual (microsoft/mdeberta-v3-base backbone)
    Detects specific aspect terms + polarity per sentence.
    Mixed = review contains BOTH Positive and Negative aspect mentions.

    Requires: pip install pyabsa  +  transformers>=4.35,<5.0
    First run downloads ~1.3 GB (mdeberta-v3-base + ATEPC checkpoint).
    """
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        from pyabsa import AspectTermExtraction as ATEPC
        logger.info("[PyABSA] Loading ATEPC multilingual model (first run downloads ~1.3 GB)...")
        model = ATEPC.AspectExtractor("multilingual", auto_device=False, cal_perplexity=False)
        logger.info("[PyABSA] Model loaded successfully.")
    except Exception as e:
        logger.warning(f"[PyABSA] Model load failed: {e}. "
                       f"Ensure transformers>=4.35,<5.0 is installed.")
        return None

    # Sample customers
    sample_ids = reviews_df["CustomerID"].drop_duplicates().sample(
        min(sample_customers, reviews_df["CustomerID"].nunique()), random_state=42
    )
    sample_reviews = reviews_df[reviews_df["CustomerID"].isin(sample_ids)].copy()
    logger.info(f"[PyABSA] Scoring {len(sample_ids):,} customers "
                f"({len(sample_reviews):,} reviews) on CPU...")

    rows = []
    for cid, grp in sample_reviews.groupby("CustomerID"):
        all_pairs, mixed_count = [], 0
        for text in grp["reviewText"].tolist():
            try:
                r    = model.predict(str(text)[:400], print_result=False, ignore_error=True)
                pairs = list(zip(r.get("aspect", []), r.get("sentiment", [])))
            except Exception:
                pairs = []
            if not pairs:
                continue
            pols = [p for _, p in pairs]
            if "Positive" in pols and "Negative" in pols:
                mixed_count += 1
            all_pairs.extend(pairs)

        neg_pairs = [(a, p) for a, p in all_pairs if p == "Negative"]
        pos_pairs = [(a, p) for a, p in all_pairs if p == "Positive"]
        critical  = int(any(a.lower() in {"quality","cd","sound","delivery","shipping"}
                            for a, _ in neg_pairs))
        rows.append({
            "CustomerID":             cid,
            "neg_aspect_count_absa":  len(neg_pairs),
            "pos_aspect_count_absa":  len(pos_pairs),
            "mixed_review_count_absa":mixed_count,
            "critical_neg_flag_absa": critical,
            "has_pyabsa":             1,
        })

    absa_df = pd.DataFrame(rows)
    absa_df["mixed_customer_flag_absa"] = (absa_df["mixed_review_count_absa"] > 0).astype(int)

    # Report
    mixed_pct = absa_df["mixed_customer_flag_absa"].mean() * 100
    crit_pct  = absa_df["critical_neg_flag_absa"].mean() * 100
    logger.info(f"\n[PyABSA] Results on {len(sample_ids):,}-customer sample:")
    logger.info(f"  Customers with >=1 mixed review : {mixed_pct:.1f}%  (keyword: 0%)")
    logger.info(f"  critical_neg_flag_absa=1        : {crit_pct:.1f}%")

    # Top aspects
    from collections import Counter
    all_neg_aspects = [a for r in rows for a, p in
                       zip([""]*r["neg_aspect_count_absa"],
                           [""]*r["neg_aspect_count_absa"])]  # placeholder
    logger.info(f"  Avg neg aspects/customer        : "
                f"{absa_df['neg_aspect_count_absa'].mean():.2f}")

    # Merge into customer table
    out_df = customer_df.merge(absa_df, on="CustomerID", how="left")
    for col in ["neg_aspect_count_absa","pos_aspect_count_absa",
                "mixed_review_count_absa","critical_neg_flag_absa",
                "mixed_customer_flag_absa"]:
        out_df[col] = out_df[col].fillna(0).astype(int)
    out_df["has_pyabsa"] = out_df["has_pyabsa"].fillna(0).astype(int)

    # Churn analysis
    if "E" in out_df.columns:
        scored = out_df[out_df["has_pyabsa"] == 1]
        for flag_val, label in [(1, "mixed"), (0, "non-mixed")]:
            sub = scored[scored["mixed_customer_flag_absa"] == flag_val]
            if len(sub) > 0:
                logger.info(f"  Churn {label} ({len(sub):,}): {sub['E'].mean()*100:.1f}%")

    absa_df.to_csv(OUT / "phase2" / "absa_pyabsa_sample.csv", index=False)
    return out_df


def run_phase2(force=False, use_absa_model=False):
    _sep("PHASE 2  ABSA Aspect-Level Sentiment")
    out = OUT / "phase2" / "customer_augmented_v3.csv"

    if not force and _exists(out):
        # If model upgrade requested but not done yet, continue
        if use_absa_model and not _exists(OUT / "phase2" / "absa_model_sample.csv"):
            logger.info("Running PyABSA model upgrade on existing keyword output...")
        else:
            logger.info("Phase 2 outputs exist — skipping.")
            return

    (OUT / "phase2").mkdir(parents=True, exist_ok=True)

    # ── Step 1: Keyword-based ABSA (fast, full dataset) ──────────────────────
    from sentiment.absa_extractor import run_absa_vectorized

    scored_path = (OUT / "phase1" / "phase1_reviews_scored.csv"
                   if (OUT / "phase1" / "phase1_reviews_scored.csv").exists()
                   else OUT / "phase1" / "reviews_scored_v3.csv")
    scored      = pd.read_csv(scored_path, parse_dates=["purchase_date"])
    customer_df = pd.read_csv(OUT / "phase1" / "customer_augmented_v3.csv")

    augmented = run_absa_vectorized(scored, customer_df, save_dir=str(OUT / "phase2"))
    augmented.to_csv(out, index=False)
    logger.info(f"Keyword ABSA done: {len(augmented):,} customers")

    # ── Step 2: PyABSA model upgrade on sample (optional) ────────────────────
    if use_absa_model:
        logger.info("\n[ABSA Upgrade] PyABSA execution skipped. Sentence-level VADER is now natively computed in Phase 1.")

    logger.info(f"Saved -> {OUT}/phase2/")



