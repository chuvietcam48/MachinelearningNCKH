import os
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.amazon_pipeline.config import logger, TAU, SNAP_DATE, WINDOW_YEARS, MIN_FREQ, RAW_REVIEWS, RAW_META, OUT, BEHAVIORAL, SENTIMENT, _sep, _exists



def run_phase1(force=False):
    _sep("PHASE 1  Coarse Sentiment (VADER)")
    out = OUT / "phase1" / "customer_augmented_v3.csv"

    if not force and _exists(out):
        logger.info("Phase 1 output exists — skipping.")
        return

    (OUT / "phase1").mkdir(parents=True, exist_ok=True)
    from sentiment.extractor import run_sentiment_phase1

    reviews_df  = pd.read_csv(OUT / "phase0" / "reviews.csv",        parse_dates=["purchase_date"])
    customer_df = pd.read_csv(OUT / "phase0" / "customer_table.csv")
    logger.info(f"Input: {len(reviews_df):,} reviews | {len(customer_df):,} customers")

    augmented = run_sentiment_phase1(
        reviews_df=reviews_df,
        customer_df=customer_df,
        snapshot_date=SNAP_DATE,
        save_dir=str(OUT / "phase1"),
        backend="vader",
        chunk_size=100_000,
    )
    
    # Run sentence-level ABSA (Mixed Sentiment)
    (OUT / "phase2").mkdir(parents=True, exist_ok=True)
    augmented = _run_absa_sentence_level(
        reviews_df=reviews_df,
        customer_df=augmented,
        snapshot_date=SNAP_DATE
    )
    
    augmented.to_csv(out, index=False)
    logger.info(f"Saved -> {OUT}/phase1/customer_augmented_v3.csv  ({len(augmented):,} rows)")


def _run_absa_sentence_level(reviews_df: pd.DataFrame,
                              customer_df: pd.DataFrame,
                              snapshot_date: str) -> pd.DataFrame:
    """
    Sentence-level VADER ABSA.
    Splits each review into sentences, scores each with VADER.
    Computes mixed_flag and mixed_intensity.
    """
    import re
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    snap = pd.Timestamp(snapshot_date)
    # Temporal leak check
    df = reviews_df[reviews_df["purchase_date"] < snap].copy()
    if len(df) == 0:
        logger.info("[Sentence-VADER] No reviews before cutoff.")
        return customer_df.copy()

    sia = SentimentIntensityAnalyzer()

    def _score_sentences(text):
        sents = re.split(r'[.!?;]+', str(text))
        sents = [s.strip() for s in sents if len(s.strip()) > 8]
        if not sents:
            return 0, 0, 0
        pos = sum(1 for s in sents if sia.polarity_scores(s)["compound"] >= 0.05)
        neg = sum(1 for s in sents if sia.polarity_scores(s)["compound"] <= -0.05)
        return pos, neg, len(sents)

    logger.info(f"[Sentence-VADER] Processing {len(df):,} reviews...")

    rows = []
    for cid, grp in df.groupby("CustomerID"):
        # We focus on the *latest* review for latest_mixed_flag, or all reviews?
        # The easiest is to compute it globally per customer.
        tot_pos, tot_neg, tot_sents = 0, 0, 0
        latest_mixed = 0
        
        # Sort by date
        grp = grp.sort_values("purchase_date")
        
        for idx, row in grp.iterrows():
            text = row["reviewText"]
            p, n, total = _score_sentences(text)
            tot_pos += p
            tot_neg += n
            tot_sents += total
            
            # Update latest mixed flag for the last review
            if p > 0 and n > 0:
                latest_mixed = 1
            else:
                latest_mixed = 0
                
        mixed_intensity = min(tot_pos, tot_neg) / tot_sents if tot_sents > 0 else 0
        
        # Strict definition of mixed review
        mixed_flag = 1 if (tot_sents >= 2 and tot_pos >= 1 and tot_neg >= 1 and mixed_intensity >= 0.25) else 0
        
        rows.append({
            "CustomerID": cid, 
            "positive_sentence_count": tot_pos, 
            "negative_sentence_count": tot_neg,
            "mixed_flag": mixed_flag,
            "mixed_intensity": mixed_intensity,
            "latest_mixed_flag": latest_mixed
        })

    absa_df = pd.DataFrame(rows)

    mixed_pct = absa_df["mixed_flag"].mean() * 100
    logger.info(f"[Sentence-VADER] Customers with >=1 mixed review: {mixed_pct:.1f}%")

    out_df = customer_df.merge(absa_df, on="CustomerID", how="left")
    
    fill_cols = ["positive_sentence_count", "negative_sentence_count", "mixed_flag", "mixed_intensity", "latest_mixed_flag"]
    for col in fill_cols:
        out_df[col] = out_df[col].fillna(0)

    absa_df.to_csv(OUT / "phase2" / "absa_sentence_full.csv", index=False)
    return out_df



