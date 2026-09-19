import os
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.amazon_pipeline.config import logger, TAU, SNAP_DATE, WINDOW_YEARS, MIN_FREQ, RAW_REVIEWS, RAW_META, OUT, BEHAVIORAL, SENTIMENT, _sep, _exists



def run_phase0(force=False):
    _sep("PHASE 0  Data Ingestion")
    out_cust = OUT / "phase0" / "customer_table.csv"
    out_rev  = OUT / "phase0" / "reviews.csv"

    if not force and _exists(out_cust, out_rev):
        logger.info("Phase 0 outputs exist — skipping. Use --force to rerun.")
        return

    (OUT / "phase0").mkdir(parents=True, exist_ok=True)

    from data.loader_amazon_mcauley_v2 import (
        load_reviews_streaming, load_metadata, join_reviews_metadata,
    )

    reviews  = load_reviews_streaming("CDs_and_Vinyl", data_dir="data/amazon_reviews")
    metadata = load_metadata("CDs_and_Vinyl",           data_dir="data/amazon_reviews")
    joined   = join_reviews_metadata(reviews, metadata)

    snap   = pd.Timestamp(SNAP_DATE)
    cutoff = snap - pd.DateOffset(years=WINDOW_YEARS)

    # Deduplicate same-day purchases per customer
    rdf_daily = joined.groupby(["reviewerID", "purchase_date"]).agg(
        price=("price", "sum"), overall=("overall", "mean")
    ).reset_index()

    grp           = rdf_daily.groupby("reviewerID")
    last_purchase = grp["purchase_date"].max()
    frequency     = grp["purchase_date"].count()

    mask = (frequency >= MIN_FREQ) & (last_purchase >= cutoff) & (last_purchase <= snap)
    valid_ids = frequency[mask].index
    rdf_cohort = rdf_daily[rdf_daily["reviewerID"].isin(valid_ids)].copy()

    logger.info(f"Cohort: {len(valid_ids):,} customers  |  {len(rdf_cohort):,} purchase-day events")

    # Build survival features
    grp2  = rdf_cohort.groupby("reviewerID")
    last2 = grp2["purchase_date"].max()
    first2= grp2["purchase_date"].min()
    freq2 = grp2["purchase_date"].count()
    mon2  = grp2["price"].sum()
    rec2  = (snap - last2).dt.days
    T     = (last2 - first2).dt.days.clip(lower=0.5)
    E     = (rec2 > TAU).astype(int)

    def ipt_stats(dates):
        gaps = dates.sort_values().diff().dt.days.dropna()
        return pd.Series({"ipt": float(gaps.mean()) if len(gaps) else np.nan,
                          "gap_dev": float(gaps.std(ddof=0)) if len(gaps) else 0.0})

    ipt_df = grp2["purchase_date"].apply(ipt_stats).unstack()
    ipt_df["ipt"] = ipt_df["ipt"].fillna(ipt_df["ipt"].median())
    ipt_df["gap_dev"] = ipt_df["gap_dev"].fillna(0)

    customer_df = pd.DataFrame({
        "CustomerID":          last2.index,
        "T":                   T,
        "E":                   E,
        "Recency":             rec2,
        "Frequency":           freq2,
        "Monetary":            mon2,
        "SinglePurchase":      (freq2 == 1).astype(int),
        "InterPurchaseTime":   ipt_df["ipt"],
        "GapDeviation":        ipt_df["gap_dev"],
        "last_purchase_date":  last2,
        "first_purchase_date": first2,
    }).reset_index(drop=True)

    logger.info(f"Churn E=1: {customer_df['E'].mean()*100:.1f}%  |  "
                f"Active E=0: {(1-customer_df['E'].mean())*100:.1f}%")
    logger.info(f"Median T={customer_df['T'].median():.0f}d  "
                f"Monetary=${customer_df['Monetary'].median():.2f}  "
                f"IPT={customer_df['InterPurchaseTime'].median():.0f}d")

    # Full reviews (with text) for cohort
    reviews_full = joined[joined["reviewerID"].isin(valid_ids)].rename(
        columns={"reviewerID": "CustomerID"}
    )[["CustomerID","asin","reviewText","overall","purchase_date","helpful_votes"]]

    customer_df.to_csv(out_cust, index=False)
    reviews_full.to_csv(out_rev,  index=False)
    logger.info(f"Saved -> {OUT}/phase0/")



