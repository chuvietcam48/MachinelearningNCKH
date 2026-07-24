"""
Amazon CDs & Vinyl — Sentiment-Augmented Survival Pipeline
==========================================================
Dataset : Amazon Reviews 2018 (McAuley Lab)  →  CDs_and_Vinyl.json.gz
Maps to : CDNOW (existing pipeline)

Cohort  : repeat buyers (Freq>=2) with last purchase in last 2 years
tau     : 365 days  →  churn ~61% (vs 93% on full dataset)

Usage:
    python run_amazon_sentiment_pipeline.py                      # run all phases
    python run_amazon_sentiment_pipeline.py --phase 0            # single phase
    python run_amazon_sentiment_pipeline.py --skip 0             # skip Phase 0
    python run_amazon_sentiment_pipeline.py --force              # re-run even if outputs exist
    python run_amazon_sentiment_pipeline.py --phase 2 --absa-model  # Phase 2 with PyABSA model
    python run_amazon_sentiment_pipeline.py --phase 4 --x-learner   # Phase 4 + X-Learner

Phases:
    0  Data ingestion + cohort filtering
    1  Coarse sentiment (VADER)
    2  Aspect-Based Sentiment (ABSA, vectorized keyword; --absa-model for PyABSA)
    3  Weibull AFT ablation: V1 behavioral vs V2 +sentiment
    4  T-Learner uplift ablation: T1/T2 + X-Learner X1/X2 (--x-learner flag)
    5  Personalized EVI (two-pass)
    6  Full ablation table
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/amazon_cds_v3/pipeline.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)
sys.path.insert(0, "src")

# ── Global config ─────────────────────────────────────────────────────────────
TAU          = 270
SNAP_DATE    = "2018-10-02"
WINDOW_YEARS = 2      # last purchase within this many years of snapshot
MIN_FREQ     = 3      # distinct purchase days required
RAW_REVIEWS  = "data/amazon_reviews/CDs_and_Vinyl.json.gz"
RAW_META     = "data/amazon_reviews/meta_CDs_and_Vinyl.json.gz"
OUT          = Path("outputs/amazon_cds_v3")

BEHAVIORAL   = ["Recency", "Monetary", "InterPurchaseTime", "GapDeviation"]
SENTIMENT    = ["sentiment_latest", "sentiment_trend", "days_since_last_neg",
                "neg_count_90d", "avg_star_rating", "has_review"]


def _sep(title=""):
    logger.info("\n" + "=" * 55)
    if title:
        logger.info(f"  {title}")
        logger.info("=" * 55)


def _exists(*paths):
    return all(Path(p).exists() for p in paths)


