"""
Phase 0 — Amazon Reviews 2018 (McAuley Lab) Data Ingestion & Join

Streaming approach: processes JSONL in chunks to avoid OOM on large files.

Categories (map to existing pipeline datasets):
  CDs_and_Vinyl            -> CDNOW
  Grocery_and_Gourmet_Food -> TaFeng
"""

import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    "CDs_and_Vinyl":            "CDNOW",
    "Grocery_and_Gourmet_Food": "TaFeng",
}

_DATA_DIR  = Path(__file__).resolve().parents[2] / "data" / "amazon_reviews"
_CHUNK     = 200_000   # rows per chunk — keeps RAM under ~1 GB


# =============================================================================
# 1. Streaming loader (chunk-based, low RAM)
# =============================================================================

def _iter_gz_jsonl(path: str):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _detect_path(data_dir: Path, category: str, prefix: str = "", suffix: str = ".json.gz") -> Path:
    """Try _5 variant first, fall back to full dataset."""
    for name in (f"{prefix}{category}_5{suffix}", f"{prefix}{category}{suffix}"):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No file found for category '{category}' in {data_dir}.\n"
        f"Expected: {prefix}{category}_5{suffix} or {prefix}{category}{suffix}"
    )


def load_reviews_streaming(category: str, data_dir: str = None) -> pd.DataFrame:
    """
    Stream review JSONL in chunks, keep only verified + non-empty text.
    Returns a lean DataFrame — only columns needed for survival + NLP.
    """
    data_dir = Path(data_dir or _DATA_DIR)
    path     = _detect_path(data_dir, category)
    size_mb  = path.stat().st_size / 1024 / 1024
    logger.info(f"[Loader] Reading reviews: {path.name}  ({size_mb:.0f} MB)")

    keep_cols = ["reviewerID", "asin", "reviewText", "overall",
                 "unixReviewTime", "verified"]

    chunks   = []
    buf      = []
    total    = 0
    kept     = 0

    for rec in _iter_gz_jsonl(str(path)):
        total += 1

        # verified filter (field may be absent in older files -> keep)
        if rec.get("verified") is False:
            continue

        text = (rec.get("reviewText") or rec.get("review_body") or "").strip()
        if not text:
            continue

        buf.append({
            "reviewerID":      rec.get("reviewerID") or rec.get("reviewer_id"),
            "asin":            rec.get("asin"),
            "reviewText":      text,
            "overall":         float(rec.get("overall", np.nan)),
            "unixReviewTime":  rec.get("unixReviewTime") or rec.get("unix_review_time"),
            "vote":            rec.get("vote") or rec.get("helpful"),
        })
        kept += 1

        if len(buf) >= _CHUNK:
            chunks.append(pd.DataFrame(buf))
            buf = []
            logger.info(f"[Loader]   {total:,} scanned | {kept:,} kept")

    if buf:
        chunks.append(pd.DataFrame(buf))

    if not chunks:
        raise ValueError("No valid reviews found after filtering.")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"[Loader] Done: {total:,} scanned -> {len(df):,} kept "
                f"({len(df)/total*100:.1f}% pass rate)")

    # Parse helpful_votes
    def _parse_vote(v):
        if isinstance(v, list):  return v[0] if v else 0
        if isinstance(v, (int, float)): return int(v)
        if isinstance(v, str):
            return int(v.replace(",", "")) if v.replace(",", "").isdigit() else 0
        return 0
    df["helpful_votes"] = df["vote"].apply(_parse_vote)

    df["purchase_date"] = pd.to_datetime(df["unixReviewTime"], unit="s", errors="coerce")
    df = df.dropna(subset=["purchase_date", "reviewerID"])

    logger.info(f"[Loader] {df['reviewerID'].nunique():,} unique users | "
                f"date range: {df['purchase_date'].min().date()} to "
                f"{df['purchase_date'].max().date()}")

    return df[["reviewerID", "asin", "reviewText", "overall",
               "purchase_date", "helpful_votes"]].reset_index(drop=True)


def load_metadata(category: str, data_dir: str = None) -> pd.DataFrame:
    """Load product metadata, extract price. Streaming to save RAM."""
    data_dir = Path(data_dir or _DATA_DIR)
    path     = _detect_path(data_dir, category, prefix="meta_")
    logger.info(f"[Loader] Reading metadata: {path.name}  "
                f"({path.stat().st_size/1024/1024:.0f} MB)")

    rows = []
    for rec in _iter_gz_jsonl(str(path)):
        asin  = rec.get("asin")
        price = rec.get("price")
        if asin:
            rows.append({"asin": asin, "price_raw": str(price) if price else ""})

    meta = pd.DataFrame(rows).drop_duplicates("asin")
    def _parse_price(s):
        # Extract first valid decimal number from string, ignore junk
        import re
        m = re.search(r"\d+\.\d+|\d+", str(s))
        if m:
            try:
                v = float(m.group())
                return v if 0 < v < 10_000 else np.nan   # sanity range
            except ValueError:
                pass
        return np.nan

    meta["price"] = meta["price_raw"].apply(_parse_price)
    median_price      = meta["price"].median()
    n_missing         = meta["price"].isna().sum()
    meta["price"]     = meta["price"].fillna(median_price)

    logger.info(f"[Loader] Metadata: {len(meta):,} products | "
                f"price missing: {n_missing:,} ({n_missing/len(meta)*100:.1f}%) "
                f"-> filled with median=${median_price:.2f}")
    return meta[["asin", "price"]].reset_index(drop=True)


# =============================================================================
# 2. Join
# =============================================================================

def join_reviews_metadata(reviews: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    df = reviews.merge(metadata, on="asin", how="left")
    still_missing = df["price"].isna().sum()
    if still_missing:
        fallback = df["price"].median()
        df["price"] = df["price"].fillna(fallback)
        logger.warning(f"[Join] {still_missing:,} rows still missing price "
                       f"-> filled with overall median=${fallback:.2f}")
    return df


# =============================================================================
# 3. Customer survival table (mirrors feature_engine.py logic)
# =============================================================================

def build_customer_table(df: pd.DataFrame, tau: int,
                          snapshot_date: str = None) -> pd.DataFrame:
    """
    Build per-customer survival features.

    T  : repeat buyers = first->last purchase span (days)
         single buyers = recency (days)
    E  : 1 if Recency > tau (churned)
    """
    snap = pd.Timestamp(snapshot_date) if snapshot_date else df["purchase_date"].max()
    logger.info(f"[CustomerTable] snapshot={snap.date()} | tau={tau}d")

    grp   = df.groupby("reviewerID")
    last  = grp["purchase_date"].max()
    first = grp["purchase_date"].min()
    freq  = grp["purchase_date"].count()
    mon   = grp["price"].sum()
    rec   = (snap - last).dt.days

    single   = freq == 1
    T        = (last - first).dt.days.where(~single, other=rec)
    E        = (rec > tau).astype(int)

    # Inter-purchase time stats
    def _ipt(dates):
        gaps = dates.sort_values().diff().dt.days.dropna()
        if len(gaps) == 0:
            return pd.Series({"ipt": np.nan, "gap_dev": 0.0})
        return pd.Series({"ipt": gaps.mean(), "gap_dev": gaps.std(ddof=0)})

    ipt_df = grp["purchase_date"].apply(_ipt).unstack()
    median_ipt = ipt_df.loc[~single, "ipt"].median()

    cdf = pd.DataFrame({
        "CustomerID":         last.index,
        "T":                  T.values,
        "E":                  E.values,
        "Recency":            rec.values,
        "Frequency":          freq.values,
        "Monetary":           mon.values,
        "SinglePurchase":     single.astype(int).values,
        "InterPurchaseTime":  ipt_df["ipt"].fillna(median_ipt).values,
        "GapDeviation":       ipt_df["gap_dev"].fillna(0).values,
        "last_purchase_date": last.values,
        "first_purchase_date": first.values,
    }).reset_index(drop=True)

    logger.info(
        f"[CustomerTable] {len(cdf):,} customers | "
        f"repeat: {(~single).sum():,} ({(~single).mean()*100:.1f}%) | "
        f"E=1: {E.sum():,} ({E.mean()*100:.1f}%)"
    )
    return cdf


# =============================================================================
# 4. Main entry point
# =============================================================================

def load_amazon_dataset(category: str, tau: int,
                         snapshot_date: str = None,
                         data_dir: str = None) -> dict:
    """
    Full Phase 0 pipeline for one Amazon category.

    Returns dict with keys:
      reviews_df    : [CustomerID, asin, reviewText, overall, purchase_date, helpful_votes]
      customer_df   : survival features per customer
      dataset_label : e.g. "CDs_and_Vinyl (CDNOW)"
      tau, snapshot_date, category
    """
    label = f"{category} ({CATEGORY_MAP.get(category, category)})"
    logger.info(f"\n{'='*60}\n  Phase 0: {label}\n{'='*60}")

    reviews  = load_reviews_streaming(category, data_dir)
    metadata = load_metadata(category, data_dir)
    joined   = join_reviews_metadata(reviews, metadata)

    reviews_df = (joined[["reviewerID", "asin", "reviewText", "overall",
                           "purchase_date", "helpful_votes"]]
                  .rename(columns={"reviewerID": "CustomerID"})
                  .copy())

    customer_df = build_customer_table(joined, tau=tau, snapshot_date=snapshot_date)

    _print_summary(customer_df, reviews_df, label, tau)

    snap = snapshot_date or str(joined["purchase_date"].max().date())
    return {"reviews_df": reviews_df, "customer_df": customer_df,
            "dataset_label": label, "tau": tau,
            "snapshot_date": snap, "category": category}


def _print_summary(cdf, rdf, label, tau):
    repeat = cdf[cdf["SinglePurchase"] == 0]
    sep = "-" * 50
    logger.info(f"\n{sep}")
    logger.info(f"  Dataset      : {label}")
    logger.info(f"  Customers    : {len(cdf):,}")
    logger.info(f"  Repeat buyers: {len(repeat):,}  ({len(repeat)/len(cdf)*100:.1f}%)")
    logger.info(f"  Churn (tau={tau}d): {cdf['E'].mean()*100:.1f}%")
    logger.info(f"  Median T     : {cdf['T'].median():.0f} d")
    logger.info(f"  Median Mon.  : ${cdf['Monetary'].median():.2f}")
    logger.info(f"  Avg text len : {rdf['reviewText'].str.len().mean():.0f} chars")
    logger.info(f"  Reviews >=10w: "
                f"{(rdf['reviewText'].str.split().str.len()>=10).mean()*100:.1f}%")
    logger.info(sep)
