"""
src/data_loader_x5.py
=====================
Data loader for the X5 RetailHero dataset.

Source Files (data/raw/x5retail/)
-----------------------------------
- purchases.csv     : transaction history (~45M rows)
    Cols: client_id, transaction_datetime, purchase_sum, trn_sum_from_iss, trn_sum_from_red, ...
- clients.csv       : customer demographics (client_id, first_issue_date, age, gender)
- uplift_train.csv  : RCT labels if present (client_id, treatment_flg, target_flag)
- products.csv      : product catalog (not used for survival modeling)

Performance Strategy
---------------------
purchases.csv has ~45M rows. We:
  1. Load uplift_train.csv if available → filter purchases to only those clients.
  2. Otherwise load ALL data in chunks (500k/chunk), cap at MAX_ROWS_PER_RUN.
  3. Drop customers with < 2 transactions (survival model needs inter-purchase gap).
"""

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_TRANSACTIONS  = 2         # survival model minimum
_MAX_ROWS_PER_RUN  = 3_000_000 # cap for pipeline speed (~5-10 min run)
_CHUNKSIZE         = 500_000   # chunk size for purchases.csv

_PURCHASE_COLS = {
    "client_id", "transaction_datetime", "purchase_sum",
    "trn_sum_from_iss", "trn_sum_from_red",
}


def load_data(purchases_path: str) -> pd.DataFrame:
    """
    Load X5 RetailHero purchases and normalize into the standard pipeline schema.

    Returns
    -------
    pd.DataFrame with columns:
        [CustomerID, InvoiceNo, InvoiceDate, Quantity, TotalSpend,
         treatment_flg*, target_flag*]   (* = present only if uplift_train.csv exists)
    """
    if not os.path.exists(purchases_path):
        raise FileNotFoundError(f"[X5] purchases.csv not found: {purchases_path}")

    base_dir = os.path.dirname(purchases_path)

    # ── 1. Try to load uplift_train.csv (RCT labels) ─────────────────────────
    uplift_df    = None
    valid_clients = None

    uplift_path = os.path.join(base_dir, "uplift_train.csv")
    exists = os.path.exists(uplift_path)
    logger.info(f"[DEBUG X5 Loader] base_dir='{base_dir}', uplift_path='{uplift_path}', exists={exists}")
    
    if exists:
        logger.info(f"[X5] uplift_train.csv found — loading RCT labels...")
        uplift_df = pd.read_csv(
            uplift_path, dtype={"client_id": str},
            usecols=lambda c: c in {"client_id", "treatment_flg", "target", "target_flag"},
        )
        if "target" in uplift_df.columns:
            uplift_df = uplift_df.rename(columns={"target": "target_flag"})
        
        uplift_cols  = [c for c in ["client_id", "treatment_flg", "target_flag"] if c in uplift_df.columns]
        uplift_df    = uplift_df[uplift_cols].drop_duplicates(subset=["client_id"])
        valid_clients = set(uplift_df["client_id"].astype(str))
        treat_rate   = uplift_df["treatment_flg"].mean() * 100 if "treatment_flg" in uplift_df.columns else float("nan")
        logger.info(
            f"[X5] uplift_train: {len(uplift_df):,} clients | "
            f"treatment rate: {treat_rate:.1f}%"
        )
    else:
        logger.warning(
            "[X5] uplift_train.csv not found — running in PROXY mode "
            "(Weibull INTERVENE used as treatment, Qini will be negative as with other datasets)."
        )

    # ── 2. Chunked read of purchases.csv ─────────────────────────────────────
    logger.info(f"[X5] Reading purchases.csv in chunks of {_CHUNKSIZE:,}...")
    chunks      = []
    total_read  = 0
    total_kept  = 0

    reader = pd.read_csv(
        purchases_path,
        chunksize=_CHUNKSIZE,
        dtype={"client_id": str},
        usecols=lambda c: c in _PURCHASE_COLS,
        low_memory=False,
    )

    for chunk in reader:
        total_read += len(chunk)
        if valid_clients is not None:
            chunk = chunk[chunk["client_id"].isin(valid_clients)]
        total_kept += len(chunk)
        if len(chunk) > 0:
            chunks.append(chunk)
        # Log progress every 5M rows
        if total_read % 5_000_000 == 0:
            logger.info(f"[X5]   {total_read:,} rows read, {total_kept:,} kept...")
        if total_kept >= _MAX_ROWS_PER_RUN:
            logger.info(f"[X5] Capped at {_MAX_ROWS_PER_RUN:,} kept rows after {total_read:,} raw rows.")
            break

    if not chunks:
        raise ValueError("[X5] No valid purchase rows found.")

    purchases = pd.concat(chunks, ignore_index=True)
    logger.info(
        f"[X5] Raw read complete: {total_read:,} rows scanned → "
        f"{len(purchases):,} kept | {purchases['client_id'].nunique():,} customers"
    )

    # ── 3. Parse datetime ────────────────────────────────────────────────────
    purchases["InvoiceDate"] = pd.to_datetime(purchases["transaction_datetime"], errors="coerce")
    n_bad = purchases["InvoiceDate"].isna().sum()
    if n_bad:
        logger.warning(f"[X5] Dropped {n_bad:,} rows with unparseable datetime.")
    purchases = purchases[purchases["InvoiceDate"].notna()].copy()

    # ── 4. Clean TotalSpend ──────────────────────────────────────────────────
    purchases["TotalSpend"] = pd.to_numeric(purchases["purchase_sum"], errors="coerce")
    bad = (purchases["TotalSpend"].isna() | (purchases["TotalSpend"] <= 0)).sum()
    if bad:
        logger.info(f"[X5] Dropped {bad:,} rows with invalid purchase_sum.")
    purchases = purchases[purchases["TotalSpend"] > 0].copy()

    # ── 5. Synthesize Quantity + InvoiceNo ────────────────────────────────────
    purchases = purchases.reset_index(drop=True)
    purchases["Quantity"]   = 1
    purchases["CustomerID"] = purchases["client_id"].astype(str)
    purchases["InvoiceNo"]  = (
        purchases["CustomerID"] + "-"
        + purchases["InvoiceDate"].dt.strftime("%Y%m%d") + "-"
        + purchases.index.astype(str)
    )

    # ── 6. Drop low-frequency customers ──────────────────────────────────────
    freq   = purchases.groupby("CustomerID")["InvoiceNo"].count()
    valid  = freq[freq >= _MIN_TRANSACTIONS].index
    before = purchases["CustomerID"].nunique()
    purchases = purchases[purchases["CustomerID"].isin(valid)].copy()
    logger.info(
        f"[X5] Frequency filter ({_MIN_TRANSACTIONS}+ txns): "
        f"{before:,} → {purchases['CustomerID'].nunique():,} customers"
    )

    # ── 7. Merge uplift labels (if available) ────────────────────────────────
    extra_cols = []
    if uplift_df is not None:
        purchases = purchases.merge(
            uplift_df.rename(columns={"client_id": "CustomerID"}).set_index("CustomerID"),
            left_on="CustomerID", right_index=True, how="left",
        )
        extra_cols = [c for c in ["treatment_flg", "target_flag"] if c in purchases.columns]
        coverage = purchases["treatment_flg"].notna().mean() * 100 if "treatment_flg" in purchases.columns else 0
        logger.info(f"[X5] Uplift label coverage: {coverage:.1f}% of rows")

    # ── 8. Final output ───────────────────────────────────────────────────────
    base_cols = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "TotalSpend"]
    out_df    = purchases[base_cols + extra_cols].copy()

    n_cust   = out_df["CustomerID"].nunique()
    med_txn  = out_df.groupby("CustomerID")["InvoiceNo"].count().median()
    d_min    = out_df["InvoiceDate"].min().date()
    d_max    = out_df["InvoiceDate"].max().date()

    logger.info(
        f"[X5] FINAL: {len(out_df):,} txns | {n_cust:,} customers | "
        f"{d_min} → {d_max} | median txns/customer: {med_txn:.1f}"
    )
    
    if med_txn < 3:
        logger.warning(
            f"[X5] Median txns = {med_txn:.1f} — recommend: main.py --dataset x5retail --tau 30"
        )
    return out_df
