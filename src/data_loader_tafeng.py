"""
src/data_loader_tafeng.py
=========================
Ingests the Ta Feng Grocery Dataset (line-item level), applies strict
preprocessing rules, and returns a clean transaction-level DataFrame that
conforms to the Standard Schema expected by feature_engine.build_customer_features().

Ta Feng Column Map (raw -> standard):
  CUSTOMER_ID   -> CustomerID
  TRANSACTION_DT -> InvoiceDate
  AMOUNT        -> Quantity   (items per line-item)
  SALES_PRICE   -> SalesAmount (price for the line-item, not per unit)

Standard Schema Output:
  CustomerID  : int
  InvoiceDate : datetime
  InvoiceNo   : str  (Synthetic: "<date_str>_<customer_id>")
  Quantity    : int   (summed per synthetic invoice)
  SalesAmount : float (summed per synthetic invoice; equivalent to TotalSpend)
  TotalSpend  : float (alias for SalesAmount -- required by feature_engine)

Synthetic Invoice Strategy:
  Ta Feng has no InvoiceNo. We assume one visit per customer per day.
  InvoiceNo = str(TRANSACTION_DT.date()) + "_" + str(CUSTOMER_ID)
  This collapses all line-items for a (customer, date) pair into one invoice.

Cleaning Rules (Strict):
  1. Drop rows where CUSTOMER_ID is NaN.
  2. Keep only AMOUNT > 0 AND SALES_PRICE > 0 (filter returns & data errors).
  3. Drop exact duplicate rows.
  4. Drop rows with unparseable TRANSACTION_DT.
"""

import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name constants (raw Ta Feng CSV)
# ---------------------------------------------------------------------------
_COL_DATE       = "TRANSACTION_DT"
_COL_CUSTOMER   = "CUSTOMER_ID"
_COL_AMOUNT     = "AMOUNT"
_COL_PRICE      = "SALES_PRICE"

# Required raw columns
_REQUIRED_COLS = {_COL_DATE, _COL_CUSTOMER, _COL_AMOUNT, _COL_PRICE}


def load_and_clean_tafeng(filepath: str) -> pd.DataFrame:
    """
    Load and clean the Ta Feng Grocery dataset.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to 'ta_feng_all_months_merged.csv'.

    Returns
    -------
    pd.DataFrame
        Clean transaction DataFrame with Standard Schema columns:
        [CustomerID, InvoiceDate, InvoiceNo, Quantity, SalesAmount, TotalSpend]
        Aggregated to synthetic-invoice level (one row per customer per day).

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing after loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Ta Feng dataset not found at: {filepath}")

    logger.info(f"[TaFeng] Loading dataset from: {filepath}")

    # ── Load CSV ─────────────────────────────────────────────────────────────
    # Read bytes manually first to bypass Windows OSError on non-ASCII paths.
    import io
    with open(filepath, "rb") as _f:
        _raw = io.BytesIO(_f.read())
    df = pd.read_csv(
        _raw,
        dtype=str,          # Read everything as string first
        encoding="utf-8",
    )
    logger.info(f"[TaFeng] Raw dataset shape: {df.shape}")

    # ── Normalize column names (strip whitespace + uppercase) ─────────────────
    df.columns = df.columns.str.strip().str.upper()

    # ── Validate required columns ─────────────────────────────────────────────
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[TaFeng] Missing required columns: {missing}")

    # ── Step 1: Task A – Parse TRANSACTION_DT strictly ───────────────────────
    n_before = len(df)
    df[_COL_DATE] = pd.to_datetime(df[_COL_DATE], errors="coerce")
    n_bad_dates = df[_COL_DATE].isna().sum()
    df = df.dropna(subset=[_COL_DATE])
    logger.info(
        f"[TaFeng] Dropped {n_bad_dates:,} rows with unparseable TRANSACTION_DT. "
        f"Remaining: {len(df):,}"
    )

    # ── Step 2: Task C.1 – Drop missing CUSTOMER_ID ──────────────────────────
    n_before = len(df)
    df = df.dropna(subset=[_COL_CUSTOMER])
    df = df[df[_COL_CUSTOMER].str.strip() != ""]
    logger.info(
        f"[TaFeng] Dropped {n_before - len(df):,} rows with null/empty CUSTOMER_ID. "
        f"Remaining: {len(df):,}"
    )

    # ── Step 3: Task A – Force numeric AMOUNT and SALES_PRICE ────────────────
    df[_COL_AMOUNT] = pd.to_numeric(df[_COL_AMOUNT], errors="coerce")
    df[_COL_PRICE]  = pd.to_numeric(df[_COL_PRICE],  errors="coerce")

    # Drop rows where forcing numeric failed
    n_before = len(df)
    df = df.dropna(subset=[_COL_AMOUNT, _COL_PRICE])
    logger.info(
        f"[TaFeng] Dropped {n_before - len(df):,} rows with non-numeric AMOUNT/SALES_PRICE. "
        f"Remaining: {len(df):,}"
    )

    # ── Step 4: Task C.2 – Handle returns (strict positive filter) ───────────
    n_before = len(df)
    df = df[(df[_COL_AMOUNT] > 0) & (df[_COL_PRICE] > 0)]
    logger.info(
        f"[TaFeng] Dropped {n_before - len(df):,} rows with AMOUNT<=0 or SALES_PRICE<=0. "
        f"Remaining: {len(df):,}"
    )

    # ── Step 5: Task C.3 – Drop exact duplicate rows ─────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    logger.info(
        f"[TaFeng] Dropped {n_before - len(df):,} exact duplicate rows. "
        f"Remaining: {len(df):,}"
    )

    # ── Step 6: Task B – Synthetic Invoice ID ────────────────────────────────
    # InvoiceNo = "<YYYY-MM-DD>_<CUSTOMER_ID>"
    # Assumes one store visit per customer per day (grocery proxy).
    df["InvoiceNo"] = (
        df[_COL_DATE].dt.strftime("%Y-%m-%d")
        + "_"
        + df[_COL_CUSTOMER].str.strip()
    )

    # ── Step 7: Task B – Aggregation to invoice level ────────────────────────
    # One row per synthetic invoice: sum Quantity and SalesAmount.
    agg = (
        df.groupby(["InvoiceNo", _COL_CUSTOMER, _COL_DATE], as_index=False)
        .agg(
            Quantity   =(_COL_AMOUNT, "sum"),
            SalesAmount=(_COL_PRICE,  "sum"),
        )
    )

    # ── Step 8: Standardise column names to match feature_engine schema ───────
    agg = agg.rename(columns={
        _COL_CUSTOMER: "CustomerID",
        _COL_DATE:     "InvoiceDate",
    })

    # CustomerID: keep as string (consistent with synthetic invoice key)
    # feature_engine groups by CustomerID, so type only needs to be consistent.
    agg["CustomerID"] = agg["CustomerID"].str.strip()

    # TotalSpend alias: feature_engine.build_customer_features uses TotalSpend
    agg["TotalSpend"] = agg["SalesAmount"]

    # ── Step 9: Final reset + log summary ────────────────────────────────────
    agg = agg.reset_index(drop=True)

    logger.info(
        f"[TaFeng] Clean synthetic-invoice dataset shape: {agg.shape} | "
        f"Unique customers: {agg['CustomerID'].nunique():,} | "
        f"Unique invoices:  {agg['InvoiceNo'].nunique():,} | "
        f"Date range: {agg['InvoiceDate'].min().date()} → {agg['InvoiceDate'].max().date()}"
    )
    return agg


def get_snapshot_date_tafeng(df: pd.DataFrame) -> pd.Timestamp:
    """
    Compute the snapshot date as max(InvoiceDate) + 1 day.
    Mirrors the UCI get_snapshot_date() interface.

    Parameters
    ----------
    df : pd.DataFrame
        Clean Ta Feng transaction DataFrame (invoice-level).

    Returns
    -------
    pd.Timestamp
        Snapshot date.
    """
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"[TaFeng] Snapshot date: {snapshot.date()}")
    return snapshot
