"""
src/data_loader.py
==================
Ingests the UCI Online Retail Excel file, applies strict preprocessing rules,
and returns a clean transaction-level DataFrame.

Preprocessing Rules (from project spec):
  1. Drop rows where CustomerID is null (guest checkouts).
  2. Remove cancellations: InvoiceNo starts with 'C'.
  3. Remove rows where Quantity <= 0 or UnitPrice <= 0.
  4. Compute TotalSpend = Quantity * UnitPrice.
  5. Parse InvoiceDate to datetime.
"""

import os
import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def load_and_clean(filepath: str) -> pd.DataFrame:
    """
    Load and clean the UCI Online Retail dataset.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to 'Online Retail.xlsx' (or .csv).

    Returns
    -------
    pd.DataFrame
        Clean transaction DataFrame with columns:
        [InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
         UnitPrice, CustomerID, Country, TotalSpend]

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If required columns are missing after loading.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    logger.info(f"Loading dataset from: {filepath}")
    _path = Path(filepath)  # pathlib handles Windows non-ASCII paths correctly

    if ext in (".xlsx", ".xls"):
        import io
        with _path.open("rb") as f:
            data = io.BytesIO(f.read())
        df = pd.read_excel(data, dtype={"CustomerID": str})
    elif ext == ".csv":
        df = pd.read_csv(_path, dtype={"CustomerID": str}, encoding="latin-1")
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .xlsx or .csv")

    logger.info(f"Raw dataset shape: {df.shape}")

    # ── Validate required columns ────────────────────────────────────────────
    required_cols = {
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── Step 1: Parse InvoiceDate ────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)

    # ── Step 2: Drop null CustomerID (guest checkouts) ───────────────────────
    n_before = len(df)
    df = df.dropna(subset=["CustomerID"])
    logger.info(f"Dropped {n_before - len(df):,} rows with null CustomerID.")

    # ── Step 3: Remove cancellations (InvoiceNo starts with 'C') ────────────
    n_before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    logger.info(f"Removed {n_before - len(df):,} cancellation transactions.")

    # ── Step 4: Sanitize Quantity and UnitPrice ──────────────────────────────
    n_before = len(df)
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    logger.info(f"Removed {n_before - len(df):,} rows with Quantity<=0 or UnitPrice<=0.")

    # ── Step 5: Compute TotalSpend ───────────────────────────────────────────
    df["TotalSpend"] = df["Quantity"] * df["UnitPrice"]

    # ── Step 6: Ensure CustomerID is integer ─────────────────────────────────
    df["CustomerID"] = df["CustomerID"].astype(float).astype(int)

    # ── Step 7: Reset index ──────────────────────────────────────────────────
    df = df.reset_index(drop=True)

    logger.info(
        f"Clean dataset shape: {df.shape} | "
        f"Unique customers: {df['CustomerID'].nunique():,} | "
        f"Date range: {df['InvoiceDate'].min().date()} → {df['InvoiceDate'].max().date()}"
    )
    return df


def get_snapshot_date(df: pd.DataFrame) -> pd.Timestamp:
    """
    Compute the snapshot date as max(InvoiceDate) + 1 day.
    Used as the reference point for Recency calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction DataFrame.

    Returns
    -------
    pd.Timestamp
        Snapshot date.
    """
    snapshot = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    logger.info(f"Snapshot date: {snapshot.date()}")
    return snapshot
