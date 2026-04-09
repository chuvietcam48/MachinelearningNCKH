"""
src/data_loader_cdnow.py
========================
Data loader specifically for the CDNOW dataset.

Source Format
-------------
Text file (space-delimited), no header.
Columns:
  1. Customer ID (string/int)
  2. Date (YYYYMMDD)
  3. Quantity (int)
  4. Dollar Value (float)

Target Schema (Standard)
------------------------
  - CustomerID   (string)
  - InvoiceNo    (string) — synthetic
  - InvoiceDate  (datetime)
  - Quantity     (int)
  - SalesAmount  (float)
"""

import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants for cleaner code
_COLS_RAW = ["CustomerID", "DateStr", "Quantity", "SalesAmount"]

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean CDNOW transaction data.

    Supports two file formats:
      1. CSV with header: ,customer_id,date,quantity,price  (cdnow.csv)
      2. Space-delimited, no header: CustomerID DateStr Quantity SalesAmount (CDNOW_master.txt)

    Parameters
    ----------
    file_path : str
        Path to CDNOW data file.

    Returns
    -------
    pd.DataFrame
        Standardized DataFrame with columns:
        ['CustomerID', 'InvoiceNo', 'InvoiceDate', 'Quantity', 'TotalSpend']
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[CDNOW] File not found: {file_path}")

    logger.info(f"Loading dataset from: {file_path}")

    # ── Auto-detect format ────────────────────────────────────────────────────
    # Read raw bytes to bypass Windows C-level parser OSError on non-ASCII paths.
    import io as _io
    with open(file_path, "rb") as _fh:
        _raw_bytes = _fh.read()
    first_line = _raw_bytes.split(b"\n", 1)[0].decode("utf-8", errors="replace").strip()

    if "customer_id" in first_line.lower() or first_line.startswith(","):
        # CSV format: ,customer_id,date,quantity,price
        df = pd.read_csv(_io.BytesIO(_raw_bytes), index_col=0)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={
            "customer_id": "CustomerID",
            "date": "InvoiceDate",
            "quantity": "Quantity",
            "price": "SalesAmount",
        })
        df["CustomerID"] = df["CustomerID"].astype(str)
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    else:
        # Legacy space-delimited format (CDNOW_master.txt)
        try:
            df = pd.read_csv(
                _io.BytesIO(_raw_bytes),
                sep=r"\s+",
                header=None,
                names=_COLS_RAW,
                dtype={
                    "CustomerID": str,
                    "DateStr": str,
                    "Quantity": int,
                    "SalesAmount": float
                },
                engine="python"
            )
        except Exception as e:
            logger.error(f"[CDNOW] Parsing failed: {e}")
            raise
        df["InvoiceDate"] = pd.to_datetime(df["DateStr"], format="%Y%m%d", errors="coerce")

    total_rows_raw = len(df)

    # 2. Drop invalid dates if any
    mask_date = df["InvoiceDate"].notna()
    if (~mask_date).sum() > 0:
        logger.warning(f"[CDNOW] Dropped {(~mask_date).sum()} rows with invalid dates")
        df = df[mask_date].copy()

    # 3. Create synthetic InvoiceNo if not present
    if "InvoiceNo" not in df.columns:
        df["InvoiceNo"] = df["CustomerID"] + "-" + df["InvoiceDate"].dt.strftime("%Y%m%d")

    # 4. Filter for positive Quantity/SalesAmount
    mask_valid = (df["Quantity"] > 0) & (df["SalesAmount"] > 0)
    n_invalid  = (~mask_valid).sum()
    if n_invalid > 0:
        logger.info(f"[CDNOW] Dropped {n_invalid} rows with <= 0 Quantity/Amount")
        df = df[mask_valid].copy()

    # 5. Select final columns
    out_df = df[["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "SalesAmount"]].copy()
    out_df = out_df.rename(columns={"SalesAmount": "TotalSpend"})

    # 7. Log summary stats
    n_cust = out_df["CustomerID"].nunique()
    n_inv  = out_df["InvoiceNo"].nunique()
    d_min  = out_df["InvoiceDate"].min().date()
    d_max  = out_df["InvoiceDate"].max().date()

    logger.info(
        f"[CDNOW] Loaded {len(out_df):,} rows (from {total_rows_raw:,} raw) | "
        f"Unique customers: {n_cust:,} | "
        f"Date range: {d_min} → {d_max}"
    )

    return out_df
