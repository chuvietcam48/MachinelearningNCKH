"""
src/feature_engine.py
=====================
Transforms clean transaction-level data into a customer-level feature matrix
suitable for survival analysis.

Features produced:
  RFM (Static):
    - Recency          : Days from last purchase to snapshot date
    - Frequency        : Number of unique invoices
    - Monetary         : Total spend (MU)

  Temporal (Dynamic):
    - InterPurchaseTime: Mean inter-purchase gap (days); 0.0 for single-visit
    - GapDeviation     : Std dev of inter-purchase gaps (days); 0.0 if < 2 gaps
    - SinglePurchase   : Binary flag — customer made only 1 purchase

  Survival Target:
    - T  : Observation window in days.
           Repeat purchasers : days from first to last purchase (active span).
           Single purchasers : Recency (days from only purchase to snapshot).
           Rationale: T must represent how long the customer has been *observed*,
           not just their inter-purchase span.  Using T=clip(1) for single-buyers
           creates an artificial spike at T=1 that biases rho < 1.
    - E  : Event indicator (1 = churned, 0 = censored)

Churn Definition:
    E_i = 1  if  Recency_i > tau  (customer has been inactive > tau days)
    E_i = 0  if  Recency_i <= tau (customer is still considered active)

Performance Note (Phase 7):
    InterPurchaseTime and GapDeviation are computed with fully vectorized
    pandas operations — no .apply() on groups.  The approach:
      1. Sort df by [CustomerID, InvoiceDate] once.
      2. Compute per-row date diff via groupby(...).diff().dt.days.
      3. Aggregate [mean, std] with a single groupby().agg() call.
    This is O(N log N) vs the previous O(N * k) .apply() loop.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_dynamic_tau(df: pd.DataFrame) -> int:
    """
    Calculate a data-driven inactivity threshold (tau) based on the 95th
    percentile of inter-purchase gaps for repeat customers.

    Rationale:
    ----------
    Reviewers often criticize arbitrary thresholds (e.g., 90 days). Using the
    95th percentile ensures the definition of 'churn' captures the tail end
    of the expected return distribution for this specific dataset.
    """
    # Unique (CustomerID, Date) pairs to avoid 0-day gaps from same-day orders
    df_sorted = (
        df[["CustomerID", "InvoiceDate"]]
        .drop_duplicates()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    df_sorted["gap_days"] = (
        df_sorted.groupby("CustomerID")["InvoiceDate"]
        .diff()
        .dt.days
    )
    avg_gaps = df_sorted.groupby("CustomerID")["gap_days"].mean().dropna()

    if not avg_gaps.empty:
        tau = int(np.percentile(avg_gaps, 95))
        logger.info(
            f"[AutoTau] DYNAMIC threshold calculated: {tau} days "
            f"(95th percentile of InterPurchaseTime, n={len(avg_gaps):,})"
        )
        return tau
    else:
        logger.warning("[AutoTau] No repeat customers found; falling back to 90 days.")
        return 90



def build_customer_features(
    df: pd.DataFrame,
    snapshot: pd.Timestamp,
    tau: int = 90,
    df_raw: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Aggregate transaction-level data to customer-level RFM + Survival features.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction DataFrame from data_loader.load_and_clean().
    snapshot : pd.Timestamp
        Reference date for Recency calculation (max date + 1 day).
    tau : int, optional
        Inactivity threshold in days to define churn event.
        If None or 0, it is dynamically calculated as the 95th percentile
        of InterPurchaseTime for repeat customers. (default: 90).
    df_raw : pd.DataFrame, optional
        Raw transaction DataFrame (same as df, or the unfiltered full set).
        When provided, computes ``future_spend`` per customer — the actual total
        spend in the lookforward window (snapshot - tau, snapshot].
        Used as the CLV regression target; customers with no future transactions
        receive 0.  If None, ``future_spend`` is omitted from the output.

    Returns
    -------
    pd.DataFrame
        Customer-level DataFrame indexed by CustomerID with columns:
        [Recency, Frequency, Monetary, InterPurchaseTime, GapDeviation,
         SinglePurchase, T, E]  + optionally ``future_spend``.

        If ``df`` contains ``treatment_flg`` and/or ``target_flag`` columns
        (X5 RetailHero RCT labels), they are propagated to the customer-level
        DataFrame via a per-customer ``first()`` aggregation.  This enables
        ``src.uplift.run_uplift_analysis()`` to switch to real-treatment mode
        automatically, producing an unbiased Qini coefficient.
    """
    logger.info(f"Building customer features | snapshot={snapshot.date()} | tau={tau} days")

    # ── Group by customer ────────────────────────────────────────────────────
    grp = df.groupby("CustomerID")

    # ── RFM Features ─────────────────────────────────────────────────────────
    recency   = (snapshot - grp["InvoiceDate"].max()).dt.days.rename("Recency")
    frequency = grp["InvoiceNo"].nunique().rename("Frequency")
    monetary  = grp["TotalSpend"].sum().rename("Monetary")

    # ── Temporal Features: Vectorized (Phase 7 performance fix) ──────────────
    # Step 1: Sort the full DataFrame once by CustomerID + InvoiceDate.
    df_sorted = (
        df[["CustomerID", "InvoiceDate"]]
        .drop_duplicates()
        .sort_values(["CustomerID", "InvoiceDate"])
    )
    df_sorted["gap_days"] = (
        df_sorted.groupby("CustomerID")["InvoiceDate"]
        .diff()
        .dt.days
    )

    # Step 2: Aggregate mean and std of gaps in one pass.
    gap_agg = (
        df_sorted.groupby("CustomerID")["gap_days"]
        .agg(
            InterPurchaseTime=("mean"),
            GapDeviation=("std"),
        )
    )

    # ── Dynamic Tau Logic (if requested) ─────────────────────────────────────
    if tau is None or tau <= 0:
        tau = calculate_dynamic_tau(df)
    else:
        logger.info(f"Building customer features | snapshot={snapshot.date()} | tau={tau} days")

    # ── SinglePurchase Flag ───────────────────────────────────────────────────
    single_purchase = (frequency == 1).astype(int).rename("SinglePurchase")

    # ── Survival Target: T (observation window) ───────────────────────────────
    # Repeat purchasers  : T = days from first to last purchase (active span)
    # Single purchasers  : T = Recency (days from only purchase to snapshot)
    # Rationale: T.clip(1) for single-buyers creates a spike at T=1 that
    #   biases rho < 1, inverting the Weibull hazard direction.
    first_purchase = grp["InvoiceDate"].min()
    last_purchase  = grp["InvoiceDate"].max()

    T_repeat = (last_purchase - first_purchase).dt.days  # 0 for single buyers
    T_single = recency                                    # observation window
    T = T_repeat.where(single_purchase == 0, other=T_single).rename("T")
    T = T.clip(lower=1)  # safety floor; T=0 causes log(0) in Weibull

    # ── Survival Target: E (event indicator) ─────────────────────────────────
    # E = 1 if customer has been inactive for more than tau days (churned)
    # E = 0 if customer is still within the active window (censored)
    E = (recency > tau).astype(int).rename("E")

    # ── Assemble customer DataFrame ───────────────────────────────────────────
    customer_df = pd.concat(
        [recency, frequency, monetary, gap_agg, single_purchase, T, E],
        axis=1,
    )

    # ── Anti-Leakage Imputation ───────────────────────────────────────────────
    # Impute InterPurchaseTime and GapDeviation with 0.0 for single-purchase
    # customers (NOT cross-customer median — that is leakage).
    # Zero is semantically correct: single-buyers have no inter-purchase gaps.
    # The SinglePurchase=1 flag lets the model learn a separate effect.
    n_single = int(customer_df["InterPurchaseTime"].isna().sum())
    customer_df["InterPurchaseTime"] = customer_df["InterPurchaseTime"].fillna(0.0)
    customer_df["GapDeviation"]      = customer_df["GapDeviation"].fillna(0.0)
    if n_single > 0:
        logger.info(
            f"Imputed InterPurchaseTime=0.0 and GapDeviation=0.0 for "
            f"{n_single:,} single-purchase customers (anti-leakage guard)."
        )

    # ── Future Spend (CLV Regression Target) ─────────────────────────────────
    # future_spend_i = total spend by customer i in the window (snapshot-tau, snapshot]
    # This represents the revenue that a successful intervention could recover.
    # Computed from df_raw when provided; otherwise omitted.
    if df_raw is not None:
        window_start = snapshot - pd.Timedelta(days=tau)
        window_end   = snapshot
        future_txns = df_raw[
            (df_raw["InvoiceDate"] > window_start) &
            (df_raw["InvoiceDate"] <= window_end)
        ]
        future_spend = (
            future_txns.groupby("CustomerID")["TotalSpend"]
            .sum()
            .rename("future_spend")
        )
        # Left-join: customers with no future transactions get 0
        customer_df = customer_df.join(future_spend, how="left")
        customer_df["future_spend"] = customer_df["future_spend"].fillna(0.0)
        n_with_future = (customer_df["future_spend"] > 0).sum()
        logger.info(
            f"[CLV Target] future_spend computed: "
            f"{n_with_future:,} / {len(customer_df):,} customers "
            f"have spend > 0 in the ({tau}d) forward window. "
            f"Mean={customer_df['future_spend'].mean():.2f} | "
            f"Median={customer_df['future_spend'].median():.2f}"
        )

    # ── RCT Label Propagation (X5 RetailHero) ────────────────────────────────
    # If the raw transaction df contains ground-truth treatment/outcome labels
    # (present in X5 uplift_train.csv merge), attach them at the customer level.
    # These columns are constant per customer (RCT-assigned), so .first() is correct.
    # This is a no-op for UCI / TaFeng / CDNOW which have no such columns.
    _rct_cols = [c for c in ("treatment_flg", "target_flag") if c in df.columns]
    if _rct_cols:
        rct_per_customer = (
            df.groupby("CustomerID")[_rct_cols]
            .first()
        )
        customer_df = customer_df.join(rct_per_customer, how="left")
        for col in _rct_cols:
            customer_df[col] = customer_df[col].fillna(0).astype(int)
        logger.info(
            f"[X5 RCT] Propagated {_rct_cols} to customer_df "
            f"({len(customer_df):,} customers) — uplift real-treatment mode enabled."
        )

    # ── Log summary statistics ────────────────────────────────────────────────
    n_customers = len(customer_df)
    n_churned   = customer_df["E"].sum()
    n_censored  = n_customers - n_churned
    churn_rate  = n_churned / n_customers * 100

    logger.info(
        f"Customer features built: {n_customers:,} customers | "
        f"Churned (E=1): {n_churned:,} ({churn_rate:.1f}%) | "
        f"Censored (E=0): {n_censored:,} ({100 - churn_rate:.1f}%)"
    )
    logger.info(
        f"T stats — mean: {customer_df['T'].mean():.1f}d | "
        f"median: {customer_df['T'].median():.1f}d | "
        f"max: {customer_df['T'].max():.1f}d | "
        f"single-purchase (T=Recency): {n_single:,} customers"
    )

    return customer_df


def sensitivity_analysis_tau(
    df: pd.DataFrame,
    snapshot: pd.Timestamp,
    tau_values: list = None,
) -> dict:
    """
    Run feature engineering across multiple tau thresholds.
    Used to assess robustness of the churn definition.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transaction DataFrame.
    snapshot : pd.Timestamp
        Snapshot date.
    tau_values : list of int, optional
        List of inactivity thresholds to test.
        If None, computes adaptively from the dataset duration:
        [duration//5, duration//3, duration//2].
        Rationale: fixed values like {60, 90, 120} are meaningless for
        a dataset shorter than 120 days (e.g. Ta Feng = 120d total).

    Returns
    -------
    dict
        Mapping {tau: customer_df} for each threshold.
    """
    if tau_values is None:
        duration = (df["InvoiceDate"].max() - df["InvoiceDate"].min()).days
        tau_values = sorted(set([
            max(duration // 5, 1),
            max(duration // 3, 1),
            max(duration // 2, 1),
        ]))
        logger.info(
            f"[SensitivityTau] Adaptive tau values (dataset duration={duration}d): "
            f"{tau_values}"
        )

    results = {}
    for tau in tau_values:
        logger.info(f"--- Sensitivity: tau = {tau} days ---")
        results[tau] = build_customer_features(df, snapshot, tau=tau)
    return results
