"""
src/causal/causal_data_prep.py
================================
Principled Data Preparation for Causal Inference.

Fixes three methodological issues in the naive approach
--------------------------------------------------------
ISSUE 1 — Endogenous outcome (Y = Monetary)
  EVI = p_response * Monetary * (1-S) - cost
  T (INTERVENE) is selected when EVI > 0, which is driven by Monetary.
  If we then use Y = Monetary, the estimator finds T predicts Y trivially
  (high-Monetary customers were selected for T=1).

  FIX: Use log1p(Monetary) to reduce scale extremes, AND exclude Monetary
       from the covariate matrix X.

ISSUE 2 — Endogenous features in X
  survival, hazard_now, evi are all direct determinants of T.
  Including them in X makes the propensity model trivially correct
  (e(X) -> 0 or 1) and destroys IPW/DR identification.

  FIX: Exclude {survival, hazard_now, evi, predicted_clv} from X.
       Keep only raw customer behaviour features:
         [Frequency, InterPurchaseTime, GapDeviation, SinglePurchase]
       Optionally add Recency (weakly endogenous — borderline).

ISSUE 3 — LOST customers in the analysis
  LOST customers (S(t) < floor) are not actionable and their inclusion
  distorts the control distribution.

  FIX: Restrict to INTERVENE + WAIT only (exclude LOST).

Outcome options
---------------
"log_monetary"   : log1p(Monetary)       — less scale-sensitive, main choice
"retention"      : 1 - E                 — binary, cleanest causal interpretation
"survival"       : S(t)                  — probability scale [0,1]
"monetary_std"   : (Monetary - mu)/sigma — z-scored continuous
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Features that CAUSE T — must be excluded from X to satisfy ignorability
_ENDOGENOUS_FEATURES = {
    "survival", "survival_now", "hazard_now", "evi", "predicted_clv",
    "clv_used", "optimal_window_days", "decision",
    # Monetary is excluded because EVI = f(Monetary) → T = f(EVI)
    "Monetary",
}

# Safe behavioural features (no direct causal path through EVI)
_SAFE_FEATURES = [
    "Frequency",
    "InterPurchaseTime",
    "GapDeviation",
    "SinglePurchase",
]
# Recency is borderline: correlated with survival but not directly in EVI
_BORDERLINE_FEATURES = ["Recency"]


def prepare_observational(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
    outcome: str = "log_monetary",
    include_recency: bool = False,
    restrict_active: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare (X, Y, T) for causal analysis of the Weibull INTERVENE decision.

    Parameters
    ----------
    weibull_decisions : pd.DataFrame
        Output of policy.make_intervention_decisions().
        Columns: CustomerID, decision, Monetary, survival, evi, hazard_now.
    customer_df : pd.DataFrame
        Original customer features indexed by CustomerID.
    outcome : str
        'log_monetary' | 'retention' | 'survival' | 'monetary_std'
    include_recency : bool
        Whether to include Recency in X (borderline endogenous).
    restrict_active : bool
        If True, restrict to INTERVENE + WAIT (exclude LOST).

    Returns
    -------
    X : ndarray (n, p)   — covariate matrix (no endogenous features)
    Y : ndarray (n,)     — outcome
    T : ndarray (n,)     — binary treatment (1=INTERVENE, 0=WAIT)
    feature_names : list[str]
    """
    # ── 1. Merge decisions with customer features ──────────────────────────
    cdf = customer_df.reset_index() if "CustomerID" not in customer_df.columns \
          else customer_df.copy()
    merged = weibull_decisions.merge(cdf, on="CustomerID", how="inner",
                                     suffixes=("_dec", ""))

    # ── 2. Restrict population ─────────────────────────────────────────────
    if restrict_active:
        before = len(merged)
        merged = merged[merged["decision"] != "LOST"].reset_index(drop=True)
        logger.info(
            "[CausalDataPrep] Restricted to INTERVENE+WAIT: %d -> %d rows (-%d LOST)",
            before, len(merged), before - len(merged),
        )

    # ── 3. Build T ─────────────────────────────────────────────────────────
    T = (merged["decision"] == "INTERVENE").astype(int).values

    # ── 4. Build Y ─────────────────────────────────────────────────────────
    # Resolve Monetary column (might be suffixed after merge)
    monetary_col = "Monetary_dec" if "Monetary_dec" in merged.columns else "Monetary"
    monetary     = merged[monetary_col].fillna(0).values.astype(float)

    if outcome == "log_monetary":
        Y = np.log1p(monetary)
        y_label = "log(1+Monetary)"
    elif outcome == "retention":
        # Y = 1 if customer did NOT churn (E=0), 0 if churned
        if "E" in merged.columns:
            Y = (1 - merged["E"].fillna(0)).values.astype(float)
            y_label = "Retention (1-E)"
        else:
            logger.warning("[CausalDataPrep] 'E' column not found — falling back to log_monetary.")
            Y = np.log1p(monetary)
            y_label = "log(1+Monetary) [fallback]"
    elif outcome == "survival":
        surv_col = "survival_dec" if "survival_dec" in merged.columns else "survival"
        if surv_col in merged.columns:
            Y = merged[surv_col].fillna(0.5).values.astype(float)
            y_label = "Survival S(t)"
        else:
            logger.warning("[CausalDataPrep] survival column not found — falling back.")
            Y = np.log1p(monetary)
            y_label = "log(1+Monetary) [fallback]"
    elif outcome == "monetary_std":
        mu, sig = monetary.mean(), monetary.std() + 1e-9
        Y = (monetary - mu) / sig
        y_label = "Standardized Monetary"
    else:
        raise ValueError(f"Unknown outcome '{outcome}'.")

    logger.info(
        "[CausalDataPrep] Outcome: %s | mean=%.4f | std=%.4f",
        y_label, Y.mean(), Y.std(),
    )

    # ── 5. Build X (safe, non-endogenous features) ─────────────────────────
    feat_candidates = _SAFE_FEATURES.copy()
    if include_recency:
        feat_candidates = _BORDERLINE_FEATURES + feat_candidates

    feature_names = [f for f in feat_candidates if f in merged.columns]

    if len(feature_names) == 0:
        logger.warning(
            "[CausalDataPrep] No safe features found! Falling back to all numeric cols."
        )
        feature_names = [c for c in merged.select_dtypes(include=np.number).columns
                         if c not in _ENDOGENOUS_FEATURES
                         and c not in {"CustomerID", "T", "E", "treatment"}]

    X = merged[feature_names].fillna(merged[feature_names].median()).values.astype(float)

    logger.info(
        "[CausalDataPrep] Final dataset | n=%d | n_treated=%d (%.1f%%) | "
        "features=%s | outcome=%s",
        len(X), int(T.sum()), T.mean() * 100, feature_names, y_label,
    )

    return X, Y, T, feature_names


def prepare_observational_multi_outcome(
    weibull_decisions: pd.DataFrame,
    customer_df: pd.DataFrame,
    include_recency: bool = False,
) -> dict:
    """
    Prepare all outcome variants for sensitivity analysis.

    Returns
    -------
    dict mapping outcome name -> (X, Y, T, feature_names)
    """
    results = {}
    for outcome in ["log_monetary", "retention", "survival", "monetary_std"]:
        try:
            X, Y, T, feats = prepare_observational(
                weibull_decisions, customer_df,
                outcome=outcome, include_recency=include_recency,
            )
            results[outcome] = (X, Y, T, feats)
        except Exception as e:
            logger.warning("[CausalDataPrep] Outcome '%s' failed: %s", outcome, e)
    return results


def build_x5_features(
    max_clients: int = 40_000,
    seed: int = 42,
    purchases_chunksize: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build X5 RCT dataset with RFM features from purchases.csv.

    Improves on the naive age+gender features by computing:
      - Frequency  : # unique purchase dates
      - Monetary   : total spend
      - Recency    : days since last purchase (relative to dataset end)
      - IPT        : mean inter-purchase interval

    Parameters
    ----------
    max_clients : int
        Max clients to include (stratified sample).
    seed : int
    purchases_chunksize : int
        Rows per chunk when streaming purchases.csv.

    Returns
    -------
    X, Y, T, feature_names
    """
    import os

    uplift_path    = os.path.join("data", "raw", "x5retail", "uplift_train.csv")
    clients_path   = os.path.join("data", "raw", "x5retail", "clients.csv")
    purchases_path = os.path.join("data", "raw", "x5retail", "purchases.csv")

    if not os.path.exists(uplift_path):
        raise FileNotFoundError(f"X5 uplift file not found: {uplift_path}")

    # ── Load uplift labels ─────────────────────────────────────────────────
    uplift = pd.read_csv(uplift_path)
    rng    = np.random.default_rng(seed)

    if len(uplift) > max_clients:
        n_per = max_clients // 2
        t1  = uplift[uplift["treatment_flg"] == 1].sample(
            min(n_per, (uplift["treatment_flg"] == 1).sum()), random_state=seed)
        t0  = uplift[uplift["treatment_flg"] == 0].sample(
            min(n_per, (uplift["treatment_flg"] == 0).sum()), random_state=seed)
        uplift = pd.concat([t1, t0]).reset_index(drop=True)

    client_set = set(uplift["client_id"].astype(str))
    logger.info("[X5 RFM] Loaded %d clients from uplift_train.csv", len(uplift))

    # ── Load demographic features ──────────────────────────────────────────
    if os.path.exists(clients_path):
        clients = pd.read_csv(clients_path, usecols=["client_id", "age", "gender"])
        clients["age"]            = clients["age"].fillna(40.0)
        clients["gender_encoded"] = clients["gender"].map(
            {"M": 1.0, "F": 0.0, "U": 0.5}).fillna(0.5)
        uplift = uplift.merge(
            clients[["client_id", "age", "gender_encoded"]],
            on="client_id", how="left",
        )
        uplift["age"]            = uplift["age"].fillna(40.0)
        uplift["gender_encoded"] = uplift["gender_encoded"].fillna(0.5)

    # ── Stream purchases.csv for RFM ───────────────────────────────────────
    rfm_data = {}
    if os.path.exists(purchases_path):
        logger.info("[X5 RFM] Streaming purchases.csv (chunksize=%d)...",
                    purchases_chunksize)
        n_chunks = 0
        try:
            for chunk in pd.read_csv(
                purchases_path,
                chunksize=purchases_chunksize,
                usecols=lambda c: c in {
                    "client_id", "transaction_datetime", "purchase_sum",
                    "trn_sum_from_iss",   # alternate column names
                },
                dtype={"client_id": str},
                on_bad_lines="skip",
            ):
                # Filter to target clients
                chunk = chunk[chunk["client_id"].isin(client_set)]
                if chunk.empty:
                    n_chunks += 1
                    if n_chunks > 30:
                        break
                    continue

                # Parse date
                date_col = ("transaction_datetime"
                            if "transaction_datetime" in chunk.columns
                            else chunk.columns[1])
                chunk[date_col] = pd.to_datetime(chunk[date_col], errors="coerce")

                # Sum column
                sum_col = ("purchase_sum" if "purchase_sum" in chunk.columns
                           else "trn_sum_from_iss" if "trn_sum_from_iss" in chunk.columns
                           else None)
                if sum_col is None:
                    n_chunks += 1
                    continue

                for cid, grp in chunk.groupby("client_id"):
                    if cid not in rfm_data:
                        rfm_data[cid] = {
                            "dates":  [], "spend": 0.0,
                        }
                    rfm_data[cid]["dates"].extend(
                        grp[date_col].dropna().tolist()
                    )
                    rfm_data[cid]["spend"] += grp[sum_col].fillna(0).sum()
                n_chunks += 1

            logger.info("[X5 RFM] Processed %d purchase chunks | "
                        "%d clients with purchase data", n_chunks, len(rfm_data))
        except Exception as exc:
            logger.warning("[X5 RFM] Purchase streaming failed: %s", exc)

    # ── Compute RFM per client ─────────────────────────────────────────────
    if rfm_data:
        # Find global max date for Recency
        all_dates = [d for v in rfm_data.values() for d in v["dates"]]
        max_date  = max(all_dates) if all_dates else pd.Timestamp("2019-03-18")

        rfm_rows = []
        for cid, v in rfm_data.items():
            dates_sorted = sorted(v["dates"])
            freq   = len(dates_sorted)
            monetary = v["spend"]
            last_date = dates_sorted[-1] if dates_sorted else max_date
            recency  = max((max_date - last_date).days, 0)
            if len(dates_sorted) >= 2:
                gaps = [(dates_sorted[i+1] - dates_sorted[i]).days
                        for i in range(len(dates_sorted)-1)]
                ipt = float(np.mean(gaps))
                gap_std = float(np.std(gaps))
            else:
                ipt     = recency
                gap_std = 0.0
            rfm_rows.append({
                "client_id": str(cid),
                "Frequency": freq,
                "Monetary_rfm": monetary,
                "Recency": recency,
                "InterPurchaseTime": ipt,
                "GapDeviation": gap_std,
                "SinglePurchase": int(freq == 1),
            })

        rfm_df = pd.DataFrame(rfm_rows)
        uplift = uplift.merge(rfm_df, on="client_id", how="left")
        logger.info("[X5 RFM] RFM features computed for %d clients", len(rfm_df))
    else:
        # Fallback: dummy RFM
        uplift["Frequency"] = 1.0
        uplift["Monetary_rfm"] = 0.0
        uplift["Recency"] = 30.0
        uplift["InterPurchaseTime"] = 30.0
        uplift["GapDeviation"] = 0.0
        uplift["SinglePurchase"] = 1.0
        logger.warning("[X5 RFM] No purchase data — using dummy RFM features.")

    # ── Assemble final arrays ──────────────────────────────────────────────
    feature_names = ["age", "gender_encoded",
                     "Frequency", "Recency", "InterPurchaseTime",
                     "GapDeviation", "SinglePurchase"]
    available = [f for f in feature_names if f in uplift.columns]

    # Fill any NaNs from missing purchase records with median
    for col in available:
        uplift[col] = uplift[col].fillna(uplift[col].median())

    X = uplift[available].values.astype(float)
    T = uplift["treatment_flg"].values.astype(int)
    Y = uplift["target"].values.astype(float)

    logger.info(
        "[X5 RFM] Final X5 dataset | n=%d | features=%s | "
        "treatment_rate=%.1f%% | target_rate=%.1f%%",
        len(X), available, T.mean()*100, Y.mean()*100,
    )
    return X, Y, T, available
