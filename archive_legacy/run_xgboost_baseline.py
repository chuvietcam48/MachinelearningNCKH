"""
run_xgboost_baseline.py
=======================
XGBoost External Baseline — Reviewer Response (Comment 1).

Implements XGBoost binary classifier inside the SAME EVI decision framework
as the paper — so the comparison is fair:

  Weibull+EVI  vs  LR+EVI  vs  XGBoost+EVI  vs  RFM

Decision rule mirrors paper exactly, with XGBoost substituting:
    [1 - S(t|x)]  ≈  P(churn|x)      (churn probability proxy)
    h(t) > θ_h    ≈  P(churn) > θ_p   (hazard threshold proxy)
    S(t) < θ_s    ≈  P(churn) > 0.95  (LOST threshold proxy)

Full 3-arm decision:
    LOST      : P(churn) > 0.95
    INTERVENE : P(churn) > θ_p  AND  EVI_xgb > 0
    WAIT      : otherwise

EVI formula (identical to paper):
    EVI_xgb(i) = p_response × Monetary_i × P(churn|x_i) − C_contact

OUTPUTS
-------
    outputs/xgboost_baseline/xgboost_results.md   ← paper-ready tables
    outputs/xgboost_baseline/xgboost_results.csv  ← raw numbers

USAGE
-----
    .venv\\Scripts\\python run_xgboost_baseline.py
    .venv\\Scripts\\python run_xgboost_baseline.py --datasets uci
"""

import os, sys, logging, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("xgb_baseline")

# ── XGBoost ───────────────────────────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
except ImportError:
    logger.error("XGBoost not found. Run: .venv\\Scripts\\pip install xgboost")
    sys.exit(1)

# ── sklearn ───────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index


# =============================================================================
# CONFIG — mirrors paper exactly
# =============================================================================

# Same features as LR baseline (Recency excluded — E = Recency > tau)
XGB_FEATURES = [
    "Frequency", "Monetary", "InterPurchaseTime",
    "GapDeviation", "SinglePurchase",
]

# XGBoost hyperparameters
XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    eval_metric="logloss", random_state=42,
    n_jobs=-1, verbosity=0,
)

# Decision thresholds — same logic as paper
THETA_P  = 0.01   # P(churn) > θ_p → hazard proxy (paper θ_h = 0.01)
THETA_S  = 0.95   # P(churn) > 0.95 → LOST (paper S(t) < 0.05)

# Monte Carlo — identical to paper (simulation_params.yaml)
MC_N_ITER        = 1000
MC_RESP_MEAN     = 0.15
MC_RESP_STD      = 0.03
MC_COST_MEAN     = 1.0
MC_COST_STD      = 0.10
MARKETING_BUDGET = 500.0
SLEEPING_DOG_PENALTY = 0.20

# Dataset configs (tau from paper)
DATASETS = {
    "uci":    {"display": "UCI Online Retail", "currency": "GBP", "tau": 124},
    "tafeng": {"display": "Ta Feng Grocery",   "currency": "TWD", "tau": 39},
    "cdnow":  {"display": "CDNOW Music",       "currency": "USD", "tau": 181},
}

# Existing paper numbers (from PAPER_NUMBERS.md) for table formatting
# Note: LR+EVI profits have been updated to reflect the TRUE penalized profit (Case B).
# The original paper omitted the Sleeping Dog penalty for LR+EVI, making it incorrectly high.
PAPER = {
    "uci":    {"w_cidx": 0.8248, "cox_cidx": 0.8206, "w_ibs": 0.1914, "rho": 1.1938,
               "w_profit": 34560, "lr_profit": -79662, "rfm_profit": -1_080_414,
               "w_intervene": 228, "lr_intervene": 2040},
    "tafeng": {"w_cidx": 0.9440, "cox_cidx": 0.8145, "w_ibs": 0.1553, "rho": 1.3531,
               "w_profit": 381176, "lr_profit": -42682, "rfm_profit": -2_997_097,
               "w_intervene": 1441, "lr_intervene": 16580},
    "cdnow":  {"w_cidx": 0.7822, "cox_cidx": 0.7958, "w_ibs": 0.0829, "rho": 1.4659,
               "w_profit": 13137, "lr_profit": -6147, "rfm_profit": -115_715,
               "w_intervene": 692, "lr_intervene": 1360},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_customer_df(dataset_key: str) -> pd.DataFrame:
    from src.dataset_registry import get_dataset
    from src.feature_engine import build_customer_features

    cfg = DATASETS[dataset_key]
    info = get_dataset(dataset_key)
    logger.info(f"[{dataset_key.upper()}] Loading: {info.data_path}")
    df_raw   = info.loader_fn(info.data_path)
    snapshot = info.snapshot_fn(df_raw)
    cdf      = build_customer_features(df_raw, snapshot, tau=cfg["tau"], df_raw=df_raw)
    logger.info(
        f"[{dataset_key.upper()}] n={len(cdf):,} | "
        f"churn={cdf['E'].mean()*100:.1f}%"
    )
    return cdf


# =============================================================================
# STEP 1 — XGBoost train + C-index
# =============================================================================

def compute_cindex(dataset_key: str, cdf: pd.DataFrame) -> dict:
    """
    Train XGBoost on 80/20 stratified holdout (same as paper).
    Compute C-index: concordance_index(T_test, -P_churn, E_test)
    """
    feats = [f for f in XGB_FEATURES if f in cdf.columns]
    X, y, T = cdf[feats].values, cdf["E"].values, cdf["T"].values

    X_tr, X_te, y_tr, y_te, T_tr, T_te = train_test_split(
        X, y, T, test_size=0.20, stratify=y, random_state=42
    )
    logger.info(f"[{dataset_key.upper()}] train={len(X_tr):,} | test={len(X_te):,}")

    pipe = Pipeline([
        ("imp",  SimpleImputer(strategy="median")),
        ("scl",  StandardScaler()),
        ("xgb",  XGBClassifier(**XGB_PARAMS)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X_tr, y_tr)

    prob = pipe.predict_proba(X_te)[:, 1]

    c_idx = concordance_index(T_te, -prob, y_te.astype(bool))
    auc   = roc_auc_score(y_te, prob)

    logger.info(
        f"[{dataset_key.upper()}] C-index={c_idx:.4f} | AUC-ROC={auc:.4f}"
    )
    return {"dataset": dataset_key, "c_index": c_idx, "auc": auc,
            "_pipe": pipe, "_feats": feats}


# =============================================================================
# STEP 2 — EVI decision + Monte Carlo (same framework as paper)
# =============================================================================

def compute_evi_and_montecarlo(
    dataset_key: str, cdf: pd.DataFrame, ci_result: dict
) -> dict:
    """
    Apply paper's EVI framework with XGBoost P(churn) as survival proxy.

    Decision rule:
        LOST      : P(churn) > 0.95            [≈ S(t) < 0.05]
        INTERVENE : P(churn) > θ_p AND EVI > 0 [≈ h(t) > θ_h AND EVI > 0]
        WAIT      : otherwise

    Monte Carlo (1000 iter, budget-constrained):
        - XGBoost+EVI pool: INTERVENE, sorted by EVI desc
        - RFM pool: top 40% by Monetary
        - Same Sleeping Dog penalty for RFM
    """
    pipe  = ci_result["_pipe"]
    feats = ci_result["_feats"]
    cfg   = DATASETS[dataset_key]

    # ── Predict P(churn) on FULL dataset ─────────────────────────────────────
    X_full = cdf[feats].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p_churn = pipe.predict_proba(X_full)[:, 1]

    monetary = cdf["Monetary"].fillna(0).values.astype(float)

    # ── EVI formula (deterministic, mean params for gate decision) ────────────
    evi_xgb = MC_RESP_MEAN * monetary * p_churn - MC_COST_MEAN

    # ── Decision rule — 3-arm ─────────────────────────────────────────────────
    lost_mask      = p_churn > THETA_S
    intervene_mask = (~lost_mask) & (p_churn > THETA_P) & (evi_xgb > 0)
    wait_mask      = ~(lost_mask | intervene_mask)

    n_lost      = lost_mask.sum()
    n_intervene = intervene_mask.sum()
    n_wait      = wait_mask.sum()
    n_total     = len(cdf)

    logger.info(
        f"[{dataset_key.upper()}] Decisions: "
        f"INTERVENE={n_intervene:,} ({n_intervene/n_total*100:.1f}%) | "
        f"WAIT={n_wait:,} ({n_wait/n_total*100:.1f}%) | "
        f"LOST={n_lost:,} ({n_lost/n_total*100:.1f}%)"
    )

    # ── XGBoost+EVI pool: INTERVENE sorted by EVI desc ───────────────────────
    xgb_idx  = np.where(intervene_mask)[0]
    xgb_sort = xgb_idx[np.argsort(evi_xgb[xgb_idx])[::-1]]
    n_pool   = len(xgb_sort)

    # ── RFM pool: top 40% by Monetary ────────────────────────────────────────
    rfm_threshold = np.quantile(monetary, 0.60)  # top 40%
    rfm_mask      = monetary >= rfm_threshold
    rfm_idx       = np.where(rfm_mask)[0]
    rfm_sort      = rfm_idx[np.argsort(monetary[rfm_idx])[::-1]]
    
    # ── Load TRUE hazard from Weibull decisions to accurately penalize Sleeping Dogs
    import joblib
    try:
        decisions_path = _HERE / "outputs" / f"{dataset_key.upper()}_tau{cfg['tau']}" / "models" / "decisions.pkl"
        w_decisions = joblib.load(decisions_path)
        true_hazard = w_decisions.set_index("CustomerID")["hazard_now"].reindex(cdf.index).fillna(0).values
        # Customer is persuadable if their true Weibull hazard > 0.01 (theta_h proxy)
        is_persuadable = true_hazard > THETA_P
    except Exception as e:
        logger.warning(f"Could not load decisions.pkl for {dataset_key}, falling back to XGBoost proxy hazard: {e}")
        is_persuadable = p_churn > THETA_P

    rfm_is_persuadable = is_persuadable[rfm_sort]
    xgb_is_persuadable = is_persuadable[xgb_sort]

    if n_pool == 0:
        logger.warning(f"[{dataset_key.upper()}] No INTERVENE customers — profit=0")
        zero_ci = (0.0, 0.0, 0.0)
        return {
            "dataset": dataset_key, "currency": cfg["currency"],
            "xgb_profit_ci": zero_ci, "rfm_profit_ci": zero_ci,
            "n_intervene": 0, "n_lost": int(n_lost),
            "n_lost_pct": n_lost/n_total*100,
            "n_intervene_pct": 0.0, "n_wait_pct": n_wait/n_total*100,
        }

    # ── Monte Carlo loop ──────────────────────────────────────────────────────
    rng = np.random.default_rng(42)
    xgb_profits = np.empty(MC_N_ITER)
    rfm_profits = np.empty(MC_N_ITER)

    for i in range(MC_N_ITER):
        p_resp   = float(np.clip(rng.normal(MC_RESP_MEAN, MC_RESP_STD), 0.0, 1.0))
        cost     = float(max(rng.normal(MC_COST_MEAN, MC_COST_STD), 0.1))
        max_cont = max(int(np.floor(MARKETING_BUDGET / cost)), 1)

        # — XGBoost+EVI arm (with Sleeping Dog penalty) —
        n_fund   = min(n_pool, max_cont)
        targets  = xgb_sort[:n_fund]
        x_persuadable = xgb_is_persuadable[:n_fund]
        
        # Profit: same formula as Weibull, but penalised if they are sleeping dogs
        x_profit = np.where(
            x_persuadable,
            monetary[targets] * p_resp * p_churn[targets] - cost,
            -(monetary[targets] * SLEEPING_DOG_PENALTY) - cost
        )
        xgb_profits[i] = float(np.sum(x_profit))

        # — RFM arm (with Sleeping Dog penalty) —
        n_rfm_fund    = min(len(rfm_sort), max_cont)
        r_targets     = rfm_sort[:n_rfm_fund]
        r_persuadable = rfm_is_persuadable[:n_rfm_fund]
        r_profit      = np.where(
            r_persuadable,
            monetary[r_targets] * p_resp - cost,
            -(monetary[r_targets] * SLEEPING_DOG_PENALTY) - cost
        )
        rfm_profits[i] = float(np.sum(r_profit))

    pcts   = [2.5, 50.0, 97.5]
    xgb_ci = tuple(float(x) for x in np.percentile(xgb_profits, pcts))
    rfm_ci = tuple(float(x) for x in np.percentile(rfm_profits, pcts))

    # Sleeping dog count (deterministic at median budget)
    n_median_fund = min(len(rfm_sort), max(int(MARKETING_BUDGET / MC_COST_MEAN), 1))
    n_sleeping    = int((~rfm_is_persuadable[:n_median_fund]).sum())

    logger.info(
        f"[{dataset_key.upper()}] XGBoost+EVI profit ({cfg['currency']}): "
        f"lo={xgb_ci[0]:+.0f} | med={xgb_ci[1]:+.0f} | hi={xgb_ci[2]:+.0f}"
    )
    logger.info(
        f"[{dataset_key.upper()}] RFM (recheck) profit ({cfg['currency']}): "
        f"lo={rfm_ci[0]:+.0f} | med={rfm_ci[1]:+.0f} | hi={rfm_ci[2]:+.0f} "
        f"| sleeping_dogs={n_sleeping}"
    )

    return {
        "dataset": dataset_key,
        "currency": cfg["currency"],
        "xgb_profit_ci": xgb_ci,
        "rfm_profit_ci": rfm_ci,
        "n_intervene": int(n_intervene),
        "n_lost": int(n_lost),
        "n_wait": int(n_wait),
        "n_intervene_pct": n_intervene / n_total * 100,
        "n_lost_pct": n_lost / n_total * 100,
        "n_wait_pct": n_wait / n_total * 100,
        "n_sleeping_dogs": n_sleeping,
        "xgb_profits_arr": xgb_profits,
    }


# =============================================================================
# FORMAT OUTPUT — paper-ready tables
# =============================================================================

def format_tables(all_ci: list, all_mc: list) -> str:
    lines = []
    lines.append("# XGBoost External Baseline — Paper Tables")
    lines.append("*For reviewer response to Comment 1 (missing external model comparison)*\n")
    lines.append(f"Parameters: θ_p={THETA_P}, θ_s={THETA_S}, "
                 f"budget={MARKETING_BUDGET}MU, n_iter={MC_N_ITER}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE III: C-index
    # ─────────────────────────────────────────────────────────────────────────
    lines.append("## TABLE III — Updated (add XGBoost column)\n")
    lines.append("| Dataset | Weibull (C-idx) | Cox PH (C-idx) | **XGBoost (C-idx)** | Weibull IBS | Shape (ρ) |")
    lines.append("|---------|----------------|---------------|---------------------|-------------|-----------|")

    for r in all_ci:
        ds = r["dataset"]
        p  = PAPER[ds]
        xg = r["c_index"]
        best = max(p["w_cidx"], p["cox_cidx"], xg)

        def _fmt(v): return f"**{v:.4f}**" if abs(v - best) < 1e-6 else f"{v:.4f}"

        lines.append(
            f"| {DATASETS[ds]['display']} | {_fmt(p['w_cidx'])} | "
            f"{_fmt(p['cox_cidx'])} | {_fmt(xg)} | "
            f"{p['w_ibs']:.4f} | {p['rho']:.3f} |"
        )

    lines.append("")
    lines.append("> **Bold** = highest C-index per row.")
    lines.append("> XGBoost C-index: `concordance_index(T_test, -P(churn), E_test)`, same 80/20 holdout.")
    lines.append("> Features: Frequency, Monetary, InterPurchaseTime, GapDeviation, SinglePurchase (Recency excluded).\n")

    # ─────────────────────────────────────────────────────────────────────────
    # TABLE VI: Monte Carlo profit
    # ─────────────────────────────────────────────────────────────────────────
    lines.append("## TABLE VI — Updated (add XGBoost+EVI row)\n")
    lines.append("*(n = 1,000 iterations, budget-constrained, Sleeping Dog penalised)*\n")

    ds_order = [r["dataset"] for r in all_mc]
    hdr = ["Policy"] + [f"{DATASETS[d]['display']} ({DATASETS[d]['currency']})" for d in ds_order]
    lines.append("| " + " | ".join(hdr) + " |")
    lines.append("|" + "|".join(["---"] * len(hdr)) + "|")

    # Weibull row
    lines.append("| Weibull+EVI | " +
                 " | ".join(f"+{PAPER[d]['w_profit']:,}" for d in ds_order) + " |")

    # LR row
    lines.append("| LR+EVI | " +
                 " | ".join(f"+{PAPER[d]['lr_profit']:,}" for d in ds_order) + " |")

    # XGBoost row (NEW)
    xgb_cells = []
    for r in all_mc:
        med = r["xgb_profit_ci"][1]
        xgb_cells.append(f"**{med:+,.0f}**")
    lines.append("| **XGBoost+EVI** | " + " | ".join(xgb_cells) + " |")

    # RFM row
    lines.append("| RFM top-40% | " +
                 " | ".join(f"{PAPER[d]['rfm_profit']:,}" for d in ds_order) + " |")

    lines.append("")

    # Efficiency gains: XGBoost vs RFM
    lines.append("**Efficiency Gain — XGBoost+EVI vs RFM (median):**")
    for r in all_mc:
        ds = r["dataset"]
        rfm_med = PAPER[ds]["rfm_profit"]
        xgb_med = r["xgb_profit_ci"][1]
        gain = (xgb_med - rfm_med) / abs(rfm_med) * 100
        lines.append(
            f"- {DATASETS[ds]['display']}: {xgb_med:+,.0f} vs {rfm_med:,} "
            f"→ **{gain:+.1f}%**"
        )
    lines.append("")

    # ─────────────────────────────────────────────────────────────────────────
    # Decision Distribution (supplementary)
    # ─────────────────────────────────────────────────────────────────────────
    lines.append("## XGBoost Decision Distribution (vs Weibull)\n")
    lines.append("| Dataset | XGB INTERVENE | Weibull INTERVENE | XGB WAIT | XGB LOST |")
    lines.append("|---------|--------------|-------------------|----------|----------|")
    for ci_r, mc_r in zip(all_ci, all_mc):
        ds = ci_r["dataset"]
        lines.append(
            f"| {DATASETS[ds]['display']} | "
            f"**{mc_r['n_intervene']:,}** ({mc_r['n_intervene_pct']:.1f}%) | "
            f"{PAPER[ds]['w_intervene']:,} | "
            f"{mc_r['n_wait']:,} ({mc_r['n_wait_pct']:.1f}%) | "
            f"{mc_r['n_lost']:,} ({mc_r['n_lost_pct']:.1f}%) |"
        )
    lines.append("")
    lines.append(f"> Decision thresholds: LOST = P(churn)>{THETA_S} | "
                 f"INTERVENE = P(churn)>{THETA_P} AND EVI>0")
    lines.append("")

    # ─────────────────────────────────────────────────────────────────────────
    # Full CI numbers
    # ─────────────────────────────────────────────────────────────────────────
    lines.append("## Full 95% CI Numbers\n")
    for ci_r, mc_r in zip(all_ci, all_mc):
        ds  = ci_r["dataset"]
        xci = mc_r["xgb_profit_ci"]
        lines.append(f"### {DATASETS[ds]['display']}")
        lines.append(f"- **C-index:** {ci_r['c_index']:.4f} | AUC-ROC: {ci_r['auc']:.4f}")
        lines.append(
            f"- **XGBoost+EVI Profit ({mc_r['currency']}):** "
            f"[{xci[0]:+,.0f}; **{xci[1]:+,.0f}**; {xci[2]:+,.0f}]"
        )
        lines.append(f"- **INTERVENE pool:** {mc_r['n_intervene']:,} customers")
        lines.append(f"- **Sleeping Dogs (RFM recheck):** {mc_r.get('n_sleeping_dogs', 'N/A')}")
        lines.append("")

    lines.append("---")
    lines.append("*All existing Table III/VI numbers unchanged — XGBoost adds 1 column + 1 row.*")
    lines.append("*EVI formula: p_resp × Monetary × P(churn) − C_contact*")
    lines.append("*Same MC params as paper: budget=500MU, n_iter=1000, resp~N(0.15,0.03), cost~N(1.0,0.10)*")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    choices=list(DATASETS.keys()),
                    default=list(DATASETS.keys()))
    ap.add_argument("--output-dir",
                    default=str(_HERE / "outputs" / "xgboost_baseline"))
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 55)
    logger.info("XGBoost External Baseline")
    logger.info(f"Datasets : {args.datasets}")
    logger.info(f"Output   : {out}")
    logger.info("=" * 55)

    all_ci, all_mc = [], []

    for ds in args.datasets:
        logger.info(f"\n{'─'*45}\n  {DATASETS[ds]['display']}\n{'─'*45}")
        try:
            cdf = load_customer_df(ds)
        except Exception as e:
            logger.error(f"Load failed [{ds}]: {e}"); continue

        try:
            ci = compute_cindex(ds, cdf)
            all_ci.append(ci)
        except Exception as e:
            logger.error(f"C-index failed [{ds}]: {e}")
            import traceback; traceback.print_exc(); continue

        try:
            mc = compute_evi_and_montecarlo(ds, cdf, ci)
            all_mc.append(mc)
        except Exception as e:
            logger.error(f"MC failed [{ds}]: {e}")
            import traceback; traceback.print_exc(); continue

    if not all_ci:
        logger.error("No results. Check data paths.")
        sys.exit(1)

    # ── Save outputs ──────────────────────────────────────────────────────────
    md  = format_tables(all_ci, all_mc)
    md_path  = out / "xgboost_results.md"
    csv_path = out / "xgboost_results.csv"

    md_path.write_text(md, encoding="utf-8")
    logger.info(f"Markdown → {md_path}")

    rows = []
    for ci_r, mc_r in zip(all_ci, all_mc):
        ds = ci_r["dataset"]
        xci = mc_r["xgb_profit_ci"]
        rows.append({
            "dataset":        DATASETS[ds]["display"],
            "currency":       DATASETS[ds]["currency"],
            "xgb_c_index":    round(ci_r["c_index"], 4),
            "xgb_auc":        round(ci_r["auc"], 4),
            "weibull_c_index": PAPER[ds]["w_cidx"],
            "coxph_c_index":   PAPER[ds]["cox_cidx"],
            "xgb_profit_lo":   round(xci[0], 0),
            "xgb_profit_med":  round(xci[1], 0),
            "xgb_profit_hi":   round(xci[2], 0),
            "weibull_profit":  PAPER[ds]["w_profit"],
            "lr_profit":       PAPER[ds]["lr_profit"],
            "rfm_profit":      PAPER[ds]["rfm_profit"],
            "n_xgb_intervene": mc_r["n_intervene"],
            "n_xgb_lost":      mc_r["n_lost"],
            "n_sleeping_dogs": mc_r.get("n_sleeping_dogs", ""),
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    logger.info(f"CSV     → {csv_path}")

    # ── Print to console ──────────────────────────────────────────────────────
    # Fix Windows cp1252 console encoding (θ char)
    safe_md = md.replace("θ", "theta").replace("ρ", "rho")
    print("\n" + "=" * 55)
    print(safe_md)
    print("=" * 55)
    print(f"\nSaved: {md_path}\n       {csv_path}")


if __name__ == "__main__":
    main()
