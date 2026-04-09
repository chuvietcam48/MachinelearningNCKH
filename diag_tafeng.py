"""Quick step-by-step diagnostic for tafeng benchmark failure."""
import sys, os, traceback, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.CRITICAL)

from src.dataset_registry import get_dataset
from src.feature_engine import build_customer_features, calculate_dynamic_tau
from src.models import train_weibull_aft, get_survival_features
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

ds = get_dataset("tafeng")
print("Step 1: loading data...")
df = ds.loader_fn(ds.data_path)
print(f"  OK: {df.shape}")

snap = ds.snapshot_fn(df)
tau = calculate_dynamic_tau(df)
print(f"  tau={tau}")

print("Step 2: feature engineering...")
cdf = build_customer_features(df, snap, tau=tau, df_raw=df)
print(f"  OK: {cdf.shape}, churn={cdf['E'].mean():.3f}")

print("Step 3: train/test split...")
min_cls = cdf["E"].value_counts().min()
strat = cdf["E"] if min_cls >= 2 else None
tr, te = train_test_split(cdf, test_size=0.2, random_state=42, stratify=strat)
print(f"  OK: train={len(tr)}, test={len(te)}")

print("Step 4: train Weibull...")
try:
    waf, dsc_tr, prep, af = train_weibull_aft(tr)
    print(f"  OK: C-index={waf.concordance_index_:.4f}, rho={getattr(waf,'rho_',None)}")
except Exception as e:
    print(f"  FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 5: IBS...")
try:
    input_feats = get_survival_features(tr)
    X_te = prep.transform(te[input_feats])
    df_sc = pd.DataFrame(X_te, columns=input_feats, index=te.index)
    df_sc = df_sc[af].copy()
    df_sc["T"] = te["T"].values
    df_sc["E"] = te["E"].values
    from src.evaluation import compute_integrated_brier_score
    ibs = compute_integrated_brier_score(waf, df_sc)
    print(f"  OK: IBS={ibs:.4f}")
except Exception as e:
    print(f"  FAILED at IBS: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 6: policy...")
try:
    X_all = prep.transform(cdf[input_feats])
    df_all = pd.DataFrame(X_all, columns=input_feats, index=cdf.index)[af].copy()
    df_all["T"] = cdf["T"].values
    df_all["E"] = cdf["E"].values
    from src.policy import make_intervention_decisions, rfm_intervention_decisions
    from src.models import rfm_segment
    from src.evaluation import load_config_with_overrides
    policy_cfg = load_config_with_overrides("tafeng").get("policy", {})
    min_evi = policy_cfg.get("min_evi_threshold", 0.0)
    wdec = make_intervention_decisions(waf, df_all, cdf, min_evi_threshold=min_evi)
    rfm_df = rfm_segment(cdf)
    rdec = rfm_intervention_decisions(rfm_df)
    from src.evaluation import compute_outreach_efficiency, compute_revenue_lift
    out = compute_outreach_efficiency(wdec, rdec)
    rev = compute_revenue_lift(wdec, rdec)
    print(f"  OK: Eff={out['efficiency_gain_pct']:.1f}%, Lift={rev['revenue_precision_lift_pct']:.1f}%")
except Exception as e:
    print(f"  FAILED at policy: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 7: uplift...")
try:
    from src.uplift import run_uplift_analysis
    ures = run_uplift_analysis(wdec, cdf)
    print(f"  OK: Qini={ures['qini_auc_ratio']:.4f}, Persuadables={ures['persuadable_pct']:.3f}")
except Exception as e:
    print(f"  FAILED at uplift: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Step 8: Monte Carlo...")
try:
    from src.simulator import run_monte_carlo_simulation
    mc = run_monte_carlo_simulation(df_decisions=wdec, n_iterations=200)
    print(f"  OK: weibull_profit_ci={mc.get('weibull_profit_ci')}")
except Exception as e:
    print(f"  FAILED at MC: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n=== ALL STEPS PASSED ===")
