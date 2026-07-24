import os
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.amazon_pipeline.config import logger, TAU, SNAP_DATE, WINDOW_YEARS, MIN_FREQ, RAW_REVIEWS, RAW_META, OUT, BEHAVIORAL, SENTIMENT, _sep, _exists



def run_phase5(force=False):
    _sep("PHASE 5  Personalized EVI (Two-Pass)")
    out = OUT / "phase5" / "phase5_results.json"

    if not force and _exists(out):
        logger.info("Phase 5 output exists — skipping.")
        return

    (OUT / "phase5").mkdir(parents=True, exist_ok=True)

    from lifelines import WeibullAFTFitter
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    P_RESP = 0.15
    COST   = 1.0

    df = pd.read_csv(OUT / "phase2" / "customer_augmented_v3.csv")
    df = df.drop(columns=["last_purchase_date","first_purchase_date",
                           "dominant_complaint"], errors="ignore")
    df["T"] = df["T"].clip(lower=0.5)
    if "CustomerID" not in df.columns:
        df["CustomerID"] = np.arange(len(df))
        
    preds = pd.read_csv(OUT / "phase3" / "phase3_predictions.csv")
    df = df.merge(preds, on="CustomerID", how="left")
    
    churn_prob = df["calib_churn_prob_1095d"].values
    
    logger.info("Loaded calibrated 1095-day churn probabilities from Phase 3.")

    monetary   = df["Monetary"].fillna(0).values
    # Use hard survival threshold: LOST = churn_prob > 0.95 (equivalent to S(t) < 0.05 at 1095d)
    # This is consistent with θ_s = 0.05 used in other datasets (see PAPER_NUMBERS.md Sec 3)
    is_lost    = churn_prob > 0.95

    # ── Pass 1: constant p_resp EVI ──────────────────────────────────────────
    evi_1     = P_RESP * monetary * churn_prob - COST
    dec_1     = np.where(is_lost, "LOST", np.where(evi_1 > 0, "INTERVENE", "WAIT"))
    logger.info(f"Pass 1: {pd.Series(dec_1).value_counts().to_dict()}")

    # ── Load T2 tau_hat from Phase 4 ─────────────────────────────────────────
    t2_path = OUT / "phase4" / "uplift_df_T2_+sentiment.csv"
    t2_df   = pd.read_csv(t2_path)
    if "CustomerID" in t2_df.columns and "CustomerID" in df.columns:
        df_tau = df[["CustomerID"]].merge(t2_df[["CustomerID","tau_hat"]], on="CustomerID", how="left")
        tau    = df_tau["tau_hat"].fillna(t2_df["tau_hat"].median()).values
    else:
        tau = t2_df["tau_hat"].values[:len(df)]

    # Winsorise tau at P1-P99 to remove extreme outliers
    tau = np.clip(tau, np.percentile(tau, 1), np.percentile(tau, 99))
    logger.info(f"tau_hat (winsorized): mean={tau.mean():.3f}  "
                f"range=[{tau.min():.2f}, {tau.max():.2f}]")

    # ── Pass 2: personalized EVI (p_resp_personalized * CLV * churn - cost) ─────────────
    rank_pct = pd.Series(tau).rank(pct=True).values
    # Formula: P_RESP * (0.5 + rank_pct) → range [P_RESP*0.5, P_RESP*1.5]
    # This is the same relative-scaling formula used in the Sensitivity Grid (rr_base * (0.5 + rank_pct))
    # so that Pass 2 is a special case of the Grid with rr_base = P_RESP = 0.15
    p_resp_personalized = P_RESP * (0.5 + rank_pct)
    evi_2 = p_resp_personalized * monetary * churn_prob - COST
    dec_2 = np.where(is_lost, "LOST", np.where(evi_2 > 0, "INTERVENE", "WAIT"))
    logger.info(f"Pass 2: {pd.Series(dec_2).value_counts().to_dict()}")

    # ── Convergence ───────────────────────────────────────────────────────────
    non_lost  = ~is_lost
    overlap   = (dec_1[non_lost] == dec_2[non_lost]).mean() * 100
    switched  = (dec_1[non_lost] != dec_2[non_lost]).sum()
    logger.info(f"Decision overlap (non-LOST): {overlap:.1f}%  |  switches: {switched:,}")

    for from_d, to_d in [("INTERVENE","WAIT"), ("WAIT","INTERVENE")]:
        mask = (dec_1 == from_d) & (dec_2 == to_d)
        if mask.sum() > 0:
            logger.info(f"  {from_d}->>{to_d}: {mask.sum():,}  "
                        f"avg_tau={tau[mask].mean():.3f}  "
                        f"avg_Monetary=${monetary[mask].mean():.2f}")

    # ── Monte Carlo profit comparison ─────────────────────────────────────────
    np.random.seed(42)

    def mc_profit(decisions, true_r, n_sim=1000):
        mask = decisions == "INTERVENE"
        if mask.sum() == 0: return np.zeros(n_sim)
        M, C = monetary[mask], churn_prob[mask]
        R = true_r[mask]
        noise = np.random.normal(0, 0.03, size=(n_sim, len(M)))
        R_sim = np.clip(R + noise, 0.01, 0.99)
        return np.array([float(np.sum(M * C * R_sim[i] - COST)) for i in range(n_sim)])

    p1_arr = mc_profit(dec_1, p_resp_personalized)
    p2_arr = mc_profit(dec_2, p_resp_personalized)
    ci = lambda a: np.percentile(a, [2.5, 50, 97.5])
    ci1, ci2 = ci(p1_arr), ci(p2_arr)

    logger.info(f"Profit Pass1 (median): ${ci1[1]:,.0f}  [{ci1[0]:,.0f}, {ci1[2]:,.0f}]")
    logger.info(f"Profit Pass2 (median): ${ci2[1]:,.0f}  [{ci2[0]:,.0f}, {ci2[2]:,.0f}]")
    eff = (ci2[1] - ci1[1]) / abs(ci1[1]) * 100 if ci1[1] != 0 else 0
    logger.info(f"Efficiency gain Pass1->Pass2: {eff:+.1f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_out = df[["CustomerID","T","E","Monetary"]].copy()
    df_out["churn_prob_raw"]   = df["raw_churn_prob_1095d"]
    df_out["churn_prob_calib"] = df["calib_churn_prob_1095d"]
    df_out["tau_hat"]    = tau
    df_out["decision_1"] = dec_1
    df_out["evi_1"]      = evi_1
    df_out["decision_2"] = dec_2
    df_out["evi_2"]      = evi_2
    df_out.to_csv(OUT / "phase5" / "phase5_decisions.csv", index=False)

    results = {
        "pass1_intervene":    int((dec_1=="INTERVENE").sum()),
        "pass2_intervene":    int((dec_2=="INTERVENE").sum()),
        "overlap_pct":        round(overlap, 2),
        "switched":           int(switched),
        "profit_pass1":       round(float(ci1[1]), 2),
        "profit_pass2":       round(float(ci2[1]), 2),
        "efficiency_gain_pct":round(eff, 2),
    }
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, ax = plt.subplots(figsize=(7,4))
    ax.hist(p1_arr/1000, bins=50, alpha=0.6, color="#e74c3c", label="Pass 1 (constant p_resp)")
    ax.hist(p2_arr/1000, bins=50, alpha=0.6, color="#2ecc71", label="Pass 2 (personalized EVI)")
    ax.axvline(ci1[1]/1000, color="#c0392b", lw=2, ls="--")
    ax.axvline(ci2[1]/1000, color="#27ae60", lw=2, ls="--")
    ax.set_xlabel("Profit (K USD)"); ax.set_ylabel("Frequency")
    ax.set_title("Profit Distribution: Pass 1 vs Pass 2")
    ax.legend(); plt.tight_layout()
    ax.legend(); plt.tight_layout()
    plt.savefig(OUT / "phase5" / "profit_comparison.png", dpi=150)
    plt.close()

    # ── 3D Sensitivity Grid with Monte Carlo Noise ───────────────────────────
    logger.info("\n  === Running 3D Sensitivity Grid (Monte Carlo) ===")
    costs = [2, 5, 10]
    rrs = [0.05, 0.10, 0.15, 0.20, 0.25]
    margins = [0.20, 0.30, 0.40, 0.50]
    sigmas = [0.00, 0.03, 0.05, 0.10]
    
    n_simulations_per_scenario = 100
    grid_results = []
    
    for v_cost in costs:
        for rr_base in rrs:
            for g_margin in margins:
                for sigma in sigmas:
                    margin_mon = monetary * g_margin
                    
                    # EVI 1 (Constant estimated_p_resp)
                    e1 = rr_base * margin_mon * churn_prob - v_cost
                    d1 = np.where(is_lost, "LOST", np.where(e1 > 0, "INTERVENE", "WAIT"))
                    mask1 = (d1 == "INTERVENE")
                    
                    # EVI 2 (Personalized estimated_p_resp)
                    estimated_p_resp = rr_base * (0.5 + rank_pct)
                    e2 = estimated_p_resp * margin_mon * churn_prob - v_cost
                    d2 = np.where(is_lost, "LOST", np.where(e2 > 0, "INTERVENE", "WAIT"))
                    mask2 = (d2 == "INTERVENE")
                    
                    # EVI 3 (Guardrail - Hybrid: default to V1, override only when EVI > risk_buffer)
                    risk_buffer_map = {2: 0.0, 5: 1.0, 10: 3.0}
                    base_buffer = risk_buffer_map.get(v_cost, 0.0)
                    
                    # Extra penalty when base response rate too low (< 0.10)
                    penalty = 2.0 if rr_base < 0.10 else 0.0
                    risk_buffer = base_buffer + penalty
                    
                    # Hybrid logic: start from V1 decisions, override to V4 only when EVI > buffer
                    # This ensures lift_guardrail = pure delta vs V1; fallback cases contribute 0 lift
                    override_to_v4 = (~is_lost) & (e2 > risk_buffer)   # V4 confident enough
                    
                    # Run MC Simulations for this scenario
                    lifts_raw = []
                    lifts_guardrail = []
                    for _ in range(n_simulations_per_scenario):
                        # True response rate with noise
                        true_p_resp_1 = np.clip(rr_base + np.random.normal(0, sigma, size=len(churn_prob)), 0.01, 0.40)
                        true_p_resp_2 = np.clip(estimated_p_resp + np.random.normal(0, sigma, size=len(churn_prob)), 0.01, 0.40)
                        
                        profit_1 = np.sum(margin_mon[mask1] * churn_prob[mask1] * true_p_resp_1[mask1] - v_cost)
                        profit_2 = np.sum(margin_mon[mask2] * churn_prob[mask2] * true_p_resp_2[mask2] - v_cost)
                        
                        # Guardrail: cases overridden to V4 use personalized p_resp,
                        # cases falling back to V1 use V1's constant p_resp (same as profit_1 for those rows)
                        profit_3_override = np.sum(margin_mon[override_to_v4 & mask2] * churn_prob[override_to_v4 & mask2] * true_p_resp_2[override_to_v4 & mask2] - v_cost)
                        profit_3_fallback  = np.sum(margin_mon[(~override_to_v4) & mask1] * churn_prob[(~override_to_v4) & mask1] * true_p_resp_1[(~override_to_v4) & mask1] - v_cost)
                        profit_3 = profit_3_override + profit_3_fallback
                        
                        lifts_raw.append(profit_2 - profit_1)
                        lifts_guardrail.append(profit_3 - profit_1)
                        
                    lifts_raw = np.array(lifts_raw)
                    lifts_guardrail = np.array(lifts_guardrail)
                    
                    grid_results.append({
                        "voucher_cost": v_cost,
                        "response_rate_base": rr_base,
                        "gross_margin": g_margin,
                        "noise_sigma": sigma,
                        "mean_lift_raw": np.mean(lifts_raw),
                        "median_lift_raw": np.median(lifts_raw),
                        "pct5_lift_raw": np.percentile(lifts_raw, 5),
                        "worst_lift_raw": np.min(lifts_raw),
                        "win_rate_pct_raw": (lifts_raw > 0).mean() * 100,
                        "mean_lift_guardrail": np.mean(lifts_guardrail),
                        "median_lift_guardrail": np.median(lifts_guardrail),
                        "pct5_lift_guardrail": np.percentile(lifts_guardrail, 5),
                        "worst_lift_guardrail": np.min(lifts_guardrail),
                        "win_rate_pct_guardrail": (lifts_guardrail > 0).mean() * 100
                    })
                
    df_grid = pd.DataFrame(grid_results)
    logger.info(f"Sensitivity Grid Metrics ({len(df_grid)} scenarios):")
    logger.info(f"  Note: We assume true response rates deviate from estimated response rates under bounded Gaussian noise.")
    logger.info(f"  Avg Win Rate (Raw vs V1)       : {df_grid['win_rate_pct_raw'].mean():.1f}%")
    logger.info(f"  Avg Win Rate (Guardrail vs V1) : {df_grid['win_rate_pct_guardrail'].mean():.1f}%")
    logger.info(f"  Median Lift (Raw)              : ${df_grid['median_lift_raw'].median():,.2f}")
    logger.info(f"  Median Lift (Guardrail)        : ${df_grid['median_lift_guardrail'].median():,.2f}")
    logger.info(f"  Worst-case Lift (Raw)          : ${df_grid['worst_lift_raw'].min():,.2f}")
    logger.info(f"  Worst-case Lift (Guardrail)    : ${df_grid['worst_lift_guardrail'].min():,.2f}")
    
    df_grid.to_csv(OUT / "phase5" / "sensitivity_grid.csv", index=False)
    
    metrics = {
        "group_win_rate_pct_raw": float((df_grid['mean_lift_raw'] > 0).mean() * 100),
        "run_win_rate_pct_raw": float(df_grid['win_rate_pct_raw'].mean()),
        "median_lift_raw": float(df_grid['median_lift_raw'].median()),
        "pct5_lift_raw": float(df_grid['pct5_lift_raw'].median()),
        "worst_lift_raw": float(df_grid['worst_lift_raw'].min()),
        "group_win_rate_pct_guardrail": float((df_grid['mean_lift_guardrail'] > 0).mean() * 100),
        "run_win_rate_pct_guardrail": float(df_grid['win_rate_pct_guardrail'].mean()),
        "median_lift_guardrail": float(df_grid['median_lift_guardrail'].median()),
        "pct5_lift_guardrail": float(df_grid['pct5_lift_guardrail'].median()),
        "worst_lift_guardrail": float(df_grid['worst_lift_guardrail'].min())
    }
    
    with open(OUT / "phase5" / "sensitivity_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved -> {OUT}/phase5/")



