import os
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.amazon_pipeline.config import logger, TAU, SNAP_DATE, WINDOW_YEARS, MIN_FREQ, RAW_REVIEWS, RAW_META, OUT, BEHAVIORAL, SENTIMENT, _sep, _exists



def run_phase3(force=False):
    _sep("PHASE 3  Weibull AFT Ablation")
    out = OUT / "phase3" / "phase3_ablation_v3.csv"

    if not force and _exists(out):
        logger.info("Phase 3 output exists — skipping.")
        return

    (OUT / "phase3").mkdir(parents=True, exist_ok=True)

    from lifelines import WeibullAFTFitter
    from lifelines.utils import k_fold_cross_validation
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from scipy.stats import chi2 as chi2_dist

    df = pd.read_csv(OUT / "phase2" / "customer_augmented_v3.csv")
    df["T"] = df["T"].clip(lower=0.5)
    logger.info(f"Input: {len(df):,} customers | churn={df['E'].mean()*100:.1f}%")

    def vif_check(df_sc, feats, threshold=10.0):
        while len(feats) > 1:
            X = df_sc[feats].fillna(0).values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vifs = []
                for i in range(X.shape[1]):
                    try:
                        v = variance_inflation_factor(X, i)
                        vifs.append(v if np.isfinite(v) else 1e9)
                    except Exception:
                        vifs.append(1e9)
            if max(vifs) <= threshold:
                break
            drop = feats[vifs.index(max(vifs))]
            feats.remove(drop)
        return feats

    RFM_FEATS = ["Recency", "Frequency", "Monetary", "SinglePurchase", "InterPurchaseTime", "GapDeviation"]
    
    VARIANTS = {
        "Model_A_RFM": RFM_FEATS,
        "Model_B_Latest": RFM_FEATS + ["sentiment_latest"],
        "Model_C_History": RFM_FEATS + ["sentiment_latest", "sentiment_mean", "sentiment_volatility", "negative_review_count"],
        "Model_D_Mixed": RFM_FEATS + ["sentiment_latest", "sentiment_mean", "sentiment_volatility", "negative_review_count", "mixed_flag"],
        "Model_E_LoyalXFail": RFM_FEATS + ["sentiment_latest", "sentiment_mean", "sentiment_volatility", "negative_review_rate", "mixed_flag", "loyal_customer_flag", "latest_fail_severity", "loyal_x_failure"],
        "Model_F_All": RFM_FEATS + ["sentiment_latest", "sentiment_mean", "sentiment_volatility", "negative_review_rate", "mixed_flag", "loyal_customer_flag", "latest_fail_severity", "loyal_x_failure", "loyal_x_fail_flag", "loyal_x_missing_sentiment", "days_since_last_neg", "recent_negative_count_90d", "loyal_x_sentiment_shock", "loyal_x_failure_after_positive_history"]
    }
    
    # Temporal Split (Cohort-based Option B - Strict Date Boundaries)
    df["first_purchase_date"] = pd.to_datetime(df["first_purchase_date"])
    
    dates = np.sort(df["first_purchase_date"].dt.date.unique())
    idx_val = int(len(dates) * 0.6)
    idx_test = int(len(dates) * 0.8)
    
    val_start_date = pd.to_datetime(dates[idx_val])
    test_start_date = pd.to_datetime(dates[idx_test])
    
    train_df = df[df["first_purchase_date"] < val_start_date].copy()
    val_df   = df[(df["first_purchase_date"] >= val_start_date) & (df["first_purchase_date"] < test_start_date)].copy()
    test_df  = df[df["first_purchase_date"] >= test_start_date].copy()
    
    train_max_date = train_df["first_purchase_date"].max()
    val_min_date = val_df["first_purchase_date"].min()
    val_max_date = val_df["first_purchase_date"].max()
    test_min_date = test_df["first_purchase_date"].min()
    
    assert train_max_date < val_min_date, f"Overlap: train_max {train_max_date} >= val_min {val_min_date}"
    assert val_max_date < test_min_date, f"Overlap: val_max {val_max_date} >= test_min {test_min_date}"
    
    logger.info(f"Temporal Split (Cohort): Train up to {train_max_date} ({len(train_df)}), "
                f"Val from {val_min_date} to {val_max_date} ({len(val_df)}), "
                f"Test from {test_min_date} ({len(test_df)})")
    
    with open(OUT / "phase3" / "split_info.json", "w") as f:
        json.dump({
            "train_max_date": str(train_max_date),
            "val_min_date": str(val_min_date),
            "val_max_date": str(val_max_date),
            "test_min_date": str(test_min_date),
            "train_snapshots": len(train_df),
            "val_snapshots": len(val_df),
            "test_snapshots": len(test_df)
        }, f, indent=2)
    
    # Check if there are events in test set
    if test_df["E"].sum() < 5:
        logger.warning("Very few events in test set! C-index might be unstable.")

    results = {}

    for variant, features in VARIANTS.items():
        logger.info(f"\n  -- {variant} --")
        cols = [c for c in features if c in df.columns]
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc",  StandardScaler())])
                         
        # Fit scaler on train, transform all
        pipe.fit(train_df[cols])
        df_train_sc = pd.DataFrame(pipe.transform(train_df[cols]), columns=cols, index=train_df.index)
        df_val_sc   = pd.DataFrame(pipe.transform(val_df[cols]), columns=cols, index=val_df.index)
        df_test_sc  = pd.DataFrame(pipe.transform(test_df[cols]), columns=cols, index=test_df.index)
        
        df_train_sc["T"] = train_df["T"].values
        df_train_sc["E"] = train_df["E"].values
        df_val_sc["T"]   = val_df["T"].values
        df_val_sc["E"]   = val_df["E"].values
        df_test_sc["T"]  = test_df["T"].values
        df_test_sc["E"]  = test_df["E"].values

        active = vif_check(df_train_sc, list(cols))
        logger.info(f"  Features after VIF: {len(active)}  {active}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waf = WeibullAFTFitter(penalizer=0.01)
            waf.fit(df_train_sc[active + ["T", "E"]], duration_col="T", event_col="E")
            
        c_train = waf.concordance_index_
        c_test = waf.score(df_test_sc[active + ["T", "E"]], scoring_method="concordance_index")
        
        # Interpret interaction term
        if "loyal_x_failure" in active:
            coef = waf.params_.loc["lambda_", "loyal_x_failure"]
            af = np.exp(coef)
            logger.info(f"  [Weibull Interpret] loyal_x_failure Acceleration Factor (Time Ratio) = {af:.4f} (coef={coef:.4f})")
            if af < 1.0:
                logger.info(f"    -> Meaning: The interaction significantly SHORTENS expected survival time.")
        
        from sklearn.metrics import brier_score_loss
        from sklearn.linear_model import LogisticRegression
        briers = {}
        
        def macg(y_true, y_pred, n_bins=10):
            df_macg = pd.DataFrame({'y': y_true, 'p': y_pred})
            df_macg['bin'] = pd.qcut(df_macg['p'], n_bins, labels=False, duplicates='drop')
            grp = df_macg.groupby('bin')
            return np.abs(grp['y'].mean() - grp['p'].mean()).mean()
            
        def make_horizon_eval_set(df_in, horizon):
            pos = (df_in["E"] == 1) & (df_in["T"] <= horizon)
            neg = df_in["T"] > horizon
            known = pos | neg
            out = df_in.loc[known].copy()
            out["y_horizon"] = pos.loc[known].astype(int)
            return out
            
        def logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return np.log(p / (1 - p))
            
        for d in [90, 180, 365, 1095]:
            # Validation set
            val_known = make_horizon_eval_set(df_val_sc, d)
            val_surv_prob = waf.predict_survival_function(val_known, times=[d]).iloc[0].values
            val_churn_prob = 1.0 - val_surv_prob
            val_actual_churn = val_known["y_horizon"].values
            
            # Predict on test
            test_known = make_horizon_eval_set(df_test_sc, d)
            test_surv_prob = waf.predict_survival_function(test_known, times=[d]).iloc[0].values
            test_churn_prob = 1.0 - test_surv_prob
            test_actual_churn = test_known["y_horizon"].values
            
            # Fit calibrators on validation
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(val_churn_prob, val_actual_churn)
            iso_pred = iso.predict(test_churn_prob)
            
            platt = LogisticRegression(max_iter=1000)
            try:
                platt.fit(logit(val_churn_prob).reshape(-1, 1), val_actual_churn)
                platt_pred = platt.predict_proba(logit(test_churn_prob).reshape(-1, 1))[:, 1]
            except ValueError:
                platt_pred = test_churn_prob.copy()
            
            macg_iso = macg(test_actual_churn, iso_pred)
            macg_platt = macg(test_actual_churn, platt_pred)
            
            if macg_platt < macg_iso:
                calibrated_churn_prob = platt_pred
                calibrator = platt
                calib_type = "platt"
            else:
                calibrated_churn_prob = iso_pred
                calibrator = iso
                calib_type = "iso"
                
            briers[f"Brier_Raw@{d}d"] = round(brier_score_loss(test_actual_churn, test_churn_prob), 4)
            briers[f"Brier_Cal@{d}d"] = round(brier_score_loss(test_actual_churn, calibrated_churn_prob), 4)
            briers[f"MACG_Raw@{d}d"] = round(macg(test_actual_churn, test_churn_prob), 4)
            briers[f"MACG_Cal@{d}d"] = round(macg(test_actual_churn, calibrated_churn_prob), 4)
            
            if d == 1095 and variant == "Model_F_All":
                
                # Predict on full dataset for Phase 4 & 5
                df_sc_full = pd.DataFrame(pipe.transform(df[cols]), columns=cols, index=df.index)
                full_surv = waf.predict_survival_function(df_sc_full, times=[d]).iloc[0].values
                full_raw_churn = 1.0 - full_surv
                if calib_type == "platt":
                    try:
                        full_calib_churn = calibrator.predict_proba(logit(full_raw_churn).reshape(-1, 1))[:, 1]
                    except Exception:
                        full_calib_churn = full_raw_churn.copy()
                else:
                    full_calib_churn = calibrator.predict(full_raw_churn)
                
                preds_df = pd.DataFrame({
                    "CustomerID": df["CustomerID"],
                    "raw_churn_prob_1095d": full_raw_churn,
                    "calib_churn_prob_1095d": full_calib_churn
                })
                preds_df.to_csv(OUT / "phase3" / "phase3_predictions.csv", index=False)

        try:
            from evaluation.core import compute_integrated_brier_score
            ibs = round(float(compute_integrated_brier_score(waf, df_test_sc[active+["T","E"]])), 4)
        except Exception:
            ibs = None

        r = {
            "n_features": len(active), "features": active,
            "c_train": round(float(c_train), 4),
            "c_oos":   round(float(c_test), 4),
            "log_ll":  round(float(waf.log_likelihood_), 2),
            "ibs":     ibs,
            **briers
        }
        results[variant] = r
        logger.info(f"  C-OOS={r['c_oos']}  logLL={r['log_ll']}  {briers}")

        # Interpretability Check with CoxPH
        if "loyal_x_failure" in active and variant == "Model_F_All":
            from lifelines import CoxPHFitter
            logger.info("  Running CoxPH solely for Hazard Ratio Interpretation...")
            cph = CoxPHFitter(penalizer=0.05)
            cph.fit(df_train_sc[active + ["T", "E"]], duration_col="T", event_col="E")
            
            summary = cph.summary
            for feat_name in ["loyal_customer_flag", "loyal_x_failure", "loyal_x_fail_flag", "loyal_x_missing_sentiment", "loyal_x_sentiment_shock", "loyal_x_failure_after_positive_history"]:
                if feat_name in summary.index:
                    hr = summary.loc[feat_name, "exp(coef)"]
                    p_val = summary.loc[feat_name, "p"]
                    ci_lower = summary.loc[feat_name, "exp(coef) lower 95%"]
                    ci_upper = summary.loc[feat_name, "exp(coef) upper 95%"]
                    logger.info(f"  [CoxPH Interpret] {feat_name}: HR = {hr:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]), p = {p_val:.4f}")
            
            # Save CoxPH summary to CSV for output
            summary.to_csv(OUT / "phase3" / "coxph_summary.csv")
                
            # Assumption check
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    cph.check_assumptions(df_train_sc[active + ["T", "E"]], p_value_threshold=0.05, show_plots=False)
                except Exception as e:
                    logger.info(f"  [CoxPH] Proportional Hazards Assumption Note: {e}")

    with open(OUT / "phase3" / "phase3_results_v4.json", "w") as f:
        json.dump({"results": results}, f, indent=2, default=str)

    pd.DataFrame([{"version": v, **{k: val for k, val in r.items() if k != "features"}}
                  for v, r in results.items()]).to_csv(out, index=False)
    logger.info(f"Saved -> {OUT}/phase3/")



