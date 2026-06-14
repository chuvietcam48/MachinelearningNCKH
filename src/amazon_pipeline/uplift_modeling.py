import os
import sys
import warnings
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.amazon_pipeline.config import logger, TAU, SNAP_DATE, WINDOW_YEARS, MIN_FREQ, RAW_REVIEWS, RAW_META, OUT, BEHAVIORAL, SENTIMENT, _sep, _exists



def _fit_x_learner(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    IPTW-corrected X-Learner Proxy Policy Estimation.
    
    DISCLAIMER: Because Amazon review data does not contain randomized 
    voucher assignment, this module should be interpreted strictly as a 
    policy simulation / prioritization layer, not a causal estimate of 
    true voucher treatment effect.

    Improvement over T-Learner: cross-imputes simulation effects between groups,
    reducing bias in unbalanced pseudo-treatment settings.

    Algorithm:
      1. Fit mu_1(x) on treated, mu_0(x) on control (IPTW-weighted)
      2. Imputed proxy effects: d_1 = Y - mu_0(x) for treated
                                d_0 = mu_1(x) - Y for control
      3. Fit tau_1(x) on treated predicting d_1
         Fit tau_0(x) on control  predicting d_0
      4. Combined: tau(x) = g(x)*tau_0(x) + (1-g(x))*tau_1(x)
                   where g(x) = propensity score
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    treatment = df["treatment"].values.astype(int)
    y         = df["Monetary"].values.astype(float)
    X_raw     = df[feature_cols].values

    imp    = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(imp.fit_transform(X_raw))
    T      = treatment.astype(bool)

    if T.sum() < 10 or (~T).sum() < 10:
        logger.warning("[X-Learner] Too few treated or control samples — falling back to zeros.")
        df = df.copy()
        df["tau_hat"] = 0.0
        df.attrs["response_threshold_1"] = 0.0
        df.attrs["response_threshold_0"] = 0.0
        return df

    # Propensity scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ps = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        ps.fit(X_sc, treatment)
    propensity = ps.predict_proba(X_sc)[:, 1].clip(0.01, 0.99)
    logger.info(f"[X-Learner] Propensity: mean={propensity.mean():.3f}  "
                f"min={propensity.min():.3f}  max={propensity.max():.3f}")

    # IPTW weights (stabilized + winsorized)
    p_t  = treatment.mean()
    iptw = np.where(T, p_t / propensity, (1 - p_t) / (1 - propensity))
    iptw = np.clip(iptw, 0, np.percentile(iptw, 99))

    def _gbr():
        return GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                          learning_rate=0.1, random_state=42)

    # Step 1: Outcome models
    mu_1, mu_0 = _gbr(), _gbr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu_1.fit(X_sc[T],  y[T],  sample_weight=iptw[T])
        mu_0.fit(X_sc[~T], y[~T], sample_weight=iptw[~T])

    # Step 2: Imputed effects
    d_1 = y[T]  - mu_0.predict(X_sc[T])    # treated: actual - counterfactual(no treat)
    d_0 = mu_1.predict(X_sc[~T]) - y[~T]   # control: counterfactual(treat) - actual

    # Step 3: CATE models
    tau_1, tau_0 = _gbr(), _gbr()
    tau_1.fit(X_sc[T],  d_1)
    tau_0.fit(X_sc[~T], d_0)

    # Step 4: Combine
    tau_hat = propensity * tau_0.predict(X_sc) + (1 - propensity) * tau_1.predict(X_sc)

    df = df.copy()
    df["tau_hat"]    = tau_hat
    df["mu_1"]       = mu_1.predict(X_sc)
    df["mu_0"]       = mu_0.predict(X_sc)
    df["propensity"] = propensity
    
    is_binary_y = set(np.unique(y)).issubset({0, 1, 0.0, 1.0})
    df.attrs["is_binary_y"] = is_binary_y
    if is_binary_y:
        df.attrs["response_threshold_1"] = 0.5
        df.attrs["response_threshold_0"] = 0.5
        df.attrs["tau_thr"] = 0.0
    else:
        df.attrs["response_threshold_1"] = float(np.median(df["mu_1"]))
        df.attrs["response_threshold_0"] = float(np.median(df["mu_0"]))
        tau_thr = float(np.percentile(tau_hat, 75))
        df.attrs["tau_thr"] = tau_thr if tau_thr > 0 else 0.0001
    logger.info(f"[X-Learner] tau_hat: mean={tau_hat.mean():.3f}  "
                f"std={tau_hat.std():.3f}  pos={( tau_hat>0).mean()*100:.1f}%")
    return df





def run_phase4(force=False, run_xlearner=False):
    _sep("PHASE 4  Proxy Policy Simulation (T-Learner + X-Learner)")
    logger.info("DISCLAIMER: This module generates an Intervention Prioritization Score "
                "based on observational data. It is a simulated EVI policy, "
                "not a true causal treatment effect.")
    
    out1 = OUT / "phase4" / "uplift_df_T1_behavioral.csv"
    out2 = OUT / "phase4" / "uplift_df_T2_+sentiment.csv"
    out3 = OUT / "phase4" / "uplift_df_X1_behavioral.csv"
    out4 = OUT / "phase4" / "uplift_df_X2_+sentiment.csv"

    t_done = _exists(out1, out2)
    x_done = _exists(out3, out4)

    if not force and t_done and (not run_xlearner or x_done):
        logger.info("Phase 4 outputs exist — skipping.")
        return

    (OUT / "phase4").mkdir(parents=True, exist_ok=True)

    from lifelines import WeibullAFTFitter
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from uplift import _fit_t_learner, _compute_qini, _assign_uplift_segment, _RESPONSE_THR, _trapz

    df = pd.read_csv(OUT / "phase2" / "customer_augmented_v3.csv")
    df = df.drop(columns=["last_purchase_date", "first_purchase_date",
                           "dominant_complaint"], errors="ignore")
    df["T"] = df["T"].clip(lower=0.5)
    if "CustomerID" not in df.columns:
        df["CustomerID"] = np.arange(len(df))

    # Fit Weibull V2 for decision generation
    wf_cols = [c for c in BEHAVIORAL + SENTIMENT if c in df.columns]
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc",  StandardScaler())])
    df_sc = pd.DataFrame(pipe.fit_transform(df[wf_cols]),
                          columns=wf_cols, index=df.index)
    df_sc["T"] = df["T"].values
    df_sc["E"] = df["E"].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        waf = WeibullAFTFitter(penalizer=0.001)
        waf.fit(df_sc, duration_col="T", event_col="E")

    logger.info(f"Weibull V2 C-index={waf.concordance_index_:.4f}")

    median_T   = df["T"].median()
    surv       = waf.predict_survival_function(df_sc, times=[median_T]).iloc[0].values
    churn_prob = 1.0 - surv
    monetary   = df["Monetary"].fillna(0).values
    evi        = 0.15 * monetary * churn_prob - 1.0
    is_lost    = surv < np.percentile(surv, 5)

    decisions = np.where(is_lost, "LOST", np.where(evi > 0, "INTERVENE", "WAIT"))
    logger.info(f"Decisions: {pd.Series(decisions).value_counts().to_dict()}")

    dec_df = pd.DataFrame({
        "CustomerID": df["CustomerID"].values,
        "decision": decisions, "survival": surv, "evi": evi,
    })

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Define all learner variants: (name, feature_set, fitter_fn)
    T_VARIANTS = {
        "T1_behavioral": BEHAVIORAL,
        "T2_+sentiment": BEHAVIORAL + SENTIMENT,
    }
    X_VARIANTS = {
        "X1_behavioral": BEHAVIORAL,
        "X2_+sentiment": BEHAVIORAL + SENTIMENT,
    } if run_xlearner else {}

    all_qinis = {}

    def _run_variant(variant, feats, fitter, marker):
        avail = [c for c in feats if c in df.columns]
        right = list(dict.fromkeys(["CustomerID"] + avail + ["T", "E", "Monetary"]))
        m = dec_df.merge(df[right], on="CustomerID", how="left")
        # Resolve Monetary duplication
        mon_cols = [c for c in m.columns if "Monetary" in c and c != "Monetary"]
        if mon_cols:
            m["Monetary"] = m.get("Monetary", pd.Series(dtype=float)).fillna(0)
            m = m.drop(columns=mon_cols)
        m = m[m["decision"] != "LOST"].copy()
        m["treatment"] = (m["decision"] == "INTERVENE").astype(int)

        uplift_df = fitter(m, avail)
        theta_1   = uplift_df.attrs.get("response_threshold_1", _RESPONSE_THR)
        theta_0   = uplift_df.attrs.get("response_threshold_0", _RESPONSE_THR)
        is_binary = uplift_df.attrs.get("is_binary_y", False)
        tau_thr   = uplift_df.attrs.get("tau_thr", 0.0)
        uplift_df["uplift_segment"] = uplift_df.apply(
            _assign_uplift_segment, axis=1, theta_1=theta_1, theta_0=theta_0,
            is_binary_y=is_binary, tau_thr=tau_thr
        )

        qini_df = _compute_qini(uplift_df)
        qa  = _trapz(qini_df["qini_gain"],      qini_df["pct_targeted"])
        ra  = _trapz(qini_df["random_baseline"], qini_df["pct_targeted"])
        qc  = qa / ra if ra != 0 else 0.0

        int_df  = uplift_df[uplift_df["treatment"] == 1]
        persu   = (int_df["uplift_segment"] == "Persuadables").mean() * 100 if len(int_df) else 0
        tau_pos = (uplift_df["tau_hat"] > 0).mean() * 100
        logger.info(f"  [{marker}] Qini={qc:.4f}  Persuadables={persu:.1f}%  tau_pos={tau_pos:.1f}%")
        logger.info(f"  [{marker}] Segments: {uplift_df['uplift_segment'].value_counts().to_dict()}")

        if "sentiment_latest" in uplift_df.columns:
            uplift_df["sg"] = uplift_df["sentiment_latest"].apply(
                lambda s: "pos" if s > 0.05 else ("neg" if s < -0.05 else "neu")
            )
            tg = uplift_df.groupby("sg")["tau_hat"].mean().round(3)
            logger.info(f"  [{marker}] tau by sentiment: {tg.to_dict()}")

        if marker == "T" and "+sentiment" in variant:
            try:
                import shap
                mu_1 = uplift_df.attrs.get("mu_1_model")
                mu_0 = uplift_df.attrs.get("mu_0_model")
                scaler = uplift_df.attrs.get("scaler")
                imputer = uplift_df.attrs.get("imputer")
                if mu_1 and mu_0 and scaler and imputer:
                    logger.info(f"  [SHAP] Generating SHAP summary plot for {variant}...")
                    X_raw = m[avail].values
                    X_sc = scaler.transform(imputer.transform(X_raw))
                    explainer_1 = shap.TreeExplainer(mu_1)
                    explainer_0 = shap.TreeExplainer(mu_0)
                    shap_1 = explainer_1.shap_values(X_sc)
                    shap_0 = explainer_0.shap_values(X_sc)
                    shap_uplift = shap_1 - shap_0
                    
                    # Create plot
                    plt.figure(figsize=(8, 6))
                    shap.summary_plot(shap_uplift, pd.DataFrame(X_raw, columns=avail), show=False)
                    plt.title(f"Uplift SHAP Summary — {variant}")
                    plt.tight_layout()
                    plt.savefig(OUT / "phase4" / f"shap_summary_{variant}.png", dpi=150, bbox_inches="tight")
                    plt.close()
            except ImportError:
                logger.warning("  [SHAP] 'pip install shap' is required to generate SHAP plots.")
            except Exception as e:
                logger.warning(f"  [SHAP] Failed to generate SHAP plots: {e}")

        uplift_df.to_csv(OUT / "phase4" / f"uplift_df_{variant}.csv", index=False)

        # Qini plot
        color = "#3498db" if marker == "T" else "#e74c3c"
        _, ax = plt.subplots(figsize=(6, 4))
        ax.plot(qini_df["pct_targeted"]*100, qini_df["qini_gain"],
                color=color, lw=2, label=f"{marker}-Learner (Qini={qc:.3f})")
        ax.plot(qini_df["pct_targeted"]*100, qini_df["random_baseline"],
                "--", color="gray", lw=1.5, label="Random")
        ax.fill_between(qini_df["pct_targeted"]*100,
                        qini_df["random_baseline"], qini_df["qini_gain"],
                        alpha=0.15, color=color)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("% Customers Targeted")
        ax.set_ylabel("Qini Gain")
        ax.set_title(f"Qini — {variant}  (Qini={qc:.3f})")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT / "phase4" / f"qini_{variant}.png", dpi=150)
        plt.close()

        return qc

    logger.info("\n  === T-Learner variants ===")
    for v, feats in T_VARIANTS.items():
        logger.info(f"\n  -- {v} --")
        if not force and _exists(OUT / "phase4" / f"uplift_df_{v}.csv"):
            logger.info(f"  {v} output exists — skipping.")
            # Still read Qini from existing file
            tmp = pd.read_csv(OUT / "phase4" / f"uplift_df_{v}.csv")
            qdf = _compute_qini(tmp)
            qa  = _trapz(qdf["qini_gain"],      qdf["pct_targeted"])
            ra  = _trapz(qdf["random_baseline"], qdf["pct_targeted"])
            all_qinis[v] = qa/ra if ra != 0 else 0.0
        else:
            all_qinis[v] = _run_variant(v, feats, _fit_t_learner, "T")

    if run_xlearner:
        logger.info("\n  === X-Learner variants ===")
        for v, feats in X_VARIANTS.items():
            logger.info(f"\n  -- {v} --")
            all_qinis[v] = _run_variant(v, feats, _fit_x_learner, "X")

    # Summary comparison
    logger.info("\n  === PHASE 4 SUMMARY ===")
    for v, qc in all_qinis.items():
        logger.info(f"  {v:<25} Qini = {qc:+.4f}")

    if run_xlearner and "T1_behavioral" in all_qinis and "X1_behavioral" in all_qinis:
        delta = all_qinis["X1_behavioral"] - all_qinis["T1_behavioral"]
        logger.info(f"\n  X-Learner vs T-Learner (behavioral): {delta:+.4f}  "
                    f"({'improvement' if delta > 0 else 'decrease'})")
        if all_qinis["X1_behavioral"] < 0:
            logger.info("  [X-Learner Note] Negative Qini observed. This may be due to "
                        "continuous monetary outcome noise, treatment imbalance, or "
                        "learner unsuitability for this proxy setup.")
            
    if run_xlearner and "T2_+sentiment" in all_qinis and "X2_+sentiment" in all_qinis:
        delta = all_qinis["X2_+sentiment"] - all_qinis["T2_+sentiment"]
        logger.info(f"  X-Learner vs T-Learner (+sentiment): {delta:+.4f}  "
                    f"({'improvement' if delta > 0 else 'decrease'})")
        if all_qinis["X2_+sentiment"] < 0:
            logger.info("  [X-Learner Note] Negative Qini observed. This may be due to "
                        "continuous monetary outcome noise, treatment imbalance, or "
                        "learner unsuitability for this proxy setup.")

    # Save summary
    pd.DataFrame([{"variant": v, "qini": qc} for v, qc in all_qinis.items()]).to_csv(
        OUT / "phase4" / "phase4_qini_summary.csv", index=False
    )
    logger.info(f"Saved -> {OUT}/phase4/")



