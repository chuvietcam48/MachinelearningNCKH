"""
tests/test_integration.py
==========================
Integration test: synthetic data → full pipeline mini-run.

Tests the critical path:
  make_transactions_df
    → build_customer_features
    → train_weibull_aft (small, fast)
    → make_intervention_decisions
    → compute_outreach_efficiency

Validates cross-module schema contracts without running the heavyweight
CLI (no SHAP, no MLflow, no file I/O).
"""
import unittest
import numpy as np
import pandas as pd
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")  # suppress lifelines convergence warnings in tests

from tests.conftest import make_transactions_df, make_snapshot
from src.feature_engine import build_customer_features
from src.models import train_weibull_aft, rfm_segment, get_survival_features
from src.policy import make_intervention_decisions, rfm_intervention_decisions
from src.evaluation import compute_outreach_efficiency, compute_revenue_lift


class TestMinimalEndToEnd(unittest.TestCase):
    """
    Runs a minimal but complete pipeline on synthetic data.
    Validates that all modules produce the expected schemas and types.
    """

    @classmethod
    def setUpClass(cls):
        """Run pipeline once; reuse results across all test methods."""
        snap = make_snapshot()
        df_tx = make_transactions_df(n_customers=60, n_rows=300, seed=7)
        cls.customer_df = build_customer_features(df_tx, snap, tau=90)
        cls.rfm_df = rfm_segment(cls.customer_df)

        # Train Weibull (minimal CV for speed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.waf, cls.df_scaled, cls.preprocessor, cls.active_features = \
                train_weibull_aft(cls.customer_df, penalizer_grid=[0.1])

        cls.weibull_decisions = make_intervention_decisions(
            cls.waf, cls.df_scaled, cls.customer_df
        )
        cls.rfm_decisions = rfm_intervention_decisions(cls.rfm_df)

    # ── Feature Engineering ──────────────────────────────────────────────────

    def test_customer_df_required_columns(self):
        """customer_df must have all required feature + target columns."""
        for col in ("Recency", "Frequency", "Monetary",
                    "InterPurchaseTime", "GapDeviation", "SinglePurchase", "T", "E"):
            self.assertIn(col, self.customer_df.columns,
                          f"Missing column: {col}")

    def test_customer_df_no_negative_T(self):
        """Survival time T must be > 0 for all customers."""
        self.assertTrue((self.customer_df["T"] > 0).all(),
                        "Found T <= 0 in customer_df")

    def test_event_col_binary(self):
        """E must only contain 0 and 1."""
        unique_vals = set(self.customer_df["E"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}),
                        f"E contains unexpected values: {unique_vals}")

    # ── Auto-discover features ───────────────────────────────────────────────

    def test_auto_discovered_features_subset_of_df(self):
        """All auto-discovered features must be real columns in customer_df."""
        features = get_survival_features(self.customer_df)
        for f in features:
            self.assertIn(f, self.customer_df.columns)

    # ── RFM Segmentation ────────────────────────────────────────────────────

    def test_rfm_df_valid_segments(self):
        """RFM segments must be valid categories."""
        valid = {"Champions", "Loyal", "At Risk", "Lost"}
        actual = set(self.rfm_df["RFM_Segment"].unique())
        self.assertTrue(actual.issubset(valid))

    # ── Weibull Training ────────────────────────────────────────────────────

    def test_waf_has_concordance_index(self):
        """Fitted Weibull model must have concordance_index_ attribute."""
        self.assertTrue(hasattr(self.waf, "concordance_index_"))

    def test_c_index_reasonable(self):
        """C-index must be in a reasonable range [0.4, 1.0]."""
        c = self.waf.concordance_index_
        self.assertGreaterEqual(c, 0.4)
        self.assertLessEqual(c, 1.0)

    def test_active_features_nonempty(self):
        """VIF check must leave at least 1 active feature."""
        self.assertGreater(len(self.active_features), 0)

    # ── Intervention Decisions ───────────────────────────────────────────────

    def test_decision_output_columns(self):
        """Decision table must have all required columns."""
        for col in ("CustomerID", "hazard_now", "survival", "evi", "decision"):
            self.assertIn(col, self.weibull_decisions.columns)

    def test_decision_values_valid(self):
        """All decisions must be INTERVENE, WAIT, or LOST."""
        valid = {"INTERVENE", "WAIT", "LOST"}
        actual = set(self.weibull_decisions["decision"].unique())
        self.assertTrue(actual.issubset(valid),
                        f"Unexpected decisions: {actual - valid}")

    def test_same_n_customers_in_decisions(self):
        """Decision table must have same number of rows as customer_df."""
        self.assertEqual(len(self.weibull_decisions), len(self.customer_df))

    # ── Evaluation ───────────────────────────────────────────────────────────

    def test_outreach_efficiency_all_keys(self):
        """compute_outreach_efficiency must return all expected keys."""
        metrics = compute_outreach_efficiency(self.weibull_decisions, self.rfm_decisions)
        for key in ("weibull_intervene_rate", "rfm_intervene_rate",
                    "contacts_avoided", "contacts_avoided_pct", "efficiency_gain_pct"):
            self.assertIn(key, metrics)

    def test_revenue_lift_all_keys(self):
        """compute_revenue_lift must return all expected keys."""
        metrics = compute_revenue_lift(self.weibull_decisions, self.rfm_decisions)
        for key in ("avg_evi_weibull", "avg_evi_rfm_proxy",
                    "total_evi_weibull", "total_evi_rfm_proxy",
                    "revenue_precision_lift_pct"):
            self.assertIn(key, metrics)

    def test_survival_values_in_0_1(self):
        """All survival values must be in [0, 1]."""
        s = self.weibull_decisions["survival"]
        self.assertTrue((s >= 0).all() and (s <= 1).all())



class TestX5UpliftLabelPropagation(unittest.TestCase):
    """
    Validates that build_customer_features() correctly propagates X5 RCT labels
    (treatment_flg, target_flag) from transaction-level data to the customer_df.

    These columns must survive aggregation so uplift.py can switch to
    real-treatment mode and prove customer segment classification.
    """

    @classmethod
    def setUpClass(cls):
        import warnings
        from tests.conftest import make_transactions_df, make_snapshot
        from src.feature_engine import build_customer_features
        warnings.filterwarnings("ignore")

        snap = make_snapshot()
        df_tx = make_transactions_df(n_customers=40, n_rows=200, seed=77)

        # Simulate X5: assign a treatment_flg and target_flag per customer
        # (one value per customer — same across all their rows)
        rng = np.random.default_rng(77)
        unique_custs = df_tx["CustomerID"].unique()
        treat_map  = {c: int(rng.integers(0, 2)) for c in unique_custs}
        target_map = {c: int(rng.integers(0, 2)) for c in unique_custs}

        df_tx["treatment_flg"] = df_tx["CustomerID"].map(treat_map)
        df_tx["target_flag"]   = df_tx["CustomerID"].map(target_map)

        cls.df_tx      = df_tx
        cls.snap       = snap
        cls.treat_map  = treat_map
        cls.target_map = target_map
        cls.customer_df = build_customer_features(df_tx, snap, tau=90)

    def test_treatment_flg_present(self):
        """treatment_flg must be present in customer_df after propagation."""
        self.assertIn("treatment_flg", self.customer_df.columns,
                      "treatment_flg not propagated to customer_df")

    def test_target_flag_present(self):
        """target_flag must be present in customer_df after propagation."""
        self.assertIn("target_flag", self.customer_df.columns,
                      "target_flag not propagated to customer_df")

    def test_treatment_flg_values_binary(self):
        """treatment_flg must only contain 0 and 1."""
        vals = set(self.customer_df["treatment_flg"].unique())
        self.assertTrue(vals.issubset({0, 1}), f"Non-binary values: {vals}")

    def test_treatment_flg_values_match_source(self):
        """treatment_flg in customer_df must match the per-customer assignment."""
        # Reset index so CustomerID is a column
        cdf = self.customer_df.reset_index()
        for _, row in cdf.iterrows():
            expected = self.treat_map[row["CustomerID"]]
            self.assertEqual(row["treatment_flg"], expected,
                             f"Mismatch for {row['CustomerID']}: "
                             f"got {row['treatment_flg']}, expected {expected}")

    def test_standard_columns_still_present(self):
        """Core RFM + survival columns must still be present after RCT propagation."""
        for col in ("Recency", "Frequency", "Monetary",
                    "InterPurchaseTime", "GapDeviation", "SinglePurchase", "T", "E"):
            self.assertIn(col, self.customer_df.columns)

    def test_no_rct_cols_for_plain_transactions(self):
        """Without treatment_flg/target_flag in raw df, customer_df must NOT have them."""
        from tests.conftest import make_transactions_df, make_snapshot
        from src.feature_engine import build_customer_features
        df_plain = make_transactions_df(n_customers=20, n_rows=100, seed=1)
        snap = make_snapshot()
        cdf = build_customer_features(df_plain, snap, tau=90)
        self.assertNotIn("treatment_flg", cdf.columns)
        self.assertNotIn("target_flag", cdf.columns)


if __name__ == "__main__":
    unittest.main()
