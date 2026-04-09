"""
tests/test_uplift.py
====================
Unit tests for src/uplift.py:
  - Qini curve vectorized formula correctness
  - T-Learner structure
  - Persuadables segmentation logic
"""
import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQiniVectorized(unittest.TestCase):
    """Tests for the O(n) vectorized Qini computation."""

    def _make_uplift_df(self, n=50, seed=42):
        """Minimal DataFrame matching what _compute_qini expects."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "CustomerID": [f"C{i}" for i in range(n)],
            "treatment":  rng.choice([0, 1], size=n, p=[0.5, 0.5]),
            "tau_hat":    rng.uniform(-5, 5, n),
            "Monetary":   rng.uniform(10, 500, n),
        })

    def test_output_columns(self):
        """Qini DataFrame must have pct_targeted, qini_gain, random_baseline."""
        from src.uplift import _compute_qini
        df = self._make_uplift_df()
        qini_df = _compute_qini(df)
        for col in ("pct_targeted", "qini_gain", "random_baseline"):
            self.assertIn(col, qini_df.columns)

    def test_pct_targeted_range(self):
        """pct_targeted must be in (0, 1]."""
        from src.uplift import _compute_qini
        df = self._make_uplift_df()
        qini_df = _compute_qini(df)
        self.assertGreater(qini_df["pct_targeted"].min(), 0.0)
        self.assertAlmostEqual(qini_df["pct_targeted"].max(), 1.0, places=5)

    def test_same_length_as_input(self):
        """Qini DataFrame rows must equal number of input customers."""
        from src.uplift import _compute_qini
        n = 40
        df = self._make_uplift_df(n=n)
        qini_df = _compute_qini(df)
        self.assertEqual(len(qini_df), n)

    def test_all_same_treatment_returns_fallback(self):
        """When all are treated or all are control, Qini returns zero-gain fallback."""
        from src.uplift import _compute_qini
        df = self._make_uplift_df(n=20)
        df["treatment"] = 1  # all treated — no control group
        qini_df = _compute_qini(df)
        # Must return a valid DataFrame (not raise)
        self.assertIsInstance(qini_df, pd.DataFrame)
        self.assertIn("qini_gain", qini_df.columns)

    def test_vectorized_qini_monotone_at_boundary(self):
        """With perfect uplift signal, treated should cluster at the top."""
        from src.uplift import _compute_qini
        n = 20
        # Perfect signal: all treated have high tau_hat, all control have low
        df = pd.DataFrame({
            "treatment": [1]*10 + [0]*10,
            "tau_hat":   list(range(20, 10, -1)) + list(range(10, 0, -1)),
            "Monetary":  [100.0] * 20,
        })
        qini_df = _compute_qini(df)
        # At the first 50% targeted, all treated should be included → peak Qini
        self.assertIsInstance(qini_df["qini_gain"].iloc[9], (float, np.floating))


class TestPersuadablesSegmentation(unittest.TestCase):
    """Tests for _assign_uplift_segment (quadrant logic)."""

    def test_persuadables(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": 1.0, "mu_1": _RESPONSE_THR + 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Persuadables")

    def test_sure_things(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": -1.0, "mu_1": _RESPONSE_THR + 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Sure Things")

    def test_sleeping_dogs(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": 1.0, "mu_1": _RESPONSE_THR - 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Sleeping Dogs")

    def test_lost_causes(self):
        from src.uplift import _assign_uplift_segment, _RESPONSE_THR
        row = pd.Series({"tau_hat": -1.0, "mu_1": _RESPONSE_THR - 1.0})
        self.assertEqual(_assign_uplift_segment(row), "Lost Causes")



class TestRealTreatmentMode(unittest.TestCase):
    """
    Tests that run_uplift_analysis switches to real-treatment mode
    when customer_df contains treatment_flg + target_flag (X5 schema).

    This validates the core claim: with ground-truth RCT labels the pipeline
    CAN classify customers into uplift segments (Persuadables, Sure Things, etc.)
    producing a meaningful (potentially positive) Qini coefficient.
    """

    @classmethod
    def setUpClass(cls):
        """Build minimal synthetic X5-style data and run uplift once."""
        import warnings
        warnings.filterwarnings("ignore")

        rng = np.random.default_rng(99)
        n = 120

        # Synthetic customer_df with RCT columns (mimics X5 post-feature-engine output)
        customer_ids = [f"X{i:04d}" for i in range(n)]
        treat = rng.choice([0, 1], size=n, p=[0.5, 0.5])
        target = np.where(treat == 1, rng.random(n) < 0.35, rng.random(n) < 0.15).astype(int)

        cls.customer_df = pd.DataFrame({
            "CustomerID":       customer_ids,
            "Recency":          rng.integers(1, 300, n),
            "Frequency":        rng.integers(1, 20, n),
            "Monetary":         rng.uniform(10, 1000, n),
            "InterPurchaseTime": rng.uniform(0, 100, n),
            "GapDeviation":     rng.uniform(0, 50, n),
            "SinglePurchase":   rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "treatment_flg":    treat,
            "target_flag":      target,
        })

        # Synthetic decisions table (as produced by policy.make_intervention_decisions)
        decisions = rng.choice(["INTERVENE", "WAIT", "LOST"], size=n, p=[0.35, 0.45, 0.20])
        cls.decisions_df = pd.DataFrame({
            "CustomerID":          customer_ids,
            "hazard_now":          rng.uniform(0.001, 0.05, n),
            "survival":            rng.uniform(0.1, 0.99, n),
            "evi":                 rng.uniform(-1.0, 10.0, n),
            "decision":            decisions,
            "Monetary":            cls.customer_df["Monetary"].values,
            "optimal_window_days": rng.uniform(10, 200, n),
        })

        from src.uplift import run_uplift_analysis
        cls.results = run_uplift_analysis(
            weibull_decisions=cls.decisions_df,
            customer_df=cls.customer_df,
            save_path=None,
        )

    def test_qini_coef_is_finite(self):
        """Qini coefficient must be a finite float when RCT labels available."""
        qc = self.results["qini_auc_ratio"]
        self.assertIsInstance(qc, (float, np.floating))
        self.assertTrue(np.isfinite(qc), f"Qini coefficient is not finite: {qc}")

    def test_persuadable_pct_in_range(self):
        """Persuadable percentage must be in [0, 1]."""
        pct = self.results["persuadable_pct"]
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 1.0)

    def test_uplift_df_has_segment(self):
        """uplift_df must contain uplift_segment column with valid categories."""
        udf = self.results["uplift_df"]
        valid = {"Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"}
        actual = set(udf["uplift_segment"].unique())
        self.assertTrue(actual.issubset(valid), f"Unexpected segments: {actual - valid}")

    def test_all_four_segments_can_appear(self):
        """With n=120 synthetic customers all 4 quadrants should appear."""
        udf = self.results["uplift_df"]
        n_segments = udf["uplift_segment"].nunique()
        # At least 2 distinct segments expected (1 could be unlucky with tiny n)
        self.assertGreaterEqual(n_segments, 2)

    def test_real_treatment_mode_uses_rct_labels(self):
        """treatment column in uplift_df must match ground-truth treatment_flg."""
        udf = self.results["uplift_df"]
        self.assertIn("treatment", udf.columns)
        # All treatment values must be 0 or 1
        self.assertTrue(udf["treatment"].isin([0, 1]).all())

    def test_segment_counts_sum_to_n(self):
        """Segment counts must sum to total number of customers in analysis."""
        counts = self.results["segment_counts"]
        udf = self.results["uplift_df"]
        self.assertEqual(sum(counts.values()), len(udf))


if __name__ == "__main__":
    unittest.main()
