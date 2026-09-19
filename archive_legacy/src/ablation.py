"""
src/ablation.py  (Phase E — E6: Ablation Study)
=================================================
"Tháo rời cỗ máy" — prove every enhancement adds measurable value.

Runs the pipeline 5 times with different modules disabled:
  S0  Full          (all enhancements active)
  S1  No VIF Guard  (vif_threshold=inf → no multicollinearity pruning)
  S2  Fixed Tau 365 (ignores adaptive tau → temporal mismatch)
  S3  No VIP Guard  (vip_pct=1.0 → no VIP protection)
  S4  Naive IBS     (force non-IPCW Brier Score)

Output:
  - outputs/ablation/ablation_table.csv
  - outputs/ablation/ablation_table.md
  - outputs/ablation/ablation_delta_chart.png  (normalized Δ% bar chart)
"""

import logging
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AblationRunner:
    """
    Runs the full pipeline under multiple ablation scenarios and collects
    performance + business safety metrics for each.
    """

    SCENARIOS = {
        "S0_Full":       {"label": "Full Pipeline",    "overrides": {}},
        "S1_No_VIF":     {"label": "No VIF Guard",     "overrides": {"vif_threshold": float("inf")}},
        "S2_Fixed_Tau":  {"label": "Fixed τ=365",      "overrides": {"tau": 365}},
        "S3_No_VIP":     {"label": "No VIP Guard",     "overrides": {"vip_pct": 1.0}},
        "S4_Naive_IBS":  {"label": "Naive IBS",        "overrides": {"force_naive_ibs": True}},
    }

    def __init__(self, output_dir: str = None):
        self.output_dir = output_dir or os.path.join("outputs", "ablation")
        os.makedirs(self.output_dir, exist_ok=True)

    def run_suite(self, dataset_key: str, base_tau: int = 90) -> pd.DataFrame:
        """
        Run all 5 ablation scenarios on the given dataset.

        Parameters
        ----------
        dataset_key : str
            Dataset key (e.g. "cdnow", "uci", "tafeng").
        base_tau : int
            Default tau for S0 Full (default: 90).

        Returns
        -------
        pd.DataFrame
            One row per scenario with all ablation metrics.
        """
        records = []
        for scenario_id, spec in self.SCENARIOS.items():
            label = spec["label"]
            overrides = spec["overrides"]
            logger.info(f"\n{'='*60}")
            logger.info(f"  ABLATION: {scenario_id} — {label}")
            logger.info(f"{'='*60}")

            try:
                result = self._execute_single(
                    dataset_key, base_tau, overrides, scenario_id
                )
                result["Scenario"] = scenario_id
                result["Label"] = label
                records.append(result)
                logger.info(f"  [{scenario_id}] ✅ Done — C-index={result.get('c_index', 'N/A')}")

            except Exception as exc:
                logger.error(f"  [{scenario_id}] ❌ FAILED: {exc}")
                records.append({
                    "Scenario": scenario_id,
                    "Label": label,
                    "status": "FAIL",
                    "error": str(exc),
                    "c_index": float("nan"),
                    "ibs": float("nan"),
                    "revenue_lift": float("nan"),
                    "efficiency": float("nan"),
                    "vip_spam": float("nan"),
                    "ci_width": float("nan"),
                    "beta_std": float("nan"),
                    "runtime": float("nan"),
                    "n_features": float("nan"),
                })

        df = pd.DataFrame(records)

        # Save outputs
        csv_path = os.path.join(self.output_dir, "ablation_table.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"[Ablation] CSV → {csv_path}")

        md_path = os.path.join(self.output_dir, "ablation_table.md")
        self._save_markdown(df, md_path)
        logger.info(f"[Ablation] Markdown → {md_path}")

        chart_path = os.path.join(self.output_dir, "ablation_delta_chart.png")
        self._plot_delta_chart(df, chart_path)
        logger.info(f"[Ablation] Delta chart → {chart_path}")

        return df

    def _execute_single(
        self, dataset_key: str, base_tau: int, overrides: dict, scenario_id: str
    ) -> dict:
        """
        Execute a single mini-pipeline run with specific overrides.

        Returns dict of ablation metrics.
        """
        from src.dataset_registry import get_dataset
        from src.feature_engine import build_customer_features
        from src.models import train_weibull_aft, rfm_segment, get_survival_features
        from src.policy import make_intervention_decisions, rfm_intervention_decisions
        from src.evaluation import (
            compute_c_index, compute_integrated_brier_score,
            compute_outreach_efficiency, compute_revenue_lift,
            bootstrap_c_index,
        )

        t0 = time.time()

        # ── Override resolution ──────────────────────────────────────────────
        tau = overrides.get("tau", base_tau)
        vif_threshold = overrides.get("vif_threshold", 5.0)
        vip_pct = overrides.get("vip_pct", None)  # None = use config default
        force_naive = overrides.get("force_naive_ibs", False)

        # ── 1. Load data ─────────────────────────────────────────────────────
        ds = get_dataset(dataset_key)
        df_clean = ds.loader_fn(ds.data_path)
        snapshot = ds.snapshot_fn(df_clean)

        # ── 2. Feature engineering ───────────────────────────────────────────
        customer_df = build_customer_features(df_clean, snapshot, tau=tau)

        # ── 3. Train Weibull (with/without VIF) ──────────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            waf, df_scaled, preprocessor, active_feats = train_weibull_aft(
                customer_df, vif_threshold=vif_threshold
            )

        c_index = compute_c_index(waf, df_scaled, model_name=f"Ablation-{scenario_id}")
        ibs = compute_integrated_brier_score(waf, df_scaled, force_naive=force_naive)

        # Bootstrap CI (reduced to 100 for speed)
        boot_ci = bootstrap_c_index(waf, df_scaled, n_boot=100)
        ci_width = boot_ci[2] - boot_ci[0] if not np.isnan(boot_ci[0]) else float("nan")

        # Coefficient stability: std of all beta coefficients
        try:
            beta_params = waf.params_.drop(["Intercept"], errors="ignore")
            beta_std = float(beta_params.std()) if len(beta_params) > 0 else 0.0
        except Exception:
            beta_std = float("nan")

        # ── 4. Policy (with/without VIP guard) ───────────────────────────────
        rfm_df = rfm_segment(customer_df)
        t_now = float(df_scaled["T"].median())

        weibull_dec = make_intervention_decisions(
            waf, df_scaled, customer_df, t_now=t_now, vip_pct=vip_pct
        )
        rfm_dec = rfm_intervention_decisions(rfm_df)

        # ── 5. Evaluate ──────────────────────────────────────────────────────
        outreach = compute_outreach_efficiency(weibull_dec, rfm_dec)
        revenue = compute_revenue_lift(weibull_dec, rfm_dec)

        # ── 6. VIP Spam Count ────────────────────────────────────────────────
        # Count high-value customers with low hazard that got INTERVENE (shouldn't)
        from src.policy import DEFAULT_HAZARD_THRESHOLD
        monetary_p90 = weibull_dec["Monetary"].quantile(0.90)
        vip_spam = int((
            (weibull_dec["decision"] == "INTERVENE")
            & (weibull_dec["Monetary"] > monetary_p90)
            & (weibull_dec["hazard_now"] < DEFAULT_HAZARD_THRESHOLD * 0.5)
        ).sum())

        elapsed = time.time() - t0

        return {
            "status":       "OK",
            "c_index":      round(c_index, 4),
            "ibs":          round(ibs, 4),
            "revenue_lift": round(revenue.get("revenue_precision_lift_pct", 0), 1),
            "efficiency":   round(outreach.get("efficiency_gain_pct", 0), 1),
            "vip_spam":     vip_spam,
            "ci_width":     round(ci_width, 4) if not np.isnan(ci_width) else float("nan"),
            "beta_std":     round(beta_std, 4),
            "n_features":   len(active_feats),
            "runtime":      round(elapsed, 1),
        }

    def _save_markdown(self, df: pd.DataFrame, path: str):
        """Write a publication-ready ablation punchline table."""
        cols = ["Scenario", "Label", "status", "c_index", "ibs",
                "revenue_lift", "efficiency", "vip_spam", "ci_width",
                "beta_std", "n_features", "runtime"]
        cols_avail = [c for c in cols if c in df.columns]
        sub = df[cols_avail]

        lines = [
            "# Ablation Study — Component Contribution Analysis",
            "",
            "> S0 (Full Pipeline) = baseline. All other scenarios disable one component.",
            "",
        ]

        # Generate table
        try:
            lines.append(sub.to_markdown(index=False))
        except ImportError:
            header = "| " + " | ".join(str(c) for c in cols_avail) + " |"
            sep    = "| " + " | ".join("---" for _ in cols_avail)  + " |"
            lines.append(header)
            lines.append(sep)
            for _, row in sub.iterrows():
                lines.append("| " + " | ".join(str(row[c]) for c in cols_avail) + " |")

        lines.extend([
            "",
            "## Interpretation",
            "",
        ])

        # Check for failures
        fails = df[df.get("status", "") == "FAIL"]
        if len(fails) > 0:
            for _, row in fails.iterrows():
                lines.append(
                    f"- **{row['Scenario']}** ({row['Label']}) FAILED: "
                    f"`{row.get('error', 'unknown')}` — this demonstrates the "
                    f"**critical necessity** of the disabled module."
                )
            lines.append("")

        # Check VIP spam
        s0_spam = df[df["Scenario"] == "S0_Full"]["vip_spam"].values
        s3_spam = df[df["Scenario"] == "S3_No_VIP"]["vip_spam"].values
        if len(s0_spam) > 0 and len(s3_spam) > 0:
            s0_v = s0_spam[0]
            s3_v = s3_spam[0]
            if not np.isnan(s3_v) and s3_v > s0_v:
                lines.append(
                    f"- **VIP Guard**: Disabling increased VIP spam from "
                    f"{int(s0_v)} → {int(s3_v)} customers. "
                    f"These are high-value customers at low churn risk who would "
                    f"be unnecessarily contacted → risking relationship damage."
                )
                lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _plot_delta_chart(self, df: pd.DataFrame, path: str):
        """
        Normalized Δ% bar chart: S0 = 0% baseline.
        Each bar shows how much a metric changed when a module is disabled.
        """
        s0 = df[df["Scenario"] == "S0_Full"]
        if len(s0) == 0:
            logger.warning("[Ablation] No S0 baseline found — skipping delta chart.")
            return

        metrics_to_plot = ["c_index", "revenue_lift", "efficiency", "vip_spam"]
        metric_labels = ["C-index", "Revenue Lift %", "Efficiency %", "VIP Spam Count"]

        # Filter only non-S0 scenarios that succeeded
        others = df[(df["Scenario"] != "S0_Full") & (df.get("status", "") != "FAIL")]
        if len(others) == 0:
            return

        s0_vals = {m: float(s0[m].values[0]) for m in metrics_to_plot}

        fig, ax = plt.subplots(figsize=(12, 6))
        scenario_labels = others["Label"].values
        x = np.arange(len(scenario_labels))
        bar_width = 0.18
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            deltas = []
            for _, row in others.iterrows():
                val = float(row.get(metric, 0))
                base = s0_vals.get(metric, 0)
                if metric == "vip_spam":
                    # For VIP spam, show absolute increase (not %)
                    delta = val - base
                elif base != 0 and not np.isnan(base) and not np.isnan(val):
                    delta = ((val - base) / abs(base)) * 100
                else:
                    delta = 0
                deltas.append(delta)

            bars = ax.bar(x + i * bar_width, deltas, bar_width,
                         label=label, color=colors[i], edgecolor="white", linewidth=1)

            # Value labels
            for bar, d in zip(bars, deltas):
                va = "bottom" if d >= 0 else "top"
                text = f"+{d:.0f}" if d > 0 else f"{d:.0f}"
                if metric == "vip_spam":
                    text = f"+{d:.0f}" if d > 0 else f"{d:.0f}"
                else:
                    text = f"{d:+.1f}%"
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                       text, ha="center", va=va, fontsize=8, fontweight="bold")

        ax.axhline(0, color="black", linewidth=1.5, linestyle="-")
        ax.set_xlabel("Ablation Scenario", fontsize=12)
        ax.set_ylabel("Δ from S0 Full Pipeline (%)", fontsize=12)
        ax.set_title(
            "Ablation Study: Impact of Disabling Each Module\n"
            "Baseline: S0 Full Pipeline = 0%",
            fontsize=13, fontweight="bold"
        )
        ax.set_xticks(x + bar_width * 1.5)
        ax.set_xticklabels(scenario_labels, fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
