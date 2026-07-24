"""
Gate 8 — Semantic Distribution Analysis (EDA)
==============================================
Purpose : Analyze the raw semantic annotation outputs to understand the
          distribution of aspects, polarities, UNKNOWN rates, and their
          relationship to churn (Y_dormant_270).

Input   : mass_results_v224_qa.jsonl  (or full corpus JSONL after Dataset Freeze)
Output  : gate8_eda_report.json, plots/, corpus_qa_report.json

DO NOT hardcode features. DO NOT define thresholds.
All feature engineering decisions must be deferred to Gate 10,
guided by the distributions observed here.

Run this script AFTER each annotation batch to monitor distributions.
Run it FINALLY on the full annotated corpus as the Corpus QA Report.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Serialize numpy scalar types that json.dumps cannot handle natively."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

# ── Paths ──────────────────────────────────────────────────────────────────
ANNOTATION_JSONL = Path(
    "outputs/amazon_v5_rebuild/annotation/mass_workspace/mass_results_v224_qa.jsonl"
)
EPISODE_PARQUET = Path(
    "outputs/amazon_v5_rebuild/data/verified_episode_snapshots_h270_v1.parquet"
)
OUT_DIR = Path("outputs/amazon_v5_rebuild/gate8_semantic_eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Taxonomy from v2.2.4 (reference only — do NOT change)
KNOWN_ASPECTS = {
    "Domain_Experience",
    "Product_Condition_Quality",
    "Listing_Expectation_Compatibility",
    "Packaging_Presentation",
    "Price_Value",
    "Product_Performance_Usability",
    "Delivery_Fulfillment",
    "Customer_Service_Returns",
    "Other_Specific",
}


# ── 1. Load Annotations ────────────────────────────────────────────────────
def load_annotations(path: Path) -> pd.DataFrame:
    """Parse JSONL into a flat DataFrame of (annotation_item_id, aspect, polarity)."""
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            aid = item.get("annotation_item_id", "")
            targets = item.get("final_targets", [])
            mixed   = item.get("Review_Mixed_Flag", False)

            if not targets:
                records.append({
                    "annotation_item_id": aid,
                    "aspect": "__EMPTY__",
                    "polarity": None,
                    "is_mixed": mixed,
                    "is_unknown": False,
                })
            for t in targets:
                asp = t.get("aspect", "")
                pol = t.get("polarity", "")
                records.append({
                    "annotation_item_id": aid,
                    "aspect": asp,
                    "polarity": pol,
                    "is_mixed": mixed,
                    "is_unknown": asp not in KNOWN_ASPECTS,
                })

    return pd.DataFrame(records)


# ── 2. Load Churn Labels ───────────────────────────────────────────────────
def load_churn_labels(path: Path) -> pd.DataFrame:
    """Load episode snapshots for churn label join."""
    if not path.exists():
        print(f"  [WARN] Episode parquet not found: {path}. Churn analysis will be skipped.")
        return pd.DataFrame(columns=["reviewerID", "episode_id", "Y_dormant_270"])

    df = pd.read_parquet(path, columns=["reviewerID", "episode_id", "Y_dormant_270"])
    df["annotation_item_id"] = df["reviewerID"].astype(str) + "_" + df["episode_id"].astype(str)
    return df[["annotation_item_id", "Y_dormant_270"]]


# ── 3. Compute EDA Sections ────────────────────────────────────────────────
def section_overview(ann: pd.DataFrame) -> dict:
    """High-level counts."""
    items = ann["annotation_item_id"].nunique()
    real  = ann[ann["aspect"] != "__EMPTY__"]
    empty = ann[ann["aspect"] == "__EMPTY__"]["annotation_item_id"].nunique()
    unknown = ann[ann["is_unknown"] == True]["annotation_item_id"].nunique()

    return {
        "total_annotated_items": items,
        "total_aspect_instances": len(real),
        "avg_aspects_per_item": round(len(real) / max(items, 1), 3),
        "empty_extraction_count": empty,
        "empty_extraction_rate": round(empty / max(items, 1), 4),
        "unknown_aspect_count": int(ann["is_unknown"].sum()),
        "unknown_aspect_rate": round(ann["is_unknown"].mean(), 4),
        "mixed_item_count": int(ann.drop_duplicates("annotation_item_id")["is_mixed"].sum()),
        "mixed_item_rate": round(
            float(ann.drop_duplicates("annotation_item_id")["is_mixed"].mean()), 4
        ),
    }


def section_aspect_frequency(ann: pd.DataFrame) -> dict:
    """Aspect frequency table (excluding EMPTY)."""
    real = ann[ann["aspect"] != "__EMPTY__"]
    counts = real["aspect"].value_counts()
    total  = counts.sum()
    return {
        asp: {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
        for asp, cnt in counts.items()
    }


def section_polarity_distribution(ann: pd.DataFrame) -> dict:
    """Global polarity split."""
    real = ann[(ann["aspect"] != "__EMPTY__") & ann["polarity"].notna()]
    counts = real["polarity"].value_counts()
    total  = counts.sum()
    return {
        pol: {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
        for pol, cnt in counts.items()
    }


def section_aspect_by_polarity(ann: pd.DataFrame) -> dict:
    """Aspect × Polarity cross-table — identifies dominant signals."""
    real = ann[(ann["aspect"] != "__EMPTY__") & ann["polarity"].notna()]
    pivot = real.groupby(["aspect", "polarity"]).size().unstack(fill_value=0)
    return {
        asp: {pol: int(pivot.loc[asp, pol]) for pol in pivot.columns if asp in pivot.index}
        for asp in pivot.index
    }


def section_aspect_by_churn(ann: pd.DataFrame, churn: pd.DataFrame) -> dict:
    """
    Aspect × Churn cross-table.
    Shows churn rate for customers who mentioned each aspect.
    NOTE: These are raw correlations — do NOT use as threshold for feature engineering.
    Threshold decisions are deferred to Gate 10.
    """
    if churn.empty or "Y_dormant_270" not in churn.columns:
        return {"note": "Churn labels not available. Skipping."}

    merged = ann.merge(churn, on="annotation_item_id", how="inner")
    real   = merged[(merged["aspect"] != "__EMPTY__") & merged["polarity"].notna()]
    if real.empty:
        return {"note": "No matched items after join."}

    result = {}
    # One row per item (deduplicate within aspect group)
    for (asp, pol), grp in real.groupby(["aspect", "polarity"]):
        items  = grp.drop_duplicates("annotation_item_id")
        n      = len(items)
        churn_rate = items["Y_dormant_270"].mean()
        result[f"{asp}|{pol}"] = {
            "n_items": n,
            "churn_rate": round(churn_rate, 4),
            "note": "Raw correlation only. Not a feature threshold."
        }
    return result


# ── 4. Corpus QA Gate Check ────────────────────────────────────────────────
def qa_gate_check(overview: dict) -> dict:
    """
    Structured gate check for Corpus QA (Gate 7.7).
    Returns pass/fail for each criterion.
    """
    checks = {
        "unknown_rate_below_5pct": overview["unknown_aspect_rate"] < 0.05,
        "empty_rate_below_10pct":  overview["empty_extraction_rate"] < 0.10,
        "avg_aspects_above_0_8":   overview["avg_aspects_per_item"] >= 0.8,
        "min_10_items_annotated":  overview["total_annotated_items"] >= 10,
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"
    return {"verdict": verdict, "checks": checks}


# ── 5. Main ────────────────────────────────────────────────────────────────
def run(annotation_path: Path = ANNOTATION_JSONL):
    print("=== Gate 8: Semantic Distribution Analysis (EDA) ===\n")

    print(f"1. Loading annotations from: {annotation_path}")
    ann = load_annotations(annotation_path)
    print(f"   {ann['annotation_item_id'].nunique():,} items | {len(ann):,} aspect rows")

    print("2. Loading churn labels...")
    churn = load_churn_labels(EPISODE_PARQUET)

    print("3. Computing EDA sections...")
    overview   = section_overview(ann)
    asp_freq   = section_aspect_frequency(ann)
    pol_dist   = section_polarity_distribution(ann)
    asp_pol    = section_aspect_by_polarity(ann)
    asp_churn  = section_aspect_by_churn(ann, churn)
    qa_verdict = qa_gate_check(overview)

    # ── Assemble report ────────────────────────────────────────────────────
    report = {
        "meta": {
            "gate": "Gate 8 — Semantic Distribution Analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "annotation_source": str(annotation_path),
            "prompt_version": "v2.2.4",
            "model": "gemini-2.5-flash",
        },
        "overview":           overview,
        "aspect_frequency":   asp_freq,
        "polarity_distribution": pol_dist,
        "aspect_by_polarity": asp_pol,
        "aspect_by_churn":    asp_churn,
        "corpus_qa_gate":     qa_verdict,
        "analyst_notes": (
            "DO NOT use these distributions to hardcode features. "
            "Use them to make data-driven decisions in Gate 10 Feature Engineering."
        ),
    }

    report_path = OUT_DIR / "gate8_eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # Also write corpus_qa_report.json (the paired artifact with manifest + parquet)
    qa_report_path = OUT_DIR / "corpus_qa_report.json"
    with open(qa_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": report["meta"],
            "overview": overview,
            "corpus_qa_gate": qa_verdict,
        }, f, indent=2, cls=NumpyEncoder)

    print(f"\n4. Reports saved:")
    print(f"   {report_path}")
    print(f"   {qa_report_path}")

    print("\n=== EDA Summary ===")
    print(f"  Total items annotated : {overview['total_annotated_items']:,}")
    print(f"  Avg aspects / item    : {overview['avg_aspects_per_item']:.2f}")
    print(f"  Unknown rate          : {overview['unknown_aspect_rate']:.2%}")
    print(f"  Empty extraction rate : {overview['empty_extraction_rate']:.2%}")
    print(f"  Mixed item rate       : {overview['mixed_item_rate']:.2%}")
    print(f"\n  Corpus QA Verdict     : {qa_verdict['verdict']}")
    for check, passed in qa_verdict["checks"].items():
        status = "✅" if passed else "❌"
        print(f"    {status} {check}")

    print("\n  Top 5 Aspects:")
    for i, (asp, info) in enumerate(list(asp_freq.items())[:5]):
        print(f"    {i+1}. {asp}: {info['count']} ({info['pct']}%)")

    print("\n  Polarity Split:")
    for pol, info in pol_dist.items():
        print(f"    {pol}: {info['pct']}%")

    print("\n=== Gate 8 EDA Complete ===")
    return report


if __name__ == "__main__":
    run()
