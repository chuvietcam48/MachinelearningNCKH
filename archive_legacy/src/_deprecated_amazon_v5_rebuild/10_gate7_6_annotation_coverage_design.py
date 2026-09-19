#!/usr/bin/env python
"""Gate 7.6 outcome-blind annotation coverage cohort design."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 20260706
PRIMARY_EPISODE_TARGET = 10_000
ENRICHMENT_PER_THEME = 1_000

ENRICHMENT_THEMES = {
    "Delivery_Fulfillment": [
        "shipping",
        "delivery",
        "delivered",
        "arrived",
        "never received",
        "missing",
        "sent wrong",
        "wrong item",
    ],
    "Customer_Service_Returns": [
        "refund",
        "return",
        "returned",
        "replacement",
        "customer service",
        "seller",
        "exchange",
    ],
    "Packaging_Presentation": [
        "packaging",
        "package",
        "box",
        "case",
        "jewel case",
        "cracked case",
        "broken case",
    ],
    "Product_Performance_Usability": [
        "skip",
        "skips",
        "skipping",
        "won't play",
        "does not play",
        "download",
        "track",
        "defective",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def assign_split(df: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["episode_start"] = pd.to_datetime(out["episode_start"])
    out["split"] = "unassigned"
    ordered_idx = out.sort_values(["episode_start", "episode_id"]).index.to_list()
    cursor = 0
    for _, row in split_manifest.iterrows():
        row_count = int(row["row_count"])
        split_idx = ordered_idx[cursor : cursor + row_count]
        out.loc[split_idx, "split"] = row["split"]
        cursor += row_count
    if cursor != len(out):
        raise ValueError(f"Split manifest assigns {cursor} rows, episode frame has {len(out)}")
    return out


def proportional_targets(split_manifest: pd.DataFrame, target: int) -> dict[str, int]:
    counts = split_manifest.set_index("split")["row_count"].astype(int)
    raw = counts / counts.sum() * target
    floors = raw.astype(int)
    remainder = target - int(floors.sum())
    fractions = (raw - floors).sort_values(ascending=False)
    out = floors.to_dict()
    for split in fractions.index[:remainder]:
        out[split] += 1
    return {str(k): int(v) for k, v in out.items()}


def item_id(df: pd.DataFrame) -> pd.Series:
    return df["reviewerID"].astype(str) + "_" + df["raw_source_row_id"].astype(str)


def make_pattern(words: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(w) for w in words]
    return re.compile("|".join(escaped), flags=re.IGNORECASE)


def strict_prior_counts(membership: pd.DataFrame) -> pd.DataFrame:
    by_date = (
        membership.groupby(["reviewerID", "review_timestamp"], sort=True)
        .size()
        .rename("same_date_review_count")
        .reset_index()
        .sort_values(["reviewerID", "review_timestamp"])
    )
    by_date["strict_prior_review_count"] = (
        by_date.groupby("reviewerID")["same_date_review_count"].cumsum()
        - by_date["same_date_review_count"]
    )
    return by_date[["reviewerID", "review_timestamp", "strict_prior_review_count"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--primary-episode-target", type=int, default=PRIMARY_EPISODE_TARGET)
    parser.add_argument("--enrichment-per-theme", type=int, default=ENRICHMENT_PER_THEME)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / "outputs" / "amazon_v5_rebuild" / "features_semantic_b3"
    data_dir = repo / "outputs" / "amazon_v5_rebuild" / "data"
    ml_dir = repo / "outputs" / "amazon_v5_rebuild" / "ml_phase3a"
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = data_dir / "verified_episode_snapshots_h270_v1.parquet"
    membership_path = data_dir / "verified_episode_review_membership_v1.parquet"
    split_path = ml_dir / "temporal_development_split_manifest_v1.csv"

    snapshots = pd.read_parquet(snapshot_path, columns=["reviewerID", "episode_id", "episode_start"])
    snapshots["episode_start"] = pd.to_datetime(snapshots["episode_start"])
    split_manifest = pd.read_csv(split_path)
    membership = pd.read_parquet(
        membership_path,
        columns=[
            "reviewerID",
            "episode_id",
            "review_timestamp",
            "asin",
            "overall",
            "reviewText",
            "summary",
            "raw_source_row_id",
        ],
    )
    membership["review_timestamp"] = pd.to_datetime(membership["review_timestamp"])
    membership["annotation_item_id"] = item_id(membership)

    strict_counts = strict_prior_counts(membership)
    snapshots = snapshots.merge(
        strict_counts,
        left_on=["reviewerID", "episode_start"],
        right_on=["reviewerID", "review_timestamp"],
        how="left",
    ).drop(columns=["review_timestamp"])
    snapshots["strict_prior_review_count"] = snapshots["strict_prior_review_count"].fillna(0).astype(int)
    snapshots = assign_split(snapshots, split_manifest)
    eligible = snapshots[snapshots["strict_prior_review_count"] > 0].copy()

    targets = proportional_targets(split_manifest, args.primary_episode_target)
    sampled_parts = []
    for split, n in targets.items():
        part = eligible[eligible["split"] == split]
        if len(part) < n:
            raise ValueError(f"Split {split} has only {len(part)} eligible episodes, target {n}")
        sampled_parts.append(part.sample(n=n, random_state=SEED))
    primary_episodes = pd.concat(sampled_parts, ignore_index=True)
    primary_episodes = primary_episodes.sort_values(["split", "episode_start", "episode_id"])

    selected = primary_episodes[
        ["reviewerID", "episode_id", "episode_start", "split"]
    ].rename(
        columns={
            "episode_id": "selected_episode_id",
            "episode_start": "selected_episode_start",
            "split": "selected_split",
        }
    )
    prior_pairs = membership.merge(selected, on="reviewerID", how="inner")
    prior_pairs = prior_pairs[
        prior_pairs["review_timestamp"] < prior_pairs["selected_episode_start"]
    ].copy()

    primary_reviews = (
        prior_pairs.sort_values(["reviewerID", "review_timestamp", "raw_source_row_id"])
        .drop_duplicates("annotation_item_id")
        .copy()
    )
    coverage_by_episode = prior_pairs.groupby("selected_episode_id")["annotation_item_id"].nunique()
    primary_episodes["prior_reviews_in_annotation_manifest"] = (
        primary_episodes["episode_id"].map(coverage_by_episode).fillna(0).astype(int)
    )
    primary_reviews["annotation_scope"] = "primary_b3_outcome_blind_random_cohort"

    primary_ids = set(primary_reviews["annotation_item_id"])
    enrichment_pool = membership[~membership["annotation_item_id"].isin(primary_ids)].copy()
    text = (
        enrichment_pool["summary"].fillna("").astype(str)
        + " "
        + enrichment_pool["reviewText"].fillna("").astype(str)
    )
    enrichment_rows = []
    used_ids: set[str] = set()
    for theme, keywords in ENRICHMENT_THEMES.items():
        pattern = make_pattern(keywords)
        mask = text.str.contains(pattern, regex=True, na=False)
        candidates = enrichment_pool[mask & ~enrichment_pool["annotation_item_id"].isin(used_ids)].copy()
        if "overall" in candidates.columns:
            candidates = candidates.sort_values(["overall", "review_timestamp", "annotation_item_id"])
        sample_n = min(args.enrichment_per_theme, len(candidates))
        sample = candidates.sample(n=sample_n, random_state=SEED + len(enrichment_rows)).copy()
        sample["annotation_scope"] = "targeted_enrichment_not_for_primary_predictive_claim"
        sample["enrichment_theme"] = theme
        sample["enrichment_keywords"] = ";".join(keywords)
        used_ids.update(sample["annotation_item_id"].tolist())
        enrichment_rows.append(sample)
    enrichment_reviews = pd.concat(enrichment_rows, ignore_index=True) if enrichment_rows else pd.DataFrame()

    primary_episode_path = out_dir / "gate7_6_primary_b3_episode_cohort_manifest.csv"
    primary_review_path = out_dir / "gate7_6_primary_b3_annotation_review_manifest.csv"
    enrichment_path = out_dir / "gate7_6_targeted_enrichment_review_manifest.csv"
    report_path = out_dir / "gate7_6_annotation_coverage_design_report.json"
    md_path = out_dir / "gate7_6_annotation_coverage_design_report.md"

    primary_episodes.to_csv(primary_episode_path, index=False)
    primary_reviews.to_csv(primary_review_path, index=False)
    enrichment_reviews.to_csv(enrichment_path, index=False)

    split_summary = (
        primary_episodes.groupby("split")
        .agg(
            selected_episode_count=("episode_id", "size"),
            selected_customer_count=("reviewerID", "nunique"),
            episode_with_at_least_one_prior_review_to_annotate=(
                "prior_reviews_in_annotation_manifest",
                lambda s: int((s > 0).sum()),
            ),
            median_prior_reviews_to_annotate=("prior_reviews_in_annotation_manifest", "median"),
            p95_prior_reviews_to_annotate=(
                "prior_reviews_in_annotation_manifest",
                lambda s: float(s.quantile(0.95)),
            ),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    enrichment_summary = (
        enrichment_reviews.groupby("enrichment_theme")
        .agg(review_count=("annotation_item_id", "nunique"), customer_count=("reviewerID", "nunique"))
        .reset_index()
        .to_dict(orient="records")
        if len(enrichment_reviews)
        else []
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_6_status": "DESIGN_READY_NO_ANNOTATION_RUN",
        "purpose": (
            "Define an outcome-blind annotation coverage cohort before expanding semantic labels. "
            "No Gemini/provider annotation is executed by this step."
        ),
        "outcome_blind_controls": {
            "selection_uses_Y_dormant_270": False,
            "selection_uses_future_disengagement_event": False,
            "primary_selection_unit": "episode snapshot sampled within temporal split using only reviewerID, episode_id, episode_start, strict_prior_review_count, split",
            "annotation_unit": "all unique prior reviews for selected reviewer/snapshot pairs where review_timestamp < selected_episode_start",
            "feature_extraction_rule": "For any B3 model, aggregate only annotated reviews with review_timestamp < episode_start.",
            "zero_semantic_feature_meaning_after_primary_annotation": (
                "Within selected primary cohort, zero can mean no labelled semantic signal in annotated prior history. "
                "Outside selected primary cohort, zero must still be treated as unannotated coverage."
            ),
        },
        "primary_b3_random_cohort": {
            "random_seed": SEED,
            "target_episode_count": args.primary_episode_target,
            "selected_episode_count": int(len(primary_episodes)),
            "selected_customer_count": int(primary_episodes["reviewerID"].nunique()),
            "unique_prior_reviews_to_annotate": int(primary_reviews["annotation_item_id"].nunique()),
            "episodes_with_at_least_one_prior_review_to_annotate": int(
                (primary_episodes["prior_reviews_in_annotation_manifest"] > 0).sum()
            ),
            "split_summary": split_summary,
        },
        "targeted_enrichment_set": {
            "purpose": "Increase rare semantic labels for taxonomy/classifier development only.",
            "not_for_primary_predictive_effect_claim": True,
            "selection_uses_review_text_and_rating_only": True,
            "selection_uses_outcome": False,
            "target_per_theme": args.enrichment_per_theme,
            "selected_review_count": int(enrichment_reviews["annotation_item_id"].nunique())
            if len(enrichment_reviews)
            else 0,
            "selected_customer_count": int(enrichment_reviews["reviewerID"].nunique())
            if len(enrichment_reviews)
            else 0,
            "theme_summary": enrichment_summary,
        },
        "feature_policy_after_expanded_annotation": {
            "rerun_gate7_5_required": True,
            "primary_models_allowed_after_gate7_5_pass": [
                "B0 = behavioral history",
                "B1 = B0 + ratings",
                "B3-core = B1 + broad semantic feedback history",
            ],
            "sensitivity_only": ["B3-aspect = aspect-level features when support thresholds are met"],
            "standalone_aspect_feature_minimum_support_rule": (
                "Do not use an aspect as a standalone primary feature unless it has enough train, validation, "
                "and development/test support after rerun Gate 7.5; sparse categories should roll into broad semantic features."
            ),
            "currently_sparse_examples": [
                "Packaging_Presentation",
                "Customer_Service_Returns",
                "Product_Performance_Usability",
            ],
        },
        "source_hashes": {
            "verified_episode_snapshots_h270_v1": sha256_file(snapshot_path),
            "verified_episode_review_membership_v1": sha256_file(membership_path),
            "temporal_development_split_manifest_v1": sha256_file(split_path),
        },
        "outputs": {
            "primary_episode_manifest": str(primary_episode_path),
            "primary_annotation_review_manifest": str(primary_review_path),
            "targeted_enrichment_review_manifest": str(enrichment_path),
            "design_report_json": str(report_path),
            "design_report_md": str(md_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gate 7.6 Annotation Coverage Design",
                "",
                f"Status: {report['gate7_6_status']}",
                "",
                f"Primary selected episodes: {report['primary_b3_random_cohort']['selected_episode_count']}",
                f"Primary selected customers: {report['primary_b3_random_cohort']['selected_customer_count']}",
                f"Unique prior reviews to annotate: {report['primary_b3_random_cohort']['unique_prior_reviews_to_annotate']}",
                f"Targeted enrichment reviews: {report['targeted_enrichment_set']['selected_review_count']}",
                "",
                "Controls:",
                "- Primary cohort is outcome-blind.",
                "- Annotate all prior review history for selected reviewer/snapshot pairs.",
                "- Targeted enrichment is not for primary predictive-effect claims.",
                "- Rerun Gate 7.5 after expanded annotation before B3.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate7_6_status": report["gate7_6_status"],
                "primary_selected_episodes": report["primary_b3_random_cohort"]["selected_episode_count"],
                "primary_selected_customers": report["primary_b3_random_cohort"]["selected_customer_count"],
                "primary_unique_prior_reviews_to_annotate": report["primary_b3_random_cohort"][
                    "unique_prior_reviews_to_annotate"
                ],
                "targeted_enrichment_reviews": report["targeted_enrichment_set"]["selected_review_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
