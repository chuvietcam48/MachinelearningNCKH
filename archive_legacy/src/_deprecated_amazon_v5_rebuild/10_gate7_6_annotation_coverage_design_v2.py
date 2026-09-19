#!/usr/bin/env python
"""Gate 7.6 v2 annotation coverage design with fixed B3 split allocation."""

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
PRIMARY_SPLIT_TARGETS = {
    "train": 6000,
    "validation": 2000,
    "development_evaluation": 2000,
}
ENRICHMENT_PER_THEME = 1000
ITEMS_PER_PROVIDER_REQUEST = 8

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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
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


def annotation_item_id(df: pd.DataFrame) -> pd.Series:
    return df["reviewerID"].astype(str) + "_" + df["raw_source_row_id"].astype(str)


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


def keyword_pattern(words: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(w) for w in words), flags=re.IGNORECASE)


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def write_gate7_7_report(
    out_dir: Path,
    primary_review_count: int,
    enrichment_review_count: int,
    source_report_hash: str,
) -> None:
    primary_requests = ceil_div(primary_review_count, ITEMS_PER_PROVIDER_REQUEST)
    enrichment_requests = ceil_div(enrichment_review_count, ITEMS_PER_PROVIDER_REQUEST)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_7_status": "REQUIRED_NOT_READY_FOR_PROVIDER_EXECUTION",
        "purpose": "Lock provider scale feasibility before expanded annotation. No provider request is made by this step.",
        "provider_execution_plan": {
            "items_per_request": ITEMS_PER_PROVIDER_REQUEST,
            "primary_review_count": primary_review_count,
            "primary_estimated_request_count": primary_requests,
            "enrichment_review_count": enrichment_review_count,
            "enrichment_estimated_request_count": enrichment_requests,
            "total_estimated_request_count": primary_requests + enrichment_requests,
            "primary_and_enrichment_outputs_separate": True,
        },
        "must_lock_before_annotation": {
            "quota_rate_limit": "PENDING",
            "cost_ceiling": "PENDING",
            "deterministic_8_item_batch_manifest": "PENDING",
            "batch_run_id_strategy": "PENDING",
            "checkpoint_resume_without_auto_overwrite": "PENDING",
            "failed_batch_rerun_requires_new_run_id": "PENDING",
            "sampling_audit_during_expansion": "PENDING",
            "separate_primary_and_enrichment_outputs": "REQUIRED",
        },
        "run_id_strategy_recommendation": {
            "primary": "V12EXP_PRIMARY_V2_B{batch_number:05d}_R001",
            "enrichment": "V12EXP_ENRICH_V2_{theme}_B{batch_number:04d}_R001",
            "failed_batch_policy": "Never overwrite; rerun failed batch with R002/R003 and preserve failed run folder.",
        },
        "claim_guardrail": (
            "Do not begin expanded provider annotation until quota, cost, deterministic batch manifests, "
            "checkpoint/resume, audit sampling, and output separation are locked."
        ),
        "source_gate7_6_v2_report_sha256": source_report_hash,
    }
    json_path = out_dir / "gate7_7_provider_execution_feasibility_report.json"
    md_path = out_dir / "gate7_7_provider_execution_feasibility_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gate 7.7 Provider Execution Feasibility",
                "",
                f"Status: {report['gate7_7_status']}",
                "",
                f"Primary estimated requests: {primary_requests}",
                f"Enrichment estimated requests: {enrichment_requests}",
                f"Total estimated requests: {primary_requests + enrichment_requests}",
                "",
                "Do not call provider until quota, cost ceiling, deterministic batch manifest, checkpoint/resume, rerun policy, and audit sampling are locked.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
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
    membership["annotation_item_id"] = annotation_item_id(membership)

    counts = strict_prior_counts(membership)
    snapshots = snapshots.merge(
        counts,
        left_on=["reviewerID", "episode_start"],
        right_on=["reviewerID", "review_timestamp"],
        how="left",
    ).drop(columns=["review_timestamp"])
    snapshots["strict_prior_review_count"] = snapshots["strict_prior_review_count"].fillna(0).astype(int)
    snapshots = assign_split(snapshots, split_manifest)
    eligible = snapshots[snapshots["strict_prior_review_count"] > 0].copy()

    parts = []
    for split, target in PRIMARY_SPLIT_TARGETS.items():
        part = eligible[eligible["split"] == split]
        if len(part) < target:
            raise ValueError(f"{split} has {len(part)} eligible rows, below target {target}")
        parts.append(part.sample(n=target, random_state=SEED))
    primary_episodes = pd.concat(parts, ignore_index=True).sort_values(["split", "episode_start", "episode_id"])

    selected = primary_episodes[["reviewerID", "episode_id", "episode_start", "split"]].rename(
        columns={
            "episode_id": "selected_episode_id",
            "episode_start": "selected_episode_start",
            "split": "selected_split",
        }
    )
    edges = membership.merge(selected, on="reviewerID", how="inner")
    edges = edges[edges["review_timestamp"] < edges["selected_episode_start"]].copy()
    edges["is_strict_prior"] = True
    edges = edges.rename(columns={"selected_episode_start": "episode_start"})

    edge_cols = [
        "selected_episode_id",
        "reviewerID",
        "annotation_item_id",
        "review_timestamp",
        "episode_start",
        "is_strict_prior",
    ]
    episode_membership = edges[edge_cols].sort_values(
        ["selected_episode_id", "review_timestamp", "annotation_item_id"]
    )
    primary_reviews = (
        edges.sort_values(["reviewerID", "review_timestamp", "raw_source_row_id"])
        .drop_duplicates("annotation_item_id")
        .copy()
    )
    primary_reviews["annotation_scope"] = "primary_b3_outcome_blind_random_cohort_v2"

    coverage = edges.groupby("selected_episode_id")["annotation_item_id"].nunique()
    primary_episodes["prior_reviews_in_annotation_manifest"] = (
        primary_episodes["episode_id"].map(coverage).fillna(0).astype(int)
    )

    primary_ids = set(primary_reviews["annotation_item_id"])
    enrichment_pool = membership[~membership["annotation_item_id"].isin(primary_ids)].copy()
    text = enrichment_pool["summary"].fillna("").astype(str) + " " + enrichment_pool["reviewText"].fillna("").astype(str)
    enrichment_parts = []
    used_ids: set[str] = set()
    for idx, (theme, keywords) in enumerate(ENRICHMENT_THEMES.items()):
        mask = text.str.contains(keyword_pattern(keywords), regex=True, na=False)
        candidates = enrichment_pool[mask & ~enrichment_pool["annotation_item_id"].isin(used_ids)].copy()
        sample_n = min(args.enrichment_per_theme, len(candidates))
        sample = candidates.sample(n=sample_n, random_state=SEED + idx).copy()
        sample["annotation_scope"] = "targeted_enrichment_not_for_primary_predictive_claim_v2"
        sample["enrichment_theme"] = theme
        sample["enrichment_keywords"] = ";".join(keywords)
        used_ids.update(sample["annotation_item_id"].tolist())
        enrichment_parts.append(sample)
    enrichment = pd.concat(enrichment_parts, ignore_index=True)

    primary_episode_path = out_dir / "gate7_6_v2_primary_b3_episode_cohort_manifest.csv"
    primary_review_path = out_dir / "gate7_6_v2_primary_b3_annotation_review_manifest.csv"
    enrichment_path = out_dir / "gate7_6_v2_targeted_enrichment_review_manifest.csv"
    membership_path_out = out_dir / "episode_annotation_membership_v2.parquet"
    report_path = out_dir / "gate7_6_annotation_coverage_design_report_v2.json"
    md_path = out_dir / "gate7_6_annotation_coverage_design_report_v2.md"

    primary_episodes.to_csv(primary_episode_path, index=False)
    primary_reviews.to_csv(primary_review_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    episode_membership.to_parquet(membership_path_out, index=False)

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
        enrichment.groupby("enrichment_theme")
        .agg(review_count=("annotation_item_id", "nunique"), customer_count=("reviewerID", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_6_status": "DESIGN_READY_AFTER_PRE_ANNOTATION_FIXES",
        "gate7_6a_outcome_blind_cohort_design": "PASS",
        "gate7_6b_temporal_annotation_eligibility": "PASS",
        "gate7_6c_b3_train_support_allocation": "PASS_REVISED_V2",
        "gate7_6d_episode_review_lineage_manifest": "PASS_ADDED_V2",
        "gate7_7_provider_scale_feasibility": "REQUIRED_BEFORE_ANNOTATION",
        "primary_b3_random_cohort_v2": {
            "random_seed": SEED,
            "split_targets": PRIMARY_SPLIT_TARGETS,
            "selected_episode_count": int(len(primary_episodes)),
            "selected_customer_count": int(primary_episodes["reviewerID"].nunique()),
            "unique_prior_reviews_to_annotate_once": int(primary_reviews["annotation_item_id"].nunique()),
            "episode_review_membership_edge_count": int(len(episode_membership)),
            "episodes_with_at_least_one_prior_review_to_annotate": int(
                (primary_episodes["prior_reviews_in_annotation_manifest"] > 0).sum()
            ),
            "split_summary": split_summary,
        },
        "targeted_enrichment_v2": {
            "purpose": "Rare-label taxonomy/classifier development only; not for primary predictive-effect claims.",
            "not_for_primary_predictive_effect_claim": True,
            "selected_review_count": int(enrichment["annotation_item_id"].nunique()),
            "selected_customer_count": int(enrichment["reviewerID"].nunique()),
            "theme_summary": enrichment_summary,
        },
        "lineage_policy": {
            "downstream_join_key_warning": "Do not use selected_episode_id from the unique review manifest as the primary join key.",
            "feature_builder_join_policy": "Use reviewerID + review_timestamp < episode_start, or episode_annotation_membership_v2.parquet.",
            "episode_annotation_membership_map_added": True,
        },
        "provider_scale_estimate": {
            "items_per_request": ITEMS_PER_PROVIDER_REQUEST,
            "primary_estimated_request_count": ceil_div(
                int(primary_reviews["annotation_item_id"].nunique()), ITEMS_PER_PROVIDER_REQUEST
            ),
            "enrichment_estimated_request_count": ceil_div(
                int(enrichment["annotation_item_id"].nunique()), ITEMS_PER_PROVIDER_REQUEST
            ),
        },
        "outputs": {
            "primary_episode_manifest_v2": str(primary_episode_path),
            "primary_review_manifest_v2": str(primary_review_path),
            "targeted_enrichment_manifest_v2": str(enrichment_path),
            "episode_annotation_membership_v2": str(membership_path_out),
            "design_report_json_v2": str(report_path),
            "design_report_md_v2": str(md_path),
        },
        "source_hashes": {
            "verified_episode_snapshots_h270_v1": sha256_file(snapshot_path),
            "verified_episode_review_membership_v1": sha256_file(data_dir / "verified_episode_review_membership_v1.parquet"),
            "temporal_development_split_manifest_v1": sha256_file(split_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gate 7.6 Annotation Coverage Design v2",
                "",
                f"Status: {report['gate7_6_status']}",
                "",
                f"Train/validation/development allocation: {PRIMARY_SPLIT_TARGETS}",
                f"Selected episodes: {report['primary_b3_random_cohort_v2']['selected_episode_count']}",
                f"Selected customers: {report['primary_b3_random_cohort_v2']['selected_customer_count']}",
                f"Unique prior reviews to annotate once: {report['primary_b3_random_cohort_v2']['unique_prior_reviews_to_annotate_once']}",
                f"Episode-review membership edges: {report['primary_b3_random_cohort_v2']['episode_review_membership_edge_count']}",
                f"Targeted enrichment reviews: {report['targeted_enrichment_v2']['selected_review_count']}",
                "",
                "Do not call provider until Gate 7.7 is locked.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    write_gate7_7_report(
        out_dir,
        primary_review_count=int(primary_reviews["annotation_item_id"].nunique()),
        enrichment_review_count=int(enrichment["annotation_item_id"].nunique()),
        source_report_hash=sha256_file(report_path),
    )

    print(
        json.dumps(
            {
                "gate7_6_status": report["gate7_6_status"],
                "split_targets": PRIMARY_SPLIT_TARGETS,
                "primary_selected_episodes": report["primary_b3_random_cohort_v2"]["selected_episode_count"],
                "primary_unique_prior_reviews_to_annotate_once": report["primary_b3_random_cohort_v2"]["unique_prior_reviews_to_annotate_once"],
                "episode_review_membership_edges": report["primary_b3_random_cohort_v2"]["episode_review_membership_edge_count"],
                "targeted_enrichment_reviews": report["targeted_enrichment_v2"]["selected_review_count"],
                "gate7_7_status": "REQUIRED_NOT_READY_FOR_PROVIDER_EXECUTION",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
