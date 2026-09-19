#!/usr/bin/env python
"""Create Gate 7.5 semantic coverage feasibility report.

This does not train B3. It checks whether the current 120-review semantic
calibration set has enough coverage to support a primary predictive experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


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
        raise ValueError(f"Split row counts assign {cursor} rows, but episode frame has {len(out)} rows")
    return out


def split_support(df: pd.DataFrame, semantic_col: str) -> list[dict[str, Any]]:
    rows = []
    for split_name, part in df.groupby("split", sort=False):
        sem = part[part[semantic_col] > 0]
        rows.append(
            {
                "split": split_name,
                "episode_count": int(len(part)),
                "observed_disengagement_events": int(part["Y_dormant_270"].sum()),
                "semantic_positive_episode_count": int(len(sem)),
                "semantic_positive_episode_rate": float(len(sem) / len(part)) if len(part) else 0.0,
                "semantic_positive_observed_disengagement_events": int(sem["Y_dormant_270"].sum()),
                "semantic_positive_non_events": int(len(sem) - sem["Y_dormant_270"].sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    out_dir = repo / "outputs" / "amazon_v5_rebuild" / "features_semantic_b3"
    data_dir = repo / "outputs" / "amazon_v5_rebuild" / "data"
    ml_dir = repo / "outputs" / "amazon_v5_rebuild" / "ml_phase3a"

    labels_path = out_dir / "semantic_review_labels_v1_2_final_with_overlay.parquet"
    aspect_path = out_dir / "semantic_aspect_rows_adjudicated_overlay_order_agnostic.parquet"
    core_path = out_dir / "semantic_episode_features_b3_core_v1_2.parquet"
    raw_listing_path = out_dir / "semantic_episode_features_b3_listing_raw_v1_2_sensitivity.parquet"
    overlay_listing_path = (
        out_dir / "semantic_episode_features_b3_listing_adjudicated_overlay_sensitivity.parquet"
    )
    membership_path = data_dir / "verified_episode_review_membership_v1.parquet"
    split_path = ml_dir / "temporal_development_split_manifest_v1.csv"

    labels = pd.read_parquet(labels_path)
    aspects = pd.read_parquet(aspect_path)
    membership = pd.read_parquet(membership_path, columns=["reviewerID", "raw_source_row_id"])
    split_manifest = pd.read_csv(split_path)

    core = pd.read_parquet(
        core_path,
        columns=[
            "reviewerID",
            "episode_id",
            "episode_start",
            "Y_dormant_270",
            "semantic_prior_labeled_review_count",
            "semantic_prior_aspect_row_count",
            "semantic_prior_negative_aspect_count",
            "semantic_prior_mixed_review_count",
        ],
    )
    raw_listing = pd.read_parquet(
        raw_listing_path, columns=["episode_id", "semantic_prior_labeled_review_count"]
    ).rename(columns={"semantic_prior_labeled_review_count": "raw_listing_prior_label_count"})
    overlay_listing = pd.read_parquet(
        overlay_listing_path, columns=["episode_id", "semantic_prior_labeled_review_count"]
    ).rename(columns={"semantic_prior_labeled_review_count": "overlay_listing_prior_label_count"})

    episodes = core.merge(raw_listing, on="episode_id", how="left").merge(
        overlay_listing, on="episode_id", how="left"
    )
    episodes = assign_split(episodes, split_manifest)
    labels["episode_id"] = labels["episode_id"].astype("int64")
    core["episode_id"] = core["episode_id"].astype("int64")

    universe_item_ids = membership["reviewerID"].astype(str) + "_" + membership["raw_source_row_id"].astype(str)
    unique_universe_reviews = int(universe_item_ids.nunique())
    unique_universe_customers = int(membership["reviewerID"].nunique())

    label_episode_outcomes = labels.merge(
        core[["episode_id", "Y_dormant_270"]].drop_duplicates(), on="episode_id", how="left"
    )
    core_sem = episodes[episodes["semantic_prior_labeled_review_count"] > 0]
    raw_sem = episodes[episodes["raw_listing_prior_label_count"] > 0]
    overlay_sem = episodes[episodes["overlay_listing_prior_label_count"] > 0]

    aspect_counts = {k: int(v) for k, v in aspects["aspect"].value_counts().to_dict().items()}
    polarity_by_aspect = (
        aspects.groupby(["aspect", "polarity"]).size().reset_index(name="count").to_dict(orient="records")
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_5_status": "HOLD_PRIMARY_B3_PREDICTIVE_EXPERIMENT",
        "gate7_technical_feature_construction": "PASS",
        "reason": (
            "Current semantic features are technically valid, but semantic coverage is too sparse for a "
            "primary predictive B3 experiment. Zero semantic features mostly indicate unannotated prior "
            "history, not absence of complaint."
        ),
        "source_hashes": {
            "semantic_review_labels": sha256_file(labels_path),
            "semantic_aspect_rows_adjudicated_overlay": sha256_file(aspect_path),
            "b3_core_features": sha256_file(core_path),
            "b3_listing_raw_sensitivity_features": sha256_file(raw_listing_path),
            "b3_listing_adjudicated_overlay_sensitivity_features": sha256_file(overlay_listing_path),
            "verified_episode_review_membership": sha256_file(membership_path),
            "temporal_development_split_manifest": sha256_file(split_path),
        },
        "coverage": {
            "semantic_label_unique_review_count": int(labels["annotation_item_id"].nunique()),
            "semantic_label_unique_customer_count": int(labels["reviewerID"].nunique()),
            "review_universe_unique_review_count": unique_universe_reviews,
            "review_universe_unique_customer_count": unique_universe_customers,
            "semantic_label_review_universe_coverage_rate": float(
                labels["annotation_item_id"].nunique() / unique_universe_reviews
            ),
            "semantic_label_customer_universe_coverage_rate": float(
                labels["reviewerID"].nunique() / unique_universe_customers
            ),
            "episode_universe_count": int(len(episodes)),
            "core_episode_with_prior_semantic_history_count": int(len(core_sem)),
            "core_episode_with_prior_semantic_history_rate": float(len(core_sem) / len(episodes)),
            "raw_listing_episode_with_prior_semantic_history_count": int(len(raw_sem)),
            "overlay_listing_episode_with_prior_semantic_history_count": int(len(overlay_sem)),
        },
        "split_support": {
            "core_without_fine_grained_listing": split_support(
                episodes, "semantic_prior_labeled_review_count"
            ),
            "listing_raw_v1_2_sensitivity": split_support(episodes, "raw_listing_prior_label_count"),
            "listing_adjudicated_overlay_sensitivity": split_support(
                episodes, "overlay_listing_prior_label_count"
            ),
        },
        "selection_assessment": {
            "annotated_review_selection_design": (
                "120-review calibration/codebook-development set, not an outcome-representative annotation "
                "sample for the full episode universe."
            ),
            "selection_independent_of_outcome_confirmed": False,
            "annotated_reviews_linked_observed_disengagement_events": int(
                label_episode_outcomes["Y_dormant_270"].fillna(0).sum()
            ),
            "annotated_reviews_linked_non_events": int(
                len(label_episode_outcomes) - label_episode_outcomes["Y_dormant_270"].fillna(0).sum()
            ),
        },
        "zero_semantic_feature_interpretation": {
            "zero_means_no_prior_annotated_semantic_history": True,
            "zero_means_no_complaint": False,
            "zero_may_mean_unannotated_review_history": True,
            "missing_annotation_must_not_be_encoded_as_neutral_or_no_complaint": True,
            "missing_annotation_must_not_be_treated_as_behavioral_feature": True,
        },
        "aspect_support": {
            "adjudicated_aspect_row_count": int(len(aspects)),
            "aspect_counts": aspect_counts,
            "polarity_by_aspect": polarity_by_aspect,
            "broad_features_have_enough_support_for_primary_b3": False,
        },
        "decision": {
            "may_run_b3_as_primary_predictive_experiment": False,
            "may_use_current_artifacts_for_pipeline_demo_or_feasibility_tables": True,
            "required_before_primary_b3": [
                "Route A: annotate the relevant review population with frozen v1.2 prompt and validation.",
                "Route B: create an expanded silver-label training set and audited classifier before applying to the review universe.",
            ],
            "claim_guardrail": (
                "Do not claim semantic features improve prediction from the current 120-label coverage. "
                "Outcome remains observed verified-feedback disengagement within 270 days, not purchase churn."
            ),
        },
    }

    json_path = out_dir / "gate7_5_semantic_coverage_feasibility_report.json"
    md_path = out_dir / "gate7_5_semantic_coverage_feasibility_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gate 7.5 Semantic Coverage Feasibility",
                "",
                f"Status: {report['gate7_5_status']}",
                "",
                f"Semantic labelled reviews: {report['coverage']['semantic_label_unique_review_count']}",
                f"Semantic labelled customers: {report['coverage']['semantic_label_unique_customer_count']}",
                f"Review universe unique reviews: {report['coverage']['review_universe_unique_review_count']}",
                f"Core episodes with prior semantic history: {report['coverage']['core_episode_with_prior_semantic_history_count']} / {report['coverage']['episode_universe_count']}",
                "",
                "Decision: do not run primary B3 predictive experiment until semantic coverage is expanded.",
                "",
                "Zero semantic feature means no prior annotated semantic history, not no complaint.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate7_5_status": report["gate7_5_status"],
                "semantic_label_unique_review_count": report["coverage"][
                    "semantic_label_unique_review_count"
                ],
                "semantic_label_unique_customer_count": report["coverage"][
                    "semantic_label_unique_customer_count"
                ],
                "core_episode_with_prior_semantic_history_count": report["coverage"][
                    "core_episode_with_prior_semantic_history_count"
                ],
                "may_run_b3_as_primary_predictive_experiment": report["decision"][
                    "may_run_b3_as_primary_predictive_experiment"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
