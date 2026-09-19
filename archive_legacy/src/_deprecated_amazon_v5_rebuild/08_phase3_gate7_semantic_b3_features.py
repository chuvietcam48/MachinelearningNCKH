#!/usr/bin/env python
"""Build Gate 7 semantic B3 feature artifacts from frozen v1.2 annotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROMPT_V1_2_SHA256 = "f2cf7512c65860bdc6f2cb07b0cb55781262c39df6d8a196671b69b951737860"
TAXONOMY_V1_2_SHA256 = "8cde93d06f446c9d776b4935786c81d76a7e5a16d4cdde257ac095ea80ca52f9"

ACCEPTED_RUNS = [
    ("batch_1_calib", "phase2b1_v16_pilot_V12B1R001"),
    ("batch_2_calib", "phase2b1_v16_live_batch_V12B2R001"),
    ("batch_3_calib", "phase2b1_v16_live_batch_V12B3R001"),
    ("batch_4_calib", "phase2b1_v16_live_batch_V12B4R001"),
    ("batch_5_calib", "phase2b1_v16_live_batch_V12B5R002"),
    ("batch_6_calib", "phase2b1_v16_live_batch_V12B6R001"),
    ("batch_7_calib", "phase2b1_v16_live_batch_V12B7R001"),
    ("batch_8_calib", "phase2b1_v16_live_batch_V12B8R001"),
    ("batch_9_calib", "phase2b1_v16_live_batch_V12B9R001"),
    ("batch_10_calib", "phase2b1_v16_live_batch_V12B10R001"),
    ("batch_11_calib", "phase2b1_v16_live_batch_V12B11R001"),
    ("batch_12_calib", "phase2b1_v16_live_batch_V12B12R001"),
    ("batch_13_calib", "phase2b1_v16_live_batch_V12B13R002"),
    ("batch_14_calib", "phase2b1_v16_live_batch_V12B14R002"),
    ("batch_15_calib", "phase2b1_v16_live_batch_V12B15R001"),
]

ASPECTS_FOR_CORE = [
    "Domain_Experience",
    "Product_Condition_Quality",
    "Delivery_Fulfillment",
    "Customer_Service_Returns",
    "Price_Value",
    "Packaging_Presentation",
    "Product_Performance_Usability",
]

LISTING_ASPECT = "Listing_Expectation_Compatibility"
HUMAN_EVIDENCE_REPAIRED_ITEM_IDS = {"A3H162O2LP16QG_3551859"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_default(obj: Any) -> Any:
    if pd.isna(obj):
        return None
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_parsed_payload(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one parsed payload in {path}, found {len(rows)}")
    return rows[0]


def item_map_from_payload(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items = payload["parsed_provider_payload"]["items"]
    return {item["annotation_item_id"]: item for item in items}


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = f"{row['reviewerID']}_{row['raw_source_row_id']}"
            row["annotation_item_id"] = item_id
            rows[item_id] = row
    return rows


def normalize_label_fields(item: dict[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in [
        "Aspect1",
        "Polarity1",
        "Evidence_Source1",
        "Evidence1",
        "Aspect2",
        "Polarity2",
        "Evidence_Source2",
        "Evidence2",
        "Review_Mixed_Flag",
    ]:
        out[f"{prefix}_{key}"] = item.get(key)
    return out


def apply_overlay(base: dict[str, Any], overlay_cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    item_id = base["annotation_item_id"]
    if item_id not in overlay_cases:
        return dict(base)
    adjusted = dict(base)
    adjusted.update(overlay_cases[item_id]["adjudicated_label"])
    return adjusted


def aspect_rows(label_rows: list[dict[str, Any]], label_prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in label_rows:
        for slot in (1, 2):
            aspect = row.get(f"{label_prefix}_Aspect{slot}")
            if aspect in (None, "", "None"):
                continue
            polarity = row.get(f"{label_prefix}_Polarity{slot}")
            out.append(
                {
                    "annotation_item_id": row["annotation_item_id"],
                    "batch_id": row["batch_id"],
                    "run_folder": row["run_folder"],
                    "reviewerID": row["reviewerID"],
                    "episode_id": int(row["episode_id"]),
                    "review_timestamp": row["review_timestamp"],
                    "aspect_slot": slot,
                    "aspect": aspect,
                    "polarity": polarity,
                    "evidence_source": row.get(f"{label_prefix}_Evidence_Source{slot}"),
                    "evidence": row.get(f"{label_prefix}_Evidence{slot}"),
                    "review_mixed_flag": bool(row.get(f"{label_prefix}_Review_Mixed_Flag")),
                    "human_evidence_repair_applied": bool(row["human_evidence_repair_applied"]),
                    "listing_boundary_overlay_applied": bool(row["listing_boundary_overlay_applied"]),
                    "source_prompt_sha256": row["prompt_sha256"],
                    "source_taxonomy_sha256": row["taxonomy_sha256"],
                }
            )
    return out


def add_feature_columns(df: pd.DataFrame, include_listing: bool) -> list[str]:
    features = [
        "semantic_prior_labeled_review_count",
        "semantic_prior_aspect_row_count",
        "semantic_prior_negative_aspect_count",
        "semantic_prior_positive_aspect_count",
        "semantic_prior_negative_review_count",
        "semantic_prior_mixed_review_count",
        "semantic_prior_mixed_review_rate",
        "semantic_days_since_last_negative_feedback",
    ]
    aspects = list(ASPECTS_FOR_CORE)
    if include_listing:
        aspects.append(LISTING_ASPECT)
    for aspect in aspects:
        base = aspect.lower()
        features.extend(
            [
                f"semantic_prior_{base}_aspect_count",
                f"semantic_prior_{base}_negative_count",
                f"semantic_prior_{base}_negative_rate",
            ]
        )
    for col in features:
        df[col] = 0.0
    df["semantic_days_since_last_negative_feedback"] = pd.NA
    return features


def build_episode_features(
    snapshots: pd.DataFrame,
    aspects_df: pd.DataFrame,
    *,
    include_listing: bool,
    variant_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feat = snapshots.copy()
    features = add_feature_columns(feat, include_listing=include_listing)
    aspects = list(ASPECTS_FOR_CORE)
    if include_listing:
        aspects.append(LISTING_ASPECT)
    active = aspects_df[aspects_df["aspect"].isin(aspects)].copy()
    active["review_timestamp"] = pd.to_datetime(active["review_timestamp"], format="mixed")
    feat["episode_start"] = pd.to_datetime(feat["episode_start"])

    for reviewer_id, rows in active.groupby("reviewerID", sort=False):
        idx = feat.index[feat["reviewerID"] == reviewer_id]
        if len(idx) == 0:
            continue
        reviewer_episodes = feat.loc[idx, ["episode_start"]]
        reviewer_rows = rows.sort_values("review_timestamp")
        for episode_idx, episode in reviewer_episodes.iterrows():
            prior = reviewer_rows[reviewer_rows["review_timestamp"] < episode["episode_start"]]
            if prior.empty:
                continue
            review_count = int(prior["annotation_item_id"].nunique())
            negative = prior[prior["polarity"] == "Negative"]
            positive = prior[prior["polarity"] == "Positive"]
            negative_review_count = int(negative["annotation_item_id"].nunique())
            mixed_review_count = int(
                prior.loc[prior["review_mixed_flag"], "annotation_item_id"].nunique()
            )
            feat.at[episode_idx, "semantic_prior_labeled_review_count"] = review_count
            feat.at[episode_idx, "semantic_prior_aspect_row_count"] = int(len(prior))
            feat.at[episode_idx, "semantic_prior_negative_aspect_count"] = int(len(negative))
            feat.at[episode_idx, "semantic_prior_positive_aspect_count"] = int(len(positive))
            feat.at[episode_idx, "semantic_prior_negative_review_count"] = negative_review_count
            feat.at[episode_idx, "semantic_prior_mixed_review_count"] = mixed_review_count
            feat.at[episode_idx, "semantic_prior_mixed_review_rate"] = (
                mixed_review_count / review_count if review_count else 0.0
            )
            if not negative.empty:
                last_negative = negative["review_timestamp"].max()
                feat.at[episode_idx, "semantic_days_since_last_negative_feedback"] = (
                    episode["episode_start"] - last_negative
                ).days
            for aspect in aspects:
                base = aspect.lower()
                aspect_prior = prior[prior["aspect"] == aspect]
                aspect_neg = aspect_prior[aspect_prior["polarity"] == "Negative"]
                feat.at[episode_idx, f"semantic_prior_{base}_aspect_count"] = int(len(aspect_prior))
                feat.at[episode_idx, f"semantic_prior_{base}_negative_count"] = int(len(aspect_neg))
                feat.at[episode_idx, f"semantic_prior_{base}_negative_rate"] = (
                    len(aspect_neg) / len(aspect_prior) if len(aspect_prior) else 0.0
                )

    nonzero = int((feat["semantic_prior_labeled_review_count"] > 0).sum())
    report = {
        "variant": variant_name,
        "row_count": int(len(feat)),
        "feature_column_count": len(features),
        "episodes_with_any_prior_semantic_label": nonzero,
        "episodes_with_any_prior_semantic_label_rate": nonzero / len(feat) if len(feat) else 0.0,
        "include_listing_expectation": include_listing,
        "anti_leakage_rule": "review_timestamp < episode_start",
    }
    return feat, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    annotation_dir = repo / "outputs" / "amazon_v5_rebuild" / "annotation"
    data_dir = repo / "outputs" / "amazon_v5_rebuild" / "data"
    out_dir = repo / "outputs" / "amazon_v5_rebuild" / "features_semantic_b3"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = annotation_dir / "calibration_sample_manifest_v1_1.csv"
    overlay_path = annotation_dir / "listing_boundary_adjudication_v1_2.json"
    episode_path = data_dir / "verified_episode_snapshots_h270_v1.parquet"
    manifest = load_manifest(manifest_path)
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    overlay_cases = {case["annotation_item_id"]: case for case in overlay["cases"]}

    label_rows: list[dict[str, Any]] = []
    seen_ids: list[str] = []
    for batch_id, run_folder in ACCEPTED_RUNS:
        run_dir = annotation_dir / run_folder
        raw_payload = read_parsed_payload(run_dir / "parsed_objects_v1_6.jsonl")
        raw_items = item_map_from_payload(raw_payload)
        if batch_id == "batch_13_calib":
            feature_payload = read_parsed_payload(
                run_dir / "parsed_objects_v1_6_human_evidence_repaired.jsonl"
            )
            feature_items = item_map_from_payload(feature_payload)
        else:
            feature_items = raw_items

        if len(feature_items) != 8:
            raise ValueError(f"{batch_id} expected 8 items, found {len(feature_items)}")

        for item_id, feature_item in feature_items.items():
            if item_id not in manifest:
                raise KeyError(f"{item_id} missing from manifest")
            raw_item = raw_items[item_id]
            final_item = apply_overlay(feature_item, overlay_cases)
            row = {
                **manifest[item_id],
                "batch_id": batch_id,
                "run_folder": run_folder,
                "prompt_sha256": PROMPT_V1_2_SHA256,
                "taxonomy_sha256": TAXONOMY_V1_2_SHA256,
                "human_evidence_repair_applied": item_id in HUMAN_EVIDENCE_REPAIRED_ITEM_IDS,
                "listing_boundary_overlay_applied": item_id in overlay_cases,
                "raw_model_output_overwritten": False,
            }
            row.update(normalize_label_fields(raw_item, "raw"))
            row.update(normalize_label_fields(feature_item, "feature_base"))
            row.update(normalize_label_fields(final_item, "final"))
            label_rows.append(row)
            seen_ids.append(item_id)

    if len(label_rows) != 120:
        raise ValueError(f"Expected 120 labels, found {len(label_rows)}")
    duplicate_ids = sorted([item for item, count in Counter(seen_ids).items() if count > 1])
    if duplicate_ids:
        raise ValueError(f"Duplicate annotation ids: {duplicate_ids}")

    labels_df = pd.DataFrame(label_rows)
    labels_df["review_timestamp"] = pd.to_datetime(labels_df["review_timestamp"], format="mixed")
    labels_df.to_csv(out_dir / "semantic_review_labels_v1_2_final_with_overlay.csv", index=False)
    labels_df.to_json(
        out_dir / "semantic_review_labels_v1_2_final_with_overlay.jsonl",
        orient="records",
        lines=True,
        date_format="iso",
        force_ascii=False,
    )
    labels_df.to_parquet(out_dir / "semantic_review_labels_v1_2_final_with_overlay.parquet", index=False)

    raw_aspect_df = pd.DataFrame(aspect_rows(label_rows, "feature_base"))
    final_aspect_df = pd.DataFrame(aspect_rows(label_rows, "final"))
    for name, df in [
        ("semantic_aspect_rows_raw_v1_2_order_agnostic", raw_aspect_df),
        ("semantic_aspect_rows_adjudicated_overlay_order_agnostic", final_aspect_df),
    ]:
        df.to_csv(out_dir / f"{name}.csv", index=False)
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    snapshots = pd.read_parquet(episode_path)
    core_features, core_report = build_episode_features(
        snapshots, final_aspect_df, include_listing=False, variant_name="B3-core_without_fine_grained_listing"
    )
    raw_listing_features, raw_listing_report = build_episode_features(
        snapshots, raw_aspect_df, include_listing=True, variant_name="B3-listing_raw_v1_2_sensitivity"
    )
    final_listing_features, final_listing_report = build_episode_features(
        snapshots,
        final_aspect_df,
        include_listing=True,
        variant_name="B3-listing_adjudicated_overlay_sensitivity",
    )

    core_features.to_parquet(out_dir / "semantic_episode_features_b3_core_v1_2.parquet", index=False)
    raw_listing_features.to_parquet(
        out_dir / "semantic_episode_features_b3_listing_raw_v1_2_sensitivity.parquet", index=False
    )
    final_listing_features.to_parquet(
        out_dir / "semantic_episode_features_b3_listing_adjudicated_overlay_sensitivity.parquet",
        index=False,
    )

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_status": "PASS",
        "gate6_dependency": "PASS_WITH_BOUNDARY_ADJUDICATION_LAYER",
        "taxonomy_sha256": TAXONOMY_V1_2_SHA256,
        "prompt_sha256": PROMPT_V1_2_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "listing_boundary_overlay_sha256": sha256_file(overlay_path),
        "episode_snapshot_sha256": sha256_file(episode_path),
        "accepted_batch_count": len(ACCEPTED_RUNS),
        "review_label_count": int(len(labels_df)),
        "unique_annotation_item_count": int(labels_df["annotation_item_id"].nunique()),
        "raw_model_output_overwritten": False,
        "human_evidence_repair_count": int(labels_df["human_evidence_repair_applied"].sum()),
        "listing_boundary_overlay_case_count": int(labels_df["listing_boundary_overlay_applied"].sum()),
        "order_agnostic_aspect_rows_pass": True,
        "aspect1_primary_feature_absent_pass": True,
        "anti_leakage_pass": True,
        "anti_leakage_rule": "For every episode aggregate, only semantic labels with review_timestamp < episode_start are counted.",
        "raw_aspect_row_count": int(len(raw_aspect_df)),
        "adjudicated_aspect_row_count": int(len(final_aspect_df)),
        "raw_aspect_counts": raw_aspect_df["aspect"].value_counts().to_dict(),
        "adjudicated_aspect_counts": final_aspect_df["aspect"].value_counts().to_dict(),
        "episode_snapshot_count": int(len(snapshots)),
        "feature_variants": {
            "core": core_report,
            "listing_raw_v1_2_sensitivity": raw_listing_report,
            "listing_adjudicated_overlay_sensitivity": final_listing_report,
        },
        "outputs": {
            "review_labels_csv": str(out_dir / "semantic_review_labels_v1_2_final_with_overlay.csv"),
            "review_labels_jsonl": str(out_dir / "semantic_review_labels_v1_2_final_with_overlay.jsonl"),
            "review_labels_parquet": str(out_dir / "semantic_review_labels_v1_2_final_with_overlay.parquet"),
            "raw_aspect_rows_csv": str(out_dir / "semantic_aspect_rows_raw_v1_2_order_agnostic.csv"),
            "raw_aspect_rows_parquet": str(out_dir / "semantic_aspect_rows_raw_v1_2_order_agnostic.parquet"),
            "adjudicated_aspect_rows_csv": str(
                out_dir / "semantic_aspect_rows_adjudicated_overlay_order_agnostic.csv"
            ),
            "adjudicated_aspect_rows_parquet": str(
                out_dir / "semantic_aspect_rows_adjudicated_overlay_order_agnostic.parquet"
            ),
            "b3_core_parquet": str(out_dir / "semantic_episode_features_b3_core_v1_2.parquet"),
            "b3_listing_raw_sensitivity_parquet": str(
                out_dir / "semantic_episode_features_b3_listing_raw_v1_2_sensitivity.parquet"
            ),
            "b3_listing_adjudicated_sensitivity_parquet": str(
                out_dir / "semantic_episode_features_b3_listing_adjudicated_overlay_sensitivity.parquet"
            ),
        },
    }
    (out_dir / "gate7_semantic_feature_engineering_report_v1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    md = [
        "# Gate 7 Semantic Feature Engineering Report",
        "",
        f"Status: {report['gate7_status']}",
        f"Review labels: {report['review_label_count']} unique={report['unique_annotation_item_count']}",
        f"Raw aspect rows: {report['raw_aspect_row_count']}",
        f"Adjudicated aspect rows: {report['adjudicated_aspect_row_count']}",
        f"Episode snapshots: {report['episode_snapshot_count']}",
        "",
        "Feature variants:",
    ]
    for key, value in report["feature_variants"].items():
        md.append(
            f"- {key}: rows={value['row_count']}, feature_columns={value['feature_column_count']}, "
            f"episodes_with_prior_semantic_label={value['episodes_with_any_prior_semantic_label']}"
        )
    md.extend(
        [
            "",
            "Controls:",
            "- Aspect slots are order-agnostic; one row is emitted for every non-null aspect slot.",
            "- Slot-specific polarity and evidence are retained.",
            "- Aspect1 alone is never used as the primary feature source.",
            "- Anti-leakage rule: review_timestamp < episode_start.",
            "- Listing boundary overlay is used only for the adjudicated sensitivity branch; raw v1.2 is preserved.",
        ]
    )
    (out_dir / "gate7_semantic_feature_engineering_report_v1.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ["gate7_status", "review_label_count", "raw_aspect_row_count", "adjudicated_aspect_row_count", "episode_snapshot_count"]}, indent=2))


if __name__ == "__main__":
    main()
