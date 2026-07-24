#!/usr/bin/env python
"""Gate 7.6 v3: outcome-blind cohort swap for frozen 8-item batch contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SEED = 20260706
SPLIT_TARGETS = {"train": 6000, "validation": 2000, "development_evaluation": 2000}
ITEMS_PER_REQUEST = 8
ENRICHMENT_PER_THEME = 1000
RESERVE_PER_SPLIT = 3000

ENRICHMENT_THEMES = {
    "Delivery_Fulfillment": ["shipping", "delivery", "delivered", "arrived", "never received", "missing", "sent wrong", "wrong item"],
    "Customer_Service_Returns": ["refund", "return", "returned", "replacement", "customer service", "seller", "exchange"],
    "Packaging_Presentation": ["packaging", "package", "box", "case", "jewel case", "cracked case", "broken case"],
    "Product_Performance_Usability": ["skip", "skips", "skipping", "won't play", "does not play", "download", "track", "defective"],
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


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def df_hash(df: pd.DataFrame, cols: list[str]) -> str:
    data = df.sort_values(cols)[cols].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def assign_split(df: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["episode_start"] = pd.to_datetime(out["episode_start"])
    out["split"] = "unassigned"
    ordered_idx = out.sort_values(["episode_start", "episode_id"]).index.to_list()
    cursor = 0
    for _, row in split_manifest.iterrows():
        row_count = int(row["row_count"])
        out.loc[ordered_idx[cursor : cursor + row_count], "split"] = row["split"]
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


def candidate_edges(membership: pd.DataFrame, episodes: pd.DataFrame) -> pd.DataFrame:
    selected = episodes[["reviewerID", "episode_id", "episode_start", "split"]].rename(
        columns={"episode_id": "selected_episode_id", "episode_start": "selected_episode_start", "split": "selected_split"}
    )
    edges = membership.merge(selected, on="reviewerID", how="inner")
    edges = edges[edges["review_timestamp"] < edges["selected_episode_start"]].copy()
    edges["is_strict_prior"] = True
    return edges.rename(columns={"selected_episode_start": "episode_start"})


def edge_sets(edges: pd.DataFrame) -> dict[int, set[str]]:
    return {
        int(ep): set(group["annotation_item_id"])
        for ep, group in edges.groupby("selected_episode_id", sort=False)
    }


def find_swap(
    current: pd.DataFrame,
    reserves: pd.DataFrame,
    sets: dict[int, set[str]],
    current_count: Counter[str],
    current_unique_n: int,
) -> dict[str, Any] | None:
    selected_order = current.sort_values(["split", "episode_start", "episode_id"])
    reserve_order = reserves.sort_values(["split", "reserve_order"])
    for split in SPLIT_TARGETS:
        selected_split = selected_order[selected_order["split"] == split]
        reserve_split = reserve_order[reserve_order["split"] == split]
        for _, remove_row in selected_split.iterrows():
            remove_ep = int(remove_row["episode_id"])
            remove_set = sets.get(remove_ep, set())
            removed_unique = sum(1 for item in remove_set if current_count[item] == 1)
            remove_set_lookup = remove_set
            for _, add_row in reserve_split.iterrows():
                add_ep = int(add_row["episode_id"])
                add_set = sets.get(add_ep, set())
                added_new = sum(
                    1
                    for item in add_set
                    if current_count.get(item, 0) == 0
                    or (current_count.get(item, 0) == 1 and item in remove_set_lookup)
                )
                new_n = current_unique_n - removed_unique + added_new
                if new_n % ITEMS_PER_REQUEST == 0:
                    return {
                        "split": split,
                        "removed_episode_id": remove_ep,
                        "added_episode_id": add_ep,
                        "removed_unique_review_count": int(removed_unique),
                        "added_new_unique_review_count": int(added_new),
                        "unique_count_before": int(current_unique_n),
                        "unique_count_after": int(new_n),
                    }
    return None


def apply_swap(current: pd.DataFrame, reserves: pd.DataFrame, swap: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    removed = int(swap["removed_episode_id"])
    added = int(swap["added_episode_id"])
    add_row = reserves[reserves["episode_id"] == added]
    current_next = pd.concat([current[current["episode_id"] != removed], add_row], ignore_index=True)
    reserves_next = reserves[reserves["episode_id"] != added].copy()
    return current_next, reserves_next


def keyword_pattern(words: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(re.escape(w) for w in words), flags=re.IGNORECASE)


def make_item_batch_manifest(reviews: pd.DataFrame, prefix: str) -> pd.DataFrame:
    ordered = reviews.sort_values(["reviewerID", "review_timestamp", "annotation_item_id"]).reset_index(drop=True)
    if len(ordered) % ITEMS_PER_REQUEST != 0:
        raise ValueError(f"{prefix} review count {len(ordered)} is not divisible by {ITEMS_PER_REQUEST}")
    rows = []
    for idx, row in ordered.iterrows():
        batch_idx = idx // ITEMS_PER_REQUEST + 1
        rows.append(
            {
                "campaign_run_id": "V12EXP_PRIMARY_V3" if prefix == "P" else "V12EXP_ENRICH_V3",
                "batch_id": f"{prefix}{batch_idx:06d}",
                "attempt_id": "A001",
                "item_position": idx % ITEMS_PER_REQUEST + 1,
                "annotation_item_id": row["annotation_item_id"],
                "reviewerID": row["reviewerID"],
                "review_timestamp": row["review_timestamp"],
            }
        )
    return pd.DataFrame(rows)


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
        columns=["reviewerID", "episode_id", "review_timestamp", "asin", "overall", "reviewText", "summary", "raw_source_row_id"],
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

    initial_parts = []
    reserve_parts = []
    for split, target in SPLIT_TARGETS.items():
        part = eligible[eligible["split"] == split]
        initial = part.sample(n=target, random_state=SEED)
        reserves = part[~part["episode_id"].isin(initial["episode_id"])].sample(
            frac=1, random_state=SEED + 77
        ).head(RESERVE_PER_SPLIT)
        reserves = reserves.copy()
        reserves["reserve_order"] = range(1, len(reserves) + 1)
        initial_parts.append(initial)
        reserve_parts.append(reserves)
    current = pd.concat(initial_parts, ignore_index=True)
    reserves = pd.concat(reserve_parts, ignore_index=True)
    original_hash = df_hash(current, ["split", "episode_start", "episode_id", "reviewerID"])
    reserve_order_hash = df_hash(reserves, ["split", "reserve_order", "episode_id", "reviewerID"])

    all_candidates = pd.concat([current, reserves], ignore_index=True)
    all_edges = candidate_edges(membership, all_candidates)
    sets = edge_sets(all_edges)
    selected_ids = set(current["episode_id"].astype(int))
    selected_sets = [sets[int(ep)] for ep in selected_ids]
    count = Counter(item for items in selected_sets for item in items)
    initial_unique = len(count)
    swaps: list[dict[str, Any]] = []

    if initial_unique % ITEMS_PER_REQUEST != 0:
        swap = find_swap(current, reserves, sets, count, initial_unique)
        if swap is None:
            raise RuntimeError("No deterministic single swap found within reserve candidates")
        current, reserves = apply_swap(current, reserves, swap)
        swaps.append(swap)

    final_edges = candidate_edges(membership, current)
    final_edges["episode_start"] = pd.to_datetime(final_edges["episode_start"])
    final_edges["review_timestamp"] = pd.to_datetime(final_edges["review_timestamp"])
    final_edges = final_edges.sort_values(["selected_episode_id", "review_timestamp", "annotation_item_id"])
    final_unique_reviews = final_edges.drop_duplicates("annotation_item_id").copy()
    final_unique_reviews["annotation_scope"] = "primary_b3_outcome_blind_random_cohort_v3"
    final_unique_n = int(final_unique_reviews["annotation_item_id"].nunique())
    if final_unique_n % ITEMS_PER_REQUEST != 0:
        raise RuntimeError(f"Final unique review count {final_unique_n} is not divisible by {ITEMS_PER_REQUEST}")

    coverage = final_edges.groupby("selected_episode_id")["annotation_item_id"].nunique()
    current = current.copy()
    current["prior_reviews_in_annotation_manifest"] = current["episode_id"].map(coverage).fillna(0).astype(int)
    final_hash = df_hash(current, ["split", "episode_start", "episode_id", "reviewerID"])

    primary_ids = set(final_unique_reviews["annotation_item_id"])
    enrichment_pool = membership[~membership["annotation_item_id"].isin(primary_ids)].copy()
    text = enrichment_pool["summary"].fillna("").astype(str) + " " + enrichment_pool["reviewText"].fillna("").astype(str)
    enrichment_parts = []
    used_ids: set[str] = set()
    for idx, (theme, keywords) in enumerate(ENRICHMENT_THEMES.items()):
        mask = text.str.contains(keyword_pattern(keywords), regex=True, na=False)
        candidates = enrichment_pool[mask & ~enrichment_pool["annotation_item_id"].isin(used_ids)].copy()
        sample = candidates.sample(n=min(args.enrichment_per_theme, len(candidates)), random_state=SEED + idx)
        sample["annotation_scope"] = "targeted_enrichment_not_for_primary_predictive_claim_v3"
        sample["enrichment_theme"] = theme
        sample["enrichment_keywords"] = ";".join(keywords)
        used_ids.update(sample["annotation_item_id"].tolist())
        enrichment_parts.append(sample)
    enrichment = pd.concat(enrichment_parts, ignore_index=True)

    episode_manifest_path = out_dir / "gate7_6_v3_primary_b3_episode_cohort_manifest.csv"
    review_manifest_path = out_dir / "gate7_6_v3_primary_b3_annotation_review_manifest.csv"
    enrichment_path = out_dir / "gate7_6_v3_targeted_enrichment_review_manifest.csv"
    membership_out_path = out_dir / "episode_annotation_membership_v3.parquet"
    primary_batch_path = out_dir / "gate7_7_v3_primary_batch_manifest.csv"
    enrichment_batch_path = out_dir / "gate7_7_v3_enrichment_batch_manifest.csv"
    report_path = out_dir / "gate7_6_annotation_coverage_design_report_v3.json"
    md_path = out_dir / "gate7_6_annotation_coverage_design_report_v3.md"
    gate77_path = out_dir / "gate7_7_provider_execution_feasibility_report_v3.json"
    gate77_md_path = out_dir / "gate7_7_provider_execution_feasibility_report_v3.md"

    current.to_csv(episode_manifest_path, index=False)
    final_unique_reviews.to_csv(review_manifest_path, index=False)
    enrichment.to_csv(enrichment_path, index=False)
    final_edges[["selected_episode_id", "reviewerID", "annotation_item_id", "review_timestamp", "episode_start", "is_strict_prior"]].to_parquet(
        membership_out_path, index=False
    )
    primary_batches = make_item_batch_manifest(final_unique_reviews, "P")
    enrichment_batches = make_item_batch_manifest(enrichment, "E")
    primary_batches.to_csv(primary_batch_path, index=False)
    enrichment_batches.to_csv(enrichment_batch_path, index=False)

    split_summary = (
        current.groupby("split")
        .agg(
            selected_episode_count=("episode_id", "size"),
            selected_customer_count=("reviewerID", "nunique"),
            episode_with_at_least_one_prior_review_to_annotate=("prior_reviews_in_annotation_manifest", lambda s: int((s > 0).sum())),
            median_prior_reviews_to_annotate=("prior_reviews_in_annotation_manifest", "median"),
            p95_prior_reviews_to_annotate=("prior_reviews_in_annotation_manifest", lambda s: float(s.quantile(0.95))),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_6_status": "DESIGN_READY_AFTER_PRE_ANNOTATION_FIXES",
        "gate7_6a_outcome_blind_cohort_design": "PASS",
        "gate7_6b_temporal_annotation_eligibility": "PASS",
        "gate7_6c_b3_train_support_allocation": "PASS_REVISED_V2_RETAINED",
        "gate7_6d_episode_review_lineage_manifest": "PASS_ADDED_V3",
        "gate7_6e_batch_divisibility_under_frozen_8_item_contract": "PASS_REVISED_V3",
        "gate7_7_provider_scale_feasibility": "HOLD_NOT_READY_FOR_PROVIDER_EXECUTION",
        "frozen_provider_contract": {"items_per_request": ITEMS_PER_REQUEST, "prompt_allows_partial_batch": False, "dummy_padding_allowed": False},
        "primary_b3_random_cohort_v3": {
            "random_seed": SEED,
            "split_targets": SPLIT_TARGETS,
            "selected_episode_count": int(len(current)),
            "selected_customer_count": int(current["reviewerID"].nunique()),
            "unique_prior_reviews_to_annotate_once": final_unique_n,
            "unique_prior_reviews_divisible_by_8": final_unique_n % ITEMS_PER_REQUEST == 0,
            "primary_request_count": final_unique_n // ITEMS_PER_REQUEST,
            "episode_review_membership_edge_count": int(len(final_edges)),
            "episodes_with_at_least_one_prior_review_to_annotate": int((current["prior_reviews_in_annotation_manifest"] > 0).sum()),
            "split_summary": split_summary,
        },
        "deterministic_swap_log": {
            "original_cohort_hash": original_hash,
            "reserve_order_hash": reserve_order_hash,
            "initial_unique_prior_reviews_to_annotate_once": int(initial_unique),
            "initial_mod_8": int(initial_unique % ITEMS_PER_REQUEST),
            "swap_count": len(swaps),
            "swaps": swaps,
            "final_cohort_hash": final_hash,
            "final_unique_prior_reviews_to_annotate_once": final_unique_n,
            "final_mod_8": int(final_unique_n % ITEMS_PER_REQUEST),
        },
        "targeted_enrichment_v3": {
            "selected_review_count": int(enrichment["annotation_item_id"].nunique()),
            "request_count": int(len(enrichment) // ITEMS_PER_REQUEST),
            "not_for_primary_predictive_effect_claim": True,
        },
        "lineage_policy": {
            "feature_builder_join_policy": "Use reviewerID + review_timestamp < episode_start, or episode_annotation_membership_v3.parquet.",
            "do_not_join_primary_features_by_unique_review_manifest_selected_episode_id": True,
        },
        "outputs": {
            "primary_episode_manifest_v3": str(episode_manifest_path),
            "primary_review_manifest_v3": str(review_manifest_path),
            "targeted_enrichment_manifest_v3": str(enrichment_path),
            "episode_annotation_membership_v3": str(membership_out_path),
            "primary_batch_manifest_v3": str(primary_batch_path),
            "enrichment_batch_manifest_v3": str(enrichment_batch_path),
            "design_report_v3": str(report_path),
        },
        "source_hashes": {
            "verified_episode_snapshots_h270_v1": sha256_file(snapshot_path),
            "verified_episode_review_membership_v1": sha256_file(membership_path),
            "temporal_development_split_manifest_v1": sha256_file(split_path),
        },
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gate 7.6 Annotation Coverage Design v3",
                "",
                f"Status: {report['gate7_6_status']}",
                f"Gate 7.6E: {report['gate7_6e_batch_divisibility_under_frozen_8_item_contract']}",
                f"Unique primary reviews: {final_unique_n}",
                f"Primary requests: {final_unique_n // ITEMS_PER_REQUEST}",
                f"Episode-review edges: {len(final_edges)}",
                f"Swap count: {len(swaps)}",
                "",
                "No provider call was made.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    gate77 = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_7_status": "HOLD_NOT_READY_FOR_PROVIDER_EXECUTION",
        "campaign_architecture_locked": True,
        "campaign_run_id_primary": "V12EXP_PRIMARY_V3",
        "campaign_run_id_enrichment": "V12EXP_ENRICH_V3",
        "batch_id_strategy": {
            "primary": f"P000001 ... P{final_unique_n // ITEMS_PER_REQUEST:06d}",
            "attempt_id_initial": "A001",
        },
        "failed_batch_policy": "same batch_id + new attempt_id A002/A003; never overwrite A001 raw, parsed, validation artifacts",
        "folder_strategy": "campaign-level run with immutable per-batch attempt folders",
        "deterministic_batch_manifests": {
            "primary_batch_manifest_v3": str(primary_batch_path),
            "enrichment_batch_manifest_v3": str(enrichment_batch_path),
        },
        "provider_execution_plan": {
            "items_per_request": ITEMS_PER_REQUEST,
            "primary_review_count": final_unique_n,
            "primary_request_count": final_unique_n // ITEMS_PER_REQUEST,
            "enrichment_review_count": int(enrichment["annotation_item_id"].nunique()),
            "enrichment_request_count": int(enrichment["annotation_item_id"].nunique()) // ITEMS_PER_REQUEST,
            "total_request_count": final_unique_n // ITEMS_PER_REQUEST + int(enrichment["annotation_item_id"].nunique()) // ITEMS_PER_REQUEST,
            "primary_and_enrichment_outputs_separate": True,
        },
        "still_pending_before_provider_call": {
            "quota_rate_limit": "PENDING",
            "cost_ceiling": "PENDING",
            "checkpoint_resume_implementation_review": "PENDING",
            "sampling_audit_schedule_during_expansion": "PENDING",
            "operator_approval_for_live_provider_campaign": "PENDING",
        },
        "source_gate7_6_v3_report_sha256": sha256_file(report_path),
    }
    gate77_path.write_text(json.dumps(gate77, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")
    gate77_md_path.write_text(
        "\n".join(
            [
                "# Gate 7.7 Provider Execution Feasibility v3",
                "",
                f"Status: {gate77['gate7_7_status']}",
                "Architecture locked, provider execution still blocked.",
                f"Primary requests: {gate77['provider_execution_plan']['primary_request_count']}",
                f"Enrichment requests: {gate77['provider_execution_plan']['enrichment_request_count']}",
                f"Total requests: {gate77['provider_execution_plan']['total_request_count']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "gate7_6e": report["gate7_6e_batch_divisibility_under_frozen_8_item_contract"],
        "initial_unique_reviews": initial_unique,
        "final_unique_reviews": final_unique_n,
        "final_mod_8": final_unique_n % ITEMS_PER_REQUEST,
        "primary_request_count": final_unique_n // ITEMS_PER_REQUEST,
        "episode_review_edges": len(final_edges),
        "swap_count": len(swaps),
        "gate7_7_status": gate77["gate7_7_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
