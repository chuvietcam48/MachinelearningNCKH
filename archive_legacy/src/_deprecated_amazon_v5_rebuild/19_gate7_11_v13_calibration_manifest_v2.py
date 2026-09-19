#!/usr/bin/env python
"""Build Gate 7.11 v1.3 calibration manifest v2 from immutable batch payloads."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


OUT_DIR = Path("outputs/amazon_v5_rebuild/annotation/v1_3_calibration_v2")
PRIMARY_BATCH_ROOT = Path("outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/batches")
HISTORICAL_CALIBRATION_ROOT = Path("outputs/amazon_v5_rebuild/annotation")
PRIMARY_REVIEW_MANIFEST = Path(
    "outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/manifests/gate7_6_v3_primary_b3_annotation_review_manifest.csv"
)
GATE6_AUDIT = Path("outputs/amazon_v5_rebuild/annotation/v1_2_gate6_human_audit_30_seed_20260705.json")
TAXONOMY_V13 = Path("outputs/amazon_v5_rebuild/annotation/taxonomy_v1_3_draft_for_calibration.md")
PROMPT_V13 = Path("outputs/amazon_v5_rebuild/annotation/gemini_semantic_prompt_template_v1_3_draft_for_calibration.md")
CANONICAL_SCHEMA = Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json")
PROVIDER_SCHEMA = Path(
    "outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/manifests/provider_response_schema_frozen_v1_2.json"
)
VALIDATOR_SPEC = Path("outputs/amazon_v5_rebuild/annotation/semantic_validator_spec_v1_6.md")
LIVE_EXECUTOR_SOURCE = Path("src/amazon_v5_rebuild/16_v12exp_primary_pilot_live_executor.py")
POLICY_MANIFEST = Path("outputs/amazon_v5_rebuild/annotation/v1_3_policy_manifest.json")

SEED = 20260706
P000007_ID = "A102XSQH2IW56B_3720498"
KEYWORDS = ["production", "sound", "mix", "mastering", "recording", "style", "genre", "raw", "garage", "audio"]
PRIOR_MONITORING_IDS = [
    "A2LQQKWT4NROUW_3741396",
    "AZQHNUHSC3OWC_3985858",
    "AE37E22PLKYKN_4302748",
    "A2F26XZ573IAZW_2625034",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_new(path: Path, data: bytes) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def load_batch_payload_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for provider_path in sorted(PRIMARY_BATCH_ROOT.glob("P*/A001/provider_input.json")):
        batch_id = provider_path.parents[1].name
        provider_sha = sha256_file(provider_path)
        rows = json.loads(provider_path.read_text(encoding="utf-8-sig"))
        for pos, row in enumerate(rows, start=1):
            item_id = row["annotation_item_id"]
            if item_id not in lookup:
                lookup[item_id] = {
                    "annotation_item_id": item_id,
                    "summary": row.get("summary", "") or "",
                    "reviewText": row.get("reviewText", "") or "",
                    "source_batch_id": batch_id,
                    "source_item_position": pos,
                    "source_provider_input_path": str(provider_path),
                    "source_provider_input_sha256": provider_sha,
                }
    for snapshot_path in sorted(HISTORICAL_CALIBRATION_ROOT.glob("phase2b1_v16_live_batch_*/batch_input_snapshot_*_calib.json")):
        snapshot_sha = sha256_file(snapshot_path)
        rows = json.loads(snapshot_path.read_text(encoding="utf-8-sig"))
        for pos, row in enumerate(rows, start=1):
            item_id = row["annotation_item_id"]
            if item_id not in lookup:
                lookup[item_id] = {
                    "annotation_item_id": item_id,
                    "summary": row.get("summary", "") or "",
                    "reviewText": row.get("reviewText", "") or "",
                    "source_batch_id": row.get("batch_id"),
                    "source_item_position": pos,
                    "source_provider_input_path": str(snapshot_path),
                    "source_provider_input_sha256": snapshot_sha,
                    "source_run_id": row.get("run_id"),
                }
    return lookup


def load_primary_metadata() -> dict[str, dict[str, str]]:
    if not PRIMARY_REVIEW_MANIFEST.exists():
        return {}
    with PRIMARY_REVIEW_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["annotation_item_id"]: row for row in csv.DictReader(f)}


def make_item(row: dict[str, Any], metadata: dict[str, str], split: str, reason: str, position: int) -> dict[str, Any]:
    payload = {
        "annotation_item_id": row["annotation_item_id"],
        "summary": row.get("summary", "") or "",
        "reviewText": row.get("reviewText", "") or "",
    }
    batch_id = f"V13CAL_B{((position - 1) // 8) + 1:03d}"
    return {
        "calibration_item_id": f"V13CAL_{position:03d}",
        "calibration_batch_id": batch_id,
        "item_position": position,
        "annotation_item_id": row["annotation_item_id"],
        "reviewerID": metadata.get("reviewerID"),
        "review_timestamp": metadata.get("review_timestamp"),
        "asin": metadata.get("asin"),
        "selection_split": split,
        "selection_reason": reason,
        "source_batch_id": row.get("source_batch_id"),
        "source_item_position": row.get("source_item_position"),
        "source_provider_input_path": row.get("source_provider_input_path"),
        "source_provider_input_sha256": row.get("source_provider_input_sha256"),
        "provider_input": payload,
        "provider_input_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def main() -> None:
    payload_lookup = load_batch_payload_lookup()
    metadata_lookup = load_primary_metadata()
    gate6 = json.loads(GATE6_AUDIT.read_text(encoding="utf-8-sig"))
    gate6_ids = [row["annotation_item_id"] for row in gate6.get("rows", [])]

    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    replacement_log: list[dict[str, Any]] = []

    def add_id(item_id: str, split: str, reason: str) -> bool:
        if item_id in used:
            replacement_log.append({"annotation_item_id": item_id, "split": split, "reason": "duplicate_not_added"})
            return False
        row = payload_lookup.get(item_id)
        if not row:
            replacement_log.append({"annotation_item_id": item_id, "split": split, "reason": "payload_text_not_found"})
            return False
        used.add(item_id)
        selected.append(make_item(row, metadata_lookup.get(item_id, {}), split, reason, len(selected) + 1))
        return True

    for item_id in gate6_ids:
        if len([x for x in selected if x["selection_split"] == "A_old_regression_set"]) == 8:
            break
        if item_id in PRIOR_MONITORING_IDS:
            replacement_log.append({
                "annotation_item_id": item_id,
                "split": "A_old_regression_set",
                "reason": "reserved_for_C_prior_monitoring_boundary_cases",
            })
            continue
        add_id(item_id, "A_old_regression_set", "old_manual_audit_regression")

    add_id(P000007_ID, "B_p000007_sentinel_acceptance", "hidden_expected_label_acceptance_test")

    for item_id in PRIOR_MONITORING_IDS:
        add_id(item_id, "C_prior_monitoring_boundary_cases", "prior_monitoring_boundary_case")

    keyword_candidates = []
    for row in payload_lookup.values():
        if row["annotation_item_id"] in used:
            continue
        text = f"{row.get('summary','')} {row.get('reviewText','')}".lower()
        matched = [k for k in KEYWORDS if k in text]
        if matched:
            row = dict(row)
            row["matched_keywords"] = matched
            keyword_candidates.append(row)
    keyword_candidates.sort(key=lambda r: sha256_bytes((r["annotation_item_id"] + "keyword" + str(SEED)).encode("utf-8")))
    for row in keyword_candidates[:27]:
        used.add(row["annotation_item_id"])
        selected.append(make_item(row, metadata_lookup.get(row["annotation_item_id"], {}), "D_keyword_boundary_holdout", "keyword_selected_boundary_holdout", len(selected) + 1))

    random_candidates = sorted(payload_lookup.values(), key=lambda r: sha256_bytes((r["annotation_item_id"] + "random" + str(SEED)).encode("utf-8")))
    for row in random_candidates:
        if len([x for x in selected if x["selection_split"] == "E_random_regression_set"]) == 8:
            break
        if row["annotation_item_id"] not in used:
            used.add(row["annotation_item_id"])
            selected.append(make_item(row, metadata_lookup.get(row["annotation_item_id"], {}), "E_random_regression_set", "seeded_random_regression", len(selected) + 1))

    expected_counts = {
        "A_old_regression_set": 8,
        "B_p000007_sentinel_acceptance": 1,
        "C_prior_monitoring_boundary_cases": 4,
        "D_keyword_boundary_holdout": 27,
        "E_random_regression_set": 8,
    }
    actual_counts = {name: 0 for name in expected_counts}
    for item in selected:
        actual_counts[item["selection_split"]] += 1
    if actual_counts != expected_counts:
        raise RuntimeError(f"Split counts mismatch: {actual_counts}")
    if len(selected) != 48 or len({x["annotation_item_id"] for x in selected}) != 48:
        raise RuntimeError("Expected 48 unique calibration items")

    # Reassign positions after grouped selection to keep batch order stable.
    for idx, item in enumerate(selected, start=1):
        item["item_position"] = idx
        item["calibration_item_id"] = f"V13CAL_{idx:03d}"
        item["calibration_batch_id"] = f"V13CAL_B{((idx - 1) // 8) + 1:03d}"

    batches = []
    for batch_idx in range(6):
        batch_items = selected[batch_idx * 8:(batch_idx + 1) * 8]
        provider_payload = [item["provider_input"] for item in batch_items]
        batches.append({
            "calibration_batch_id": f"V13CAL_B{batch_idx + 1:03d}",
            "item_count": 8,
            "annotation_item_ids": [item["annotation_item_id"] for item in batch_items],
            "provider_input_sha256": sha256_bytes(canonical_json_bytes(provider_payload)),
            "planned_base_provider_requests": 1,
            "retry_policy": "no_blind_retry",
        })

    hidden_expected = {
        P000007_ID: {
            "split": "B_p000007_sentinel_acceptance",
            "expected_v1_3": {
                "Aspect1": "Domain_Experience",
                "Polarity1": "Positive",
                "Aspect2": None,
                "Polarity2": None,
                "Review_Mixed_Flag": False,
                "must_not_emit": ["Product_Condition_Quality Positive"],
            },
            "not_counted_as_generalization_evidence": True,
        }
    }

    manifest = {
        "gate": "Gate 7.11",
        "status": "MANIFEST_READY_NO_PROVIDER_CALLS",
        "policy_version": "v1.3",
        "manifest_version": "v2",
        "calibration_manifest_id": "V13CAL_MANIFEST_V2_20260706_SEED20260706",
        "sample_seed": SEED,
        "item_count": 48,
        "batch_count": 6,
        "batch_size": 8,
        "provider_calls_made": False,
        "provider_calls_allowed_now": False,
        "selection_splits": expected_counts,
        "analysis_groups_must_be_reported_separately": list(expected_counts),
        "schema_validator_locks": {
            "canonical_schema_path": str(CANONICAL_SCHEMA),
            "canonical_schema_sha256": sha256_file(CANONICAL_SCHEMA),
            "provider_projection_schema_path": str(PROVIDER_SCHEMA),
            "provider_projection_schema_sha256": sha256_file(PROVIDER_SCHEMA),
            "semantic_validator_spec_path": str(VALIDATOR_SPEC),
            "semantic_validator_spec_sha256": sha256_file(VALIDATOR_SPEC),
            "live_executor_validator_source_path": str(LIVE_EXECUTOR_SOURCE),
            "live_executor_validator_source_sha256": sha256_file(LIVE_EXECUTOR_SOURCE),
            "schema_policy": "reuse_v1_2_schema_projection_and_validator_no_new_schema_version",
        },
        "policy_locks": {
            "taxonomy_v1_3_sha256": sha256_file(TAXONOMY_V13),
            "prompt_template_v1_3_sha256": sha256_file(PROMPT_V13),
            "policy_manifest_sha256": sha256_file(POLICY_MANIFEST),
        },
        "pass_rules_locked_before_provider": {
            "raw_outputs_schema_evidence_unique_span_pass": "48/48",
            "p000007_sentinel": "Domain_Experience Positive only; mixed=false",
            "creative_style_double_count_errors_allowed": 0,
            "human_repair_allowed_in_calibration": False,
            "base_provider_requests_per_batch": 1,
            "blind_retry_allowed": False,
            "keyword_boundary_reported_separately": True,
            "do_not_report_single_accuracy_number": True,
            "metric_name": "agreement against one-person blind adjudication",
        },
        "batches": batches,
        "items": selected,
        "replacement_log": replacement_log,
    }

    human_rows = []
    for item in selected:
        human_rows.append({
            "calibration_item_id": item["calibration_item_id"],
            "calibration_batch_id": item["calibration_batch_id"],
            "annotation_item_id": item["annotation_item_id"],
            "selection_split": item["selection_split"],
            "summary": item["provider_input"]["summary"],
            "reviewText": item["provider_input"]["reviewText"],
            "human_Aspect1": "",
            "human_Polarity1": "",
            "human_Evidence_Source1": "",
            "human_Evidence1": "",
            "human_Aspect2": "",
            "human_Polarity2": "",
            "human_Evidence_Source2": "",
            "human_Evidence2": "",
            "human_Review_Mixed_Flag": "",
            "human_rationale": "",
        })

    OUT_DIR.mkdir(parents=True, exist_ok=False)
    manifest_sha = write_new(OUT_DIR / "v13_calibration_manifest_48_v2.json", canonical_json_bytes(manifest))
    hidden_sha = write_new(OUT_DIR / "v13_hidden_acceptance_expected_labels_v2.json", canonical_json_bytes(hidden_expected))
    template_path = OUT_DIR / "v13_blind_human_adjudication_template_v2.csv"
    with template_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(human_rows[0]))
        writer.writeheader()
        writer.writerows(human_rows)
    template_sha = sha256_file(template_path)
    report = {
        "status": "GATE7_11_MANIFEST_READY_NO_PROVIDER_CALLS",
        "manifest_version": "v2",
        "manifest_sha256": manifest_sha,
        "hidden_expected_labels_sha256": hidden_sha,
        "blind_human_template_sha256": template_sha,
        "item_count": 48,
        "batch_count": 6,
        "replacement_log_count": len(replacement_log),
        "provider_calls_made": False,
    }
    report_sha = write_new(OUT_DIR / "gate7_11_v13_calibration_manifest_report_v2.json", canonical_json_bytes(report))
    report["report_sha256"] = report_sha
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
