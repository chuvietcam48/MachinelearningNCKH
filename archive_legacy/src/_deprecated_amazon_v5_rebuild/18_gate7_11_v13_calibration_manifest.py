#!/usr/bin/env python
"""Build Gate 7.11 v1.3 calibration manifest without provider calls."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any


OUT_DIR = Path("outputs/amazon_v5_rebuild/annotation/v1_3_calibration")
PRIMARY_MANIFEST = Path(
    "outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/manifests/gate7_6_v3_primary_b3_annotation_review_manifest.csv"
)
GATE6_AUDIT = Path("outputs/amazon_v5_rebuild/annotation/v1_2_gate6_human_audit_30_seed_20260705.json")
P000007_PROVIDER = Path(
    "outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/batches/P000007/A001/provider_input.json"
)
TAXONOMY_V13 = Path("outputs/amazon_v5_rebuild/annotation/taxonomy_v1_3_draft_for_calibration.md")
PROMPT_V13 = Path("outputs/amazon_v5_rebuild/annotation/gemini_semantic_prompt_template_v1_3_draft_for_calibration.md")
CANONICAL_SCHEMA = Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json")
PROVIDER_SCHEMA = Path(
    "outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/manifests/provider_response_schema_frozen_v1_2.json"
)
VALIDATOR_SPEC = Path("outputs/amazon_v5_rebuild/annotation/semantic_validator_spec_v1_6.md")
LIVE_EXECUTOR_SOURCE = Path("src/amazon_v5_rebuild/16_v12exp_primary_pilot_live_executor.py")
POLICY_MANIFEST = Path("outputs/amazon_v5_rebuild/annotation/v1_3_policy_manifest.json")

KEYWORDS = ["production", "sound", "mix", "mastering", "recording", "style", "genre", "raw", "garage", "audio"]
PRIOR_MONITORING_IDS = [
    "A2LQQKWT4NROUW_3741396",
    "AZQHNUHSC3OWC_3985858",
    "AE37E22PLKYKN_4302748",
    "A2F26XZ573IAZW_2625034",
]
P000007_ID = "A102XSQH2IW56B_3720498"
SEED = 20260706


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


def load_primary_rows() -> list[dict[str, str]]:
    with PRIMARY_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def row_to_item(row: dict[str, str], split: str, reason: str, batch_id: str, position: int) -> dict[str, Any]:
    payload = {
        "annotation_item_id": row["annotation_item_id"],
        "summary": row.get("summary", "") or "",
        "reviewText": row.get("reviewText", "") or "",
    }
    return {
        "calibration_item_id": f"V13CAL_{position:03d}",
        "calibration_batch_id": batch_id,
        "item_position": position,
        "annotation_item_id": row["annotation_item_id"],
        "reviewerID": row.get("reviewerID"),
        "review_timestamp": row.get("review_timestamp"),
        "asin": row.get("asin"),
        "selection_split": split,
        "selection_reason": reason,
        "provider_input": payload,
        "provider_input_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def p000007_row() -> dict[str, str]:
    rows = json.loads(P000007_PROVIDER.read_text(encoding="utf-8-sig"))
    for row in rows:
        if row["annotation_item_id"] == P000007_ID:
            return {
                "annotation_item_id": row["annotation_item_id"],
                "summary": row.get("summary", ""),
                "reviewText": row.get("reviewText", ""),
                "reviewerID": None,
                "review_timestamp": None,
                "asin": None,
            }
    raise RuntimeError(f"Missing P000007 sentinel {P000007_ID}")


def main() -> None:
    primary = load_primary_rows()
    by_id = {row["annotation_item_id"]: row for row in primary}
    used: set[str] = set()
    items: list[dict[str, Any]] = []
    replacement_log: list[dict[str, Any]] = []

    gate6 = json.loads(GATE6_AUDIT.read_text(encoding="utf-8-sig"))
    gate6_ids = [row["annotation_item_id"] for row in gate6.get("rows", [])]
    old_regression_rows = []
    for item_id in gate6_ids:
        if item_id in by_id and item_id not in used:
            old_regression_rows.append(by_id[item_id])
            used.add(item_id)
        else:
            replacement_log.append({
                "requested_split": "old_manual_audit_regression",
                "requested_annotation_item_id": item_id,
                "reason": "not_found_in_v3_primary_review_manifest_or_duplicate",
            })
        if len(old_regression_rows) == 8:
            break

    # Deterministic replacements for old regression rows if old audited reviews are outside the v3 primary universe.
    random_rows = sorted(primary, key=lambda r: sha256_bytes((r["annotation_item_id"] + str(SEED)).encode("utf-8")))
    for row in random_rows:
        if len(old_regression_rows) == 8:
            break
        if row["annotation_item_id"] not in used:
            old_regression_rows.append(row)
            used.add(row["annotation_item_id"])
            replacement_log.append({
                "requested_split": "old_manual_audit_regression",
                "replacement_annotation_item_id": row["annotation_item_id"],
                "reason": "deterministic_primary_manifest_replacement_for_missing_old_audit_text",
            })

    def add_rows(rows: list[dict[str, str]], split: str, reason: str) -> None:
        for row in rows:
            position = len(items) + 1
            batch_number = ((position - 1) // 8) + 1
            batch_id = f"V13CAL_B{batch_number:03d}"
            items.append(row_to_item(row, split, reason, batch_id, position))

    add_rows(old_regression_rows, "A_old_regression_set", "old_manual_audit_regression_or_seeded_replacement")

    sentinel = p000007_row()
    used.add(P000007_ID)
    add_rows([sentinel], "B_p000007_sentinel_acceptance", "hidden_expected_label_acceptance_test")

    prior_rows = []
    for item_id in PRIOR_MONITORING_IDS:
        if item_id in by_id and item_id not in used:
            prior_rows.append(by_id[item_id])
            used.add(item_id)
        else:
            replacement_log.append({
                "requested_split": "C_prior_monitoring_boundary_cases",
                "requested_annotation_item_id": item_id,
                "reason": "not_found_in_v3_primary_review_manifest_or_duplicate",
            })
    for row in random_rows:
        if len(prior_rows) == 4:
            break
        text = f"{row.get('summary','')} {row.get('reviewText','')}".lower()
        if row["annotation_item_id"] not in used and any(k in text for k in ["expect", "edition", "format", "listing", "sound"]):
            prior_rows.append(row)
            used.add(row["annotation_item_id"])
            replacement_log.append({
                "requested_split": "C_prior_monitoring_boundary_cases",
                "replacement_annotation_item_id": row["annotation_item_id"],
                "reason": "deterministic_boundary_text_replacement_for_missing_prior_monitoring_text",
            })
    add_rows(prior_rows, "C_prior_monitoring_boundary_cases", "prior_monitoring_boundary_or_seeded_replacement")

    keyword_rows = []
    for row in sorted(primary, key=lambda r: sha256_bytes((r["annotation_item_id"] + "keyword" + str(SEED)).encode("utf-8"))):
        if len(keyword_rows) == 27:
            break
        text = f"{row.get('summary','')} {row.get('reviewText','')}".lower()
        matched = [k for k in KEYWORDS if k in text]
        if matched and row["annotation_item_id"] not in used:
            row = dict(row)
            row["_matched_keywords"] = ",".join(matched)
            keyword_rows.append(row)
            used.add(row["annotation_item_id"])
    add_rows(keyword_rows, "D_keyword_boundary_holdout", "keyword_selected_boundary_holdout")

    regression_rows = []
    for row in random_rows:
        if len(regression_rows) == 8:
            break
        if row["annotation_item_id"] not in used:
            regression_rows.append(row)
            used.add(row["annotation_item_id"])
    add_rows(regression_rows, "E_random_regression_set", "seeded_random_regression")

    if len(items) != 48:
        raise RuntimeError(f"Expected 48 calibration items, got {len(items)}")
    if len({item["annotation_item_id"] for item in items}) != 48:
        raise RuntimeError("Calibration manifest contains duplicate annotation_item_id")
    for i, item in enumerate(items, start=1):
        expected_batch = f"V13CAL_B{((i - 1) // 8) + 1:03d}"
        if item["calibration_batch_id"] != expected_batch:
            raise RuntimeError("Batch assignment drifted")

    batches = []
    for batch_idx in range(6):
        batch_items = items[batch_idx * 8:(batch_idx + 1) * 8]
        provider_payload = [item["provider_input"] for item in batch_items]
        batches.append({
            "calibration_batch_id": f"V13CAL_B{batch_idx + 1:03d}",
            "item_count": len(batch_items),
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

    pass_rules = {
        "raw_outputs_schema_evidence_unique_span_pass": "48/48",
        "p000007_sentinel": "Domain_Experience Positive only; mixed=false",
        "creative_style_double_count_errors_allowed": 0,
        "human_repair_allowed_in_calibration": False,
        "base_provider_requests_per_batch": 1,
        "blind_retry_allowed": False,
        "keyword_boundary_reported_separately": True,
        "do_not_report_single_accuracy_number": True,
        "metric_name": "agreement against one-person blind adjudication",
    }

    manifest = {
        "gate": "Gate 7.11",
        "status": "MANIFEST_READY_NO_PROVIDER_CALLS",
        "policy_version": "v1.3",
        "calibration_manifest_id": "V13CAL_MANIFEST_20260706_SEED20260706",
        "sample_seed": SEED,
        "item_count": 48,
        "batch_count": 6,
        "batch_size": 8,
        "provider_calls_made": False,
        "provider_calls_allowed_now": False,
        "selection_splits": {
            "A_old_regression_set": 8,
            "B_p000007_sentinel_acceptance": 1,
            "C_prior_monitoring_boundary_cases": 4,
            "D_keyword_boundary_holdout": 27,
            "E_random_regression_set": 8,
        },
        "analysis_groups_must_be_reported_separately": [
            "A_old_regression_set",
            "B_p000007_sentinel_acceptance",
            "C_prior_monitoring_boundary_cases",
            "D_keyword_boundary_holdout",
            "E_random_regression_set",
        ],
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
            "policy_manifest_sha256_at_build_start": sha256_file(POLICY_MANIFEST),
        },
        "pass_rules_locked_before_provider": pass_rules,
        "batches": batches,
        "items": items,
        "replacement_log": replacement_log,
    }

    human_template_rows = []
    for item in items:
        human_template_rows.append({
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
    manifest_sha = write_new(OUT_DIR / "v13_calibration_manifest_48.json", canonical_json_bytes(manifest))
    hidden_sha = write_new(OUT_DIR / "v13_hidden_acceptance_expected_labels.json", canonical_json_bytes(hidden_expected))
    with (OUT_DIR / "v13_blind_human_adjudication_template.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(human_template_rows[0]))
        writer.writeheader()
        writer.writerows(human_template_rows)
    template_sha = sha256_file(OUT_DIR / "v13_blind_human_adjudication_template.csv")
    report = {
        "status": "GATE7_11_MANIFEST_READY_NO_PROVIDER_CALLS",
        "manifest_sha256": manifest_sha,
        "hidden_expected_labels_sha256": hidden_sha,
        "blind_human_template_sha256": template_sha,
        "item_count": 48,
        "batch_count": 6,
        "provider_calls_made": False,
    }
    report_sha = write_new(OUT_DIR / "gate7_11_v13_calibration_manifest_report.json", canonical_json_bytes(report))
    report["report_sha256"] = report_sha
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
