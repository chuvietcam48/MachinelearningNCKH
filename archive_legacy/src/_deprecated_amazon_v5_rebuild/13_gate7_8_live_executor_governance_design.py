#!/usr/bin/env python
"""Gate 7.8 live executor governance design.

This step does not create a live executor and never calls any provider. It
materializes the governance contract that must be implemented before any live
campaign execution is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FEATURE_DIR = Path("outputs/amazon_v5_rebuild/features_semantic_b3")
STATIC_VERIFY_REPORT = FEATURE_DIR / "gate7_7_static_payload_contract_verify_only_report_v3.json"
PAYLOAD_SCRIPT = Path("src/amazon_v5_rebuild/12_gate7_7_payload_contract_v3.py")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    feature_dir = repo / FEATURE_DIR
    verify_report_path = repo / STATIC_VERIFY_REPORT
    payload_script_path = repo / PAYLOAD_SCRIPT
    verify_report = json.loads(verify_report_path.read_text(encoding="utf-8"))

    if verify_report.get("gate7_7_static_payload_contract_status") != "VERIFY_ONLY_PASS":
        raise RuntimeError("Gate 7.8 requires Gate 7.7 static verify-only PASS.")
    if verify_report.get("provider_request_sent") is not False:
        raise RuntimeError("Gate 7.7 verify report indicates provider request was sent.")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_8_status": "GOVERNANCE_DESIGN_LOCKED_EXECUTOR_NOT_IMPLEMENTED",
        "provider_request_sent": False,
        "live_executor_created": False,
        "live_campaign_allowed_now": False,
        "dependency_gate7_7": {
            "static_payload_contract_status": verify_report["gate7_7_static_payload_contract_status"],
            "verify_only_report_sha256": sha256_file(verify_report_path),
            "payload_contract_source_sha256": sha256_file(payload_script_path),
        },
        "campaign_scope": {
            "primary_campaign_run_id": "V12EXP_PRIMARY_V3",
            "primary_batch_count": verify_report["campaigns"]["primary"]["batch_count"],
            "primary_review_count": verify_report["campaigns"]["primary"]["review_count"],
            "enrichment_campaign_run_id": "V12EXP_ENRICH_V3",
            "enrichment_batch_count": verify_report["campaigns"]["enrichment"]["batch_count"],
            "enrichment_review_count": verify_report["campaigns"]["enrichment"]["review_count"],
            "items_per_batch": 8,
        },
        "append_only_execution_ledger_contract": {
            "status": "REQUIRED_BEFORE_LIVE_EXECUTOR",
            "ledger_file_pattern": "V12EXP_{CAMPAIGN}/execution_ledger/ledger_YYYYMMDD.jsonl",
            "append_only": True,
            "forbidden_operations": [
                "edit prior ledger rows",
                "delete ledger rows",
                "rewrite checkpoint_state_static_v3.json",
                "overwrite A001 raw/parsed/validation artifacts",
            ],
            "required_row_fields": [
                "ledger_event_id",
                "event_timestamp_utc",
                "campaign_run_id",
                "batch_id",
                "attempt_id",
                "event_type",
                "provider_request_sent",
                "provider_response_received",
                "provider_input_sha256",
                "rendered_prompt_sha256",
                "provider_response_schema_sha256",
                "raw_response_sha256",
                "parsed_output_sha256",
                "derived_validation_sha256",
                "terminal_reason",
                "operator_approval_id",
            ],
            "allowed_event_types": [
                "ATTEMPT_PLANNED",
                "REQUEST_SENT",
                "RESPONSE_RECEIVED",
                "VALIDATION_PASS",
                "VALIDATION_FAIL",
                "ATTEMPT_TERMINAL",
                "AUDIT_REQUIRED",
                "AUDIT_PASS",
                "AUDIT_HOLD",
                "KILL_SWITCH_TRIGGERED",
            ],
        },
        "resume_policy": {
            "status": "REQUIRED_BEFORE_LIVE_EXECUTOR",
            "resume_source_of_truth": "append-only ledger + immutable artifact hashes",
            "static_checkpoint_mutation_allowed": False,
            "completed_batch_definition": [
                "ledger has VALIDATION_PASS for campaign_run_id/batch_id/attempt_id",
                "raw_response, parsed_output, derived_validation artifacts exist",
                "artifact hashes match ledger",
                "payload contract hashes match static contract index",
            ],
            "failed_batch_policy": {
                "same_batch_id_new_attempt_id": True,
                "next_attempt_sequence": "A002, A003, ...",
                "never_overwrite_failed_attempt": True,
                "no_automatic_retry": True,
            },
        },
        "quota_rate_limit_policy": {
            "status": "MUST_LOCK_VALUE_BEFORE_LIVE",
            "max_requests_per_minute": "PENDING_OPERATOR_VALUE",
            "max_requests_per_day": "PENDING_OPERATOR_VALUE",
            "concurrency": "PENDING_OPERATOR_VALUE",
            "provider_429_or_quota_error_action": "stop campaign, write terminal ledger event, require operator review",
        },
        "cost_ceiling_and_kill_switch": {
            "status": "MUST_LOCK_VALUE_BEFORE_LIVE",
            "max_primary_requests": 7745,
            "max_enrichment_requests": 500,
            "max_total_requests": 8245,
            "budget_currency": "PENDING_OPERATOR_VALUE",
            "max_total_cost": "PENDING_OPERATOR_VALUE",
            "kill_switch_conditions": [
                "cost ceiling reached",
                "quota/rate limit terminal error",
                "schema validation failure rate exceeds locked threshold",
                "evidence/offset validation failure appears systematic",
                "operator stop file present",
            ],
        },
        "pilot_at_scale_audit_rule": {
            "status": "REQUIRED_BEFORE_FULL_SCALE",
            "pilot_scope": "first 10 primary batches only",
            "pilot_batch_ids": [f"P{i:06d}" for i in range(1, 11)],
            "pilot_review_count": 80,
            "merge_into_b3_before_gate7_7_pass": False,
            "pass_requirements": [
                "10/10 batches terminal VALIDATION_PASS",
                "80/80 records schema valid",
                "no missing/duplicate annotation_item_id",
                "raw/parsed/derived hashes written",
                "evidence and offset validation pass",
                "audit sample reviewed and no new systematic semantic failure",
            ],
            "on_fail": "HOLD campaign; write failure report; fix executor/governance only, never patch taxonomy/prompt/schema mid-campaign",
        },
        "periodic_audit_cadence": {
            "status": "REQUIRED_BEFORE_FULL_SCALE",
            "audit_every_n_primary_batches": "PENDING_OPERATOR_VALUE",
            "minimum_random_reviews_per_audit": "PENDING_OPERATOR_VALUE",
            "always_audit": [
                "first 10 primary batches",
                "any failed batch before retry attempt",
                "any provider/API behavior change",
                "any validation failure cluster",
            ],
            "audit_log_location": "V12EXP_{CAMPAIGN}/audits/",
        },
        "explicit_operator_approval": {
            "status": "REQUIRED_BEFORE_EACH_LIVE_CAMPAIGN",
            "required_flags": [
                "--ack-live-provider",
                "--campaign-run-id",
                "--operator-approval-id",
                "--cost-ceiling",
                "--rate-limit-profile",
            ],
            "approval_must_state": [
                "campaign id",
                "batch range",
                "attempt id",
                "cost ceiling",
                "quota/rate limit profile",
                "audit cadence",
            ],
        },
        "next_allowed_work": [
            "Implement live executor only after this governance design is accepted.",
            "Executor must be append-only and ledger-derived.",
            "First live execution, if approved later, must be 10-batch primary pilot only.",
        ],
    }

    report_path = feature_dir / "gate7_8_live_executor_governance_design_report.json"
    md_path = feature_dir / "gate7_8_live_executor_governance_design_report.md"
    write_json(report_path, report)
    md_path.write_text(
        "\n".join(
            [
                "# Gate 7.8 Live Executor Governance Design",
                "",
                f"Status: {report['gate7_8_status']}",
                "",
                "No provider call was made. No live executor was created.",
                "",
                "Required before live execution:",
                "- append-only execution ledger",
                "- resume from ledger + immutable artifact hashes",
                "- quota/rate limit values",
                "- cost ceiling and kill switch",
                "- 10-batch primary pilot audit",
                "- periodic audit cadence",
                "- explicit operator approval before each campaign",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "gate7_8_status": report["gate7_8_status"],
                "provider_request_sent": False,
                "live_executor_created": False,
                "live_campaign_allowed_now": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
