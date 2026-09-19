#!/usr/bin/env python
"""Gate 7.10 live permission package and dry-run executor audit.

This step does not call any provider and does not create a live executor. It
locks the first-live-pilot scope and guardrails, statically audits the dry-run
executor source, and proves campaign-range lock contention fails closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "V12EXP_PRIMARY_V3"
ATTEMPT_ID = "A001"
BATCH_START = "P000001"
BATCH_END = "P000010"
BATCH_COUNT = 10
REVIEW_COUNT = 80
FEATURE_DIR = Path("outputs/amazon_v5_rebuild/features_semantic_b3")
DRY_RUN_EXECUTOR = Path("src/amazon_v5_rebuild/14_gate7_9_dry_run_live_executor.py")


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
        raise RuntimeError(f"Refusing to overwrite Gate 7.10 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def latest_gate79_report(feature_dir: Path) -> Path:
    reports = sorted(
        feature_dir.glob("gate7_9_dry_run_executor_report_DRYRUN_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        raise RuntimeError("No Gate 7.9 run-specific dry-run report found")
    return reports[0]


def static_audit_executor_source(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    banned = [
        "google.genai",
        "genai.Client",
        "generate_content",
        "GEMINI_API_KEY",
        "requests.",
        "httpx.",
        "urllib.request",
    ]
    banned_present = [term for term in banned if term in text]
    required = [
        'parser.add_argument("--workspace-root", required=True)',
        'parser.add_argument("--approval-file", required=True)',
        "assert_non_sync_workspace_root",
        "campaign_locks",
        "DRY_RUN_CONTRACT_PASS",
        '"execution_mode"',
        "simulated_completed_attempts",
        "ack_live_provider",
    ]
    missing = [term for term in required if term not in text]
    if banned_present:
        raise RuntimeError(f"Dry-run executor source contains banned provider/network terms: {banned_present}")
    if missing:
        raise RuntimeError(f"Dry-run executor source missing required governance terms: {missing}")
    return {
        "source_sha256": sha256_file(path),
        "banned_provider_terms_absent": True,
        "required_governance_terms_present": True,
    }


def make_dry_run_approval(workspace_root: Path) -> Path:
    approval = {
        "approval_id": f"GATE710_CONTENTION_DRYRUN_{uuid.uuid4().hex[:12]}",
        "execution_mode": "DRY_RUN",
        "campaign_run_id": CAMPAIGN_ID,
        "batch_start": BATCH_START,
        "batch_end": BATCH_END,
        "attempt_id": ATTEMPT_ID,
        "ack_live_provider": False,
        "note": "Gate 7.10 lock contention wiring test only; not live provider approval.",
    }
    path = workspace_root / "gate7_10_contention_dry_run_approval.json"
    path.write_text(json.dumps(approval, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_lock_contention_test(repo: Path, workspace_root: Path, approval_file: Path) -> dict[str, Any]:
    lock_path = workspace_root / "campaign_locks" / CAMPAIGN_ID / f"{BATCH_START}_{BATCH_END}_{ATTEMPT_ID}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        lock_path.unlink()
    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, b"gate7_10_contention_test\n")
        os.fsync(fd)
        cmd = [
            sys.executable,
            str(repo / DRY_RUN_EXECUTOR),
            "--repo-root",
            str(repo),
            "--workspace-root",
            str(workspace_root),
            "--approval-file",
            str(approval_file),
            "--batch-start",
            BATCH_START,
            "--batch-count",
            str(BATCH_COUNT),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    finally:
        os.close(fd)
        if lock_path.exists():
            lock_path.unlink()
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode == 0:
        raise RuntimeError("Contention test failed: second dry-run unexpectedly succeeded")
    if "locked by another writer" not in combined:
        raise RuntimeError(f"Contention test failed for unexpected reason: {combined[:1000]}")
    return {
        "contention_test_pass": True,
        "second_process_return_code": result.returncode,
        "lock_path": str(lock_path),
        "failure_contains_locked_by_another_writer": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--workspace-root", required=True)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    workspace_root = Path(args.workspace_root).resolve()
    if any("onedrive" in part.lower() for part in workspace_root.parts):
        raise RuntimeError(f"Gate 7.10 workspace-root must be non-OneDrive: {workspace_root}")
    workspace_root.mkdir(parents=True, exist_ok=True)
    feature_dir = repo / FEATURE_DIR
    run_id = f"GATE710_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"

    gate79_report_path = latest_gate79_report(feature_dir)
    gate79_report = load_json(gate79_report_path)
    if gate79_report.get("gate7_9_status") != "DRY_RUN_EXECUTOR_PASS":
        raise RuntimeError("Gate 7.10 requires latest Gate 7.9 dry-run pass")
    if gate79_report.get("provider_request_sent") is not False:
        raise RuntimeError("Latest Gate 7.9 report indicates provider request was sent")

    source_audit = static_audit_executor_source(repo / DRY_RUN_EXECUTOR)
    dry_approval = make_dry_run_approval(workspace_root)
    contention = run_lock_contention_test(repo, workspace_root, dry_approval)

    approval_template = {
        "template_status": "TEMPLATE_ONLY_NOT_OPERATOR_APPROVED",
        "execution_mode": "LIVE",
        "campaign_run_id": CAMPAIGN_ID,
        "batch_start": BATCH_START,
        "batch_end": BATCH_END,
        "attempt_id": ATTEMPT_ID,
        "ack_live_provider": False,
        "operator_must_set_ack_live_provider_true_only_at_execution_time": True,
        "scope_is_only_first_live_pilot": True,
        "max_batches": BATCH_COUNT,
        "max_reviews": REVIEW_COUNT,
        "max_provider_requests": BATCH_COUNT,
        "approval_is_not_valid_until_signed_by_operator": True,
    }
    permission_package = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate7_10_status": "LIVE_PERMISSION_PACKAGE_LOCKED_PROVIDER_PERMISSION_HOLD",
        "provider_request_sent": False,
        "live_executor_created": False,
        "live_provider_permission": "HOLD",
        "scope": {
            "campaign_run_id": CAMPAIGN_ID,
            "batch_start": BATCH_START,
            "batch_end": BATCH_END,
            "attempt_id": ATTEMPT_ID,
            "batch_count": BATCH_COUNT,
            "review_count": REVIEW_COUNT,
        },
        "quota_rate_limit_profile": {
            "profile_id": "PILOT_10_BATCH_SINGLE_WRITER_CONSERVATIVE",
            "max_concurrency": 1,
            "max_requests_per_minute": 1,
            "max_requests_per_hour": 30,
            "max_requests_per_day": 10,
            "max_total_provider_requests": 10,
        },
        "cost_ceiling": {
            "currency": "USD",
            "max_total_cost": 5.0,
            "hard_stop_if_cost_tracking_unavailable": True,
            "hard_stop_if_request_count_exceeds": 10,
        },
        "kill_switch_thresholds": {
            "any_schema_failure": "STOP",
            "any_missing_raw_response": "STOP",
            "any_duplicate_or_missing_annotation_item_id": "STOP",
            "any_evidence_or_offset_failure": "STOP_AND_AUDIT",
            "any_provider_quota_or_rate_limit_error": "STOP",
            "operator_stop_file_present": "STOP",
        },
        "audit_cadence": {
            "pilot_batches_to_audit": "P000001-P000010",
            "pilot_reviews_to_audit": 80,
            "audit_before_release_to_corpus": True,
            "release_requires": [
                "PILOT_AUDIT_PASS",
                "separate explicit operator release approval",
            ],
        },
        "source_audit": source_audit,
        "contention_test": contention,
        "latest_gate7_9_report": {
            "path": str(gate79_report_path),
            "sha256": sha256_file(gate79_report_path),
        },
        "next_allowed_step": "Static-audit live executor implementation only; no provider run until operator signs live approval for this exact pilot scope.",
    }

    template_path = feature_dir / f"gate7_10_live_pilot_operator_approval_template_{run_id}.json"
    package_path = feature_dir / f"gate7_10_live_permission_package_{run_id}.json"
    write_new(template_path, canonical_json_bytes(approval_template))
    permission_package["approval_template"] = {
        "path": str(template_path),
        "sha256": sha256_file(template_path),
    }
    write_new(package_path, canonical_json_bytes(permission_package))
    print(json.dumps({
        "gate7_10_status": permission_package["gate7_10_status"],
        "live_provider_permission": "HOLD",
        "contention_test_pass": True,
        "provider_request_sent": False,
        "package_path": str(package_path),
        "approval_template_path": str(template_path),
    }, indent=2))


if __name__ == "__main__":
    main()
