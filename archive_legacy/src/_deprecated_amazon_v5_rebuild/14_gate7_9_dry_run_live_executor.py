#!/usr/bin/env python
"""Gate 7.9 append-only dry-run executor.

No provider calls are made. This dry-run verifies approval wiring, non-sync
workspace enforcement, campaign-level locking, static contract integrity, and
hash-chained ledger resume behavior for the first 10 primary batches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "V12EXP_PRIMARY_V3"
ATTEMPT_ID = "A001"
EXECUTION_MODE = "DRY_RUN"
PILOT_BATCH_IDS = [f"P{i:06d}" for i in range(1, 11)]
FEATURE_DIR = Path("outputs/amazon_v5_rebuild/features_semantic_b3")
STATIC_CONTRACT_REPORT = FEATURE_DIR / "gate7_7_static_payload_contract_verify_only_report_v3.json"
GOVERNANCE_REPORT = FEATURE_DIR / "gate7_8_live_executor_governance_design_report.json"


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
        raise RuntimeError(f"Refusing to overwrite existing dry-run artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def assert_non_sync_workspace_root(path: Path) -> Path:
    resolved = path.resolve()
    if any("onedrive" in part.lower() for part in resolved.parts):
        raise RuntimeError(f"Workspace root must be non-sync; rejected OneDrive path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


class SingleWriterLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "SingleWriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise RuntimeError(f"Campaign range is locked by another writer: {self.path}") from e
        os.write(self.fd, f"{os.getpid()}\n".encode("utf-8"))
        os.fsync(self.fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.fd is not None:
            os.close(self.fd)
        if self.path.exists():
            self.path.unlink()


class AppendOnlyLedger:
    def __init__(self, ledger_path: Path, execution_workspace_id: str, approval: dict[str, Any]) -> None:
        self.ledger_path = ledger_path
        self.execution_workspace_id = execution_workspace_id
        self.approval = approval
        self.sequence = 0
        self.previous_event_sha256 = "0" * 64
        if ledger_path.exists():
            raise RuntimeError(f"Dry-run ledger already exists; refusing overwrite: {ledger_path}")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        payload = {
            **event,
            "execution_mode": EXECUTION_MODE,
            "ledger_event_id": str(uuid.uuid4()),
            "event_sequence": self.sequence,
            "previous_event_sha256": self.previous_event_sha256,
            "execution_workspace_id": self.execution_workspace_id,
            "operator_approval_id": self.approval["approval_id"],
            "approval_file_sha256": self.approval["approval_file_sha256"],
            "event_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        event_payload_sha256 = sha256_bytes(canonical_json_bytes(payload))
        payload["event_payload_sha256"] = event_payload_sha256
        line = canonical_json_bytes(payload) + b"\n"
        with self.ledger_path.open("ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.previous_event_sha256 = sha256_bytes(line.rstrip(b"\n"))
        return payload


def validate_approval(path: Path, batch_ids: list[str]) -> dict[str, Any]:
    approval = load_json(path)
    required = {
        "execution_mode": EXECUTION_MODE,
        "campaign_run_id": CAMPAIGN_ID,
        "batch_start": batch_ids[0],
        "batch_end": batch_ids[-1],
        "attempt_id": ATTEMPT_ID,
        "ack_live_provider": False,
    }
    for key, expected in required.items():
        if approval.get(key) != expected:
            raise RuntimeError(f"Dry-run approval mismatch for {key}: expected {expected!r}, got {approval.get(key)!r}")
    if not approval.get("approval_id"):
        raise RuntimeError("Dry-run approval file must include approval_id")
    approval["approval_file_sha256"] = sha256_file(path)
    return approval


def campaign_paths(repo: Path, batch_id: str) -> dict[str, Path]:
    attempt_dir = repo / FEATURE_DIR / CAMPAIGN_ID / "batches" / batch_id / ATTEMPT_ID
    manifest_dir = repo / FEATURE_DIR / CAMPAIGN_ID / "manifests"
    return {
        "provider_input.json": attempt_dir / "provider_input.json",
        "provenance_snapshot.json": attempt_dir / "provenance_snapshot.json",
        "rendered_prompt.md": attempt_dir / "rendered_prompt.md",
        "payload_contract_static_v3.json": attempt_dir / "payload_contract_static_v3.json",
        "checkpoint_state_static_v3.json": attempt_dir / "checkpoint_state_static_v3.json",
        "provider_response_schema_frozen_v1_2.json": manifest_dir / "provider_response_schema_frozen_v1_2.json",
        "payload_contract_index_static_v3.json": manifest_dir / "payload_contract_index_static_v3.json",
    }


def copy_static_contract_bundle(repo: Path, workspace: Path, batch_id: str) -> dict[str, Path]:
    paths = campaign_paths(repo, batch_id)
    dst_dir = workspace / "static_contracts" / CAMPAIGN_ID / "batches" / batch_id / ATTEMPT_ID
    schema_dir = workspace / "static_contracts" / CAMPAIGN_ID / "manifests"
    out: dict[str, Path] = {}
    for name, src in paths.items():
        dst = schema_dir / name if name.endswith("schema_frozen_v1_2.json") or name.endswith("index_static_v3.json") else dst_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"Copied static artifact hash mismatch for {batch_id}/{name}")
        out[name] = dst
    return out


def verify_ids_and_positions(provider_input: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> None:
    if len(provider_input) != 8 or len(provenance) != 8:
        raise RuntimeError("Provider input and provenance must each contain exactly 8 rows")
    input_ids = [row["annotation_item_id"] for row in provider_input]
    if len(set(input_ids)) != 8:
        raise RuntimeError("Provider input must contain exactly 8 unique annotation IDs")
    provenance_sorted = sorted(provenance, key=lambda row: row["item_position"])
    if [row["item_position"] for row in provenance_sorted] != list(range(1, 9)):
        raise RuntimeError("Provenance positions must be exactly 1..8")
    if input_ids != [row["annotation_item_id"] for row in provenance_sorted]:
        raise RuntimeError("Provider input order/IDs must match provenance item_position order")


def verify_static_contract(paths: dict[str, Path], batch_id: str) -> dict[str, str]:
    contract = load_json(paths["payload_contract_static_v3.json"])
    index = load_json(paths["payload_contract_index_static_v3.json"])
    index_entry = next((row for row in index if row["batch_id"] == batch_id and row["attempt_id"] == ATTEMPT_ID), None)
    if index_entry is None:
        raise RuntimeError(f"Batch {batch_id} missing from campaign-level static contract index")

    provider_input = load_json(paths["provider_input.json"])
    provenance = load_json(paths["provenance_snapshot.json"])
    checkpoint = load_json(paths["checkpoint_state_static_v3.json"])
    verify_ids_and_positions(provider_input, provenance)

    hashes = {
        "provider_input_sha256": sha256_file(paths["provider_input.json"]),
        "provenance_snapshot_sha256": sha256_file(paths["provenance_snapshot.json"]),
        "rendered_prompt_sha256": sha256_file(paths["rendered_prompt.md"]),
        "payload_contract_sha256": sha256_file(paths["payload_contract_static_v3.json"]),
        "static_checkpoint_sha256": sha256_file(paths["checkpoint_state_static_v3.json"]),
        "provider_response_schema_sha256": sha256_file(paths["provider_response_schema_frozen_v1_2.json"]),
    }
    expected = {
        "provider_input_sha256": contract["provider_input_sha256"],
        "provenance_snapshot_sha256": contract["provenance_snapshot_sha256"],
        "rendered_prompt_sha256": contract["rendered_prompt_sha256"],
        "provider_response_schema_sha256": contract["provider_response_schema_sha256"],
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise RuntimeError(f"{batch_id} {key} mismatch against static contract")
        if key in index_entry and index_entry.get(key) != value:
            raise RuntimeError(f"{batch_id} {key} mismatch against campaign index")
    if checkpoint.get("state") != "PLANNED_NOT_EXECUTED" or checkpoint.get("provider_request_sent") is not False:
        raise RuntimeError(f"{batch_id} static checkpoint is not PLANNED_NOT_EXECUTED")
    return hashes


def derive_resume_state(ledger_path: Path, workspace: Path) -> dict[str, Any]:
    simulated_completed: set[str] = set()
    previous = "0" * 64
    event_count = 0
    with ledger_path.open("rb") as f:
        for raw_line in f:
            event_count += 1
            line = raw_line.rstrip(b"\n")
            event = json.loads(line.decode("utf-8"))
            if event.get("execution_mode") != EXECUTION_MODE:
                raise RuntimeError("DRY_RUN resume derivation received non-DRY_RUN event")
            if event["previous_event_sha256"] != previous:
                raise RuntimeError(f"Ledger hash chain broken at sequence {event['event_sequence']}")
            event_payload_hash = event.pop("event_payload_sha256")
            if sha256_bytes(canonical_json_bytes(event)) != event_payload_hash:
                raise RuntimeError(f"Ledger payload hash mismatch at sequence {event['event_sequence']}")
            event["event_payload_sha256"] = event_payload_hash
            previous = sha256_bytes(line)
            if event["event_type"] == "DRY_RUN_CONTRACT_PASS":
                paths = copy_static_contract_bundle(Path(event["repo_root"]), workspace, event["batch_id"])
                verify_static_contract(paths, event["batch_id"])
                simulated_completed.add(f"{event['campaign_run_id']}:{event['batch_id']}:{event['attempt_id']}")
    return {"event_count": event_count, "simulated_completed_attempts": sorted(simulated_completed)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--approval-file", required=True)
    parser.add_argument("--batch-start", default="P000001")
    parser.add_argument("--batch-count", type=int, default=10)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    workspace_root = assert_non_sync_workspace_root(Path(args.workspace_root))
    static_report = load_json(repo / STATIC_CONTRACT_REPORT)
    governance_report = load_json(repo / GOVERNANCE_REPORT)
    if static_report["gate7_7_static_payload_contract_status"] != "VERIFY_ONLY_PASS":
        raise RuntimeError("Gate 7.9 requires Gate 7.7 verify-only PASS")
    if governance_report["gate7_8_status"] != "GOVERNANCE_DESIGN_LOCKED_EXECUTOR_NOT_IMPLEMENTED":
        raise RuntimeError("Gate 7.9 requires Gate 7.8 governance design locked")

    batch_ids = PILOT_BATCH_IDS[: args.batch_count]
    if args.batch_start != "P000001" or batch_ids != PILOT_BATCH_IDS:
        raise RuntimeError("Gate 7.9 dry-run is intentionally limited to P000001-P000010/A001")
    approval = validate_approval(Path(args.approval_file).resolve(), batch_ids)

    range_id = f"{batch_ids[0]}_{batch_ids[-1]}_{ATTEMPT_ID}"
    lock_path = workspace_root / "campaign_locks" / CAMPAIGN_ID / f"{range_id}.lock"
    workspace_id = f"DRYRUN_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"
    workspace = workspace_root / "executor_workspaces" / workspace_id
    ledger_path = workspace / "execution_ledger" / "ledger_dry_run.jsonl"
    feature_dir = repo / FEATURE_DIR
    report_path = feature_dir / f"gate7_9_dry_run_executor_report_{workspace_id}.json"
    archive_path = feature_dir / f"gate7_9_dry_run_executor_archive_manifest_{workspace_id}.json"

    with SingleWriterLock(lock_path):
        ledger = AppendOnlyLedger(ledger_path, workspace_id, approval)
        batch_results = []
        for batch_id in batch_ids:
            paths = copy_static_contract_bundle(repo, workspace, batch_id)
            hashes = verify_static_contract(paths, batch_id)
            base_event = {
                "repo_root": str(repo),
                "campaign_run_id": CAMPAIGN_ID,
                "batch_id": batch_id,
                "attempt_id": ATTEMPT_ID,
                "provider_request_sent": False,
                "provider_response_received": False,
                "provider_input_sha256": hashes["provider_input_sha256"],
                "provenance_snapshot_sha256": hashes["provenance_snapshot_sha256"],
                "rendered_prompt_sha256": hashes["rendered_prompt_sha256"],
                "provider_response_schema_sha256": hashes["provider_response_schema_sha256"],
                "raw_response_sha256": None,
                "parsed_output_sha256": None,
                "derived_validation_sha256": None,
            }
            ledger.append({**base_event, "event_type": "DRY_RUN_ATTEMPT_PLANNED", "terminal_reason": None})
            ledger.append({**base_event, "event_type": "DRY_RUN_STATIC_CONTRACT_VERIFIED", "terminal_reason": None})
            ledger.append({**base_event, "event_type": "DRY_RUN_CONTRACT_PASS", "terminal_reason": None})
            ledger.append({**base_event, "event_type": "DRY_RUN_ATTEMPT_TERMINAL", "terminal_reason": "DRY_RUN_COMPLETE_NO_PROVIDER_CALL"})
            batch_results.append({"batch_id": batch_id, "attempt_id": ATTEMPT_ID, **hashes})

        resume_state = derive_resume_state(ledger_path, workspace)
        if len(resume_state["simulated_completed_attempts"]) != len(batch_ids):
            raise RuntimeError("Resume derivation did not recover all simulated completed attempts")

        archive = {
            "execution_workspace_id": workspace_id,
            "workspace_root": str(workspace_root),
            "workspace_path": str(workspace),
            "campaign_lock_path": str(lock_path),
            "ledger_path": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
            "approval_id": approval["approval_id"],
            "approval_file_sha256": approval["approval_file_sha256"],
            "static_contract_copy_only": True,
            "provider_request_sent": False,
            "batch_count": len(batch_ids),
        }
        write_new(archive_path, canonical_json_bytes(archive))

        report = {
            "gate7_9_status": "DRY_RUN_EXECUTOR_PASS",
            "execution_mode": EXECUTION_MODE,
            "provider_request_sent": False,
            "live_executor_created": False,
            "dry_run_only": True,
            "execution_workspace_id": workspace_id,
            "workspace_root": str(workspace_root),
            "workspace_path": str(workspace),
            "workspace_root_rejects_onedrive": True,
            "single_writer_lock_scope": str(lock_path),
            "single_writer_lock_used": True,
            "ledger_append_only_verified": True,
            "resume_derived_from_ledger_and_hashes": True,
            "simulated_completed_attempt_count": len(resume_state["simulated_completed_attempts"]),
            "static_checkpoint_mutated": False,
            "approval_wiring": {
                "approval_id": approval["approval_id"],
                "approval_file_sha256": approval["approval_file_sha256"],
                "ack_live_provider": approval["ack_live_provider"],
                "approval_is_live_provider_permission": False,
            },
            "pilot_scope": {"campaign_run_id": CAMPAIGN_ID, "batch_ids": batch_ids, "attempt_id": ATTEMPT_ID, "review_count": 80},
            "pilot_output_release_rule": (
                "No pilot output may enter the expanded semantic-label corpus, Gate 7.5 re-run, "
                "or B3 modelling until PILOT_AUDIT_PASS and a separate explicit operator release approval are recorded."
            ),
            "ledger": {
                "event_count": resume_state["event_count"],
                "ledger_path": str(ledger_path),
                "ledger_sha256": sha256_file(ledger_path),
            },
            "batch_results": batch_results,
            "archive_manifest_path": str(archive_path),
            "next_gate_status": "LIVE_PROVIDER_PERMISSION_STILL_HOLD",
        }
        write_new(report_path, canonical_json_bytes(report))

    print(json.dumps({
        "gate7_9_status": "DRY_RUN_EXECUTOR_PASS",
        "provider_request_sent": False,
        "batch_count": len(batch_ids),
        "ledger_events": len(batch_ids) * 4,
        "report_path": str(report_path),
        "archive_manifest_path": str(archive_path),
        "live_provider_permission": "HOLD",
    }, indent=2))


if __name__ == "__main__":
    main()
