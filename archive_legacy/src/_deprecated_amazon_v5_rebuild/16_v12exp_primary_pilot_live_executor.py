#!/usr/bin/env python
"""Minimal live pilot executor for V12EXP_PRIMARY_V3 P000001-P000010/A001.

Default use for now is --static-audit only. Live execution is hard scoped to the
first 10 primary batches and requires an explicit LIVE approval file with
ack_live_provider=true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


CAMPAIGN_ID = "V12EXP_PRIMARY_V3"
CONTRACT_ATTEMPT_ID = "A001"
EXECUTION_MODE_LIVE = "LIVE"
PILOT_BATCH_IDS = [f"P{i:06d}" for i in range(1, 11)]
FEATURE_DIR = Path("outputs/amazon_v5_rebuild/features_semantic_b3")
STATIC_VERIFY_REPORT = FEATURE_DIR / "gate7_7_static_payload_contract_verify_only_report_v3.json"
SCHEMA_PATH = Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json")


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, data: bytes) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


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
            raise RuntimeError(f"Campaign pilot range is locked by another writer: {self.path}") from e
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
            raise RuntimeError(f"Ledger already exists; refusing overwrite: {ledger_path}")
        ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        payload = {
            **event,
            "execution_mode": EXECUTION_MODE_LIVE,
            "ledger_event_id": str(uuid.uuid4()),
            "event_sequence": self.sequence,
            "previous_event_sha256": self.previous_event_sha256,
            "execution_workspace_id": self.execution_workspace_id,
            "operator_approval_id": self.approval["approval_id"],
            "approval_file_sha256": self.approval["approval_file_sha256"],
            "event_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload_hash = sha256_bytes(canonical_json_bytes(payload))
        payload["event_payload_sha256"] = payload_hash
        line = canonical_json_bytes(payload) + b"\n"
        with self.ledger_path.open("ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        self.previous_event_sha256 = sha256_bytes(line.rstrip(b"\n"))
        return payload


def validate_live_approval(path: Path) -> dict[str, Any]:
    approval = load_json(path)
    required = {
        "execution_mode": EXECUTION_MODE_LIVE,
        "campaign_run_id": CAMPAIGN_ID,
        "ack_live_provider": True,
    }
    for key, expected in required.items():
        if approval.get(key) != expected:
            raise RuntimeError(f"Live approval mismatch for {key}: expected {expected!r}, got {approval.get(key)!r}")
    attempt_id = approval.get("attempt_id")
    if not isinstance(attempt_id, str) or not re.fullmatch(r"A\d{3}", attempt_id):
        raise RuntimeError(f"Live approval attempt_id must look like A001/A002/etc; got {attempt_id!r}")
    batch_start = approval.get("batch_start")
    batch_end = approval.get("batch_end")
    if batch_start not in PILOT_BATCH_IDS or batch_end not in PILOT_BATCH_IDS:
        raise RuntimeError("Live approval batch range must stay inside P000001-P000010")
    if PILOT_BATCH_IDS.index(batch_start) > PILOT_BATCH_IDS.index(batch_end):
        raise RuntimeError("Live approval batch_start must not come after batch_end")
    selected_count = PILOT_BATCH_IDS.index(batch_end) - PILOT_BATCH_IDS.index(batch_start) + 1
    if approval.get("max_provider_requests") != selected_count:
        raise RuntimeError(f"Live approval max_provider_requests must equal selected batch count {selected_count}")
    if approval.get("max_concurrency") != 1:
        raise RuntimeError("Live approval must set max_concurrency=1")
    if not approval.get("approval_id"):
        raise RuntimeError("Live approval must include approval_id")
    approval["approval_file_sha256"] = sha256_file(path)
    return approval


def contract_paths(repo: Path, batch_id: str) -> dict[str, Path]:
    attempt_dir = repo / FEATURE_DIR / CAMPAIGN_ID / "batches" / batch_id / CONTRACT_ATTEMPT_ID
    manifest_dir = repo / FEATURE_DIR / CAMPAIGN_ID / "manifests"
    return {
        "provider_input": attempt_dir / "provider_input.json",
        "provenance": attempt_dir / "provenance_snapshot.json",
        "rendered_prompt": attempt_dir / "rendered_prompt.md",
        "contract": attempt_dir / "payload_contract_static_v3.json",
        "checkpoint": attempt_dir / "checkpoint_state_static_v3.json",
        "schema": manifest_dir / "provider_response_schema_frozen_v1_2.json",
    }


def verify_contract(repo: Path, batch_id: str) -> dict[str, Any]:
    paths = contract_paths(repo, batch_id)
    contract = load_json(paths["contract"])
    provider_input = load_json(paths["provider_input"])
    provenance = load_json(paths["provenance"])
    checkpoint = load_json(paths["checkpoint"])
    if len(provider_input) != 8 or len({row["annotation_item_id"] for row in provider_input}) != 8:
        raise RuntimeError(f"{batch_id}: provider_input must have 8 unique IDs")
    provenance_sorted = sorted(provenance, key=lambda row: row["item_position"])
    if [row["item_position"] for row in provenance_sorted] != list(range(1, 9)):
        raise RuntimeError(f"{batch_id}: provenance positions must be 1..8")
    if [row["annotation_item_id"] for row in provider_input] != [row["annotation_item_id"] for row in provenance_sorted]:
        raise RuntimeError(f"{batch_id}: provider_input order must match provenance positions")
    if checkpoint.get("state") != "PLANNED_NOT_EXECUTED" or checkpoint.get("provider_request_sent") is not False:
        raise RuntimeError(f"{batch_id}: static checkpoint is not planned/not executed")
    hashes = {
        "provider_input_sha256": sha256_file(paths["provider_input"]),
        "provenance_snapshot_sha256": sha256_file(paths["provenance"]),
        "rendered_prompt_sha256": sha256_file(paths["rendered_prompt"]),
        "provider_response_schema_sha256": sha256_file(paths["schema"]),
    }
    for key, actual in hashes.items():
        if actual != contract[key]:
            raise RuntimeError(f"{batch_id}: {key} mismatch against static contract")
    return {"paths": paths, "hashes": hashes, "contract": contract}


def build_run_schema(base_schema: dict[str, Any], batch_size: int) -> dict[str, Any]:
    run_schema = json.loads(json.dumps(base_schema))
    run_schema["properties"]["items"]["minItems"] = batch_size
    run_schema["properties"]["items"]["maxItems"] = batch_size
    return run_schema


def validate_schema(payload: dict[str, Any], run_schema: dict[str, Any]) -> tuple[bool, str]:
    errors = sorted(Draft7Validator(run_schema).iter_errors(payload), key=lambda e: e.path)
    if errors:
        return False, errors[0].message
    return True, ""


def validate_semantics(item: dict[str, Any], source_row: dict[str, str]) -> tuple[bool, str, dict[str, int | None]]:
    a1 = item.get("Aspect1")
    p1 = item.get("Polarity1")
    e_src1 = item.get("Evidence_Source1")
    e1 = item.get("Evidence1")
    a2 = item.get("Aspect2")
    p2 = item.get("Polarity2")
    e_src2 = item.get("Evidence_Source2")
    e2 = item.get("Evidence2")
    mixed = item.get("Review_Mixed_Flag")

    polarities = []
    if a1 != "None" and p1 in ("Positive", "Negative"):
        polarities.append(p1)
    if a2 is not None and p2 in ("Positive", "Negative"):
        polarities.append(p2)
    has_pos = "Positive" in polarities
    has_neg = "Negative" in polarities
    if mixed is True and not (has_pos and has_neg):
        return False, "Rule D: mixed=true requires at least one Positive and one Negative aspect", {}
    if mixed is False and has_pos and has_neg:
        return False, "Rule D: mixed=false despite Positive and Negative aspects", {}
    if not isinstance(mixed, bool):
        return False, "Rule D: missing or non-boolean mixed flag", {}

    def check_evidence(e_str: Any, e_src: Any) -> tuple[bool, str, int | None, int | None]:
        if e_str is None:
            return True, "", None, None
        if not isinstance(e_str, str):
            return False, "Rule C: evidence must be string", None, None
        stripped = e_str.strip()
        if not stripped:
            return False, "Rule C: evidence empty after strip", None, None
        if len(stripped.split()) > 7:
            return False, "Rule C: evidence quote exceeds 7 words", None, None
        if e_src not in ("summary", "reviewText"):
            return False, "Rule C: Invalid evidence source", None, None
        src_text = str(source_row.get(e_src, ""))
        count = src_text.count(stripped)
        if count == 0:
            return False, "Rule 14: evidence quote absent from source", None, None
        if count > 1:
            return False, "Rule 15: evidence quote repeated in source", None, None
        start_idx = src_text.find(stripped)
        end_idx = start_idx + len(stripped)
        if start_idx > 0 and src_text[start_idx - 1].isalnum():
            return False, "Rule C: quote starts inside an alphanumeric word", None, None
        if end_idx < len(src_text) and src_text[end_idx].isalnum():
            return False, "Rule C: quote ends inside an alphanumeric word", None, None
        return True, "", start_idx, end_idx

    ok, msg, start1, end1 = check_evidence(e1, e_src1)
    if not ok:
        return False, msg, {}
    ok, msg, start2, end2 = check_evidence(e2, e_src2)
    if not ok:
        return False, msg, {}
    return True, "", {"e1_start": start1, "e1_end": end1, "e2_start": start2, "e2_end": end2}


def validate_live_output(raw_text: str, provider_input: list[dict[str, str]], base_schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_ids = [row["annotation_item_id"] for row in provider_input]
    source_by_id = {row["annotation_item_id"]: row for row in provider_input}
    derived = {
        "status": "PASS",
        "layer": "All",
        "error_code": "None",
        "items": [],
    }
    try:
        parsed = json.loads(raw_text)
    except Exception:
        derived.update({"status": "REJECT", "layer": "JSON Parser", "error_code": "MalformedJSON"})
        return {}, derived

    ok, msg = validate_schema(parsed, build_run_schema(base_schema, len(requested_ids)))
    if not ok:
        derived.update({"status": "REJECT", "layer": "Schema Validator", "error_code": msg})
        return parsed, derived

    returned_ids = [item.get("annotation_item_id") for item in parsed.get("items", [])]
    if len(returned_ids) != len(requested_ids) or len(set(returned_ids)) != len(returned_ids) or set(returned_ids) != set(requested_ids):
        err = "Rule 4: duplicate ID" if len(set(returned_ids)) != len(returned_ids) else "Rule 5: missing/altered annotation_item_id"
        derived.update({"status": "REJECT", "layer": "ID Integrity", "error_code": err})
        return parsed, derived

    for item in parsed["items"]:
        ok, msg, offsets = validate_semantics(item, source_by_id[item["annotation_item_id"]])
        derived["items"].append({
            "annotation_item_id": item["annotation_item_id"],
            "valid": ok,
            "error": None if ok else msg,
            **offsets,
        })
        if not ok:
            derived.update({"status": "REJECT", "layer": "Semantic Validator", "error_code": msg})
            return parsed, derived
    return parsed, derived


def call_gemini_live(prompt_text: str, provider_schema: dict[str, Any]) -> str:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_json_schema=provider_schema,
    )
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_text,
        config=config,
    )
    return response.text


def build_serving_provider_schema(_frozen_provider_schema: dict[str, Any]) -> dict[str, Any]:
    """Use serving-light field constraints; frozen/local validation still enforces the full contract."""
    short_evidence_schema = {
        "anyOf": [
            {"type": "string", "maxLength": 28, "pattern": r"^\S+(?:\s+\S+){0,6}$"},
            {"type": "null"},
        ]
    }
    item_properties: dict[str, Any] = {
        "annotation_item_id": {"type": "string"},
        "Aspect1": {"type": "string"},
        "Polarity1": {"type": "string"},
        "Evidence_Source1": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Evidence1": short_evidence_schema,
        "Aspect2": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Polarity2": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Evidence_Source2": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Evidence2": short_evidence_schema,
        "Review_Mixed_Flag": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": 8,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "properties": item_properties,
                    "required": list(item_properties),
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def connectivity_precheck() -> None:
    """Fail before recording REQUEST_SENT when the local environment blocks sockets."""
    try:
        with socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=10):
            return
    except OSError as e:
        raise RuntimeError(f"CONNECTIVITY_PRECHECK_FAILED:{type(e).__name__}:{e}") from e


def static_audit_source(repo: Path) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    text = source_path.read_text(encoding="utf-8")
    required_terms = [
        "validate_live_approval",
        '"ack_live_provider": True',
        "batch_start",
        "batch_end",
        "CONTRACT_ATTEMPT_ID",
        "CONNECTIVITY_PRECHECK_FAILED",
        "build_serving_provider_schema",
        "provider_request_schema_sha256",
        "maxLength",
        "pattern",
        "max_provider_requests",
        "SingleWriterLock",
        "AppendOnlyLedger",
        "call_gemini_live",
        "response_json_schema",
        "PILOT_BATCH_IDS",
        "validate_live_output",
        "VALIDATION_PASS",
        "VALIDATION_FAIL",
        "PROVIDER_REQUEST_FAILED",
        "derived_validation_output.json",
        "Draft7Validator",
    ]
    missing = [term for term in required_terms if term not in text]
    if missing:
        raise RuntimeError(f"Live executor source audit missing terms: {missing}")
    report = {
        "live_executor_static_audit_status": "PASS",
        "source_sha256": sha256_file(source_path),
        "hard_scope_campaign": CAMPAIGN_ID,
        "hard_scope_batches": PILOT_BATCH_IDS,
        "contract_attempt": CONTRACT_ATTEMPT_ID,
        "live_attempt_from_approval_file": True,
        "requires_live_approval_ack_true": True,
        "uses_single_writer_lock": True,
        "uses_append_only_ledger": True,
        "contains_provider_call_only_inside_call_gemini_live": True,
        "uses_serving_light_provider_schema_with_strict_local_validation": True,
        "provider_request_sent": False,
        "live_run_executed": False,
    }
    out = repo / FEATURE_DIR / f"v12exp_primary_pilot_live_executor_static_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}.json"
    write_new(out, canonical_json_bytes(report))
    report["report_path"] = str(out)
    report["report_sha256"] = sha256_file(out)
    return report


def execute_live(repo: Path, workspace_root: Path, approval_file: Path) -> None:
    static_report = load_json(repo / STATIC_VERIFY_REPORT)
    if static_report["gate7_7_static_payload_contract_status"] != "VERIFY_ONLY_PASS":
        raise RuntimeError("Live executor requires Gate 7.7 static verify-only PASS")
    base_schema = load_json(repo / SCHEMA_PATH)
    approval = validate_live_approval(approval_file)
    workspace_root = workspace_root.resolve()
    if any("onedrive" in part.lower() for part in workspace_root.parts):
        raise RuntimeError(f"Workspace root must be non-sync; rejected OneDrive path: {workspace_root}")
    workspace_root.mkdir(parents=True, exist_ok=True)
    execution_attempt_id = approval["attempt_id"]
    selected_batch_ids = PILOT_BATCH_IDS[
        PILOT_BATCH_IDS.index(approval["batch_start"]): PILOT_BATCH_IDS.index(approval["batch_end"]) + 1
    ]
    range_id = f"{approval['batch_start']}_{approval['batch_end']}_{execution_attempt_id}"
    lock_path = workspace_root / "campaign_locks" / CAMPAIGN_ID / f"{range_id}.lock"
    workspace_id = f"LIVEPILOT_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:12]}"
    workspace = workspace_root / "executor_workspaces" / workspace_id
    ledger = AppendOnlyLedger(workspace / "execution_ledger" / "ledger_live.jsonl", workspace_id, approval)

    with SingleWriterLock(lock_path):
        sent_requests = 0
        for batch_id in selected_batch_ids:
            verified = verify_contract(repo, batch_id)
            paths = verified["paths"]
            hashes = verified["hashes"]
            base = {
                "campaign_run_id": CAMPAIGN_ID,
                "batch_id": batch_id,
                "attempt_id": execution_attempt_id,
                "contract_attempt_id": CONTRACT_ATTEMPT_ID,
                "provider_input_sha256": hashes["provider_input_sha256"],
                "provenance_snapshot_sha256": hashes["provenance_snapshot_sha256"],
                "rendered_prompt_sha256": hashes["rendered_prompt_sha256"],
                "provider_response_schema_sha256": hashes["provider_response_schema_sha256"],
                "raw_response_sha256": None,
                "parsed_output_sha256": None,
                "derived_validation_sha256": None,
            }
            ledger.append({**base, "event_type": "ATTEMPT_PLANNED", "provider_request_sent": False, "provider_response_received": False, "terminal_reason": None})
            if sent_requests >= approval["max_provider_requests"]:
                raise RuntimeError("Kill switch: max live pilot request count exceeded")
            prompt_text = paths["rendered_prompt"].read_text(encoding="utf-8")
            provider_schema = load_json(paths["schema"])
            serving_provider_schema = build_serving_provider_schema(provider_schema)
            request_schema_path = workspace / "provider_request_schemas" / CAMPAIGN_ID / batch_id / execution_attempt_id / "provider_request_schema.json"
            provider_request_schema_sha = write_new(request_schema_path, canonical_json_bytes(serving_provider_schema))
            base["provider_request_schema_sha256"] = provider_request_schema_sha
            try:
                connectivity_precheck()
            except Exception as e:
                terminal_reason = str(e)
                ledger.append({**base, "event_type": "CONNECTIVITY_PRECHECK_FAILED", "provider_request_sent": False, "provider_response_received": False, "terminal_reason": terminal_reason})
                ledger.append({**base, "event_type": "ATTEMPT_TERMINAL", "provider_request_sent": False, "provider_response_received": False, "terminal_reason": terminal_reason})
                raise
            ledger.append({**base, "event_type": "REQUEST_SENT", "provider_request_sent": True, "provider_response_received": False, "terminal_reason": None})
            try:
                raw_text = call_gemini_live(prompt_text, serving_provider_schema)
            except Exception as e:
                terminal_reason = f"PROVIDER_REQUEST_FAILED:{type(e).__name__}:{e}"
                ledger.append({**base, "event_type": "PROVIDER_REQUEST_FAILED", "provider_request_sent": True, "provider_response_received": False, "terminal_reason": terminal_reason})
                ledger.append({**base, "event_type": "ATTEMPT_TERMINAL", "provider_request_sent": True, "provider_response_received": False, "terminal_reason": terminal_reason})
                raise
            sent_requests += 1
            batch_out_dir = workspace / "live_outputs" / CAMPAIGN_ID / batch_id / execution_attempt_id
            raw_path = batch_out_dir / "raw_provider_response.txt"
            raw_sha = write_new(raw_path, raw_text.encode("utf-8"))
            provider_input = load_json(paths["provider_input"])
            parsed, derived = validate_live_output(raw_text, provider_input, base_schema)
            parsed_path = batch_out_dir / "parsed_output.json"
            derived_path = batch_out_dir / "derived_validation_output.json"
            parsed_sha = write_new(parsed_path, canonical_json_bytes(parsed))
            derived_sha = write_new(derived_path, canonical_json_bytes(derived))
            response_base = {
                **base,
                "raw_response_sha256": raw_sha,
                "parsed_output_sha256": parsed_sha,
                "derived_validation_sha256": derived_sha,
            }
            ledger.append({**response_base, "event_type": "RESPONSE_RECEIVED", "provider_request_sent": True, "provider_response_received": True, "terminal_reason": None})
            if derived["status"] != "PASS":
                ledger.append({**response_base, "event_type": "VALIDATION_FAIL", "provider_request_sent": True, "provider_response_received": True, "terminal_reason": derived["error_code"]})
                ledger.append({**response_base, "event_type": "ATTEMPT_TERMINAL", "provider_request_sent": True, "provider_response_received": True, "terminal_reason": f"STOP_ON_VALIDATION_FAIL:{derived['layer']}:{derived['error_code']}"})
                raise RuntimeError(f"{batch_id} validation failed: {derived['layer']} {derived['error_code']}")
            ledger.append({**response_base, "event_type": "VALIDATION_PASS", "provider_request_sent": True, "provider_response_received": True, "terminal_reason": None})
            ledger.append({**response_base, "event_type": "ATTEMPT_TERMINAL", "provider_request_sent": True, "provider_response_received": True, "terminal_reason": "LIVE_VALIDATION_PASS"})
    print(json.dumps({"live_pilot_status": "LIVE_VALIDATION_PASS", "requests_sent": sent_requests, "workspace": str(workspace)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--workspace-root")
    parser.add_argument("--approval-file")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--static-audit", action="store_true")
    mode.add_argument("--execute-live", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    if args.static_audit:
        print(json.dumps(static_audit_source(repo), indent=2))
        return
    if not args.workspace_root or not args.approval_file:
        raise RuntimeError("--execute-live requires --workspace-root and --approval-file")
    execute_live(repo, Path(args.workspace_root), Path(args.approval_file).resolve())


if __name__ == "__main__":
    main()
