#!/usr/bin/env python
"""Document one human evidence-only repair for V12EXP primary pilot P000002.

This does not modify raw provider output. It writes a separate adjudicated layer
and repair ledger entry, then validates the repaired output with the unchanged
local validator.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "V12EXP_PRIMARY_V3"
BATCH_ID = "P000002"
SOURCE_ATTEMPT_ID = "A006"
ANNOTATION_ITEM_ID = "A101OMG474Q26I_1420194"
OLD_EVIDENCE = "leave off three of this top 40 hits"
NEW_EVIDENCE = "leave off three"
FEATURE_DIR = Path("outputs/amazon_v5_rebuild/features_semantic_b3")
SCHEMA_PATH = Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json")


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
        raise RuntimeError(f"Refusing to overwrite repair artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def load_executor_module(repo: Path):
    source = repo / "src/amazon_v5_rebuild/16_v12exp_primary_pilot_live_executor.py"
    spec = importlib.util.spec_from_file_location("v12exp_live_executor", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load executor module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_source_artifacts(workspace_root: Path) -> dict[str, Path]:
    matches = sorted(
        workspace_root.glob(f"executor_workspaces/*/live_outputs/{CAMPAIGN_ID}/{BATCH_ID}/{SOURCE_ATTEMPT_ID}/parsed_output.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise RuntimeError(f"No source parsed output found for {BATCH_ID}/{SOURCE_ATTEMPT_ID} under {workspace_root}")
    parsed = matches[0]
    base = parsed.parent
    return {
        "parsed": parsed,
        "raw": base / "raw_provider_response.txt",
        "derived": base / "derived_validation_output.json",
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    workspace_root = Path(os.environ.get("LOCALAPPDATA", "")) / "MLNCKH_GeminiPilotWorkspace"
    if not workspace_root.exists():
        raise RuntimeError(f"Workspace root not found: {workspace_root}")

    source_paths = find_source_artifacts(workspace_root)
    parsed = load_json(source_paths["parsed"])
    raw_sha = sha256_file(source_paths["raw"])
    parsed_sha = sha256_file(source_paths["parsed"])
    source_derived = load_json(source_paths["derived"])
    if source_derived.get("status") != "REJECT" or source_derived.get("error_code") != "Rule C: evidence quote exceeds 7 words":
        raise RuntimeError("Source A006 derived output is not the expected evidence-length rejection")

    repaired = json.loads(json.dumps(parsed))
    target = None
    for item in repaired["items"]:
        if item.get("annotation_item_id") == ANNOTATION_ITEM_ID:
            target = item
            break
    if target is None:
        raise RuntimeError(f"Missing target annotation item: {ANNOTATION_ITEM_ID}")
    if target.get("Evidence1") != OLD_EVIDENCE:
        raise RuntimeError(f"Unexpected old Evidence1: {target.get('Evidence1')!r}")
    before_semantic = {k: v for k, v in target.items() if k != "Evidence1"}
    target["Evidence1"] = NEW_EVIDENCE
    after_semantic = {k: v for k, v in target.items() if k != "Evidence1"}
    if before_semantic != after_semantic:
        raise RuntimeError("Repair changed semantic fields beyond Evidence1")
    if target.get("Evidence_Source1") != "reviewText":
        raise RuntimeError("Evidence source changed or unexpected")

    executor = load_executor_module(repo)
    provider_input = load_json(repo / FEATURE_DIR / CAMPAIGN_ID / "batches" / BATCH_ID / "A001" / "provider_input.json")
    base_schema = load_json(repo / SCHEMA_PATH)
    repaired_text = json.dumps(repaired, ensure_ascii=False)
    _, repaired_derived = executor.validate_live_output(repaired_text, provider_input, base_schema)
    if repaired_derived.get("status") != "PASS":
        raise RuntimeError(f"Repaired output did not validate: {repaired_derived}")

    timestamp = datetime.now(timezone.utc).isoformat()
    out_dir = repo / FEATURE_DIR / CAMPAIGN_ID / "human_evidence_repairs" / BATCH_ID / SOURCE_ATTEMPT_ID
    repaired_output_path = out_dir / "adjudicated_repaired_output.json"
    repaired_derived_path = out_dir / "adjudicated_repaired_validation.json"
    ledger_path = out_dir / "human_evidence_repair_ledger.json"
    policy_path = out_dir / "pilot_exception_policy.json"

    repaired_output_sha = write_new(repaired_output_path, canonical_json_bytes(repaired))
    repaired_derived_sha = write_new(repaired_derived_path, canonical_json_bytes(repaired_derived))
    ledger = {
        "status": "RESOLVED_BY_DOCUMENTED_HUMAN_EVIDENCE_ONLY_REPAIR",
        "campaign_run_id": CAMPAIGN_ID,
        "batch_id": BATCH_ID,
        "source_attempt_id": SOURCE_ATTEMPT_ID,
        "raw_attempt_statuses": {
            "P000002/A004": "RAW_PROVIDER_VALIDATION_FAIL",
            "P000002/A005": "RAW_PROVIDER_VALIDATION_FAIL",
            "P000002/A006": "RAW_PROVIDER_VALIDATION_FAIL",
        },
        "raw_response_sha256": raw_sha,
        "raw_output_sha256": parsed_sha,
        "repaired_output_sha256": repaired_output_sha,
        "repaired_validation_sha256": repaired_derived_sha,
        "annotation_item_id": ANNOTATION_ITEM_ID,
        "field_changed": "Evidence1 only",
        "old_evidence": OLD_EVIDENCE,
        "new_evidence": NEW_EVIDENCE,
        "semantic_fields_changed": False,
        "evidence_source_unchanged": True,
        "validator_result_on_repaired_output": "PASS",
        "reason": "provider exceeded frozen max-7-word evidence rule despite transmitted schema constraint",
        "adjudicator": "human_adjudicator",
        "timestamp_utc": timestamp,
        "raw_model_output_overwritten": False,
        "repair_ledger_event_id": str(uuid.uuid4()),
    }
    ledger_sha = write_new(ledger_path, canonical_json_bytes(ledger))
    policy = {
        "pilot_human_evidence_repair_count": 1,
        "repair_scope": "evidence-only",
        "new_semantic_disagreement_or_second_repair": "STOP_AND_AUDIT",
        "pilot_completion_status_allowed_after_full_audit": "PILOT_AUDIT_PASS_WITH_DOCUMENTED_EVIDENCE_EXCEPTION",
        "timestamp_utc": timestamp,
        "linked_repair_ledger_sha256": ledger_sha,
    }
    policy_sha = write_new(policy_path, canonical_json_bytes(policy))
    print(json.dumps({
        "repair_status": ledger["status"],
        "validator_result_on_repaired_output": "PASS",
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "repaired_output_path": str(repaired_output_path),
        "repaired_output_sha256": repaired_output_sha,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha,
    }, indent=2))


if __name__ == "__main__":
    main()
