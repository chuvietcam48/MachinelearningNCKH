#!/usr/bin/env python
"""Gate 7.11 v1.3 calibration runner.

This runner is intentionally scoped to the 48-review v1.3 calibration manifest:
6 batches, exactly 8 items per batch, one base provider request per batch, no
blind retry, and no human repair. It can static-audit locally without provider
traffic; live execution requires an explicit approval file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


POLICY_VERSION = "v1.3.5"
MANIFEST_SOURCE_POLICY_VERSION = "v1.3"
CAMPAIGN_ID = "V135CALIBRATION_V1"
MODEL_NAME = "gemini-2.5-flash"
EXPECTED_BATCHES = [f"V13CAL_B{i:03d}" for i in range(1, 7)]
EXPECTED_ITEM_COUNT = 48
EXPECTED_BATCH_SIZE = 8

OUT_DIR = Path("outputs/amazon_v5_rebuild/annotation/v1_3_calibration_v2")
MANIFEST_PATH = OUT_DIR / "v13_calibration_manifest_48_v2.json"
HUMAN_ADJ_PATH = OUT_DIR / "v13_one_person_human_adjudication_completed_v2.csv"
PROMPT_PATH = Path("outputs/amazon_v5_rebuild/annotation/gemini_semantic_prompt_template_v1_3_5_draft_for_calibration.md")
TAXONOMY_PATH = Path("outputs/amazon_v5_rebuild/annotation/taxonomy_v1_3_draft_for_calibration.md")
POLICY_MANIFEST_PATH = Path("outputs/amazon_v5_rebuild/annotation/v1_3_policy_manifest.json")
CANONICAL_SCHEMA_PATH = Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json")
PROVIDER_SCHEMA_PATH = Path("outputs/amazon_v5_rebuild/features_semantic_b3/V12EXP_PRIMARY_V3/manifests/provider_response_schema_frozen_v1_2.json")

LOCKED_HASHES = {
    "manifest": "ea920f6da541575130deb65d8ed9d508aa3c0041a7d500a793b09d5738fa902c",
    "human_adjudication": "16c1e8a344943ad16ab58e1745705599d7e4ff9f19b3001808ac00fb51e405ca",
    "prompt": "9a3955ca459404022187b33a3b36efd178161e438775d788205e109ab20ac4cf",
    "taxonomy": "d130bde48fb66c9979ba79808e9e9bf87aec4c9d7f992337a5768fd1c80b66ac",
    "policy_manifest": "cdc6a312ee78f984d270d137dcdd9d33644cb61b0729737c1a7ee2591dda1b43",
    "canonical_schema": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
    "provider_schema": "003371c1939ddf14a5f453889abc193775b6bb87c96f95a65c0f7a67a2c40f37",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_new(path: Path, data: bytes) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def assert_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, got {actual}")


def assert_locked_inputs(repo: Path) -> None:
    paths = {
        "manifest": repo / MANIFEST_PATH,
        "human_adjudication": repo / HUMAN_ADJ_PATH,
        "prompt": repo / PROMPT_PATH,
        "taxonomy": repo / TAXONOMY_PATH,
        "policy_manifest": repo / POLICY_MANIFEST_PATH,
        "canonical_schema": repo / CANONICAL_SCHEMA_PATH,
        "provider_schema": repo / PROVIDER_SCHEMA_PATH,
    }
    for label, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Missing locked input {label}: {path}")
        assert_hash(path, LOCKED_HASHES[label], label)


def load_manifest(repo: Path) -> dict[str, Any]:
    manifest = load_json(repo / MANIFEST_PATH)
    if manifest.get("policy_version") != MANIFEST_SOURCE_POLICY_VERSION:
        raise RuntimeError("Manifest source policy version mismatch")
    if manifest.get("item_count") != EXPECTED_ITEM_COUNT or manifest.get("batch_count") != len(EXPECTED_BATCHES):
        raise RuntimeError("Manifest count mismatch")
    if [b["calibration_batch_id"] for b in manifest["batches"]] != EXPECTED_BATCHES:
        raise RuntimeError("Manifest batch IDs are not the expected contiguous six batches")
    return manifest


def grouped_provider_inputs(manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {batch_id: [] for batch_id in EXPECTED_BATCHES}
    for item in sorted(manifest["items"], key=lambda x: x["item_position"]):
        batch_id = item["calibration_batch_id"]
        provider_input = item["provider_input"]
        grouped[batch_id].append({
            "annotation_item_id": provider_input["annotation_item_id"],
            "summary": provider_input.get("summary") or "",
            "reviewText": provider_input.get("reviewText") or "",
        })
    for batch_id, rows in grouped.items():
        if len(rows) != EXPECTED_BATCH_SIZE:
            raise RuntimeError(f"{batch_id} does not contain exactly 8 items")
        ids = [row["annotation_item_id"] for row in rows]
        if len(set(ids)) != EXPECTED_BATCH_SIZE:
            raise RuntimeError(f"{batch_id} contains duplicate annotation_item_id")
    all_ids = [row["annotation_item_id"] for rows in grouped.values() for row in rows]
    if len(set(all_ids)) != EXPECTED_ITEM_COUNT:
        raise RuntimeError("Calibration manifest contains duplicate IDs across batches")
    return grouped


def repeated_evidence_phrases_to_avoid(*texts: str) -> list[str]:
    """Return short exact phrases that occur more than once in the item text."""
    joined = "\n".join(text or "" for text in texts)
    tokens = list(re.finditer(r"\S+", joined))
    repeated: set[str] = set()
    for n in range(2, 8):
        for i in range(0, max(0, len(tokens) - n + 1)):
            phrase = joined[tokens[i].start():tokens[i + n - 1].end()]
            if "\n" in phrase or len(phrase) > 80:
                continue
            if joined.count(phrase) > 1:
                repeated.add(phrase)
    return sorted(repeated, key=lambda x: (-len(x.split()), x.lower()))[:80]


def candidate_evidence_spans_for_source(text: str) -> list[str]:
    """Generate exact, unique, short candidate spans from a source string."""
    text = text or ""
    candidates: set[str] = set()

    def add_candidate(phrase: str) -> None:
        for variant in {phrase, phrase.strip(" \t\"'()[]{}.,;:!?")}:
            if variant and len(variant.split()) <= 7 and text.count(variant) == 1:
                candidates.add(variant)

    stripped = text.strip()
    if stripped and len(stripped.split()) <= 7 and text.count(stripped) == 1:
        candidates.add(stripped)
    for chunk in re.split(r"[\r\n.!?;:]+", text):
        phrase = chunk.strip(" \t\"'()[]{}")
        add_candidate(phrase)
    tokens = list(re.finditer(r"\S+", text))
    keywords = {
        "album", "artist", "artists", "beautiful", "best", "better", "boring",
        "broken", "creative", "disappointed", "disappointing", "excellent",
        "fantastic", "flaw", "flawless", "good", "great", "liked", "love",
        "major", "music", "performance", "performances", "quality", "raw",
        "recorded", "recording", "selection", "song", "songs", "sound",
        "sounding", "style", "superb", "terrible", "title", "worth",
    }
    for n in range(2, 8):
        for i in range(0, max(0, len(tokens) - n + 1)):
            phrase = text[tokens[i].start():tokens[i + n - 1].end()]
            if "\n" in phrase or len(phrase) > 120 or text.count(phrase) != 1:
                continue
            words = [re.sub(r"^\W+|\W+$", "", m.group(0)).lower() for m in tokens[i:i + n]]
            if any(word in keywords for word in words):
                add_candidate(phrase)
    return sorted(candidates, key=lambda x: (-len(x.split()), x.lower()))[:120]


def candidate_evidence_spans(item: dict[str, str]) -> dict[str, list[str]]:
    return {
        "summary": candidate_evidence_spans_for_source(item["summary"]),
        "reviewText": candidate_evidence_spans_for_source(item["reviewText"]),
    }


def render_prompt(prompt_template: str, items: list[dict[str, str]]) -> str:
    prompt = prompt_template.replace("{EXPECTED_BATCH_SIZE}", str(len(items)))
    for item in items:
        avoid = repeated_evidence_phrases_to_avoid(item["summary"], item["reviewText"])
        candidates = candidate_evidence_spans(item)
        prompt += (
            "\n--- ITEM ---\n"
            f"annotation_item_id: {item['annotation_item_id']}\n"
            f"summary: {item['summary']}\n"
            f"reviewText: {item['reviewText']}\n"
            f"repeated_evidence_phrases_to_avoid: {json.dumps(avoid, ensure_ascii=False)}\n"
            f"candidate_evidence_spans: {json.dumps(candidates, ensure_ascii=False)}\n"
        )
    return prompt


def build_run_schema(base_schema: dict[str, Any], batch_size: int) -> dict[str, Any]:
    run_schema = json.loads(json.dumps(base_schema))
    run_schema["properties"]["items"]["minItems"] = batch_size
    run_schema["properties"]["items"]["maxItems"] = batch_size
    return run_schema


def build_serving_provider_schema() -> dict[str, Any]:
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
                "minItems": EXPECTED_BATCH_SIZE,
                "maxItems": EXPECTED_BATCH_SIZE,
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
    if mixed is True and not ("Positive" in polarities and "Negative" in polarities):
        return False, "Rule D: mixed=true requires at least one Positive and one Negative aspect", {}
    if mixed is False and "Positive" in polarities and "Negative" in polarities:
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


def validate_provider_output(raw_text: str, provider_input: list[dict[str, str]], base_schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    requested_ids = [row["annotation_item_id"] for row in provider_input]
    source_by_id = {row["annotation_item_id"]: row for row in provider_input}
    derived = {"status": "PASS", "layer": "All", "error_code": "None", "items": []}
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
        derived["items"].append({"annotation_item_id": item["annotation_item_id"], "valid": ok, "error": None if ok else msg, **offsets})
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
    provider_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_json_schema=provider_schema,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt_text,
        config=provider_config,
    )
    return response.text


def connectivity_precheck() -> None:
    try:
        with socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=10):
            return
    except OSError as e:
        raise RuntimeError(f"CONNECTIVITY_PRECHECK_FAILED:{type(e).__name__}:{e}") from e


def load_human_rows(repo: Path) -> dict[str, dict[str, str]]:
    with (repo / HUMAN_ADJ_PATH).open("r", encoding="utf-8-sig", newline="") as f:
        return {row["annotation_item_id"]: row for row in csv.DictReader(f)}


def label_multiset_from_model(item: dict[str, Any]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if item["Aspect1"] != "None":
        pairs.append((item["Aspect1"], item["Polarity1"]))
    if item["Aspect2"] is not None:
        pairs.append((item["Aspect2"], item["Polarity2"]))
    return sorted(pairs)


def label_multiset_from_human(row: dict[str, str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    if row["human_Aspect1"] != "None":
        pairs.append((row["human_Aspect1"], row["human_Polarity1"]))
    if row["human_Aspect2"]:
        pairs.append((row["human_Aspect2"], row["human_Polarity2"]))
    return sorted(pairs)


def compare_against_human(repo: Path, run_dir: Path) -> dict[str, Any]:
    human = load_human_rows(repo)
    manifest = load_manifest(repo)
    split_by_id = {item["annotation_item_id"]: item["selection_split"] for item in manifest["items"]}
    records = []
    split_stats: dict[str, dict[str, int]] = {}
    for batch_id in EXPECTED_BATCHES:
        parsed_path = run_dir / "batches" / batch_id / "parsed_output.json"
        if not parsed_path.exists():
            raise RuntimeError(f"Missing parsed output for {batch_id}: {parsed_path}")
        parsed = load_json(parsed_path)
        for item in parsed["items"]:
            aid = item["annotation_item_id"]
            split = split_by_id[aid]
            split_stats.setdefault(split, {"n": 0, "aspect_set_agree": 0, "mixed_agree": 0})
            model_pairs = label_multiset_from_model(item)
            human_pairs = label_multiset_from_human(human[aid])
            mixed_model = item["Review_Mixed_Flag"]
            mixed_human = human[aid]["human_Review_Mixed_Flag"].lower() == "true"
            aspect_agree = model_pairs == human_pairs
            mixed_agree = mixed_model == mixed_human
            split_stats[split]["n"] += 1
            split_stats[split]["aspect_set_agree"] += int(aspect_agree)
            split_stats[split]["mixed_agree"] += int(mixed_agree)
            records.append({
                "annotation_item_id": aid,
                "selection_split": split,
                "model_aspect_polarity_multiset": model_pairs,
                "human_aspect_polarity_multiset": human_pairs,
                "aspect_set_agreement": aspect_agree,
                "mixed_flag_agreement": mixed_agree,
            })
    p000007 = next(r for r in records if r["annotation_item_id"] == "A102XSQH2IW56B_3720498")
    double_count_errors = [
        r for r in records
        if r["annotation_item_id"] == "A102XSQH2IW56B_3720498"
        and ("Product_Condition_Quality", "Positive") in r["model_aspect_polarity_multiset"]
    ]
    status = "V13_CALIBRATION_PASS" if (
        all(r["aspect_set_agreement"] and r["mixed_flag_agreement"] for r in records)
        and not double_count_errors
        and p000007["model_aspect_polarity_multiset"] == [("Domain_Experience", "Positive")]
    ) else "V13_CALIBRATION_REVIEW_REQUIRED"
    return {
        "status": status,
        "metric_name": "agreement against one-person human adjudication",
        "single_accuracy_claim": False,
        "item_count": len(records),
        "split_stats": split_stats,
        "p000007_sentinel_record": p000007,
        "creative_style_double_count_error_count": len(double_count_errors),
        "records": records,
    }


def validate_approval(path: Path) -> dict[str, Any]:
    approval = load_json(path)
    required = {
        "execution_mode": "LIVE",
        "campaign_run_id": CAMPAIGN_ID,
        "policy_version": POLICY_VERSION,
        "batch_start": EXPECTED_BATCHES[0],
        "batch_end": EXPECTED_BATCHES[-1],
        "ack_live_provider": True,
        "max_provider_requests": len(EXPECTED_BATCHES),
        "max_concurrency": 1,
        "no_blind_retry": True,
        "human_repair_allowed": False,
    }
    for key, expected in required.items():
        if approval.get(key) != expected:
            raise RuntimeError(f"Approval mismatch for {key}: expected {expected!r}, got {approval.get(key)!r}")
    attempt_id = approval.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.startswith("A") or not attempt_id[1:].isdigit():
        raise RuntimeError(f"Approval attempt_id must look like A001/A002/etc; got {attempt_id!r}")
    if not approval.get("approval_id"):
        raise RuntimeError("Approval must include approval_id")
    approval["approval_file_sha256"] = sha256_file(path)
    return approval


def assert_non_sync_workspace_root(path: Path) -> Path:
    resolved = path.resolve()
    if any("onedrive" in part.lower() for part in resolved.parts):
        raise RuntimeError(f"Workspace root must be non-sync; rejected OneDrive path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def static_audit(repo: Path) -> dict[str, Any]:
    assert_locked_inputs(repo)
    manifest = load_manifest(repo)
    grouped = grouped_provider_inputs(manifest)
    prompt_template = (repo / PROMPT_PATH).read_text(encoding="utf-8")
    rendered_hashes = {
        batch_id: sha256_bytes(render_prompt(prompt_template, items).encode("utf-8"))
        for batch_id, items in grouped.items()
    }
    report = {
        "status": "V13_CALIBRATION_STATIC_AUDIT_PASS",
        "campaign_run_id": CAMPAIGN_ID,
        "policy_version": POLICY_VERSION,
        "batch_ids": EXPECTED_BATCHES,
        "item_count": EXPECTED_ITEM_COUNT,
        "batch_size": EXPECTED_BATCH_SIZE,
        "locked_hashes": LOCKED_HASHES,
        "rendered_prompt_sha256_by_batch": rendered_hashes,
        "provider_config": {
            "response_json_schema_present": True,
            "response_format_absent": True,
            "response_schema_absent": True,
            "temperature": 0.0,
            "response_mime_type": "application/json",
        },
        "serving_light_schema_sha256": sha256_bytes(canonical_json_bytes(build_serving_provider_schema())),
        "provider_calls_made": False,
    }
    out = repo / OUT_DIR / f"gate7_11_v13_calibration_static_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}.json"
    write_new(out, canonical_json_bytes(report))
    report["report_path"] = str(out)
    report["report_sha256"] = sha256_file(out)
    return report


def execute_live(repo: Path, workspace_root: Path, approval_file: Path) -> dict[str, Any]:
    assert_locked_inputs(repo)
    approval = validate_approval(approval_file)
    workspace_root = assert_non_sync_workspace_root(workspace_root)
    attempt_id = approval["attempt_id"]
    run_id = f"{CAMPAIGN_ID}_{attempt_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"
    run_dir = workspace_root / "v13_calibration_runs" / run_id
    if run_dir.exists():
        raise RuntimeError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    manifest = load_manifest(repo)
    grouped = grouped_provider_inputs(manifest)
    base_schema = load_json(repo / CANONICAL_SCHEMA_PATH)
    prompt_template = (repo / PROMPT_PATH).read_text(encoding="utf-8")
    serving_schema = build_serving_provider_schema()
    serving_schema_sha = write_new(run_dir / "provider_request_schema.json", canonical_json_bytes(serving_schema))
    ledger_path = run_dir / "ledger_live.jsonl"
    sent_requests = 0
    terminal_reason = None
    for batch_id in EXPECTED_BATCHES:
        items = grouped[batch_id]
        batch_dir = run_dir / "batches" / batch_id
        provider_input_sha = write_new(batch_dir / "provider_input.json", canonical_json_bytes(items))
        prompt_text = render_prompt(prompt_template, items)
        rendered_prompt_sha = write_new(batch_dir / "rendered_prompt.md", prompt_text.encode("utf-8"))
        event_base = {
            "event_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "execution_mode": "LIVE",
            "campaign_run_id": CAMPAIGN_ID,
            "policy_version": POLICY_VERSION,
            "batch_id": batch_id,
            "attempt_id": attempt_id,
            "approval_id": approval["approval_id"],
            "approval_file_sha256": approval["approval_file_sha256"],
            "provider_input_sha256": provider_input_sha,
            "rendered_prompt_sha256": rendered_prompt_sha,
            "provider_request_schema_sha256": serving_schema_sha,
        }
        with ledger_path.open("ab") as f:
            f.write(canonical_json_bytes({**event_base, "event_type": "ATTEMPT_PLANNED", "provider_request_sent": False}) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        if sent_requests >= approval["max_provider_requests"]:
            raise RuntimeError("Kill switch: max provider request count exceeded")
        try:
            connectivity_precheck()
        except Exception as e:
            terminal_reason = str(e)
            with ledger_path.open("ab") as f:
                f.write(canonical_json_bytes({
                    **event_base,
                    "event_type": "CONNECTIVITY_PRECHECK_FAILED",
                    "provider_request_sent": False,
                    "provider_response_received": False,
                    "terminal_reason": terminal_reason,
                }) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            break
        with ledger_path.open("ab") as f:
            f.write(canonical_json_bytes({**event_base, "event_type": "REQUEST_SENT", "provider_request_sent": True}) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        sent_requests += 1
        try:
            raw_text = call_gemini_live(prompt_text, serving_schema)
        except Exception as e:
            terminal_reason = f"PROVIDER_REQUEST_FAILED:{type(e).__name__}:{e}"
            with ledger_path.open("ab") as f:
                f.write(canonical_json_bytes({
                    **event_base,
                    "event_type": "PROVIDER_REQUEST_FAILED",
                    "provider_request_sent": True,
                    "provider_response_received": False,
                    "terminal_reason": terminal_reason,
                }) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            break
        raw_sha = write_new(batch_dir / "raw_provider_response.txt", raw_text.encode("utf-8"))
        parsed, derived = validate_provider_output(raw_text, items, base_schema)
        parsed_sha = write_new(batch_dir / "parsed_output.json", canonical_json_bytes(parsed))
        derived_sha = write_new(batch_dir / "derived_validation_output.json", canonical_json_bytes(derived))
        with ledger_path.open("ab") as f:
            f.write(canonical_json_bytes({
                **event_base,
                "event_type": "RESPONSE_VALIDATED",
                "provider_request_sent": True,
                "raw_response_sha256": raw_sha,
                "parsed_output_sha256": parsed_sha,
                "derived_validation_sha256": derived_sha,
                "validation_status": derived["status"],
                "validation_layer": derived["layer"],
                "validation_error_code": derived["error_code"],
            }) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        if derived["status"] != "PASS":
            terminal_reason = f"STOP_ON_VALIDATION_FAIL:{batch_id}:{derived['layer']}:{derived['error_code']}"
            break
    if terminal_reason is None:
        comparison = compare_against_human(repo, run_dir)
        comparison_sha = write_new(run_dir / "v13_calibration_comparison_report.json", canonical_json_bytes(comparison))
        terminal_reason = comparison["status"]
    else:
        comparison_sha = None
    summary = {
        "status": terminal_reason,
        "campaign_run_id": CAMPAIGN_ID,
        "policy_version": POLICY_VERSION,
        "run_id": run_id,
        "workspace_run_dir": str(run_dir),
        "requests_sent": sent_requests,
        "max_provider_requests": approval["max_provider_requests"],
        "human_repair_applied": False,
        "blind_retry_used": False,
        "comparison_report_sha256": comparison_sha,
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.exists() else None,
    }
    summary_path = repo / OUT_DIR / f"gate7_11_v13_calibration_live_summary_{run_id}.json"
    write_new(summary_path, canonical_json_bytes(summary))
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


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
        print(json.dumps(static_audit(repo), indent=2, ensure_ascii=False))
        return
    if not args.workspace_root or not args.approval_file:
        raise RuntimeError("--execute-live requires --workspace-root and --approval-file")
    print(json.dumps(execute_live(repo, Path(args.workspace_root), Path(args.approval_file).resolve()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
