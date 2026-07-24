#!/usr/bin/env python
"""Gate 7.26 v2.2 aspect-only calibration runner.

Provider emits only aspect/polarity targets. Local deterministic code selects
exact evidence spans, assembles canonical output, and validates offsets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import socket
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


BASE_RUNNER_PATH = Path(__file__).with_name("22_gate7_11_v14_candidate_id_calibration_runner.py")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("gate7_11_v14_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_runner()

POLICY_VERSION = "semantic_policy_v2_2_aspect_only_atomic"
CAMPAIGN_ID = "V22ASPECTONLYCALIBRATION_FRESH_V1"
MODEL_NAME = base.MODEL_NAME
EXPECTED_BATCHES = [f"V14CAL_B{i:03d}" for i in range(1, 13)]
EXPECTED_ITEM_COUNT = 96
EXPECTED_BATCH_SIZE = 8
OUT_DIR = base.OUT_DIR
MANIFEST_PATH = OUT_DIR / "gate7_16_v14_candidate_id_execution_manifest_96.json"
HUMAN_ADJ_PATH = OUT_DIR / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
CANONICAL_SCHEMA_PATH = base.CANONICAL_SCHEMA_PATH
POLICY_PATH = OUT_DIR / "semantic_policy_v2_2_aspect_only_atomic.md"
LOCKED_HASHES = {
    "manifest": "2437e322c64c113bf7bb49416551ddd49dc52dcecb7dbe89ed5c9572d6938c48",
    "human_adjudication": "f70b57612f30651a2242195452a46c610629da82604ee18206b10729128f30a5",
    "canonical_schema": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
}
ASPECTS = [a for a in base.ASPECTS if a != "None"]
POLARITIES = ["Positive", "Negative"]


KEYWORDS_BY_ASPECT = {
    "Domain_Experience": {
        "positive": ["love", "loved", "great", "good", "excellent", "fantastic", "best", "classic", "enjoy", "fun", "recommended", "masterful", "interesting"],
        "negative": ["disappoint", "boring", "not", "lame", "weak", "bad", "mediocre", "skip", "hard", "raw", "same level"],
    },
    "Product_Condition_Quality": {
        "positive": ["sound", "quality", "sonics", "recorded", "clear", "remaster", "deep", "full"],
        "negative": ["broken", "case", "sound", "quality", "frequency", "noise", "hard to hear", "mediocre", "muted", "defect"],
    },
    "Packaging_Presentation": {
        "positive": ["booklet", "cover", "box", "photos", "notes", "illustrated"],
        "negative": ["booklet", "cover", "box", "flaw", "disintegrate", "list", "titles", "packaging"],
    },
    "Price_Value": {
        "positive": ["worth", "bargain", "price", "cash", "purchase"],
        "negative": ["worth", "cost", "price", "paid", "shipping"],
    },
    "Listing_Expectation_Compatibility": {
        "positive": ["promised", "expected", "edition", "version"],
        "negative": ["expected", "wrong", "listing", "edition", "version", "not the same", "track list"],
    },
    "Product_Performance_Usability": {
        "positive": ["works", "use", "download", "playback"],
        "negative": ["button", "elude", "download", "playback", "access", "hidden"],
    },
    "Delivery_Fulfillment": {
        "positive": ["received", "arrived", "delivery", "shipping"],
        "negative": ["received", "arrived", "delivery", "shipping", "missing"],
    },
    "Customer_Service_Returns": {
        "positive": ["service", "refund", "return", "support"],
        "negative": ["service", "refund", "return", "support"],
    },
    "Other_Specific": {
        "positive": ["ok", "fine", "good"],
        "negative": ["problem", "issue", "bad"],
    },
}


def sha256_file(path: Path) -> str:
    return base.sha256_file(path)


def canonical_json_bytes(payload: Any) -> bytes:
    return base.canonical_json_bytes(payload)


def write_new(path: Path, data: bytes) -> str:
    return base.write_new(path, data)


def load_json(path: Path) -> Any:
    return base.load_json(path)


def assert_locked_inputs(repo: Path) -> None:
    paths = {
        "manifest": repo / MANIFEST_PATH,
        "human_adjudication": repo / HUMAN_ADJ_PATH,
        "canonical_schema": repo / CANONICAL_SCHEMA_PATH,
        "policy": repo / POLICY_PATH,
    }
    for label in ("manifest", "human_adjudication", "canonical_schema"):
        if not paths[label].exists():
            raise RuntimeError(f"Missing locked input {label}: {paths[label]}")
        actual = sha256_file(paths[label])
        if actual.lower() != LOCKED_HASHES[label].lower():
            raise RuntimeError(f"{label} hash mismatch: expected {LOCKED_HASHES[label]}, got {actual}")
    if not paths["policy"].exists():
        raise RuntimeError(f"Missing policy: {paths['policy']}")


def load_manifest(repo: Path) -> dict[str, Any]:
    manifest = load_json(repo / MANIFEST_PATH)
    if manifest.get("item_count") != EXPECTED_ITEM_COUNT:
        raise RuntimeError("Manifest item count mismatch")
    if [b["calibration_batch_id"] for b in manifest["batches"]] != EXPECTED_BATCHES:
        raise RuntimeError("Manifest batch IDs mismatch")
    return manifest


def grouped_provider_inputs(manifest: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {batch_id: [] for batch_id in EXPECTED_BATCHES}
    for item in sorted(manifest["items"], key=lambda x: x["item_position"]):
        provider_input = item["provider_input"]
        grouped[item["calibration_batch_id"]].append({
            "item_position": int(item["item_position_in_batch"]),
            "annotation_item_id": provider_input["annotation_item_id"],
            "summary": provider_input.get("summary") or "",
            "reviewText": provider_input.get("reviewText") or "",
        })
    for batch_id, rows in grouped.items():
        if len(rows) != EXPECTED_BATCH_SIZE:
            raise RuntimeError(f"{batch_id} item count != 8")
    return grouped


def build_provider_schema() -> dict[str, Any]:
    target = {
        "type": "object",
        "properties": {
            "aspect": {"type": "string"},
            "polarity": {"type": "string"},
        },
        "required": ["aspect", "polarity"],
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "item_position": {"type": "integer"},
            "targets": {"type": "array", "minItems": 0, "maxItems": 2, "items": target},
        },
        "required": ["item_position", "targets"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "items": {"type": "array", "minItems": EXPECTED_BATCH_SIZE, "maxItems": EXPECTED_BATCH_SIZE, "items": item}
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_raw_local_schema() -> dict[str, Any]:
    target = {
        "type": "object",
        "properties": {
            "aspect": {"type": "string", "enum": ASPECTS},
            "polarity": {"type": "string", "enum": POLARITIES},
        },
        "required": ["aspect", "polarity"],
        "additionalProperties": False,
    }
    item = {
        "type": "object",
        "properties": {
            "item_position": {"type": "integer"},
            "targets": {"type": "array", "minItems": 0, "maxItems": 2, "items": target},
        },
        "required": ["item_position", "targets"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "items": {"type": "array", "minItems": EXPECTED_BATCH_SIZE, "maxItems": EXPECTED_BATCH_SIZE, "items": item}
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def all_unique_windows(source: str, source_name: str) -> list[dict[str, str]]:
    source = source or ""
    tokens = list(re.finditer(r"\S+", source))
    windows = []
    seen = set()
    for n in range(min(7, len(tokens)), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            phrase = source[tokens[i].start():tokens[i + n - 1].end()].strip(" \t\"'()[]{}.,;:!?")
            if not phrase or "\n" in phrase or len(phrase.split()) > 7 or len(phrase) > 140:
                continue
            if phrase in seen or source.count(phrase) != 1:
                continue
            seen.add(phrase)
            windows.append({"source": source_name, "quote": phrase})
    return windows


def score_candidate(aspect: str, polarity: str, candidate: dict[str, str]) -> tuple[int, int, str]:
    quote = candidate["quote"]
    low = quote.lower()
    polarity_key = "positive" if polarity == "Positive" else "negative"
    score = 0
    for keyword in KEYWORDS_BY_ASPECT.get(aspect, {}).get(polarity_key, []):
        if keyword in low:
            score += 10
    if polarity == "Negative" and any(cue in low for cue in ("not", "no", "n't", "disappoint", "bad", "broken", "flaw")):
        score += 6
    if polarity == "Positive" and any(cue in low for cue in ("good", "great", "love", "excellent", "best", "worth")):
        score += 6
    score += min(len(quote.split()), 7)
    return (score, -len(quote), quote.lower())


def select_evidence(row: dict[str, str], aspect: str, polarity: str, used: set[tuple[str, str]]) -> dict[str, str]:
    candidates = []
    for source_name in ("reviewText", "summary"):
        candidates.extend(all_unique_windows(row.get(source_name) or "", source_name))
    candidates = [c for c in candidates if (c["source"], c["quote"]) not in used]
    if not candidates:
        raise RuntimeError(f"no evidence candidates for {row['annotation_item_id']}")
    return sorted(candidates, key=lambda c: score_candidate(aspect, polarity, c), reverse=True)[0]


def assemble(row: dict[str, str], targets: list[dict[str, str]]) -> dict[str, Any]:
    deduped = []
    seen = set()
    for target in targets:
        key = (target["aspect"], target["polarity"])
        if key not in seen:
            seen.add(key)
            deduped.append(target)
    item: dict[str, Any] = {"annotation_item_id": row["annotation_item_id"]}
    if not deduped:
        item.update({
            "Aspect1": "None", "Polarity1": "NotApplicable", "Evidence_Source1": None, "Evidence1": None,
            "Aspect2": None, "Polarity2": None, "Evidence_Source2": None, "Evidence2": None,
            "Review_Mixed_Flag": False,
        })
        return item
    used: set[tuple[str, str]] = set()
    ev1 = select_evidence(row, deduped[0]["aspect"], deduped[0]["polarity"], used)
    used.add((ev1["source"], ev1["quote"]))
    item.update({
        "Aspect1": deduped[0]["aspect"], "Polarity1": deduped[0]["polarity"],
        "Evidence_Source1": ev1["source"], "Evidence1": ev1["quote"],
    })
    if len(deduped) == 2:
        ev2 = select_evidence(row, deduped[1]["aspect"], deduped[1]["polarity"], used)
        item.update({
            "Aspect2": deduped[1]["aspect"], "Polarity2": deduped[1]["polarity"],
            "Evidence_Source2": ev2["source"], "Evidence2": ev2["quote"],
        })
    else:
        item.update({"Aspect2": None, "Polarity2": None, "Evidence_Source2": None, "Evidence2": None})
    polarities = [target["polarity"] for target in deduped if target["polarity"] in ("Positive", "Negative")]
    item["Review_Mixed_Flag"] = "Positive" in polarities and "Negative" in polarities
    return item


def render_prompt(items: list[dict[str, str]]) -> str:
    prompt = f"""You are an expert e-commerce review annotator.

Return exactly one JSON object with one key, items, containing exactly {len(items)} objects.

Each output item must contain exactly:
- item_position
- targets

targets is an array with 0, 1, or 2 central semantic targets.
Each target has exactly:
- aspect
- polarity

Do not output evidence, evidence IDs, quotes, offsets, final Aspect1/Aspect2 fields, or mixed flags.
Local deterministic code will select evidence and derive mixed feedback.

Aspect labels:
{json.dumps(ASPECTS, ensure_ascii=False)}

Polarity labels:
["Positive", "Negative"]

Rules:
- Use no targets when there is no concrete review target.
- Use one target by default.
- Use two targets only when there are two central concrete targets or clear positive and negative sentiment about central targets.
- Domain_Experience covers music/content/style/selection/curation/performance/artist/album/listening experience/creative production style/genre feel/atmosphere/musical energy/artistic direction.
- Product_Condition_Quality covers physical item defects, damaged playable media/case, and technical sound/recording/mastering quality.
- Packaging_Presentation covers booklet, liner notes, cover/back-cover metadata, box/cardboard, or visual presentation.
- Listing_Expectation_Compatibility requires explicit mismatch with listing, delivered item, edition, format, description, pictures, track list, or promised content.
- Prior album/series/artist expectations, availability wishes, edited mixes, and format preference are not listing mismatch unless the listing or delivered item was wrong.
- Price_Value requires explicit cost/worth/price/value judgment.
- Product_Performance_Usability is only for non-content operation, download/playback function, hidden-track access, usability, or functional failure.

Input items:
"""
    for item in items:
        prompt += (
            "\n--- ITEM ---\n"
            f"item_position: {item['item_position']}\n"
            f"summary: {item['summary']}\n"
            f"reviewText: {item['reviewText']}\n"
        )
    return prompt


def validate_provider_output(raw_text: str, rows: list[dict[str, str]], base_schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    derived = {"status": "PASS", "layer": "All", "error_code": "None", "items": []}
    try:
        raw = json.loads(raw_text)
    except Exception:
        derived.update({"status": "REJECT", "layer": "JSON Parser", "error_code": "MalformedJSON"})
        return {}, {}, derived
    raw_errors = sorted(Draft7Validator(build_raw_local_schema()).iter_errors(raw), key=lambda e: e.path)
    if raw_errors:
        derived.update({"status": "REJECT", "layer": "Provider Schema Validator", "error_code": raw_errors[0].message})
        return raw, {}, derived
    if len(raw["items"]) != len(rows):
        derived.update({"status": "REJECT", "layer": "Provider Schema Validator", "error_code": "item_count_mismatch"})
        return raw, {}, derived
    resolved = {"items": []}
    for idx, raw_item in enumerate(raw["items"]):
        try:
            resolved["items"].append(assemble(rows[idx], raw_item["targets"]))
        except Exception as exc:
            derived.update({"status": "REJECT", "layer": "Local Evidence Selector", "error_code": f"{type(exc).__name__}:{exc}"})
            return raw, resolved, derived
    canonical_errors = sorted(Draft7Validator(base.build_run_schema(base_schema, len(rows))).iter_errors(resolved), key=lambda e: e.path)
    if canonical_errors:
        derived.update({"status": "REJECT", "layer": "Canonical Schema Validator", "error_code": canonical_errors[0].message})
        return raw, resolved, derived
    for item in resolved["items"]:
        source = next(row for row in rows if row["annotation_item_id"] == item["annotation_item_id"])
        ok, msg, offsets = base.validate_semantics(item, source)
        derived["items"].append({"annotation_item_id": item["annotation_item_id"], "valid": ok, "error": None if ok else msg, **offsets})
        if not ok:
            derived.update({"status": "REJECT", "layer": "Semantic Validator", "error_code": msg})
            return raw, resolved, derived
    return raw, resolved, derived


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
    response = client.models.generate_content(model=MODEL_NAME, contents=prompt_text, config=config)
    return response.text


def connectivity_precheck() -> None:
    try:
        with socket.create_connection(("generativelanguage.googleapis.com", 443), timeout=10):
            return
    except OSError as exc:
        raise RuntimeError(f"CONNECTIVITY_PRECHECK_FAILED:{type(exc).__name__}:{exc}") from exc


def load_human_rows(repo: Path) -> dict[str, dict[str, str]]:
    with (repo / HUMAN_ADJ_PATH).open("r", encoding="utf-8-sig", newline="") as f:
        return {row["annotation_item_id"]: row for row in csv.DictReader(f)}


def pairs_from_model(item: dict[str, Any]) -> list[tuple[str, str]]:
    pairs = []
    if item["Aspect1"] != "None":
        pairs.append((item["Aspect1"], item["Polarity1"]))
    if item["Aspect2"] is not None:
        pairs.append((item["Aspect2"], item["Polarity2"]))
    return sorted(pairs)


def pairs_from_human(row: dict[str, str]) -> list[tuple[str, str]]:
    pairs = []
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
        resolved = load_json(run_dir / "batches" / batch_id / "resolved_output.json")
        for item in resolved["items"]:
            aid = item["annotation_item_id"]
            split = split_by_id[aid]
            stats = split_stats.setdefault(split, {"n": 0, "aspect_set_agree": 0, "mixed_agree": 0})
            model_pairs = pairs_from_model(item)
            human_pairs = pairs_from_human(human[aid])
            mixed_human = human[aid]["human_Review_Mixed_Flag"].lower() == "true"
            aspect_agree = model_pairs == human_pairs
            mixed_agree = item["Review_Mixed_Flag"] == mixed_human
            stats["n"] += 1
            stats["aspect_set_agree"] += int(aspect_agree)
            stats["mixed_agree"] += int(mixed_agree)
            records.append({
                "annotation_item_id": aid,
                "selection_split": split,
                "model_aspect_polarity_multiset": model_pairs,
                "human_aspect_polarity_multiset": human_pairs,
                "aspect_set_agreement": aspect_agree,
                "mixed_flag_agreement": mixed_agree,
            })
    status = "V22_ASPECT_ONLY_CALIBRATION_PASS" if all(r["aspect_set_agreement"] and r["mixed_flag_agreement"] for r in records) else "V22_ASPECT_ONLY_CALIBRATION_REVIEW_REQUIRED"
    return {
        "status": status,
        "metric_name": "agreement against one-person human adjudication",
        "single_accuracy_claim": False,
        "item_count": len(records),
        "split_stats": split_stats,
        "records": records,
    }


def static_audit(repo: Path) -> dict[str, Any]:
    assert_locked_inputs(repo)
    manifest = load_manifest(repo)
    grouped = grouped_provider_inputs(manifest)
    provider_schema = build_provider_schema()
    report = {
        "status": "V22_ASPECT_ONLY_STATIC_AUDIT_PASS",
        "campaign_run_id": CAMPAIGN_ID,
        "policy_version": POLICY_VERSION,
        "batch_ids": EXPECTED_BATCHES,
        "item_count": EXPECTED_ITEM_COUNT,
        "batch_size": EXPECTED_BATCH_SIZE,
        "provider_contract": "Provider emits aspect/polarity targets only; local code selects evidence and derives mixed flag.",
        "provider_calls_made": False,
        "locked_hashes": {
            **LOCKED_HASHES,
            "policy": sha256_file(repo / POLICY_PATH),
            "provider_schema": hashlib.sha256(canonical_json_bytes(provider_schema)).hexdigest(),
        },
        "rendered_prompt_sha256_by_batch": {
            batch_id: hashlib.sha256(render_prompt(rows).encode("utf-8")).hexdigest()
            for batch_id, rows in grouped.items()
        },
    }
    out = repo / OUT_DIR / f"gate7_26_v22_aspect_only_static_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}.json"
    write_new(out, json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    report["report_path"] = str(out)
    report["report_sha256"] = sha256_file(out)
    return report


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
    if not approval.get("approval_id") or not approval.get("attempt_id"):
        raise RuntimeError("Approval must include approval_id and attempt_id")
    approval["approval_file_sha256"] = sha256_file(path)
    return approval


def assert_non_sync_workspace_root(path: Path) -> Path:
    resolved = path.resolve()
    if any("onedrive" in part.lower() for part in resolved.parts):
        raise RuntimeError(f"workspace-root must be non-sync, got OneDrive path: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def execute_live(repo: Path, workspace_root: Path, approval_file: Path) -> None:
    assert_locked_inputs(repo)
    approval = validate_approval(approval_file)
    workspace_root = assert_non_sync_workspace_root(workspace_root)
    connectivity_precheck()
    manifest = load_manifest(repo)
    grouped = grouped_provider_inputs(manifest)
    base_schema = load_json(repo / CANONICAL_SCHEMA_PATH)
    provider_schema = build_provider_schema()
    run_id = f"{CAMPAIGN_ID}_{approval['attempt_id']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}"
    run_dir = workspace_root / "v22_aspect_only_runs" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "batches").mkdir()
    ledger_path = run_dir / "ledger_live.jsonl"
    provider_schema_sha = write_new(run_dir / "provider_request_schema.json", canonical_json_bytes(provider_schema))
    requests_sent = 0
    terminal_reason = "LIVE_COMPLETE"
    comparison_sha = None
    try:
        for batch_id in EXPECTED_BATCHES:
            rows = grouped[batch_id]
            batch_dir = run_dir / "batches" / batch_id
            batch_dir.mkdir()
            prompt_text = render_prompt(rows)
            prompt_sha = write_new(batch_dir / "rendered_prompt.md", prompt_text.encode("utf-8"))
            write_new(batch_dir / "provider_input.json", canonical_json_bytes(rows))
            raw_text = call_gemini_live(prompt_text, provider_schema)
            requests_sent += 1
            raw_sha = write_new(batch_dir / "raw_provider_response.txt", raw_text.encode("utf-8"))
            raw, resolved, derived = validate_provider_output(raw_text, rows, base_schema)
            raw_obj_sha = write_new(batch_dir / "raw_provider_object.json", canonical_json_bytes(raw))
            resolved_sha = write_new(batch_dir / "resolved_output.json", canonical_json_bytes(resolved))
            derived_sha = write_new(batch_dir / "derived_validation_output.json", canonical_json_bytes(derived))
            with ledger_path.open("ab") as f:
                f.write(canonical_json_bytes({
                    "event": "BATCH_ATTEMPT_COMPLETE",
                    "batch_id": batch_id,
                    "attempt_id": approval["attempt_id"],
                    "policy_version": POLICY_VERSION,
                    "prompt_sha256": prompt_sha,
                    "provider_schema_sha256": provider_schema_sha,
                    "raw_response_sha256": raw_sha,
                    "raw_provider_object_sha256": raw_obj_sha,
                    "resolved_output_sha256": resolved_sha,
                    "derived_validation_sha256": derived_sha,
                    "validation_status": derived["status"],
                    "validation_layer": derived["layer"],
                    "validation_error_code": derived["error_code"],
                    "provider_request_sent": True,
                }) + b"\n")
            if derived["status"] != "PASS":
                terminal_reason = f"STOP_ON_VALIDATION_FAIL:{batch_id}:{derived['layer']}:{derived['error_code']}"
                break
        if terminal_reason == "LIVE_COMPLETE":
            comparison = compare_against_human(repo, run_dir)
            comparison_sha = write_new(run_dir / "v22_aspect_only_comparison_report.json", canonical_json_bytes(comparison))
            terminal_reason = comparison["status"]
    except Exception as exc:
        terminal_reason = f"PROVIDER_REQUEST_FAILED:{type(exc).__name__}:{exc}"
    summary = {
        "status": terminal_reason,
        "campaign_run_id": CAMPAIGN_ID,
        "policy_version": POLICY_VERSION,
        "run_id": run_id,
        "workspace_run_dir": str(run_dir),
        "requests_sent": requests_sent,
        "max_provider_requests": len(EXPECTED_BATCHES),
        "human_repair_applied": False,
        "blind_retry_used": False,
        "comparison_report_sha256": comparison_sha,
        "ledger_sha256": sha256_file(ledger_path) if ledger_path.exists() else None,
    }
    summary_path = repo / OUT_DIR / f"gate7_26_v22_aspect_only_live_summary_{run_id}.json"
    write_new(summary_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    summary["summary_path"] = str(summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--static-audit", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--workspace-root")
    parser.add_argument("--approval-file")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    if args.static_audit:
        print(json.dumps(static_audit(repo), ensure_ascii=False, indent=2))
        return
    if args.execute:
        if not args.workspace_root or not args.approval_file:
            raise RuntimeError("--execute requires --workspace-root and --approval-file")
        execute_live(repo, Path(args.workspace_root), Path(args.approval_file).resolve())
        return
    raise RuntimeError("Specify --static-audit or --execute")


if __name__ == "__main__":
    main()
