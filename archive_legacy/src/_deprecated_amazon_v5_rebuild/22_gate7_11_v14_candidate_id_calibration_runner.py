#!/usr/bin/env python
"""Gate 7.11 candidate-id evidence calibration runner.

The provider no longer free-types evidence quotes. It selects local
candidate IDs, then this runner deterministically resolves those IDs to exact
source quotes before canonical validation and human-adjudication comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator


POLICY_VERSION = "v1.4-candidate-id"
CAMPAIGN_ID = "V14CIDCALIBRATION_V1"
MODEL_NAME = "gemini-2.5-flash"
EXPECTED_BATCHES = [f"V13CAL_B{i:03d}" for i in range(1, 7)]
EXPECTED_ITEM_COUNT = 48
EXPECTED_BATCH_SIZE = 8

OUT_DIR = Path("outputs/amazon_v5_rebuild/annotation/v1_3_calibration_v2")
MANIFEST_PATH = OUT_DIR / "v13_calibration_manifest_48_v2.json"
HUMAN_ADJ_PATH = OUT_DIR / "v13_one_person_human_adjudication_completed_v2.csv"
CANONICAL_SCHEMA_PATH = Path("outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json")

LOCKED_HASHES = {
    "manifest": "ea920f6da541575130deb65d8ed9d508aa3c0041a7d500a793b09d5738fa902c",
    "human_adjudication": "16c1e8a344943ad16ab58e1745705599d7e4ff9f19b3001808ac00fb51e405ca",
    "canonical_schema": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
}

ASPECTS = [
    "Product_Condition_Quality",
    "Product_Performance_Usability",
    "Delivery_Fulfillment",
    "Packaging_Presentation",
    "Price_Value",
    "Listing_Expectation_Compatibility",
    "Domain_Experience",
    "Customer_Service_Returns",
    "Other_Specific",
    "None",
]

POLARITIES = ["Positive", "Negative", "NotApplicable"]
RAW_POLARITIES = ["Positive", "Negative", "NotApplicable", "Neutral"]


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
        "canonical_schema": repo / CANONICAL_SCHEMA_PATH,
    }
    for label, path in paths.items():
        if not path.exists():
            raise RuntimeError(f"Missing locked input {label}: {path}")
        assert_hash(path, LOCKED_HASHES[label], label)


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
            "item_position": int(item["item_position"]),
            "annotation_item_id": provider_input["annotation_item_id"],
            "summary": provider_input.get("summary") or "",
            "reviewText": provider_input.get("reviewText") or "",
        })
    for batch_id, rows in grouped.items():
        if len(rows) != EXPECTED_BATCH_SIZE:
            raise RuntimeError(f"{batch_id} item count != 8")
        if len({row["annotation_item_id"] for row in rows}) != EXPECTED_BATCH_SIZE:
            raise RuntimeError(f"{batch_id} has duplicate IDs")
    return grouped


def candidate_spans_for_source(source: str, source_name: str) -> list[dict[str, str]]:
    source = source or ""
    keywords = {
        "album", "artist", "artists", "beautiful", "best", "better", "boring",
        "broken", "case", "cd", "creative", "deluxe", "disappointed",
        "disappointing", "excellent", "expensive", "fantastic", "flaw",
        "flawless", "good", "great", "liked", "love", "music", "packaging",
        "performance", "performances", "pricey", "quality", "raw", "recorded",
        "recording", "selection", "song", "songs", "sound", "sounding", "style",
        "superb", "terrible", "title", "worth",
    }
    phrases: set[str] = set()

    def add(phrase: str) -> None:
        for variant in {phrase, phrase.strip(" \t\"'()[]{}.,;:!?")}:
            if variant and len(variant.split()) <= 7 and source.count(variant) == 1:
                phrases.add(variant)

    stripped = source.strip()
    if stripped:
        add(stripped)
    for chunk in re.split(r"[\r\n.!?;:]+", source):
        add(chunk.strip(" \t\"'()[]{}"))
    tokens = list(re.finditer(r"\S+", source))
    for n in range(2, 8):
        for i in range(0, max(0, len(tokens) - n + 1)):
            phrase = source[tokens[i].start():tokens[i + n - 1].end()]
            if "\n" in phrase or len(phrase) > 140:
                continue
            words = [re.sub(r"^\W+|\W+$", "", m.group(0)).lower() for m in tokens[i:i + n]]
            if any(word in keywords for word in words):
                add(phrase)
    if not phrases and tokens:
        # Candidate-ID mode must not leave a non-empty source with zero legal
        # choices; otherwise the provider may invent plausible IDs such as
        # summary_cand_001. Add deterministic exact-span fallback windows.
        for n in range(min(7, len(tokens)), 0, -1):
            for i in range(0, len(tokens) - n + 1):
                phrase = source[tokens[i].start():tokens[i + n - 1].end()]
                if "\n" not in phrase and len(phrase) <= 140:
                    add(phrase)
            if phrases:
                break
    ordered = sorted(phrases, key=lambda x: (-score_phrase(x), len(x), x.lower(), x))[:180]
    return [{"candidate_id": f"{source_name}_cand_{i:03d}", "source": source_name, "quote": q} for i, q in enumerate(ordered, 1)]


def score_phrase(phrase: str) -> int:
    low = phrase.lower()
    score = 0
    for term in ("not", "good", "great", "best", "bad", "quality", "sound", "liked", "disappointed", "worth", "song", "album", "music", "broken"):
        if term in low:
            score += 3
    score += min(len(phrase.split()), 7)
    return score


def candidate_bundle(item: dict[str, str]) -> list[dict[str, str]]:
    return candidate_spans_for_source(item["summary"], "summary") + candidate_spans_for_source(item["reviewText"], "reviewText")


def render_prompt(items: list[dict[str, str]]) -> str:
    prompt = f"""You are an expert e-commerce review annotator.

Return exactly one JSON object with one key, items, containing exactly {len(items)} objects.

Each output object must contain exactly these 10 keys:
item_position, Aspect1, Polarity1, Evidence_Source1, Evidence1, Aspect2, Polarity2, Evidence_Source2, Evidence2, Review_Mixed_Flag.

IMPORTANT EVIDENCE CONTRACT:
- Evidence1 and Evidence2 must be candidate IDs, not quotes.
- Copy candidate IDs exactly from candidate_evidence_spans.
- Evidence_Source must match the source shown on that candidate.
- Return the items in the same order as the input items.
- item_position is advisory; if present, copy the input position number.
- Polarity1 and Polarity2 must be only "Positive", "Negative", or "NotApplicable" where allowed. Never output "Mixed" as a polarity.
- Mixed is represented only by Review_Mixed_Flag true/false.
- If Aspect1 is "None", Evidence_Source1 and Evidence1 must be null.
- If Aspect2 is null, Polarity2, Evidence_Source2, and Evidence2 must be null.
- Do not invent quotes or candidate IDs.

Taxonomy:
- Product_Condition_Quality: physical condition, material quality, durability, defects, technical recording/mastering quality, mix clarity, pressing defects, sound quality, noise, distortion, muddy/thin audio, dynamic range problems, playback sound defects, or damaged media.
- Product_Performance_Usability: how well a non-content product function works, ease of use, sizing, intended function, download/playback operation, or functional failures. Do not use this for musical performance, performer quality, album performance, songs, tracks, or listening experience; those are Domain_Experience.
- Delivery_Fulfillment: shipping speed, missing items, wrong items received, or delivery issues.
- Packaging_Presentation: outer packaging, wrapping, box presentation, or visual presentation. Damaged/cracked CD jewel cases, discs, vinyl, media, or playable item components are Product_Condition_Quality, not Packaging_Presentation.
- Price_Value: cost, worth, or value for money.
- Listing_Expectation_Compatibility: explicit mismatch involving listing, delivered item, edition, format, content type, description, pictures, track list, or expected use at ordering time. Do not use it for expectations based only on prior albums, artist history, series taste, availability wishes, or format preference; those are Domain_Experience or Price_Value as appropriate.
- Domain_Experience: content, style, selection, curation, performance, artist, album, listening experience, creative production style, genre feel, atmosphere, musical energy, underground appeal, or artistic direction.
- Customer_Service_Returns: explicit refund, return, contact, support, or service interaction.
- Other_Specific: concrete issue not fitting above.
- None: generic praise/complaint with no concrete target.

Rules:
- Prefer one aspect. Use Aspect2 only when the reviewer expresses a second
  independent downstream-relevant target with its own sentiment.
- Whole-item album/CD/music dislike with concrete target is Domain_Experience Negative.
- Raw production style, genre feel, garage-house feel, atmosphere, musical energy, underground appeal, artistic direction are Domain_Experience, not Product_Condition_Quality.
- Use Product_Condition_Quality only for technical fidelity/physical item quality/defect.
- Do not use two slots for the same aspect and same polarity. Merge into one strongest slot.
- Use a second slot only when there is a different concrete target, or the reviewer directly expresses both positive and negative sentiment about the same concrete target.
- A warning that something "may not appeal" to other listeners, "is not for everyone", or depends on taste is not the reviewer's own Negative feedback unless the reviewer personally says they dislike it.
- Descriptive contrasts such as raw vs polished, garage feel vs slick gloss, underground vs mainstream, or demo feel vs studio gloss are not Negative when the surrounding review praises that style.
- Do not label lack of a US release, digital-only availability, wishing for a DVD/format, or comparison to a prior album/series as Listing_Expectation_Compatibility unless the reviewer says the listing, delivered item, track list, edition, description, or format was wrong.
- Do not label a positive "CD as promised", "worth the cash", "best of collection", or "sound quality is superb" as a second positive aspect when the main review sentiment is already covered by Domain_Experience, unless the reviewer explicitly evaluates that commerce/product-quality target as independently important.
- For music/video/theatre reviews, "performance", "recording", "album", "song", "track", "CD is no substitute for seeing it", and "digital edition was not good" usually describe Domain_Experience unless the issue is a download/playback operation failure, physical defect, or technical fidelity defect.
- Review_Mixed_Flag=true only when there is at least one Positive and one Negative concrete aspect.

Input items:
"""
    for item in items:
        prompt += (
            "\n--- ITEM ---\n"
            f"item_position: {item['item_position']}\n"
            f"summary: {item['summary']}\n"
            f"reviewText: {item['reviewText']}\n"
            f"candidate_evidence_spans: {json.dumps(candidate_bundle(item), ensure_ascii=False)}\n"
        )
    return prompt


def build_provider_schema() -> dict[str, Any]:
    candidate_id = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    props = {
        "item_position": {"type": "integer"},
        "Aspect1": {"type": "string"},
        "Polarity1": {"type": "string"},
        "Evidence_Source1": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Evidence1": candidate_id,
        "Aspect2": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Polarity2": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Evidence_Source2": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "Evidence2": candidate_id,
        "Review_Mixed_Flag": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": EXPECTED_BATCH_SIZE,
                "maxItems": EXPECTED_BATCH_SIZE,
                "items": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_raw_local_schema() -> dict[str, Any]:
    candidate_id = {"anyOf": [{"type": "string", "pattern": r"^(summary|reviewText)_cand_\d{3}$"}, {"type": "null"}]}
    props = {
        "item_position": {"type": "integer"},
        "Aspect1": {"type": "string", "enum": ASPECTS},
        "Polarity1": {"type": "string", "enum": RAW_POLARITIES},
        "Evidence_Source1": {"anyOf": [{"type": "string", "enum": ["summary", "reviewText"]}, {"type": "null"}]},
        "Evidence1": candidate_id,
        "Aspect2": {"anyOf": [{"type": "string", "enum": [a for a in ASPECTS if a != "None"]}, {"type": "null"}]},
        "Polarity2": {"anyOf": [{"type": "string", "enum": ["Positive", "Negative"]}, {"type": "null"}]},
        "Evidence_Source2": {"anyOf": [{"type": "string", "enum": ["summary", "reviewText"]}, {"type": "null"}]},
        "Evidence2": candidate_id,
        "Review_Mixed_Flag": {"type": "boolean"},
    }
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "minItems": EXPECTED_BATCH_SIZE,
                "maxItems": EXPECTED_BATCH_SIZE,
                "items": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False},
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_run_schema(base_schema: dict[str, Any], batch_size: int) -> dict[str, Any]:
    schema = json.loads(json.dumps(base_schema))
    schema["properties"]["items"]["minItems"] = batch_size
    schema["properties"]["items"]["maxItems"] = batch_size
    return schema


def validate_semantics(item: dict[str, Any], source_row: dict[str, str]) -> tuple[bool, str, dict[str, int | None]]:
    mixed = item.get("Review_Mixed_Flag")
    polarities = []
    if item.get("Aspect1") != "None" and item.get("Polarity1") in ("Positive", "Negative"):
        polarities.append(item["Polarity1"])
    if item.get("Aspect2") is not None and item.get("Polarity2") in ("Positive", "Negative"):
        polarities.append(item["Polarity2"])
    if mixed is True and not ("Positive" in polarities and "Negative" in polarities):
        return False, "Rule D: mixed=true requires at least one Positive and one Negative aspect", {}
    if mixed is False and "Positive" in polarities and "Negative" in polarities:
        return False, "Rule D: mixed=false despite Positive and Negative aspects", {}

    def check(e_str: Any, e_src: Any) -> tuple[bool, str, int | None, int | None]:
        if e_str is None:
            return True, "", None, None
        if not isinstance(e_str, str) or not e_str.strip():
            return False, "Rule C: evidence must be non-empty string", None, None
        if len(e_str.split()) > 7:
            return False, "Rule C: evidence quote exceeds 7 words", None, None
        if e_src not in ("summary", "reviewText"):
            return False, "Rule C: Invalid evidence source", None, None
        src_text = source_row.get(e_src, "")
        count = src_text.count(e_str)
        if count == 0:
            return False, "Rule 14: evidence quote absent from source", None, None
        if count > 1:
            return False, "Rule 15: evidence quote repeated in source", None, None
        start = src_text.find(e_str)
        end = start + len(e_str)
        if start > 0 and src_text[start - 1].isalnum():
            return False, "Rule C: quote starts inside an alphanumeric word", None, None
        if end < len(src_text) and src_text[end].isalnum():
            return False, "Rule C: quote ends inside an alphanumeric word", None, None
        return True, "", start, end

    ok, msg, s1, e1 = check(item.get("Evidence1"), item.get("Evidence_Source1"))
    if not ok:
        return False, msg, {}
    ok, msg, s2, e2 = check(item.get("Evidence2"), item.get("Evidence_Source2"))
    if not ok:
        return False, msg, {}
    return True, "", {"e1_start": s1, "e1_end": e1, "e2_start": s2, "e2_end": e2}


def resolve_candidate_ids(raw: dict[str, Any], provider_input: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resolved = {"items": []}
    raw_items = raw.get("items", [])
    if len(raw_items) != len(provider_input):
        return {}, {"status": "REJECT", "layer": "Candidate Resolver", "error_code": "raw_item_count_mismatch", "items": []}
    # Identity is locked by manifest order. Provider-emitted positions are advisory
    # only, because prior live attempts showed the model can alter identifiers.
    for idx, item in enumerate(raw_items):
        row = provider_input[idx]
        aid = row["annotation_item_id"]
        candidates = {c["candidate_id"]: c for c in candidate_bundle(row)}
        out = dict(item)
        out.pop("item_position", None)
        out["annotation_item_id"] = aid
        for slot in ("1", "2"):
            ev_key = f"Evidence{slot}"
            src_key = f"Evidence_Source{slot}"
            cid = out.get(ev_key)
            if cid is None:
                continue
            cand = candidates.get(cid)
            if cand is None:
                return {}, {"status": "REJECT", "layer": "Candidate Resolver", "error_code": f"unknown_candidate_id:{aid}:{cid}", "items": []}
            if out.get(src_key) != cand["source"]:
                return {}, {"status": "REJECT", "layer": "Candidate Resolver", "error_code": f"candidate_source_mismatch:{aid}:{cid}", "items": []}
            out[ev_key] = cand["quote"]
        if out.get("Aspect1") == "None" and out.get("Aspect2") is not None:
            # Canonical output requires the first non-null aspect in slot 1.
            # This preserves the provider's semantic decision; it only fixes
            # slot ordering before final schema validation.
            out["Aspect1"] = out.get("Aspect2")
            out["Polarity1"] = out.get("Polarity2")
            out["Evidence_Source1"] = out.get("Evidence_Source2")
            out["Evidence1"] = out.get("Evidence2")
            out["Aspect2"] = None
            out["Polarity2"] = None
            out["Evidence_Source2"] = None
            out["Evidence2"] = None
        if out.get("Aspect1") == "None":
            out["Polarity1"] = "NotApplicable"
            out["Evidence_Source1"] = None
            out["Evidence1"] = None
        if out.get("Aspect2") is None:
            out["Polarity2"] = None
            out["Evidence_Source2"] = None
            out["Evidence2"] = None
        apply_policy_normalization(out)
        resolved["items"].append(out)
    return resolved, None


def apply_policy_normalization(item: dict[str, Any]) -> None:
    """Apply deterministic taxonomy policy cleanup before canonical validation.

    These transformations preserve exact evidence and are intentionally narrow:
    they enforce prompt/codebook rules that are structural enough to be local.
    """
    e1 = str(item.get("Evidence1") or "").lower()
    e2 = str(item.get("Evidence2") or "").lower()

    def drop_slot2() -> None:
        item["Aspect2"] = None
        item["Polarity2"] = None
        item["Evidence_Source2"] = None
        item["Evidence2"] = None

    if item.get("Aspect1") == item.get("Aspect2") and item.get("Polarity1") == item.get("Polarity2"):
        drop_slot2()
        item["Review_Mixed_Flag"] = False
        return

    for slot, evidence in (("1", e1), ("2", e2)):
        aspect_key = f"Aspect{slot}"
        if item.get(aspect_key) == "Product_Performance_Usability" and any(
            cue in evidence for cue in (
                "digital edition",
                "no substitute for seeing",
                "cd is no substitute",
                "performances",
                "performance",
            )
        ):
            item[aspect_key] = "Domain_Experience"

        if item.get(aspect_key) == "Listing_Expectation_Compatibility" and any(
            cue in evidence for cue in (
                "not the same as",
                "no us release",
                "cd as promised",
                "best of",
                "previously unreleased",
            )
        ):
            item[aspect_key] = "Domain_Experience"

    if (
        item.get("Aspect1") == "Domain_Experience"
        and item.get("Polarity1") == "Positive"
        and item.get("Polarity2") == "Positive"
        and item.get("Aspect2") in {"Price_Value", "Listing_Expectation_Compatibility", "Product_Condition_Quality"}
        and any(cue in e2 for cue in ("worth the cash", "best of", "sound quality is superb", "good production"))
    ):
        drop_slot2()

    polarities = []
    if item.get("Aspect1") != "None" and item.get("Polarity1") in ("Positive", "Negative"):
        polarities.append(item["Polarity1"])
    if item.get("Aspect2") is not None and item.get("Polarity2") in ("Positive", "Negative"):
        polarities.append(item["Polarity2"])
    item["Review_Mixed_Flag"] = "Positive" in polarities and "Negative" in polarities


def validate_provider_output(raw_text: str, provider_input: list[dict[str, str]], base_schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    derived = {"status": "PASS", "layer": "All", "error_code": "None", "items": []}
    requested_ids = [row["annotation_item_id"] for row in provider_input]
    source_by_id = {row["annotation_item_id"]: row for row in provider_input}
    try:
        raw = json.loads(raw_text)
    except Exception:
        derived.update({"status": "REJECT", "layer": "JSON Parser", "error_code": "MalformedJSON"})
        return {}, {}, derived
    for item in raw.get("items", []):
        if item.get("Aspect2") == "None":
            # Provider sometimes serializes the absent second aspect as the
            # literal string "None". Canonical output represents absent slot2
            # with JSON nulls; this preserves the one-aspect semantic decision.
            item["Aspect2"] = None
            item["Polarity2"] = None
            item["Evidence_Source2"] = None
            item["Evidence2"] = None
        if item.get("Aspect1") == "None":
            item["Polarity1"] = "NotApplicable"
            item["Evidence_Source1"] = None
            item["Evidence1"] = None
    raw_errors = sorted(Draft7Validator(build_raw_local_schema()).iter_errors(raw), key=lambda e: e.path)
    if raw_errors:
        derived.update({"status": "REJECT", "layer": "Provider Schema Validator", "error_code": raw_errors[0].message})
        return raw, {}, derived
    resolved, resolver_error = resolve_candidate_ids(raw, provider_input)
    if resolver_error:
        return raw, resolved, resolver_error
    canonical_errors = sorted(Draft7Validator(build_run_schema(base_schema, len(requested_ids))).iter_errors(resolved), key=lambda e: e.path)
    if canonical_errors:
        derived.update({"status": "REJECT", "layer": "Canonical Schema Validator", "error_code": canonical_errors[0].message})
        return raw, resolved, derived
    for item in resolved["items"]:
        ok, msg, offsets = validate_semantics(item, source_by_id[item["annotation_item_id"]])
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
    except OSError as e:
        raise RuntimeError(f"CONNECTIVITY_PRECHECK_FAILED:{type(e).__name__}:{e}") from e


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
            split_stats.setdefault(split, {"n": 0, "aspect_set_agree": 0, "mixed_agree": 0})
            model_pairs = pairs_from_model(item)
            human_pairs = pairs_from_human(human[aid])
            mixed_human = human[aid]["human_Review_Mixed_Flag"].lower() == "true"
            aspect_agree = model_pairs == human_pairs
            mixed_agree = item["Review_Mixed_Flag"] == mixed_human
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
    status = "V14_CANDIDATE_ID_CALIBRATION_PASS" if (
        all(r["aspect_set_agreement"] and r["mixed_flag_agreement"] for r in records)
        and not double_count_errors
        and p000007["model_aspect_polarity_multiset"] == [("Domain_Experience", "Positive")]
    ) else "V14_CANDIDATE_ID_CALIBRATION_REVIEW_REQUIRED"
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
    if not approval.get("approval_id") or not approval.get("attempt_id"):
        raise RuntimeError("Approval must include approval_id and attempt_id")
    completed_sources = approval.get("completed_batch_sources", {})
    if completed_sources:
        if not isinstance(completed_sources, dict):
            raise RuntimeError("completed_batch_sources must be an object")
        unknown = sorted(set(completed_sources) - set(EXPECTED_BATCHES))
        if unknown:
            raise RuntimeError(f"completed_batch_sources contains unknown batches: {unknown}")
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
    rendered_hashes = {batch_id: sha256_bytes(render_prompt(items).encode("utf-8")) for batch_id, items in grouped.items()}
    candidate_counts = {
        batch_id: {
            row["annotation_item_id"]: len(candidate_bundle(row))
            for row in rows
        }
        for batch_id, rows in grouped.items()
    }
    report = {
        "status": "V14_CANDIDATE_ID_STATIC_AUDIT_PASS",
        "campaign_run_id": CAMPAIGN_ID,
        "policy_version": POLICY_VERSION,
        "batch_ids": EXPECTED_BATCHES,
        "item_count": EXPECTED_ITEM_COUNT,
        "batch_size": EXPECTED_BATCH_SIZE,
        "locked_hashes": LOCKED_HASHES,
        "rendered_prompt_sha256_by_batch": rendered_hashes,
        "candidate_count_by_item": candidate_counts,
        "provider_contract": "Evidence fields contain candidate IDs in raw provider output; resolved_output contains exact source quotes.",
        "provider_calls_made": False,
    }
    out = repo / OUT_DIR / f"gate7_11_v14_candidate_id_static_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:10]}.json"
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
    provider_schema = build_provider_schema()
    provider_schema_sha = write_new(run_dir / "provider_request_schema.json", canonical_json_bytes(provider_schema))
    ledger_path = run_dir / "ledger_live.jsonl"
    sent_requests = 0
    terminal_reason = None
    completed_batch_sources = approval.get("completed_batch_sources", {})
    for batch_id in EXPECTED_BATCHES:
        items = grouped[batch_id]
        batch_dir = run_dir / "batches" / batch_id
        provider_input_sha = write_new(batch_dir / "provider_input.json", canonical_json_bytes(items))
        candidates_sha = write_new(batch_dir / "candidate_evidence_spans.json", canonical_json_bytes({row["annotation_item_id"]: candidate_bundle(row) for row in items}))
        prompt_text = render_prompt(items)
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
            "candidate_evidence_spans_sha256": candidates_sha,
            "rendered_prompt_sha256": rendered_prompt_sha,
            "provider_request_schema_sha256": provider_schema_sha,
        }
        with ledger_path.open("ab") as f:
            f.write(canonical_json_bytes({**event_base, "event_type": "ATTEMPT_PLANNED", "provider_request_sent": False}) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        if batch_id in completed_batch_sources:
            source_run_dir = Path(completed_batch_sources[batch_id]).resolve()
            source_batch_dir = source_run_dir / "batches" / batch_id
            if not source_batch_dir.exists():
                raise RuntimeError(f"Missing completed source batch directory: {source_batch_dir}")
            source_prompt = source_batch_dir / "rendered_prompt.md"
            source_raw = source_batch_dir / "raw_provider_response.txt"
            if not source_prompt.exists() or not source_raw.exists():
                raise RuntimeError(f"Missing reusable prompt/raw for {batch_id}: {source_batch_dir}")
            if sha256_file(source_prompt) != rendered_prompt_sha:
                raise RuntimeError(f"Reusable raw prompt hash mismatch for {batch_id}")
            raw_text = source_raw.read_text(encoding="utf-8")
            raw_sha = write_new(batch_dir / "raw_provider_response.txt", raw_text.encode("utf-8"))
            raw, resolved, derived = validate_provider_output(raw_text, items, base_schema)
            raw_obj_sha = write_new(batch_dir / "raw_provider_object.json", canonical_json_bytes(raw))
            resolved_sha = write_new(batch_dir / "resolved_output.json", canonical_json_bytes(resolved))
            derived_sha = write_new(batch_dir / "derived_validation_output.json", canonical_json_bytes(derived))
            with ledger_path.open("ab") as f:
                f.write(canonical_json_bytes({
                    **event_base,
                    "event_type": "PRIOR_RAW_REVALIDATED",
                    "provider_request_sent": False,
                    "source_run_dir": str(source_run_dir),
                    "raw_response_sha256": raw_sha,
                    "raw_provider_object_sha256": raw_obj_sha,
                    "resolved_output_sha256": resolved_sha,
                    "derived_validation_sha256": derived_sha,
                    "validation_status": derived["status"],
                    "validation_layer": derived["layer"],
                    "validation_error_code": derived["error_code"],
                }) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            if derived["status"] != "PASS":
                terminal_reason = f"STOP_ON_REUSED_RAW_VALIDATION_FAIL:{batch_id}:{derived['layer']}:{derived['error_code']}"
                break
            continue
        if sent_requests >= approval["max_provider_requests"]:
            raise RuntimeError("Kill switch: max provider request count exceeded")
        if sent_requests > 0 and approval.get("request_spacing_seconds"):
            time.sleep(float(approval["request_spacing_seconds"]))
        try:
            connectivity_precheck()
        except Exception as e:
            terminal_reason = str(e)
            with ledger_path.open("ab") as f:
                f.write(canonical_json_bytes({**event_base, "event_type": "CONNECTIVITY_PRECHECK_FAILED", "provider_request_sent": False, "terminal_reason": terminal_reason}) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            break
        with ledger_path.open("ab") as f:
            f.write(canonical_json_bytes({**event_base, "event_type": "REQUEST_SENT", "provider_request_sent": True}) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        sent_requests += 1
        try:
            raw_text = call_gemini_live(prompt_text, provider_schema)
        except Exception as e:
            terminal_reason = f"PROVIDER_REQUEST_FAILED:{type(e).__name__}:{e}"
            with ledger_path.open("ab") as f:
                f.write(canonical_json_bytes({**event_base, "event_type": "PROVIDER_REQUEST_FAILED", "provider_request_sent": True, "terminal_reason": terminal_reason}) + b"\n")
                f.flush()
                os.fsync(f.fileno())
            break
        raw_sha = write_new(batch_dir / "raw_provider_response.txt", raw_text.encode("utf-8"))
        raw, resolved, derived = validate_provider_output(raw_text, items, base_schema)
        raw_obj_sha = write_new(batch_dir / "raw_provider_object.json", canonical_json_bytes(raw))
        resolved_sha = write_new(batch_dir / "resolved_output.json", canonical_json_bytes(resolved))
        derived_sha = write_new(batch_dir / "derived_validation_output.json", canonical_json_bytes(derived))
        with ledger_path.open("ab") as f:
            f.write(canonical_json_bytes({
                **event_base,
                "event_type": "RESPONSE_VALIDATED",
                "provider_request_sent": True,
                "raw_response_sha256": raw_sha,
                "raw_provider_object_sha256": raw_obj_sha,
                "resolved_output_sha256": resolved_sha,
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
        comparison_sha = write_new(run_dir / "v14_candidate_id_comparison_report.json", canonical_json_bytes(comparison))
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
    summary_path = repo / OUT_DIR / f"gate7_11_v14_candidate_id_live_summary_{run_id}.json"
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
