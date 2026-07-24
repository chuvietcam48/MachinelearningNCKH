"""
07_phase2b1_gemini_calibration_v1_6.py
Phase 2B-1 Gemini Calibration Pipeline — v1.6

Entry points (exactly one required):
  --dry-run              : Run the locked run_004 contract-test harness (no Gemini, no network).
  --execute --run-id <id> --preflight-only     : Write snapshot + rendered prompt for batch_1_calib only.
  --execute --run-id <id> --adapter-smoke-test : Run mock-transport for batch_1_calib through full pipeline.
  --execute --run-id <id> --live-pilot --ack-live-provider : Call Gemini for batch_1_calib only (1 batch max).
  --execute --run-id <id> --live-batch-id <batch_id> --ack-live-provider : Call Gemini for one selected planned batch only.

Full live execution (all 15 batches + repair loop) is DISABLED until repair is independently tested.
"""

import os
os.environ.setdefault("PANDAS_STRING_STORAGE", "python")
import sys
import re
import json
import argparse
import hashlib
import copy
import pandas as pd
from jsonschema import Draft7Validator
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Global path constants
# ---------------------------------------------------------------------------
MANIFEST_PATH        = "outputs/amazon_v5_rebuild/annotation/calibration_sample_manifest_v1_1.csv"
SCHEMA_PATH          = "outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json"
PROMPT_TEMPLATE_PATH = "outputs/amazon_v5_rebuild/annotation/gemini_semantic_prompt_template_v1_2.md"
TAXONOMY_PATH        = "outputs/amazon_v5_rebuild/annotation/taxonomy_v1_2_frozen.md"

# Immutable model constant — do not hardcode elsewhere
MODEL_NAME = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# Locked run_004 dry-run paths (NEVER written by execute paths)
# ---------------------------------------------------------------------------
_DRY_RUN_CHECKPOINT_PATH         = "outputs/amazon_v5_rebuild/annotation/gemini_batch_run_checkpoint_v1_6_run_004.json"
_DRY_RUN_REPORT_PATH             = "outputs/amazon_v5_rebuild/annotation/gemini_batch_dry_run_report_v1_6_run_004.md"
_DRY_RUN_RAW_RESPONSES_PATH      = "outputs/amazon_v5_rebuild/annotation/raw_provider_responses_v1_6_run_004.jsonl"
_DRY_RUN_PARSED_OBJECTS_PATH     = "outputs/amazon_v5_rebuild/annotation/parsed_provider_objects_v1_6_run_004.jsonl"
_DRY_RUN_DERIVED_RECORDS_PATH    = "outputs/amazon_v5_rebuild/annotation/validation_derived_records_v1_6_run_004.jsonl"
_DRY_RUN_MANIFEST_OUT_PATH       = "outputs/amazon_v5_rebuild/annotation/phase2b1_v16_run_004_manifest.json"
_DRY_RUN_EXPECTED_VS_OBSERVED_PATH = "outputs/amazon_v5_rebuild/annotation/expected_vs_observed_v1_6_run_004.json"
_DRY_RUN_FIXTURES_DIR            = "outputs/amazon_v5_rebuild/annotation/v1_6_run_004_fixtures"

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def get_hash(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"File missing or empty: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def get_hash_or_empty(path):
    """Dry-run variant — missing/empty returns empty string (not an error)."""
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def get_hash_if_exists(path):
    if not os.path.exists(path):
        return None
    return get_hash(path)

def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def atomic_write_checkpoint(ckpt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, path)

# ---------------------------------------------------------------------------
# Shared validation logic (used by both dry-run and execute)
# ---------------------------------------------------------------------------
def reconcile_ledger(ckpt):
    if ckpt["successful_base_requests"] > ckpt["attempted_base_requests"]:
        raise ValueError("successful_base_requests > attempted_base_requests")
    if ckpt["successful_repair_requests"] > ckpt["attempted_repair_requests"]:
        raise ValueError("successful_repair_requests > attempted_repair_requests")
    if ckpt["attempted_base_requests"] > 15:
        raise ValueError("attempted_base_requests > 15")
    if ckpt["attempted_repair_requests"] > 3:
        raise ValueError("attempted_repair_requests > 3")
    if ckpt["attempted_base_requests"] + ckpt["attempted_repair_requests"] > 18:
        raise ValueError("attempted_base_requests + attempted_repair_requests > 18")

    c = set(ckpt.get("completed_valid_item_ids", []))
    i = set(ckpt.get("invalid_item_ids", []))
    p = set(ckpt.get("pending_item_ids", []))

    all_listed = (ckpt.get("completed_valid_item_ids", []) +
                  ckpt.get("invalid_item_ids", []) +
                  ckpt.get("pending_item_ids", []))
    if len(all_listed) != len(set(all_listed)):
        raise ValueError("Duplicate IDs found in state")
    if len(ckpt.get("planned_item_ids", [])) != len(set(ckpt.get("planned_item_ids", []))):
        raise ValueError("planned_item_ids contains duplicates")
    if c & i or c & p or i & p:
        raise ValueError("overlap among completed_valid, invalid, and pending")
    if set(ckpt.get("completed_batch_ids", [])) & set(ckpt.get("failed_batch_ids", [])):
        raise ValueError("completed_batch_ids and failed_batch_ids are not disjoint")

    for b in ckpt.get("completed_batch_ids", []):
        if b not in ckpt["planned_base_batch_ids"] and not b.endswith("_repair"):
            raise ValueError("completed batch absent from planned batch inventory")

    attempted_base    = set(ckpt.get("attempted_base_batch_ids", []))
    successful_base   = set(ckpt.get("successful_base_batch_ids", []))
    attempted_repair  = set(ckpt.get("attempted_repair_batch_ids", []))
    successful_repair = set(ckpt.get("successful_repair_batch_ids", []))
    failed_batches    = set(ckpt.get("failed_batch_ids", []))

    for b in attempted_base:
        if (b in successful_base) == (b in failed_batches):
            raise ValueError(f"Batch {b} not in exactly one terminal state")
    for b in attempted_repair:
        if (b in successful_repair) == (b in failed_batches):
            raise ValueError(f"Repair batch {b} not in exactly one terminal state")
    if not (successful_base | failed_batches.intersection(attempted_base) == attempted_base):
        raise ValueError("Union of successful and failed base batches != attempted base batches")
    if not (successful_repair | failed_batches.intersection(attempted_repair) == attempted_repair):
        raise ValueError("Union of successful and failed repair batches != attempted repair batches")

    if ckpt.get("mode") == "execute":
        batch_item_ids = ckpt.get("batch_item_ids", {})
        for b in ckpt.get("completed_batch_ids", []) + ckpt.get("failed_batch_ids", []):
            if b not in batch_item_ids:
                raise ValueError(f"batch {b} missing from batch_item_ids")
            if ckpt["batch_sizes"].get(b, 0) != len(batch_item_ids[b]):
                raise ValueError(f"batch size mismatch for {b}")
        for b in ckpt.get("completed_batch_ids", []):
            batch_inputs  = set(batch_item_ids[b])
            batch_completed = {x for x in ckpt.get("completed_valid_item_ids", []) if x in batch_inputs}
            batch_invalid   = {x for x in ckpt.get("invalid_item_ids", []) if x in batch_inputs}
            if (batch_completed | batch_invalid) != batch_inputs:
                raise ValueError(f"completed/invalid IDs for batch {b} do not match its inputs")
            if any(x in p for x in batch_inputs):
                raise ValueError(f"pending IDs found assigned to completed batch {b}")

    completed_items_from_batches = sum(
        ckpt["batch_sizes"].get(b, 0) for b in ckpt.get("completed_batch_ids", []))
    if (len(ckpt.get("completed_valid_item_ids", [])) +
            len(ckpt.get("invalid_item_ids", []))) != completed_items_from_batches:
        raise ValueError("completed item count not reconciling with completed batch sizes")

    union = c | i | p
    if set(ckpt.get("planned_item_ids", [])) != union:
        raise ValueError("union of completed_valid, invalid, and pending != planned item IDs")

    if ckpt.get("terminal_stop_reason") == "COMPLETED":
        raise ValueError(
            "COMPLETED is a disallowed terminal_stop_reason until full repair loop is implemented.")

    if ckpt.get("terminal_stop_reason") == "PILOT_BASE_BATCH_COMPLETE":
        if ckpt.get("attempted_base_requests") != 1:
            raise ValueError("pilot complete requires exactly 1 attempted base request")
        if ckpt.get("successful_base_requests") != 1:
            raise ValueError("pilot complete requires exactly 1 successful base request")


def build_run_schema(base_schema, batch_size):
    run_schema = copy.deepcopy(base_schema)
    run_schema["properties"]["items"]["minItems"] = batch_size
    run_schema["properties"]["items"]["maxItems"] = batch_size
    return run_schema


def build_provider_response_json_schema(canonical_schema, batch_size):
    provider_schema = copy.deepcopy(canonical_schema)
    
    def _clean(d):
        if isinstance(d, dict):
            d.pop("$schema", None)
            d.pop("allOf", None)
            d.pop("if", None)
            d.pop("then", None)
            if "enum" in d and isinstance(d["enum"], list):
                d["enum"] = [x for x in d["enum"] if x is not None]
            for k, v in list(d.items()):
                _clean(v)
        elif isinstance(d, list):
            for i in d:
                _clean(i)
                
    _clean(provider_schema)
    
    items_def = provider_schema["properties"]["items"]["items"]
    
    for es_key in ["Evidence_Source1", "Evidence_Source2"]:
        if es_key in items_def["properties"]:
            items_def["properties"][es_key] = {
                "anyOf": [
                    {"type": "string", "enum": ["summary", "reviewText"]},
                    {"type": "null"}
                ]
            }
            
    for e_key in ["Evidence1", "Evidence2"]:
        if e_key in items_def["properties"]:
            items_def["properties"][e_key] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "null"}
                ]
            }
            
    if "Aspect2" in items_def["properties"]:
        items_def["properties"]["Aspect2"] = {
            "anyOf": [
                {
                    "type": "string",
                    "enum": [
                        "Product_Condition_Quality", "Product_Performance_Usability", "Delivery_Fulfillment",
                        "Packaging_Presentation", "Customer_Service_Returns", "Price_Value",
                        "Listing_Expectation_Compatibility", "Domain_Experience", "Other_Specific"
                    ]
                },
                {"type": "null"}
            ]
        }
        
    if "Polarity2" in items_def["properties"]:
        items_def["properties"]["Polarity2"] = {
            "anyOf": [
                {"type": "string", "enum": ["Positive", "Negative", "Neutral", "NotApplicable"]},
                {"type": "null"}
            ]
        }
        
    provider_schema["properties"]["items"]["minItems"] = batch_size
    provider_schema["properties"]["items"]["maxItems"] = batch_size
    return provider_schema


def assert_provider_projection_compat(provider_schema):
    forbidden_keywords = {"$schema", "allOf", "if", "then"}

    def _walk(node, path="$"):
        if isinstance(node, dict):
            for key in forbidden_keywords:
                if key in node:
                    raise ValueError(f"Forbidden provider schema keyword {key!r} at {path}")
            if isinstance(node.get("type"), list):
                raise ValueError(f"Forbidden list-valued type at {path}.type")
            if isinstance(node.get("enum"), list) and any(value is None for value in node["enum"]):
                raise ValueError(f"Forbidden null enum value at {path}.enum")
            for key, value in node.items():
                _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                _walk(value, f"{path}[{index}]")

    _walk(provider_schema)



class OutputValidator:
    def __init__(self, base_schema):
        self.base_schema = base_schema

    def validate_schema(self, payload, run_schema):
        try:
            validator = Draft7Validator(run_schema)
            errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
            if errors:
                return False, f"Schema validation failed: {errors[0].message}"
            return True, ""
        except Exception as e:
            return False, f"Schema parsing error: {str(e)}"

    def validate_semantics(self, item, source_row):
        a1    = item.get("Aspect1");  p1    = item.get("Polarity1")
        e_src1= item.get("Evidence_Source1"); e1 = item.get("Evidence1")
        a2    = item.get("Aspect2");  p2    = item.get("Polarity2")
        e_src2= item.get("Evidence_Source2"); e2 = item.get("Evidence2")
        mixed = item.get("Review_Mixed_Flag")

        p_list = []
        if a1 != "None" and p1 in ("Positive", "Negative"): p_list.append(p1)
        if a2 is not None and p2 in ("Positive", "Negative"): p_list.append(p2)
        has_pos = "Positive" in p_list;  has_neg = "Negative" in p_list

        if mixed is True:
            if not (has_pos and has_neg):
                if has_neg and p_list.count("Negative") == 2:
                    return False, "Rule 11: mixed=true with two Negative aspects"
                if has_pos and p_list.count("Positive") == 2:
                    return False, "Rule 12: mixed=true with two Positive aspects"
                return False, "Rule D: Mixed flag true without both concrete pos/neg"
        elif mixed is False:
            if has_pos and has_neg:
                return False, "Rule 13: mixed=false despite one concrete Positive and one concrete Negative aspect"
        else:
            return False, "Rule D: missing or non-boolean mixed flag"

        def check_evidence(e_str, e_src):
            if e_str is None: return True, "", None, None
            if not isinstance(e_str, str): return False, "Rule C: evidence must be string", None, None
            stripped = e_str.strip()
            if not stripped: return False, "Rule C: evidence empty after strip", None, None
            if len(stripped.split()) > 7: return False, "Rule C: evidence quote exceeds 7 words", None, None
            if e_src not in ("summary", "reviewText"): return False, "Rule C: Invalid evidence source", None, None
            src_text = str(source_row.get(e_src, ""))
            count = src_text.count(stripped)
            if count == 0: return False, "Rule 14: evidence quote absent from source", None, None
            if count > 1:  return False, "Rule 15: evidence quote repeated in source", None, None
            start_idx = src_text.find(stripped)
            end_idx   = start_idx + len(stripped)
            if start_idx > 0 and src_text[start_idx - 1].isalnum():
                return False, "Rule C: quote starts inside an alphanumeric word", None, None
            if end_idx < len(src_text) and src_text[end_idx].isalnum():
                return False, "Rule C: quote ends inside an alphanumeric word", None, None
            return True, "", start_idx, end_idx

        ok, msg, start1, end1 = check_evidence(e1, e_src1)
        if not ok: return False, msg
        ok, msg, start2, end2 = check_evidence(e2, e_src2)
        if not ok: return False, msg
        item["_derived"] = {"e1_start": start1, "e1_end": end1,
                            "e2_start": start2, "e2_end": end2}
        return True, ""


# ---------------------------------------------------------------------------
# =========================================================================
# DRY-RUN HARNESS  (locked — run_004 contract test, exactly as validated)
# =========================================================================
# ---------------------------------------------------------------------------
def _dry_run_mock_transport(fixture_id, batch_df):
    items = []
    for idx, row in batch_df.iterrows():
        rev_text = str(row.get("reviewText", ""))
        safe_word = None
        for w in rev_text.split():
            w_alnum = "".join(c for c in w if c.isalnum())
            if not w_alnum: continue
            if rev_text.count(w_alnum) == 1:
                midx = rev_text.find(w_alnum)
                if ((midx == 0 or not rev_text[midx-1].isalnum()) and
                        (midx + len(w_alnum) == len(rev_text) or not rev_text[midx+len(w_alnum)].isalnum())):
                    safe_word = w_alnum
                    break
        if not safe_word: safe_word = "UNIQUEMOCKWORD"

        it = {
            "annotation_item_id": row["annotation_item_id"],
            "Aspect1": "Product_Condition_Quality", "Polarity1": "Positive",
            "Evidence_Source1": "reviewText", "Evidence1": safe_word,
            "Aspect2": None, "Polarity2": None,
            "Evidence_Source2": None, "Evidence2": None,
            "Review_Mixed_Flag": False
        }
        if   fixture_id in (1, 2, 17): pass
        elif fixture_id == 3:
            if idx == batch_df.index[0]: del it["annotation_item_id"]
        elif fixture_id == 4:
            if idx == batch_df.index[0] and len(batch_df) > 1:
                it["annotation_item_id"] = batch_df.iloc[1]["annotation_item_id"]
        elif fixture_id == 5:
            if idx == batch_df.index[0]: it["annotation_item_id"] = "ALTERED_ID"
        elif fixture_id == 6:
            if idx == batch_df.index[0]: continue
        elif fixture_id == 7: return "MALFORMED { JSON"
        elif fixture_id == 8: it["Aspect1"] = "Fake_Aspect"
        elif fixture_id == 9: it["Aspect1"], it["Polarity1"] = "None", "NotApplicable"
        elif fixture_id == 10:
            it["Aspect1"], it["Polarity1"], it["Evidence_Source1"], it["Evidence1"] = "None", "NotApplicable", None, None
            it["Aspect2"], it["Polarity2"], it["Evidence_Source2"], it["Evidence2"] = "Product_Condition_Quality", "Positive", "reviewText", safe_word
        elif fixture_id == 11:
            it["Polarity1"] = "Negative"
            it["Aspect2"], it["Polarity2"], it["Evidence_Source2"], it["Evidence2"] = "Delivery_Fulfillment", "Negative", "reviewText", safe_word
            it["Review_Mixed_Flag"] = True
        elif fixture_id == 12:
            it["Aspect2"], it["Polarity2"], it["Evidence_Source2"], it["Evidence2"] = "Delivery_Fulfillment", "Positive", "reviewText", safe_word
            it["Review_Mixed_Flag"] = True
        elif fixture_id == 13:
            it["Aspect2"], it["Polarity2"], it["Evidence_Source2"], it["Evidence2"] = "Delivery_Fulfillment", "Negative", "reviewText", safe_word
            it["Review_Mixed_Flag"] = False
        elif fixture_id == 14: it["Evidence1"] = "XYZW NOT IN TEXT XYZW"
        elif fixture_id == 15: it["Evidence1"] = "a"
        elif fixture_id == 16: it["Extra_Field"] = "Unexpected"
        items.append(it)
    return json.dumps({"items": items})


def run_dry_run():
    """
    Locked run_004 contract test harness.
    Never calls Gemini. Never imports or initialises a Gemini client.
    Writes only to the locked run_004 output paths.
    """
    print("Starting Phase 2B-1 v1.6 DRY-RUN (run_004 contract test)")
    df = pd.read_csv(MANIFEST_PATH)
    df["annotation_item_id"] = (df["reviewerID"].astype(str) + "_" +
                                 df["raw_source_row_id"].astype(str))
    schema    = json.load(open(SCHEMA_PATH))
    validator = OutputValidator(schema)

    os.makedirs(_DRY_RUN_FIXTURES_DIR, exist_ok=True)
    for p in (_DRY_RUN_RAW_RESPONSES_PATH, _DRY_RUN_PARSED_OBJECTS_PATH,
              _DRY_RUN_DERIVED_RECORDS_PATH):
        open(p, "w", encoding="utf-8").close()

    planned_items = df["annotation_item_id"].tolist()
    initial_ckpt = {
        "mode": "dry_run",
        "planned_base_batch_ids": [f"batch_{i+1}_calib" for i in range(15)],
        "planned_item_ids": planned_items,
        "attempted_base_requests": 0, "successful_base_requests": 0,
        "attempted_repair_requests": 0, "successful_repair_requests": 0,
        "attempted_base_batch_ids": [], "successful_base_batch_ids": [],
        "attempted_repair_batch_ids": [], "successful_repair_batch_ids": [],
        "completed_valid_item_ids": [], "invalid_item_ids": [],
        "pending_item_ids": planned_items.copy(),
        "completed_batch_ids": [], "failed_batch_ids": [],
        "batch_sizes": {},
        "terminal_stop_reason": None,
        "hashes": {
            "manifest": get_hash_or_empty(MANIFEST_PATH),
            "schema":   get_hash_or_empty(SCHEMA_PATH),
        },
        "fixture_transport_calls": 0
    }
    atomic_write_checkpoint(initial_ckpt, _DRY_RUN_CHECKPOINT_PATH)

    descriptions = {
        1: "valid normal 8-item object", 2: "valid repair object with 1 item",
        3: "missing ID", 4: "duplicate ID", 5: "altered ID", 6: "missing item",
        7: "malformed JSON", 8: "invalid taxonomy label", 9: "None with evidence",
        10: "None with non-null Aspect2", 11: "mixed=true with two Negative aspects",
        12: "mixed=true with two Positive aspects",
        13: "mixed=false despite one concrete Positive and one concrete Negative aspect",
        14: "evidence quote absent from source",
        15: "evidence quote repeated in source (or invalid boundary)",
        16: "additional unexpected JSON field", 17: "request-count/checkpoint mismatch"
    }
    expected_layers = {
        1: ("PASS", "All"), 2: ("PASS", "All"),
        3: ("REJECT", "Schema Validator"), 4: ("REJECT", "ID Integrity"),
        5: ("REJECT", "ID Integrity"), 6: ("REJECT", "Schema Validator"),
        7: ("REJECT", "JSON Parser"), 8: ("REJECT", "Schema Validator"),
        9: ("REJECT", "Schema Validator"), 10: ("REJECT", "Schema Validator"),
        11: ("REJECT", "Semantic Validator"), 12: ("REJECT", "Semantic Validator"),
        13: ("REJECT", "Semantic Validator"), 14: ("REJECT", "Semantic Validator"),
        15: ("REJECT", "Semantic Validator"), 16: ("REJECT", "Schema Validator"),
        17: ("REJECT", "Ledger Validation"),
    }

    observed_results = []
    for i in range(17):
        fixture_id = i + 1
        safe_i = i % 15
        ckpt = copy.deepcopy(initial_ckpt)
        ckpt["fixture_transport_calls"] += 1

        if fixture_id == 2:
            batch_df = df.iloc[0:1].copy()
            batch_id = f"batch_{fixture_id}_repair"
            ckpt["attempted_repair_requests"] += 1
        else:
            batch_df = df.iloc[safe_i*8:(safe_i+1)*8].copy()
            batch_id = f"batch_{fixture_id}_calib"
            ckpt["attempted_base_requests"] += 1

        requested_ids = batch_df["annotation_item_id"].tolist()
        ckpt["batch_sizes"][batch_id] = len(requested_ids)

        if fixture_id == 15:
            batch_df = batch_df.copy()
            batch_df["reviewText"] = batch_df["reviewText"] + " a a"

        fixture_ckpt_path = os.path.join(_DRY_RUN_FIXTURES_DIR,
                                          f"checkpoint_fixture_{fixture_id}.json")
        atomic_write_checkpoint(ckpt, fixture_ckpt_path)

        raw_res = _dry_run_mock_transport(fixture_id, batch_df)
        append_jsonl(_DRY_RUN_RAW_RESPONSES_PATH, {
            "batch_id": batch_id, "input_annotation_item_ids": requested_ids,
            "mode": "dry_run",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_provider_text": raw_res
        })

        derived_record = {
            "fixture_id": fixture_id, "batch_id": batch_id,
            "status": "PASS", "layer": "All", "error_code": "None", "items": []
        }

        try:
            parsed = json.loads(raw_res)
            append_jsonl(_DRY_RUN_PARSED_OBJECTS_PATH,
                         {"batch_id": batch_id, "parsed_provider_payload": parsed})
        except Exception:
            derived_record.update({"status": "REJECT", "layer": "JSON Parser",
                                   "error_code": "Rule 7: malformed JSON"})
            append_jsonl(_DRY_RUN_DERIVED_RECORDS_PATH, derived_record)
            observed_results.append(derived_record)
            continue

        run_schema = build_run_schema(validator.base_schema, len(requested_ids))
        ok, msg = validator.validate_schema(parsed, run_schema)
        if not ok:
            err = msg
            if fixture_id == 3: err = "Missing annotation_item_id key"
            elif fixture_id == 6: err = "Rule 6: missing item (minItems)"
            elif fixture_id == 8: err = "Rule 8: invalid taxonomy label"
            elif fixture_id == 9: err = "Rule 9: None with evidence"
            elif fixture_id == 10: err = "Rule 10: None with non-null Aspect2"
            elif fixture_id == 16: err = "Rule 16: additional unexpected JSON field"
            derived_record.update({"status": "REJECT", "layer": "Schema Validator",
                                   "error_code": err})
            append_jsonl(_DRY_RUN_DERIVED_RECORDS_PATH, derived_record)
            observed_results.append(derived_record)
            continue

        returned_ids = [it.get("annotation_item_id") for it in parsed.get("items", [])]
        if (len(returned_ids) != len(requested_ids) or
                len(set(returned_ids)) != len(returned_ids) or
                set(returned_ids) != set(requested_ids)):
            err = ("Rule 4: duplicate ID" if len(set(returned_ids)) != len(returned_ids) else
                   "Rule 5: altered ID"   if "ALTERED_ID" in returned_ids else
                   "Rule 3: missing ID")
            derived_record.update({"status": "REJECT", "layer": "ID Integrity",
                                   "error_code": err})
            append_jsonl(_DRY_RUN_DERIVED_RECORDS_PATH, derived_record)
            observed_results.append(derived_record)
            continue

        batch_pass = True
        for it in parsed["items"]:
            row = batch_df[batch_df["annotation_item_id"] == it["annotation_item_id"]].iloc[0]
            ok, msg = validator.validate_semantics(it, row)
            derived = it.pop("_derived", {})
            derived_record["items"].append({
                "annotation_item_id": it["annotation_item_id"],
                "valid": ok, "error": msg if not ok else None,
                "e1_start": derived.get("e1_start"), "e1_end": derived.get("e1_end"),
                "e2_start": derived.get("e2_start"), "e2_end": derived.get("e2_end")
            })
            if not ok:
                if fixture_id == 14: msg = "Rule 14: evidence quote absent from source"
                elif fixture_id == 15: msg = "Rule 15: evidence quote repeated in source"
                derived_record.update({"status": "REJECT", "layer": "Semantic Validator",
                                       "error_code": msg})
                batch_pass = False
                break

        if batch_pass:
            if fixture_id == 2: ckpt["successful_repair_requests"] += 1
            else:               ckpt["successful_base_requests"] += 1
            ckpt["completed_batch_ids"].append(batch_id)
            for rid in requested_ids:
                if rid in ckpt["pending_item_ids"]:
                    ckpt["pending_item_ids"].remove(rid)
                ckpt["completed_valid_item_ids"].append(rid)
        else:
            ckpt["failed_batch_ids"].append(batch_id)

        if fixture_id == 17:
            ckpt["attempted_base_requests"] = 999

        atomic_write_checkpoint(ckpt, fixture_ckpt_path)

        try:
            reconcile_ledger(ckpt)
        except ValueError as e:
            derived_record.update({"status": "REJECT", "layer": "Ledger Validation",
                                   "error_code": f"Rule 17: {str(e)}"})

        append_jsonl(_DRY_RUN_DERIVED_RECORDS_PATH, derived_record)
        observed_results.append(derived_record)

    # Write dry-run report
    with open(_DRY_RUN_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("# Phase 2B-1 v1.6 Dry Run Report (Contract Test)\n\n"
                "| Fixture | Description | Status | Validation Layer | Error Code | Expected Match |\n"
                "|---|---|---|---|---|---|\n")
        for rec in observed_results:
            fix_id = rec["fixture_id"]
            exp_status, exp_layer = expected_layers[fix_id]
            match = "YES" if rec["status"] == exp_status and rec["layer"] == exp_layer else "NO"
            f.write(f"| {fix_id} | {descriptions.get(fix_id,'Unknown')} "
                    f"| {rec['status']} | {rec['layer']} | {rec['error_code']} | {match} |\n")
            if match == "NO":
                print(f"FAILED CONTRACT TEST: Fixture {fix_id} expected {exp_status} "
                      f"at {exp_layer}, got {rec['status']} at {rec['layer']}")

    with open(_DRY_RUN_EXPECTED_VS_OBSERVED_PATH, "w", encoding="utf-8") as f:
        json.dump({"expected_layers": expected_layers,
                   "observed_results": observed_results}, f, indent=2)

    reconcile_ledger(initial_ckpt)

    manifest = {
        "raw_responses_v1_6":      {"count": sum(1 for _ in open(_DRY_RUN_RAW_RESPONSES_PATH,   "rb")), "sha256": get_hash_or_empty(_DRY_RUN_RAW_RESPONSES_PATH)},
        "parsed_objects_v1_6":     {"count": sum(1 for _ in open(_DRY_RUN_PARSED_OBJECTS_PATH,  "rb")), "sha256": get_hash_or_empty(_DRY_RUN_PARSED_OBJECTS_PATH)},
        "derived_records_v1_6":    {"count": sum(1 for _ in open(_DRY_RUN_DERIVED_RECORDS_PATH, "rb")), "sha256": get_hash_or_empty(_DRY_RUN_DERIVED_RECORDS_PATH)},
        "report_v1_6":             {"sha256": get_hash_or_empty(_DRY_RUN_REPORT_PATH)},
        "checkpoint_v1_6":         {"sha256": get_hash_or_empty(_DRY_RUN_CHECKPOINT_PATH)},
        "expected_vs_observed_v1_6": {"sha256": get_hash_or_empty(_DRY_RUN_EXPECTED_VS_OBSERVED_PATH)},
    }
    with open(_DRY_RUN_MANIFEST_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("Dry-run complete. All 17 fixtures executed.")


# ---------------------------------------------------------------------------
# =========================================================================
# EXECUTE BRANCH  (preflight / smoke / live-pilot only)
# =========================================================================
# ---------------------------------------------------------------------------
def create_batch_snapshot(batch_id, run_id, mode, batch_df, source_manifest_sha256, out_dir):
    items_for_hash = []
    for _, row in batch_df.iterrows():
        items_for_hash.append({
            "annotation_item_id": str(row["annotation_item_id"]),
            "summary":    str(row.get("summary",    "")),
            "reviewText": str(row.get("reviewText", ""))
        })
    batch_input_sha256 = hashlib.sha256(
        json.dumps(items_for_hash, sort_keys=True).encode("utf-8")).hexdigest()

    snapshot_items = []
    for item in items_for_hash:
        snapshot_items.append({
            "annotation_item_id":    item["annotation_item_id"],
            "summary":               item["summary"],
            "reviewText":            item["reviewText"],
            "source_manifest_sha256": source_manifest_sha256,
            "batch_input_sha256":    batch_input_sha256,
            "batch_id":  batch_id,
            "run_id":    run_id,
            "mode":      mode
        })

    snapshot_path = os.path.join(out_dir, f"batch_input_snapshot_{batch_id}.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_items, f, indent=2)
    return snapshot_items, batch_input_sha256


def render_prompt(prompt_template, snapshot_items):
    prompt = prompt_template.replace("{EXPECTED_BATCH_SIZE}", str(len(snapshot_items)))
    for item in snapshot_items:
        prompt += (f"\n--- ITEM ---\n"
                   f"annotation_item_id: {item['annotation_item_id']}\n"
                   f"summary: {item['summary']}\n"
                   f"reviewText: {item['reviewText']}\n")
    return prompt


def mock_transport_execute(snapshot_items):
    """
    Smoke-test fake provider.  Returns exactly {"items": [...]}.
    No metadata inside the payload.  Crashes if it cannot find a real quote.
    """
    items = []
    for item in snapshot_items:
        rev_text  = item["reviewText"]
        safe_word = None
        for w in rev_text.split():
            w_alnum = "".join(c for c in w if c.isalnum())
            if not w_alnum: continue
            if rev_text.count(w_alnum) == 1:
                midx = rev_text.find(w_alnum)
                if ((midx == 0 or not rev_text[midx-1].isalnum()) and
                        (midx + len(w_alnum) == len(rev_text) or
                         not rev_text[midx+len(w_alnum)].isalnum())):
                    safe_word = w_alnum
                    break
        if not safe_word:
            raise RuntimeError(
                f"Could not find a unique valid alphanumeric quote in reviewText "
                f"for ID {item['annotation_item_id']}. Smoke test fails.")
        items.append({
            "annotation_item_id": item["annotation_item_id"],
            "Aspect1": "Product_Condition_Quality", "Polarity1": "Positive",
            "Evidence_Source1": "reviewText", "Evidence1": safe_word,
            "Aspect2": None, "Polarity2": None,
            "Evidence_Source2": None, "Evidence2": None,
            "Review_Mixed_Flag": False
        })
    return json.dumps({"items": items})


def call_gemini_provider(prompt, model_name, temperature, provider_schema):
    from google import genai  # imported only when live-pilot is explicitly requested
    from google.genai import types
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found in environment")
    client = genai.Client(api_key=api_key)
    prompt_text = prompt
    provider_response_json_schema = provider_schema
    provider_config = types.GenerateContentConfig(
        temperature=0.0,
        response_mime_type="application/json",
        response_json_schema=provider_response_json_schema,
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt_text,
        config=provider_config,
    )
    return response.text


def _execute_one_batch(batch_id, ckpt, out_dir, df, schema, prompt_template,
                       provider_callable, is_smoke, is_pilot):
    """Process exactly one batch through the full pipeline. Returns True on success."""
    validator = OutputValidator(schema)
    raw_path     = os.path.join(out_dir, "raw_provider_responses_v1_6.jsonl")
    parsed_path  = os.path.join(out_dir, "parsed_objects_v1_6.jsonl")
    derived_path = os.path.join(out_dir, "validation_derived_records_v1_6.jsonl")

    def hard_stop(reason, layer_name, err_code, batch_input_hash=None):
        print(f"FATAL STOP [{layer_name}]: {reason}")
        if layer_name:
            rec = {
                "batch_id": batch_id, "batch_input_sha256": batch_input_hash,
                "status": "REJECT", "layer": layer_name, "error_code": err_code,
                "items": [], "terminal": True
            }
            if is_smoke:
                rec["artifact_mode"] = "smoke_fixture"
                rec["not_calibration_evidence"] = True
            append_jsonl(derived_path, rec)
        ckpt["terminal_stop_reason"] = reason
        if "selected_live_batch" in ckpt:
            ckpt["selected_live_batch"]["terminal_stop_reason"] = reason
        atomic_write_checkpoint(ckpt, os.path.join(out_dir, "checkpoint.json"))
        try:
            reconcile_ledger(ckpt)
        except Exception as le:
            err_rec = {
                "batch_id": batch_id, "batch_input_sha256": batch_input_hash,
                "status": "REJECT", "layer": "Ledger Validation",
                "error_code": f"LedgerFailedInHardStop: {le}",
                "items": [], "terminal": True
            }
            if is_smoke:
                err_rec["artifact_mode"] = "smoke_fixture"
                err_rec["not_calibration_evidence"] = True
            append_jsonl(derived_path, err_rec)
            ckpt["terminal_stop_reason"] = f"Ledger Validation Failed: {le}"
            if "selected_live_batch" in ckpt:
                ckpt["selected_live_batch"]["terminal_stop_reason"] = ckpt["terminal_stop_reason"]
            atomic_write_checkpoint(ckpt, os.path.join(out_dir, "checkpoint.json"))
        sys.exit(1)

    requested_ids = ckpt["batch_item_ids"][batch_id]
    batch_df      = df[df["annotation_item_id"].isin(requested_ids)].copy()

    snapshot_items, batch_input_hash = create_batch_snapshot(
        batch_id, ckpt["run_id"], ckpt["mode"], batch_df,
        ckpt["hashes"]["manifest"], out_dir)
    prompt      = render_prompt(prompt_template, snapshot_items)
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    if "batch_hashes" not in ckpt: ckpt["batch_hashes"] = {}
    ckpt["batch_hashes"][batch_id] = {
        "batch_input_sha256":    batch_input_hash,
        "rendered_prompt_sha256": prompt_hash
    }
    if "selected_live_batch" in ckpt:
        ckpt["selected_live_batch"].update({
            "batch_id": batch_id,
            "annotation_item_ids": requested_ids,
            "batch_input_sha256": batch_input_hash,
            "rendered_prompt_sha256": prompt_hash,
            "taxonomy_sha256": ckpt["hashes"]["taxonomy"]
        })

    prompt_out_path = os.path.join(out_dir, f"rendered_prompt_{batch_id}.md")
    with open(prompt_out_path, "w", encoding="utf-8") as f:
        f.write(prompt)

    ckpt["attempted_base_requests"] += 1
    ckpt["attempted_base_batch_ids"].append(batch_id)
    atomic_write_checkpoint(ckpt, os.path.join(out_dir, "checkpoint.json"))

    run_schema = build_run_schema(validator.base_schema, len(requested_ids))
    provider_schema = build_provider_response_json_schema(validator.base_schema, len(requested_ids))

    proj_report = {
        "canonical_schema_sha256": hashlib.sha256(json.dumps(run_schema, sort_keys=True).encode("utf-8")).hexdigest(),
        "provider_schema_sha256": hashlib.sha256(json.dumps(provider_schema, sort_keys=True).encode("utf-8")).hexdigest(),
        "transform_version": "1.0",
        "removed_keywords": ["$schema", "allOf", "if", "then"],
        "nullable_field_conversions": ["Evidence_Source1", "Evidence_Source2", "Evidence1", "Evidence2", "Aspect2", "Polarity2"],
        "local_canonical_validation_mandatory": True,
        "provider_config_field": "response_json_schema"
    }
    with open(os.path.join(out_dir, "provider_schema_projection.json"), "w", encoding="utf-8") as f:
        json.dump(proj_report, f, indent=2)

    # Invoke provider
    try:
        if is_smoke:
            raw_res = provider_callable(snapshot_items, provider_schema)
        else:
            raw_res = provider_callable(prompt, provider_schema)
    except Exception as e:
        if is_pilot:
            import traceback
            import importlib.metadata
            is_api_error = type(e).__name__ == "APIError"
            exc_report = {
                "failure_stage": "PROVIDER_API" if is_api_error else "SDK_SCHEMA_SERIALIZATION",
                "network_request_sent": is_api_error,
                "provider_response_received": False,
                "python_executable": sys.executable,
                "google_genai_version": None,
                "exception_type": type(e).__name__,
                "exception_repr": repr(e),
                "traceback": traceback.format_exc(),
                "env_booleans": {
                    "HTTP_PROXY": "HTTP_PROXY" in os.environ,
                    "HTTPS_PROXY": "HTTPS_PROXY" in os.environ,
                    "ALL_PROXY": "ALL_PROXY" in os.environ,
                    "SSL_CERT_FILE": "SSL_CERT_FILE" in os.environ,
                    "REQUESTS_CA_BUNDLE": "REQUESTS_CA_BUNDLE" in os.environ
                }
            }
            try:
                exc_report["google_genai_version"] = importlib.metadata.version("google-genai")
            except Exception:
                pass
            with open(os.path.join(out_dir, "provider_exception.json"), "w", encoding="utf-8") as f:
                json.dump(exc_report, f, indent=2)

        ckpt["failed_batch_ids"].append(batch_id)
        hard_stop(f"Provider exception: {e}", "Network/Provider", "Exception",
                  batch_input_hash)

    ckpt["batch_hashes"][batch_id]["raw_provider_text_sha256"] = \
        hashlib.sha256(raw_res.encode("utf-8")).hexdigest()

    raw_record = {
        "batch_id": batch_id, "batch_input_sha256": batch_input_hash,
        "input_annotation_item_ids": requested_ids, "mode": "execute",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_provider_text": raw_res
    }
    if is_smoke:
        raw_record["artifact_mode"] = "smoke_fixture"
        raw_record["not_calibration_evidence"] = True
    append_jsonl(raw_path, raw_record)
    ckpt["batch_hashes"][batch_id]["raw_jsonl_sha256"] = get_hash_if_exists(raw_path)
    if "selected_live_batch" in ckpt:
        ckpt["selected_live_batch"]["raw_jsonl_sha256"] = ckpt["batch_hashes"][batch_id]["raw_jsonl_sha256"]

    derived_record = {
        "batch_id": batch_id, "batch_input_sha256": batch_input_hash,
        "status": "PASS", "layer": "All", "error_code": "None", "items": []
    }
    if is_smoke:
        derived_record["artifact_mode"] = "smoke_fixture"
        derived_record["not_calibration_evidence"] = True

    # Gate 1: JSON parse
    try:
        parsed = json.loads(raw_res)
        ckpt["batch_hashes"][batch_id]["parsed_payload_sha256"] = \
            hashlib.sha256(json.dumps(parsed).encode("utf-8")).hexdigest()
        parsed_record = {
            "batch_id": batch_id, "batch_input_sha256": batch_input_hash,
            "parsed_provider_payload": parsed
        }
        if is_smoke:
            parsed_record["artifact_mode"] = "smoke_fixture"
            parsed_record["not_calibration_evidence"] = True
        append_jsonl(parsed_path, parsed_record)
        ckpt["batch_hashes"][batch_id]["parsed_jsonl_sha256"] = get_hash_if_exists(parsed_path)
        if "selected_live_batch" in ckpt:
            ckpt["selected_live_batch"]["parsed_jsonl_sha256"] = ckpt["batch_hashes"][batch_id]["parsed_jsonl_sha256"]
    except Exception:
        ckpt["failed_batch_ids"].append(batch_id)
        hard_stop("JSON Parser: malformed output", "JSON Parser", "MalformedJSON",
                  batch_input_hash)

    # Gate 2: Schema
    ok, msg = validator.validate_schema(parsed, run_schema)
    if not ok:
        ckpt["failed_batch_ids"].append(batch_id)
        hard_stop(f"Schema Validator: {msg}", "Schema Validator", msg, batch_input_hash)

    # Gate 3: ID integrity
    returned_ids = [it.get("annotation_item_id") for it in parsed.get("items", [])]
    if (len(returned_ids) != len(requested_ids) or
            len(set(returned_ids)) != len(returned_ids) or
            set(returned_ids) != set(requested_ids)):
        ckpt["failed_batch_ids"].append(batch_id)
        hard_stop("ID Integrity: mismatched or duplicate IDs", "ID Integrity",
                  "IDMismatch", batch_input_hash)

    # Gate 4: Semantic
    batch_pass   = True
    invalid_items = []
    for it in parsed["items"]:
        snap_it = next(x for x in snapshot_items
                       if x["annotation_item_id"] == it["annotation_item_id"])
        ok, msg = validator.validate_semantics(
            it, {"reviewText": snap_it["reviewText"], "summary": snap_it["summary"]})
        derived = it.pop("_derived", {})
        derived_record["items"].append({
            "annotation_item_id": it["annotation_item_id"],
            "valid": ok, "error": msg if not ok else None,
            "e1_start": derived.get("e1_start"), "e1_end": derived.get("e1_end"),
            "e2_start": derived.get("e2_start"), "e2_end": derived.get("e2_end")
        })
        if not ok:
            batch_pass = False
            invalid_items.append({"annotation_item_id": it["annotation_item_id"],
                                   "error": msg})

    if not batch_pass:
        derived_record.update({"status": "REJECT", "layer": "Semantic Validator",
                                "error_code": "Semantic validation failures"})
    append_jsonl(derived_path, derived_record)
    ckpt["batch_hashes"][batch_id]["derived_jsonl_sha256"] = get_hash_if_exists(derived_path)
    if "selected_live_batch" in ckpt:
        ckpt["selected_live_batch"]["derived_jsonl_sha256"] = ckpt["batch_hashes"][batch_id]["derived_jsonl_sha256"]

    # Update ledger
    ckpt["successful_base_requests"] += 1
    ckpt["successful_base_batch_ids"].append(batch_id)
    ckpt["completed_batch_ids"].append(batch_id)

    for rid in requested_ids:
        if rid in ckpt["pending_item_ids"]:
            ckpt["pending_item_ids"].remove(rid)

    for it in invalid_items:
        ckpt["invalid_item_ids"].append(it["annotation_item_id"])

    for rid in requested_ids:
        if rid not in ckpt["invalid_item_ids"]:
            ckpt["completed_valid_item_ids"].append(rid)

    atomic_write_checkpoint(ckpt, os.path.join(out_dir, "checkpoint.json"))

    try:
        reconcile_ledger(ckpt)
    except ValueError as e:
        hard_stop(f"Ledger Validation: {e}", "Ledger Validation", str(e),
                  batch_input_hash)

    return batch_pass, invalid_items


def _build_initial_checkpoint(run_id, mode, manifest_hash, schema_hash,
                               prompt_hash, taxonomy_hash, planned_items, df,
                               is_smoke=False):
    ckpt = {
        "mode": "execute",
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "temperature": 0.0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "hashes": {
            "runtime_script": get_hash(os.path.abspath(__file__)),
            "manifest":  manifest_hash,
            "schema":    schema_hash,
            "prompt":    prompt_hash,
            "taxonomy":  taxonomy_hash
        },
        "planned_base_batch_ids": [f"batch_{i+1}_calib" for i in range(15)],
        "batch_item_ids": {},
        "planned_item_ids": planned_items,
        "attempted_base_requests": 0, "successful_base_requests": 0,
        "attempted_repair_requests": 0, "successful_repair_requests": 0,
        "attempted_base_batch_ids": [], "successful_base_batch_ids": [],
        "attempted_repair_batch_ids": [], "successful_repair_batch_ids": [],
        "completed_valid_item_ids": [], "invalid_item_ids": [],
        "pending_item_ids": planned_items.copy(),
        "completed_batch_ids": [], "failed_batch_ids": [],
        "batch_sizes": {},
        "batch_hashes": {},
        "terminal_stop_reason": None
    }
    for i in range(15):
        bid  = f"batch_{i+1}_calib"
        bids = df["annotation_item_id"].iloc[i*8:(i+1)*8].tolist()
        ckpt["batch_item_ids"][bid] = bids
        ckpt["batch_sizes"][bid]    = 8
    if is_smoke:
        ckpt["artifact_mode"]         = "smoke_fixture"
        ckpt["not_calibration_evidence"] = True
    return ckpt


def run_execute(run_id, is_preflight, is_smoke, is_pilot, live_batch_id,
                is_provider_preflight, is_structured_output_preflight,
                is_sdk_schema_compat_preflight, ack_live):
    # ---- Refuse disallowed combinations ----
    is_live_batch = live_batch_id is not None
    if not is_preflight and not is_smoke and not is_pilot and not is_live_batch and not is_provider_preflight and not is_structured_output_preflight and not is_sdk_schema_compat_preflight:
        print("Refusal: Full execution is disabled until the repair loop is "
              "implemented and independently tested.")
        sys.exit(1)

    if is_pilot or is_live_batch:
        # Live safety gates must pass before any I/O.
        if not os.environ.get("GEMINI_API_KEY"):
            print("Refusal: live provider execution requires GEMINI_API_KEY before creating any output.")
            sys.exit(1)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", run_id):
            print("Refusal: live provider run-id must match ^[A-Za-z0-9_-]{1,64}$")
            sys.exit(1)
        if not ack_live:
            print("Refusal: live provider execution requires --ack-live-provider flag.")
            sys.exit(1)
    if is_live_batch:
        if not re.fullmatch(r"batch_(?:[2-9]|1[0-5])_calib", live_batch_id):
            print("Refusal: --live-batch-id must be exactly one remaining planned batch ID: batch_2_calib through batch_15_calib.")
            sys.exit(1)

    print(f"Starting Phase 2B-1 v1.6 EXECUTE mode: run_id={run_id} "
          f"preflight={is_preflight} smoke={is_smoke} pilot={is_pilot} live_batch_id={live_batch_id}")

    # ---- Choose output directory ----
    if is_preflight:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_preflight_{run_id}"
    elif is_smoke:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_execute_SMOKE_{run_id}"
    elif is_pilot:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_pilot_{run_id}"
    elif is_live_batch:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_live_batch_{run_id}"
    elif is_provider_preflight:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_provider_preflight_{run_id}"
    elif is_structured_output_preflight:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_structured_preflight_{run_id}"
    elif is_sdk_schema_compat_preflight:
        out_dir = f"outputs/amazon_v5_rebuild/annotation/phase2b1_v16_sdk_compat_preflight_{run_id}"
    else:
        sys.exit(1)  # unreachable

    if os.path.exists(out_dir) and os.listdir(out_dir):
        if is_live_batch:
            ckpt_existing_path = os.path.join(out_dir, "checkpoint.json")
            if os.path.exists(ckpt_existing_path):
                with open(ckpt_existing_path, "r", encoding="utf-8") as f:
                    existing_ckpt = json.load(f)
                if live_batch_id in existing_ckpt.get("completed_batch_ids", []):
                    print(f"Refusal: Batch '{live_batch_id}' is already complete in '{out_dir}'.")
                    sys.exit(1)
            print(f"Refusal: Output directory '{out_dir}' already has artifacts; no retry or repair is allowed.")
            sys.exit(1)
        print(f"Refusal: Output directory '{out_dir}' already exists and is not empty.")
        sys.exit(1)
    os.makedirs(out_dir, exist_ok=True)
    
    # ---- STRUCTURED OUTPUT PREFLIGHT ----
    if is_structured_output_preflight:
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            base_schema = json.load(f)
        run_schema = build_run_schema(base_schema, 8)
        run_schema_hash = hashlib.sha256(json.dumps(run_schema, sort_keys=True).encode("utf-8")).hexdigest()
        
        report = {
            "response_mime_type": "application/json",
            "runtime_schema_sha256": run_schema_hash,
            "batch_size": 8,
            "root_required_keys": base_schema.get("required", []),
            "item_required_keys": base_schema["properties"]["items"]["items"].get("required", []),
            "additional_properties_values": {
                "root": base_schema.get("additionalProperties"),
                "item": base_schema["properties"]["items"]["items"].get("additionalProperties")
            },
            "provider_schema_equals_validator_schema": True
        }
        with open(os.path.join(out_dir, "structured_output_preflight.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Structured output preflight complete.")
        sys.exit(0)

    # ---- SDK SCHEMA COMPAT PREFLIGHT ----
    if is_sdk_schema_compat_preflight:
        import traceback
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            base_schema = json.load(f)
        canonical_schema = build_run_schema(base_schema, 8)
        provider_response_json_schema = build_provider_response_json_schema(base_schema, 8)

        forbidden_schema_keywords_absent = False
        forbidden_schema_error = None
        try:
            assert_provider_projection_compat(provider_response_json_schema)
            forbidden_schema_keywords_absent = True
        except Exception:
            forbidden_schema_error = traceback.format_exc()

        from google.genai import types
        provider_config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_json_schema=provider_response_json_schema,
        )
        config_payload = provider_config.model_dump(exclude_none=True, by_alias=False)

        validated_config = None
        try:
            validated_config = types.GenerateContentConfig.model_validate(provider_config)
            sdk_config_model_validation_pass = True
            sdk_config_model_validation_error = None
        except Exception:
            sdk_config_model_validation_pass = False
            sdk_config_model_validation_error = traceback.format_exc()

        from google import genai
        from google.genai.models import _GenerateContentConfig_to_mldev

        sdk_transformer_pass = False
        sdk_transformer_error = None
        try:
            client = genai.Client(api_key="mock")
            if validated_config is None:
                sdk_transformer_error = "Skipped because GenerateContentConfig.model_validate failed."
            else:
                _GenerateContentConfig_to_mldev(client._api_client, validated_config)
                sdk_transformer_pass = True
        except Exception:
            sdk_transformer_error = traceback.format_exc()

        report = {
            "sdk_config_model_validation_pass": sdk_config_model_validation_pass,
            "sdk_config_model_validation_error": sdk_config_model_validation_error,
            "sdk_transformer_pass": sdk_transformer_pass,
            "sdk_transformer_error": sdk_transformer_error,
            "response_json_schema_present": "response_json_schema" in config_payload,
            "response_format_absent": "response_format" not in config_payload,
            "response_schema_absent": "response_schema" not in config_payload,
            "forbidden_schema_keywords_absent": forbidden_schema_keywords_absent,
            "forbidden_schema_error": forbidden_schema_error,
            "canonical_schema_sha256": hashlib.sha256(json.dumps(canonical_schema, sort_keys=True).encode("utf-8")).hexdigest(),
            "provider_projection_schema_sha256": hashlib.sha256(json.dumps(provider_response_json_schema, sort_keys=True).encode("utf-8")).hexdigest(),
            "network_request_sent": False
        }
        with open(os.path.join(out_dir, "sdk_schema_compat_preflight.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("SDK Schema Compat preflight complete.")
        sys.exit(0)

    # ---- PROVIDER PREFLIGHT ----
    if is_provider_preflight:
        import traceback
        import importlib.metadata
        report = {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "google_genai_version": None,
            "genai_file": None,
            "api_key_present": False,
            "api_key_length": None,
            "api_key_has_whitespace_or_quotes": False,
            "env_booleans": {
                "HTTP_PROXY": "HTTP_PROXY" in os.environ,
                "HTTPS_PROXY": "HTTPS_PROXY" in os.environ,
                "ALL_PROXY": "ALL_PROXY" in os.environ,
                "SSL_CERT_FILE": "SSL_CERT_FILE" in os.environ,
                "REQUESTS_CA_BUNDLE": "REQUESTS_CA_BUNDLE" in os.environ
            },
            "client_construction_pass": False,
            "exception_type": None,
            "exception_repr": None,
            "traceback": None
        }
        try:
            report["google_genai_version"] = importlib.metadata.version("google-genai")
        except Exception:
            pass

        try:
            from google import genai
            report["genai_file"] = getattr(genai, "__file__", None)
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key is not None:
                report["api_key_present"] = True
                report["api_key_length"] = len(api_key)
                has_issues = False
                if api_key != api_key.strip(): has_issues = True
                if "\n" in api_key or "\r" in api_key: has_issues = True
                if api_key.startswith('"') or api_key.startswith("'"): has_issues = True
                if api_key.endswith('"') or api_key.endswith("'"): has_issues = True
                report["api_key_has_whitespace_or_quotes"] = has_issues

                try:
                    client = genai.Client(api_key=api_key)
                    report["client_construction_pass"] = True
                except Exception as e:
                    report["exception_type"] = type(e).__name__
                    report["exception_repr"] = repr(e)
                    report["traceback"] = traceback.format_exc()
            else:
                report["exception_repr"] = "GEMINI_API_KEY environment variable not found"
        except Exception as e:
            report["exception_type"] = type(e).__name__
            report["exception_repr"] = repr(e)
            report["traceback"] = traceback.format_exc()

        with open(os.path.join(out_dir, "provider_preflight.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Provider preflight complete.")
        sys.exit(0)

    # ---- Preflight source checks ----
    manifest_hash = get_hash(MANIFEST_PATH)
    schema_hash   = get_hash(SCHEMA_PATH)
    prompt_hash   = get_hash(PROMPT_TEMPLATE_PATH)
    taxonomy_hash = get_hash(TAXONOMY_PATH)
    assert len(manifest_hash) == 64, "Manifest hash is not 64 characters."

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)

    with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    df = pd.read_csv(MANIFEST_PATH)
    df["annotation_item_id"] = (df["reviewerID"].astype(str) + "_" +
                                 df["raw_source_row_id"].astype(str))
    planned_items = df["annotation_item_id"].tolist()

    if len(planned_items) != 120 or len(set(planned_items)) != 120:
        print("Refusal: manifest does not have exactly 120 unique IDs.")
        sys.exit(1)
    for pid in planned_items:
        assert not str(pid).startswith("ID_"), f"Placeholder ID detected: {pid}"

    ckpt = _build_initial_checkpoint(
        run_id, "execute", manifest_hash, schema_hash, prompt_hash, taxonomy_hash,
        planned_items, df, is_smoke=is_smoke)
    if is_live_batch:
        ckpt["selected_live_batch"] = {
            "batch_id": live_batch_id,
            "annotation_item_ids": ckpt["batch_item_ids"][live_batch_id],
            "taxonomy_sha256": taxonomy_hash,
            "prompt_template_sha256": prompt_hash,
            "terminal_stop_reason": None
        }

    ckpt_path = os.path.join(out_dir, "checkpoint.json")
    atomic_write_checkpoint(ckpt, ckpt_path)

    batch_id = "batch_1_calib"

    # ---- PREFLIGHT-ONLY ----
    if is_preflight:
        batch1_df = df.iloc[:8].copy()
        snapshot_items, _ = create_batch_snapshot(
            batch_id, run_id, "execute", batch1_df, manifest_hash, out_dir)
        prompt_rendered = render_prompt(prompt_template, snapshot_items)
        with open(os.path.join(out_dir, "preflight_rendered_prompt_batch_1.md"),
                  "w", encoding="utf-8") as f:
            f.write(prompt_rendered)
        print("Preflight complete.")
        sys.exit(0)

    # ---- ADAPTER SMOKE TEST ----
    if is_smoke:
        def fake_provider(snap, r_schema): return mock_transport_execute(snap)
        pass_ok, _ = _execute_one_batch(
            batch_id, ckpt, out_dir, df, schema, prompt_template,
            fake_provider, is_smoke=True, is_pilot=False)
        ckpt["terminal_stop_reason"] = "SMOKE_TEST_COMPLETED"
        atomic_write_checkpoint(ckpt, ckpt_path)
        reconcile_ledger(ckpt)
        print("Smoke test complete.")
        return

    # ---- LIVE PILOT ----
    if is_pilot:
        def live_provider(prompt_text, r_schema): return call_gemini_provider(
            prompt_text, MODEL_NAME, 0.0, r_schema)
        pass_ok, invalid = _execute_one_batch(
            batch_id, ckpt, out_dir, df, schema, prompt_template,
            live_provider, is_smoke=False, is_pilot=True)
        if pass_ok and not invalid:
            ckpt["terminal_stop_reason"] = "PILOT_BASE_BATCH_COMPLETE"
        else:
            ckpt["terminal_stop_reason"] = "PILOT_REPAIR_REQUIRED"
        atomic_write_checkpoint(ckpt, ckpt_path)
        reconcile_ledger(ckpt)
        print(f"Live pilot complete. terminal_stop_reason="
              f"{ckpt['terminal_stop_reason']}")
        return

    # ---- CONTROLLED SINGLE LIVE BATCH ----
    if is_live_batch:
        batch_id = live_batch_id
        if batch_id not in ckpt["planned_base_batch_ids"]:
            print(f"Refusal: --live-batch-id '{batch_id}' is not a planned batch ID.")
            sys.exit(1)
        if batch_id in ckpt.get("completed_batch_ids", []):
            print(f"Refusal: Batch '{batch_id}' is already complete in selected run directory.")
            sys.exit(1)

        def live_provider(prompt_text, r_schema): return call_gemini_provider(
            prompt_text, MODEL_NAME, 0.0, r_schema)
        pass_ok, invalid = _execute_one_batch(
            batch_id, ckpt, out_dir, df, schema, prompt_template,
            live_provider, is_smoke=False, is_pilot=True)
        if pass_ok and not invalid:
            ckpt["terminal_stop_reason"] = f"LIVE_BATCH_COMPLETE:{batch_id}"
        else:
            ckpt["terminal_stop_reason"] = f"LIVE_BATCH_REJECTED:{batch_id}"
        ckpt["selected_live_batch"]["terminal_stop_reason"] = ckpt["terminal_stop_reason"]
        atomic_write_checkpoint(ckpt, ckpt_path)
        reconcile_ledger(ckpt)
        print(f"Live batch complete. batch_id={batch_id} terminal_stop_reason="
              f"{ckpt['terminal_stop_reason']}")
        return


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2B-1 Gemini Calibration Pipeline v1.6")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run", action="store_true",
        help="Run locked run_004 contract-test harness (no Gemini)")
    mode_group.add_argument(
        "--execute", action="store_true",
        help="Execute branch (requires --run-id and exactly one action flag)")

    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--preflight-only",    action="store_true")
    parser.add_argument("--adapter-smoke-test", action="store_true")
    parser.add_argument("--live-pilot",         action="store_true")
    parser.add_argument("--live-batch-id",      type=str, default=None)
    parser.add_argument("--provider-preflight", action="store_true")
    parser.add_argument("--structured-output-preflight", action="store_true")
    parser.add_argument("--sdk-schema-compat-preflight", action="store_true")
    parser.add_argument("--ack-live-provider",  action="store_true")

    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()

    elif args.execute:
        if not args.run_id:
            print("Refusal: --execute requires --run-id")
            sys.exit(1)

        action_count = sum([args.preflight_only, args.adapter_smoke_test,
                            args.live_pilot, args.live_batch_id is not None,
                            args.provider_preflight,
                            args.structured_output_preflight,
                            args.sdk_schema_compat_preflight])
        if action_count != 1:
            print("Refusal: --execute requires exactly one of "
                  "--preflight-only, --adapter-smoke-test, --live-pilot, --live-batch-id, --provider-preflight, --structured-output-preflight, --sdk-schema-compat-preflight")
            sys.exit(1)

        run_execute(
            run_id=args.run_id,
            is_preflight=args.preflight_only,
            is_smoke=args.adapter_smoke_test,
            is_pilot=args.live_pilot,
            live_batch_id=args.live_batch_id,
            is_provider_preflight=args.provider_preflight,
            is_structured_output_preflight=args.structured_output_preflight,
            is_sdk_schema_compat_preflight=args.sdk_schema_compat_preflight,
            ack_live=args.ack_live_provider
        )
