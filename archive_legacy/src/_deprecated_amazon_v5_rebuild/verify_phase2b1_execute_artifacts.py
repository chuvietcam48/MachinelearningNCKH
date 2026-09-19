"""
verify_phase2b1_execute_artifacts.py

Independent disk-based replay verifier for Phase 2B-1 v1.6 execute artifacts.
Reads only from disk — trusts no pre-computed PASS strings.
Exits non-zero on any mismatch.

Usage:
  python src/amazon_v5_rebuild/verify_phase2b1_execute_artifacts.py \\
         --preflight-dir <path> \\
         --smoke-dir <path>
"""

import os
import sys
import json
import argparse
import hashlib
from jsonschema import Draft7Validator

MANIFEST_PATH        = "outputs/amazon_v5_rebuild/annotation/calibration_sample_manifest_v1_1.csv"
SCHEMA_PATH          = "outputs/amazon_v5_rebuild/annotation/semantic_output_schema_v1_6.json"
PROMPT_TEMPLATE_PATH = "outputs/amazon_v5_rebuild/annotation/gemini_semantic_prompt_template_v1_6.md"
TAXONOMY_PATH        = "outputs/amazon_v5_rebuild/annotation/taxonomy_v1_1.md"

# ---------------------------------------------------------------------------
# Local utilities (no imports from the runtime script)
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

def sha256_str(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()

def render_prompt(template, snapshot_items):
    p = template.replace("{EXPECTED_BATCH_SIZE}", str(len(snapshot_items)))
    for item in snapshot_items:
        p += (f"\n--- ITEM ---\n"
              f"annotation_item_id: {item['annotation_item_id']}\n"
              f"summary: {item['summary']}\n"
              f"reviewText: {item['reviewText']}\n")
    return p

def check_evidence(e_str, e_src, source_row):
    """Returns (ok, msg, start, end) where start/end are character offsets or None."""
    if e_str is None:
        return True, "", None, None
    if not isinstance(e_str, str):
        return False, "evidence must be string", None, None
    stripped = e_str.strip()
    if not stripped:
        return False, "evidence empty after strip", None, None
    if len(stripped.split()) > 7:
        return False, "evidence exceeds 7 words", None, None
    if e_src not in ("summary", "reviewText"):
        return False, "invalid evidence source", None, None
    src_text = str(source_row.get(e_src, ""))
    count = src_text.count(stripped)
    if count == 0:
        return False, "evidence absent from source", None, None
    if count > 1:
        return False, "evidence repeated in source", None, None
    s = src_text.find(stripped)
    e = s + len(stripped)
    if s > 0 and src_text[s-1].isalnum():
        return False, "quote starts inside alphanumeric word", None, None
    if e < len(src_text) and src_text[e].isalnum():
        return False, "quote ends inside alphanumeric word", None, None
    return True, "", s, e

def validate_semantics(item, source_row):
    a1, p1 = item.get("Aspect1"), item.get("Polarity1")
    e_src1, e1 = item.get("Evidence_Source1"), item.get("Evidence1")
    a2, p2 = item.get("Aspect2"), item.get("Polarity2")
    e_src2, e2 = item.get("Evidence_Source2"), item.get("Evidence2")
    mixed = item.get("Review_Mixed_Flag")

    p_list = []
    if a1 != "None" and p1 in ("Positive", "Negative"): p_list.append(p1)
    if a2 is not None and p2 in ("Positive", "Negative"): p_list.append(p2)
    has_pos = "Positive" in p_list; has_neg = "Negative" in p_list

    if mixed is True:
        if not (has_pos and has_neg):
            if has_neg and p_list.count("Negative") == 2:
                return False, "Rule 11: mixed=true with two Negative"
            if has_pos and p_list.count("Positive") == 2:
                return False, "Rule 12: mixed=true with two Positive"
            return False, "Rule D: mixed=true without both pos/neg"
    elif mixed is False:
        if has_pos and has_neg:
            return False, "Rule 13: mixed=false with one pos and one neg"
    else:
        return False, "Rule D: non-boolean mixed flag"

    ok1, msg1, s1, e1_end = check_evidence(e1, e_src1, source_row)
    if not ok1: return False, msg1
    ok2, msg2, s2, e2_end = check_evidence(e2, e_src2, source_row)
    if not ok2: return False, msg2
    return True, ""

def compute_offsets(item, source_row):
    """Re-derive e1/e2 offsets from evidence fields and source text."""
    def _offsets(e_str, e_src):
        if e_str is None: return None, None
        src = str(source_row.get(e_src, ""))
        s = src.find(e_str.strip())
        if s < 0: return None, None
        return s, s + len(e_str.strip())
    s1, e1 = _offsets(item.get("Evidence1"), item.get("Evidence_Source1"))
    s2, e2 = _offsets(item.get("Evidence2"), item.get("Evidence_Source2"))
    return s1, e1, s2, e2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--smoke-dir",     required=True)
    args = parser.parse_args()

    report    = {}
    all_passed = True

    def log_result(name, passed, msg=""):
        nonlocal all_passed
        report[name] = {"status": "PASS" if passed else "FAIL", "message": str(msg)}
        if not passed:
            all_passed = False
            print(f"  FAIL [{name}]: {msg}")

    def log_skip(name, reason=""):
        report[name] = {"status": "SKIPPED", "message": reason}

    try:
        # ================================================================
        # PART A: PREFLIGHT VERIFICATION
        # ================================================================
        print("\n=== PREFLIGHT VERIFICATION ===")
        pf_dir = args.preflight_dir

        # A1. Load preflight checkpoint
        pf_ckpt_path = os.path.join(pf_dir, "checkpoint.json")
        with open(pf_ckpt_path) as f:
            pf_ckpt = json.load(f)

        # A2. Runtime script SHA256 vs checkpoint
        runtime_path = os.path.join(
            os.path.dirname(__file__), "07_phase2b1_gemini_calibration_v1_6.py")
        if os.path.exists(runtime_path):
            recomputed_runtime_hash = sha256_file(runtime_path)
            stored_runtime_hash = pf_ckpt.get("hashes", {}).get("runtime_script", "")
            log_result("A1_preflight_runtime_script_hash",
                       recomputed_runtime_hash == stored_runtime_hash,
                       f"recomputed={recomputed_runtime_hash[:16]}... stored={stored_runtime_hash[:16]}...")
        else:
            log_result("A1_preflight_runtime_script_hash", False,
                       f"Runtime script not found at {runtime_path}")

        # A3. Source file hashes
        for label, path, key in [
            ("manifest",  MANIFEST_PATH,        "manifest"),
            ("schema",    SCHEMA_PATH,           "schema"),
            ("prompt",    PROMPT_TEMPLATE_PATH,  "prompt"),
            ("taxonomy",  TAXONOMY_PATH,         "taxonomy"),
        ]:
            stored = pf_ckpt.get("hashes", {}).get(key, "")
            recomputed = sha256_file(path)
            log_result(f"A2_preflight_{label}_hash", recomputed == stored,
                       f"recomputed={recomputed[:16]}... stored={stored[:16]}...")

        # A4. Preflight snapshot present and hash correct
        batch_id = "batch_1_calib"
        snap_path = os.path.join(pf_dir, f"batch_input_snapshot_{batch_id}.json")
        if not os.path.exists(snap_path):
            log_result("A3_preflight_snapshot_exists", False, "Snapshot file missing")
        else:
            log_result("A3_preflight_snapshot_exists", True)
            with open(snap_path) as f:
                pf_snap = json.load(f)

            items_for_hash = [{"annotation_item_id": it["annotation_item_id"],
                               "summary": it["summary"], "reviewText": it["reviewText"]}
                              for it in pf_snap]
            recomp_hash = hashlib.sha256(
                json.dumps(items_for_hash, sort_keys=True).encode("utf-8")).hexdigest()
            stored_hash = pf_snap[0]["batch_input_sha256"]
            log_result("A4_preflight_snapshot_hash_correct",
                       recomp_hash == stored_hash,
                       f"recomputed={recomp_hash[:16]}... stored={stored_hash[:16]}...")

            # A5. Full prompt reconstructible from template + snapshot
            with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
                template = f.read()
            recomp_prompt = render_prompt(template, pf_snap)
            disk_prompt_path = os.path.join(pf_dir, "preflight_rendered_prompt_batch_1.md")
            if os.path.exists(disk_prompt_path):
                with open(disk_prompt_path, "r", encoding="utf-8") as f:
                    disk_prompt = f.read()
                log_result("A5_preflight_prompt_reproducible",
                           recomp_prompt == disk_prompt,
                           "prompt byte-for-byte match" if recomp_prompt == disk_prompt else "mismatch")
            else:
                log_result("A5_preflight_prompt_reproducible", False,
                           "Rendered prompt file missing")

        # ================================================================
        # PART B: SMOKE VERIFICATION
        # ================================================================
        print("\n=== SMOKE VERIFICATION ===")
        sm_dir = args.smoke_dir

        sm_ckpt_path = os.path.join(sm_dir, "checkpoint.json")
        with open(sm_ckpt_path) as f:
            sm_ckpt = json.load(f)

        # B1. Smoke checkpoint fixture labeling
        log_result("B1_smoke_ckpt_artifact_mode",
                   sm_ckpt.get("artifact_mode") == "smoke_fixture",
                   sm_ckpt.get("artifact_mode"))
        log_result("B1_smoke_ckpt_not_calibration",
                   sm_ckpt.get("not_calibration_evidence") is True,
                   sm_ckpt.get("not_calibration_evidence"))

        # B2. Smoke checkpoint ledger contract
        log_result("B2_smoke_attempted_base_requests",
                   sm_ckpt.get("attempted_base_requests") == 1)
        log_result("B2_smoke_successful_base_requests",
                   sm_ckpt.get("successful_base_requests") == 1)
        log_result("B2_smoke_terminal_stop_reason",
                   sm_ckpt.get("terminal_stop_reason") == "SMOKE_TEST_COMPLETED",
                   sm_ckpt.get("terminal_stop_reason"))

        # B3. Runtime script hash vs smoke checkpoint
        if os.path.exists(runtime_path):
            recomputed_runtime_hash = sha256_file(runtime_path)
            stored_rt = sm_ckpt.get("hashes", {}).get("runtime_script", "")
            log_result("B3_smoke_runtime_script_hash",
                       recomputed_runtime_hash == stored_rt,
                       f"recomputed={recomputed_runtime_hash[:16]}... stored={stored_rt[:16]}...")

        # B4. Smoke snapshot hash
        sm_snap_path = os.path.join(sm_dir, f"batch_input_snapshot_{batch_id}.json")
        with open(sm_snap_path) as f:
            sm_snap = json.load(f)

        sm_items_for_hash = [{"annotation_item_id": it["annotation_item_id"],
                               "summary": it["summary"], "reviewText": it["reviewText"]}
                              for it in sm_snap]
        sm_recomp_hash = hashlib.sha256(
            json.dumps(sm_items_for_hash, sort_keys=True).encode("utf-8")).hexdigest()
        sm_stored_hash = sm_snap[0]["batch_input_sha256"]
        log_result("B4_smoke_snapshot_hash_correct",
                   sm_recomp_hash == sm_stored_hash,
                   f"recomputed={sm_recomp_hash[:16]}... stored={sm_stored_hash[:16]}...")

        # B5. Rendered prompt reproducible
        with open(PROMPT_TEMPLATE_PATH, "r", encoding="utf-8") as f:
            template = f.read()
        sm_recomp_prompt = render_prompt(template, sm_snap)
        sm_prompt_path   = os.path.join(sm_dir, f"rendered_prompt_{batch_id}.md")
        if os.path.exists(sm_prompt_path):
            with open(sm_prompt_path, "r", encoding="utf-8") as f:
                sm_disk_prompt = f.read()
            log_result("B5_smoke_prompt_reproducible",
                       sm_recomp_prompt == sm_disk_prompt)
            sm_prompt_hash = sha256_str(sm_disk_prompt)
        else:
            log_result("B5_smoke_prompt_reproducible", False, "rendered prompt missing")
            sm_prompt_hash = None

        # B6. Parsed JSON == json.loads(raw_provider_text)
        with open(os.path.join(sm_dir, "raw_provider_responses_v1_6.jsonl")) as f:
            raw_line = json.loads(f.readline())
        raw_payload = json.loads(raw_line["raw_provider_text"])

        with open(os.path.join(sm_dir, "parsed_objects_v1_6.jsonl")) as f:
            parsed_line = json.loads(f.readline())
        log_result("B6_parsed_equals_raw",
                   raw_payload == parsed_line["parsed_provider_payload"])

        # B7. Raw/parsed outer records carry fixture labels
        log_result("B7_raw_artifact_mode",
                   raw_line.get("artifact_mode") == "smoke_fixture",
                   raw_line.get("artifact_mode"))
        log_result("B7_raw_not_calibration",
                   raw_line.get("not_calibration_evidence") is True)
        log_result("B7_parsed_artifact_mode",
                   parsed_line.get("artifact_mode") == "smoke_fixture",
                   parsed_line.get("artifact_mode"))
        log_result("B7_parsed_not_calibration",
                   parsed_line.get("not_calibration_evidence") is True)

        # B8. Draft7 schema
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        run_schema = json.loads(json.dumps(schema))
        run_schema["properties"]["items"]["minItems"] = 8
        run_schema["properties"]["items"]["maxItems"] = 8
        validator = Draft7Validator(run_schema)
        errors = list(validator.iter_errors(parsed_line["parsed_provider_payload"]))
        schema_passed = (len(errors) == 0)
        log_result("B8_schema_passes", schema_passed, str(errors) if errors else "")

        if not schema_passed:
            for name in ("B9_id_integrity", "B10_semantics_and_offsets",
                         "B11_derived_fixture_labels", "B12_hashes_agree",
                         "B13_no_synthetic_text"):
                log_skip(name, "Schema failed")
            report["overall_status"] = "FAIL"
            print(json.dumps(report, indent=2))
            with open(os.path.join(sm_dir, "verifier_report.json"), "w") as f:
                json.dump(report, f, indent=2)
            sys.exit(1)

        # B9. ID integrity
        ret_ids  = [it["annotation_item_id"] for it in raw_payload["items"]]
        snap_ids = [it["annotation_item_id"] for it in sm_snap]
        log_result("B9_id_integrity",
                   ret_ids == snap_ids and len(ret_ids) == len(set(ret_ids)),
                   f"returned={ret_ids} expected={snap_ids}")

        # B10. Semantics, Evidence1+2 source/uniqueness/offset, derived valid/error
        with open(os.path.join(sm_dir, "validation_derived_records_v1_6.jsonl")) as f:
            derived_line = json.loads(f.readline())

        sem_ok = True
        sem_msgs = []
        for ret_it, der_it in zip(raw_payload["items"], derived_line["items"]):
            snap_it = next(x for x in sm_snap
                           if x["annotation_item_id"] == ret_it["annotation_item_id"])
            src_row = {"reviewText": snap_it["reviewText"], "summary": snap_it["summary"]}

            # Fresh semantic validate
            fresh_valid, fresh_err = validate_semantics(ret_it, src_row)
            if der_it["valid"] != fresh_valid:
                sem_ok = False
                sem_msgs.append(f"ID {ret_it['annotation_item_id']}: derived valid={der_it['valid']} fresh={fresh_valid}")

            # Recompute offsets for Evidence1 and Evidence2
            s1, e1, s2, e2 = compute_offsets(ret_it, src_row)

            if der_it["e1_start"] != s1 or der_it["e1_end"] != e1:
                sem_ok = False
                sem_msgs.append(f"ID {ret_it['annotation_item_id']}: e1 offsets stored=({der_it['e1_start']},{der_it['e1_end']}) recomputed=({s1},{e1})")
            if der_it["e2_start"] != s2 or der_it["e2_end"] != e2:
                sem_ok = False
                sem_msgs.append(f"ID {ret_it['annotation_item_id']}: e2 offsets stored=({der_it['e2_start']},{der_it['e2_end']}) recomputed=({s2},{e2})")

            # Evidence1 source check
            e1_val, e1_src = ret_it.get("Evidence1"), ret_it.get("Evidence_Source1")
            ok1, msg1, _, _ = check_evidence(e1_val, e1_src, src_row)
            if not ok1:
                sem_ok = False
                sem_msgs.append(f"Evidence1 failed for {ret_it['annotation_item_id']}: {msg1}")

            # Evidence2 source check
            e2_val, e2_src = ret_it.get("Evidence2"), ret_it.get("Evidence_Source2")
            ok2, msg2, _, _ = check_evidence(e2_val, e2_src, src_row)
            if not ok2:
                sem_ok = False
                sem_msgs.append(f"Evidence2 failed for {ret_it['annotation_item_id']}: {msg2}")

        log_result("B10_semantics_and_offsets", sem_ok, "; ".join(sem_msgs))

        # B11. Derived outer record carries fixture labels
        log_result("B11_derived_artifact_mode",
                   derived_line.get("artifact_mode") == "smoke_fixture",
                   derived_line.get("artifact_mode"))
        log_result("B11_derived_not_calibration",
                   derived_line.get("not_calibration_evidence") is True)

        # B12. All artifact hashes agree
        h_ok   = True
        h_msgs = []
        bh     = sm_ckpt.get("batch_hashes", {}).get(batch_id, {})

        if raw_line["batch_input_sha256"] != sm_stored_hash:
            h_ok = False; h_msgs.append("raw batch_input_sha256 mismatch")
        if parsed_line["batch_input_sha256"] != sm_stored_hash:
            h_ok = False; h_msgs.append("parsed batch_input_sha256 mismatch")
        if derived_line["batch_input_sha256"] != sm_stored_hash:
            h_ok = False; h_msgs.append("derived batch_input_sha256 mismatch")
        if bh.get("batch_input_sha256") != sm_stored_hash:
            h_ok = False; h_msgs.append("checkpoint batch_input_sha256 mismatch")
        if sm_prompt_hash and bh.get("rendered_prompt_sha256") != sm_prompt_hash:
            h_ok = False; h_msgs.append("rendered_prompt_sha256 mismatch")

        raw_text_hash = sha256_str(raw_line["raw_provider_text"])
        if bh.get("raw_provider_text_sha256") != raw_text_hash:
            h_ok = False; h_msgs.append("raw_provider_text_sha256 mismatch")

        parsed_pl_hash = sha256_str(json.dumps(parsed_line["parsed_provider_payload"]))
        if bh.get("parsed_payload_sha256") != parsed_pl_hash:
            h_ok = False; h_msgs.append("parsed_payload_sha256 mismatch")

        log_result("B12_hashes_agree", h_ok, "; ".join(h_msgs))

        # B13. No synthetic text unless it appears in the snapshot source
        mock_ok   = True
        mock_msgs = []
        for it in raw_payload["items"]:
            snap_it = next(x for x in sm_snap
                           if x["annotation_item_id"] == it["annotation_item_id"])
            for field in ("Evidence1", "Evidence2"):
                val = it.get(field)
                if val == "UNIQUEMOCKWORD":
                    if "UNIQUEMOCKWORD" not in snap_it.get("reviewText", ""):
                        mock_ok = False
                        mock_msgs.append(f"{field}=UNIQUEMOCKWORD not in source for {it['annotation_item_id']}")
        log_result("B13_no_synthetic_text", mock_ok, "; ".join(mock_msgs))

    except Exception as exc:
        import traceback
        traceback.print_exc()
        log_result("verifier_execution_error", False, str(exc))

    report["overall_status"] = "PASS" if all_passed else "FAIL"
    print(f"\noverall_status: {report['overall_status']}")
    print(json.dumps(report, indent=2))
    with open(os.path.join(args.smoke_dir, "verifier_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
