import csv
import datetime
import hashlib
import importlib.util
import json
import pathlib
import re
import tempfile
from typing import Any

from jsonschema import Draft7Validator


REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
RUNNER_PATH = REPO / "src" / "amazon_v5_rebuild" / "22_gate7_11_v14_candidate_id_calibration_runner.py"
MANIFEST_PATH = OUT_DIR / "gate7_16_v14_candidate_id_execution_manifest_96.json"
HUMAN_PATH = OUT_DIR / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
POLICY_PATH = OUT_DIR / "semantic_policy_v2_2_aspect_only_atomic.md"


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


def load_base_runner():
    spec = importlib.util.spec_from_file_location("gate7_11_v14_base", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POLICY_VERSION = "semantic_policy_v2_2_aspect_only_atomic"
    module.CAMPAIGN_ID = "V22ASPECT_ONLY_ORACLE_ASSEMBLER"
    module.EXPECTED_BATCHES = [f"V14CAL_B{i:03d}" for i in range(1, 13)]
    module.EXPECTED_ITEM_COUNT = 96
    module.EXPECTED_BATCH_SIZE = 8
    module.MANIFEST_PATH = module.OUT_DIR / "gate7_16_v14_candidate_id_execution_manifest_96.json"
    module.HUMAN_ADJ_PATH = module.OUT_DIR / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
    module.LOCKED_HASHES = {
        "manifest": "2437e322c64c113bf7bb49416551ddd49dc52dcecb7dbe89ed5c9572d6938c48",
        "human_adjudication": "f70b57612f30651a2242195452a46c610629da82604ee18206b10729128f30a5",
        "canonical_schema": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
    }
    return module


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def load_human() -> dict[str, dict[str, str]]:
    with HUMAN_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["annotation_item_id"]: row for row in csv.DictReader(f)}


def exact_count(source: str, phrase: str) -> int:
    return source.count(phrase)


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
            if phrase in seen or exact_count(source, phrase) != 1:
                continue
            seen.add(phrase)
            windows.append({"source": source_name, "quote": phrase})
    return windows


def score_candidate(aspect: str, polarity: str, candidate: dict[str, str]) -> tuple[int, int, str]:
    quote = candidate["quote"]
    low = quote.lower()
    polarity_key = "positive" if polarity == "Positive" else "negative"
    keywords = KEYWORDS_BY_ASPECT.get(aspect, {}).get(polarity_key, [])
    score = 0
    for keyword in keywords:
        if keyword in low:
            score += 10
    if polarity == "Negative" and any(cue in low for cue in ("not", "no", "n't", "disappoint", "bad", "broken", "flaw")):
        score += 6
    if polarity == "Positive" and any(cue in low for cue in ("good", "great", "love", "excellent", "best", "worth")):
        score += 6
    score += min(len(quote.split()), 7)
    return (score, -len(quote), quote.lower())


def select_evidence(provider_input: dict[str, str], aspect: str, polarity: str) -> dict[str, str]:
    candidates = []
    for source_name in ("reviewText", "summary"):
        candidates.extend(all_unique_windows(provider_input.get(source_name) or "", source_name))
    if not candidates:
        raise RuntimeError(f"no candidate evidence spans for {provider_input['annotation_item_id']}")
    return sorted(candidates, key=lambda c: score_candidate(aspect, polarity, c), reverse=True)[0]


def human_to_aspect_targets(row: dict[str, str]) -> list[dict[str, str]]:
    targets = []
    for slot in ("1", "2"):
        aspect = row.get(f"human_Aspect{slot}") or ""
        polarity = row.get(f"human_Polarity{slot}") or ""
        if aspect and aspect != "None":
            targets.append({"aspect": aspect, "polarity": polarity})
    return targets


def assemble(provider_input: dict[str, str], targets: list[dict[str, str]]) -> dict[str, Any]:
    deduped = []
    seen = set()
    for target in targets:
        key = (target["aspect"], target["polarity"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    if len(deduped) > 2:
        raise RuntimeError("too many targets")
    item: dict[str, Any] = {"annotation_item_id": provider_input["annotation_item_id"]}
    if not deduped:
        item.update({
            "Aspect1": "None", "Polarity1": "NotApplicable", "Evidence_Source1": None, "Evidence1": None,
            "Aspect2": None, "Polarity2": None, "Evidence_Source2": None, "Evidence2": None,
            "Review_Mixed_Flag": False,
        })
        return item
    evidence1 = select_evidence(provider_input, deduped[0]["aspect"], deduped[0]["polarity"])
    item.update({
        "Aspect1": deduped[0]["aspect"],
        "Polarity1": deduped[0]["polarity"],
        "Evidence_Source1": evidence1["source"],
        "Evidence1": evidence1["quote"],
    })
    if len(deduped) == 2:
        evidence2 = select_evidence(provider_input, deduped[1]["aspect"], deduped[1]["polarity"])
        if evidence2["source"] == evidence1["source"] and evidence2["quote"] == evidence1["quote"]:
            alternatives = [
                c for source_name in ("reviewText", "summary")
                for c in all_unique_windows(provider_input.get(source_name) or "", source_name)
                if not (c["source"] == evidence1["source"] and c["quote"] == evidence1["quote"])
            ]
            if alternatives:
                evidence2 = sorted(alternatives, key=lambda c: score_candidate(deduped[1]["aspect"], deduped[1]["polarity"], c), reverse=True)[0]
        item.update({
            "Aspect2": deduped[1]["aspect"],
            "Polarity2": deduped[1]["polarity"],
            "Evidence_Source2": evidence2["source"],
            "Evidence2": evidence2["quote"],
        })
    else:
        item.update({"Aspect2": None, "Polarity2": None, "Evidence_Source2": None, "Evidence2": None})
    polarities = [target["polarity"] for target in deduped if target["polarity"] in ("Positive", "Negative")]
    item["Review_Mixed_Flag"] = "Positive" in polarities and "Negative" in polarities
    return item


def pairs_from_item(item: dict[str, Any]) -> list[tuple[str, str]]:
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


def main() -> None:
    runner = load_base_runner()
    runner.assert_locked_inputs(REPO)
    manifest = runner.load_manifest(REPO)
    human = load_human()
    base_schema = runner.load_json(REPO / runner.CANONICAL_SCHEMA_PATH)
    run_id = f"V22_ASPECT_ONLY_ORACLE_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = pathlib.Path(tempfile.gettempdir()) / "MachineLearningNCKH" / "v13_calibration_workspace" / "v22_aspect_only_oracle_runs" / run_id
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite run dir: {run_dir}")
    (run_dir / "batches").mkdir(parents=True)
    grouped_outputs = {batch["calibration_batch_id"]: [] for batch in manifest["batches"]}
    records = []
    validation_failures = []
    evidence_selection_failures = []
    by_id = {item["annotation_item_id"]: item for item in manifest["items"]}
    atomic_packet = {"items": []}
    for item in sorted(manifest["items"], key=lambda x: x["item_position"]):
        aid = item["annotation_item_id"]
        provider_input = item["provider_input"]
        targets = human_to_aspect_targets(human[aid])
        try:
            assembled = assemble(provider_input, targets)
        except Exception as exc:
            evidence_selection_failures.append({"annotation_item_id": aid, "error": f"{type(exc).__name__}:{exc}"})
            continue
        ok, msg, offsets = runner.validate_semantics(assembled, provider_input)
        if not ok:
            validation_failures.append({"annotation_item_id": aid, "error": msg, **offsets})
        row = human[aid]
        human_pairs = pairs_from_human(row)
        assembled_pairs = pairs_from_item(assembled)
        mixed_human = row["human_Review_Mixed_Flag"].lower() == "true"
        records.append({
            "annotation_item_id": aid,
            "selection_split": item["selection_split"],
            "assembled_aspect_polarity_multiset": assembled_pairs,
            "human_aspect_polarity_multiset": human_pairs,
            "aspect_set_agreement": assembled_pairs == human_pairs,
            "mixed_flag_agreement": assembled["Review_Mixed_Flag"] == mixed_human,
        })
        grouped_outputs[item["calibration_batch_id"]].append(assembled)
        atomic_packet["items"].append({"annotation_item_id": aid, "aspect_only_targets": targets})

    for batch_id, outputs in grouped_outputs.items():
        batch_dir = run_dir / "batches" / batch_id
        batch_dir.mkdir()
        resolved = {"items": outputs}
        derived = {"status": "PASS", "layer": "All", "error_code": "None", "items": []}
        canonical_errors = sorted(Draft7Validator(runner.build_run_schema(base_schema, len(outputs))).iter_errors(resolved), key=lambda e: e.path)
        if canonical_errors:
            derived.update({"status": "REJECT", "layer": "Canonical Schema Validator", "error_code": canonical_errors[0].message})
        for assembled in outputs:
            provider_input = by_id[assembled["annotation_item_id"]]["provider_input"]
            ok, msg, offsets = runner.validate_semantics(assembled, provider_input)
            derived["items"].append({"annotation_item_id": assembled["annotation_item_id"], "valid": ok, "error": None if ok else msg, **offsets})
            if not ok:
                derived.update({"status": "REJECT", "layer": "Semantic Validator", "error_code": msg})
        (batch_dir / "resolved_output.json").write_bytes(canonical_json_bytes(resolved))
        (batch_dir / "derived_validation_output.json").write_bytes(canonical_json_bytes(derived))

    atomic_path = run_dir / "oracle_aspect_only_targets.json"
    atomic_path.write_bytes(canonical_json_bytes(atomic_packet))
    status = "V22_ASPECT_ONLY_ORACLE_ASSEMBLER_PASS" if (
        not evidence_selection_failures
        and not validation_failures
        and len(records) == 96
        and all(r["aspect_set_agreement"] and r["mixed_flag_agreement"] for r in records)
    ) else "V22_ASPECT_ONLY_ORACLE_ASSEMBLER_HOLD"
    comparison = {
        "status": status,
        "metric_name": "assembler agreement against one-person human adjudication",
        "single_accuracy_claim": False,
        "item_count": len(records),
        "records": records,
        "evidence_selection_failures": evidence_selection_failures,
        "validation_failures": validation_failures,
        "oracle_aspect_only_targets_path": str(atomic_path),
        "oracle_aspect_only_targets_sha256": sha256_file(atomic_path),
    }
    comparison_path = run_dir / "v22_aspect_only_oracle_comparison_report.json"
    comparison_path.write_bytes(canonical_json_bytes(comparison))
    report = {
        "gate": "Gate 7.25",
        "status": status,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy_version": "semantic_policy_v2_2_aspect_only_atomic",
        "provider_calls_made": False,
        "expanded_annotation_allowed_now": False,
        "b3_merge_allowed_now": False,
        "input_hashes": {
            "manifest": sha256_file(MANIFEST_PATH),
            "human_adjudication": sha256_file(HUMAN_PATH),
            "policy": sha256_file(POLICY_PATH),
        },
        "oracle_only": True,
        "item_count": len(records),
        "aspect_set_agreement_count": sum(r["aspect_set_agreement"] for r in records),
        "mixed_flag_agreement_count": sum(r["mixed_flag_agreement"] for r in records),
        "evidence_selection_failure_count": len(evidence_selection_failures),
        "validation_failure_count": len(validation_failures),
        "run_dir": str(run_dir),
        "comparison_report_path": str(comparison_path),
        "comparison_report_sha256": sha256_file(comparison_path),
        "terminal_decision": "LOCAL_ASPECT_ONLY_ASSEMBLER_READY_FOR_STATIC_PROVIDER_DESIGN" if status.endswith("_PASS") else "ASSEMBLER_HOLD_FIX_EVIDENCE_SELECTOR",
    }
    report_path = OUT_DIR / f"gate7_25_aspect_only_oracle_assembler_report_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if report_path.exists():
        raise RuntimeError(f"refusing to overwrite report: {report_path}")
    report_path.write_bytes(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    print(json.dumps({
        "status": status,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "run_dir": str(run_dir),
        "comparison_report_sha256": report["comparison_report_sha256"],
        "provider_calls_made": False,
        "aspect_set_agreement_count": report["aspect_set_agreement_count"],
        "mixed_flag_agreement_count": report["mixed_flag_agreement_count"],
        "evidence_selection_failure_count": report["evidence_selection_failure_count"],
        "validation_failure_count": report["validation_failure_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
