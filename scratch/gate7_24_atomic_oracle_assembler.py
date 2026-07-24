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
POLICY_PATH = OUT_DIR / "semantic_policy_v2_1_atomic_candidate_id.md"


def load_base_runner():
    spec = importlib.util.spec_from_file_location("gate7_11_v14_base", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POLICY_VERSION = "semantic_policy_v2_1_atomic_candidate_id"
    module.CAMPAIGN_ID = "V21ATOMIC_ORACLE_ASSEMBLER"
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


def norm_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).lower()


def load_human() -> dict[str, dict[str, str]]:
    with HUMAN_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["annotation_item_id"]: row for row in csv.DictReader(f)}


def find_candidate(runner, provider_input: dict[str, str], source: str, quote: str) -> dict[str, str] | None:
    if not quote:
        return None
    wanted = norm_text(quote)
    candidates = [c for c in runner.candidate_bundle(provider_input) if c["source"] == source]
    exact = [c for c in candidates if norm_text(c["quote"]) == wanted]
    if exact:
        return sorted(exact, key=lambda c: (len(c["quote"]), c["candidate_id"]))[0]
    containing = [c for c in candidates if wanted in norm_text(c["quote"])]
    if containing:
        return sorted(containing, key=lambda c: (len(c["quote"]), c["candidate_id"]))[0]
    return None


def human_to_atomic_targets(runner, provider_input: dict[str, str], row: dict[str, str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    targets = []
    failures = []
    for slot in ("1", "2"):
        aspect = row.get(f"human_Aspect{slot}") or ""
        polarity = row.get(f"human_Polarity{slot}") or ""
        source = row.get(f"human_Evidence_Source{slot}") or ""
        quote = row.get(f"human_Evidence{slot}") or ""
        if not aspect or aspect == "None":
            continue
        candidate = find_candidate(runner, provider_input, source, quote)
        if candidate is None:
            failures.append({
                "slot": slot,
                "aspect": aspect,
                "polarity": polarity,
                "source": source,
                "quote": quote,
                "error": "human_evidence_not_representable_by_candidate_id",
            })
            continue
        targets.append({
            "aspect": aspect,
            "polarity": polarity,
            "evidence_source": candidate["source"],
            "evidence_candidate_id": candidate["candidate_id"],
        })
    return targets, failures


def assemble_targets(runner, provider_input: dict[str, str], targets: list[dict[str, str]]) -> dict[str, Any]:
    candidates = {c["candidate_id"]: c for c in runner.candidate_bundle(provider_input)}
    deduped = []
    seen = set()
    for target in targets:
        key = (target["aspect"], target["polarity"], target["evidence_source"], target["evidence_candidate_id"])
        if key in seen:
            continue
        seen.add(key)
        candidate = candidates[target["evidence_candidate_id"]]
        deduped.append({
            "aspect": target["aspect"],
            "polarity": target["polarity"],
            "evidence_source": candidate["source"],
            "evidence": candidate["quote"],
        })
    if len(deduped) > 2:
        raise RuntimeError("assembler received more than two deduped targets")
    item: dict[str, Any] = {"annotation_item_id": provider_input["annotation_item_id"]}
    if not deduped:
        item.update({
            "Aspect1": "None",
            "Polarity1": "NotApplicable",
            "Evidence_Source1": None,
            "Evidence1": None,
            "Aspect2": None,
            "Polarity2": None,
            "Evidence_Source2": None,
            "Evidence2": None,
            "Review_Mixed_Flag": False,
        })
        return item
    first = deduped[0]
    item.update({
        "Aspect1": first["aspect"],
        "Polarity1": first["polarity"],
        "Evidence_Source1": first["evidence_source"],
        "Evidence1": first["evidence"],
    })
    if len(deduped) == 2:
        second = deduped[1]
        item.update({
            "Aspect2": second["aspect"],
            "Polarity2": second["polarity"],
            "Evidence_Source2": second["evidence_source"],
            "Evidence2": second["evidence"],
        })
    else:
        item.update({
            "Aspect2": None,
            "Polarity2": None,
            "Evidence_Source2": None,
            "Evidence2": None,
        })
    polarities = [t["polarity"] for t in deduped if t["polarity"] in ("Positive", "Negative")]
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
    run_id = f"V21_ATOMIC_ORACLE_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = pathlib.Path(tempfile.gettempdir()) / "MachineLearningNCKH" / "v13_calibration_workspace" / "v21_atomic_oracle_runs" / run_id
    if run_dir.exists():
        raise RuntimeError(f"refusing to overwrite run dir: {run_dir}")
    (run_dir / "batches").mkdir(parents=True)

    records = []
    representability_failures = []
    validation_failures = []
    grouped_outputs = {batch["calibration_batch_id"]: [] for batch in manifest["batches"]}
    atomic_packet = {"items": []}
    by_id = {item["annotation_item_id"]: item for item in manifest["items"]}
    for item in sorted(manifest["items"], key=lambda x: x["item_position"]):
        aid = item["annotation_item_id"]
        provider_input = item["provider_input"]
        row = human[aid]
        targets, failures = human_to_atomic_targets(runner, provider_input, row)
        if failures:
            representability_failures.append({"annotation_item_id": aid, "failures": failures})
        assembled = assemble_targets(runner, provider_input, targets)
        ok, msg, offsets = runner.validate_semantics(assembled, provider_input)
        if not ok:
            validation_failures.append({"annotation_item_id": aid, "error": msg, **offsets})
        model_pairs = pairs_from_item(assembled)
        human_pairs = pairs_from_human(row)
        mixed_human = row["human_Review_Mixed_Flag"].lower() == "true"
        records.append({
            "annotation_item_id": aid,
            "selection_split": item["selection_split"],
            "assembled_aspect_polarity_multiset": model_pairs,
            "human_aspect_polarity_multiset": human_pairs,
            "aspect_set_agreement": model_pairs == human_pairs,
            "mixed_flag_agreement": assembled["Review_Mixed_Flag"] == mixed_human,
        })
        grouped_outputs[item["calibration_batch_id"]].append(assembled)
        atomic_packet["items"].append({
            "annotation_item_id": aid,
            "atomic_targets": targets,
        })

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

    split_stats: dict[str, dict[str, int]] = {}
    for record in records:
        stats = split_stats.setdefault(record["selection_split"], {"n": 0, "aspect_set_agree": 0, "mixed_agree": 0})
        stats["n"] += 1
        stats["aspect_set_agree"] += int(record["aspect_set_agreement"])
        stats["mixed_agree"] += int(record["mixed_flag_agreement"])

    atomic_path = run_dir / "oracle_atomic_targets.json"
    atomic_path.write_bytes(canonical_json_bytes(atomic_packet))
    comparison = {
        "status": "V21_ATOMIC_ORACLE_ASSEMBLER_PASS" if (
            not representability_failures
            and not validation_failures
            and all(r["aspect_set_agreement"] and r["mixed_flag_agreement"] for r in records)
        ) else "V21_ATOMIC_ORACLE_ASSEMBLER_HOLD",
        "metric_name": "assembler agreement against one-person human adjudication",
        "single_accuracy_claim": False,
        "item_count": len(records),
        "split_stats": split_stats,
        "records": records,
        "representability_failures": representability_failures,
        "validation_failures": validation_failures,
        "oracle_atomic_targets_path": str(atomic_path),
        "oracle_atomic_targets_sha256": sha256_file(atomic_path),
    }
    comparison_path = run_dir / "v21_atomic_oracle_comparison_report.json"
    comparison_path.write_bytes(canonical_json_bytes(comparison))

    report = {
        "gate": "Gate 7.24",
        "status": comparison["status"],
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy_version": "semantic_policy_v2_1_atomic_candidate_id",
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
        "representability_failure_count": len(representability_failures),
        "validation_failure_count": len(validation_failures),
        "run_dir": str(run_dir),
        "comparison_report_path": str(comparison_path),
        "comparison_report_sha256": sha256_file(comparison_path),
        "terminal_decision": "ASSEMBLER_READY_FOR_PROVIDER_STATIC_DESIGN" if comparison["status"].endswith("_PASS") else "ASSEMBLER_HOLD_FIX_REPRESENTABILITY_OR_VALIDATION",
    }
    report_path = OUT_DIR / f"gate7_24_atomic_oracle_assembler_report_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if report_path.exists():
        raise RuntimeError(f"refusing to overwrite report: {report_path}")
    report_path.write_bytes(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))
    print(json.dumps({
        "status": report["status"],
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "run_dir": str(run_dir),
        "comparison_report_sha256": report["comparison_report_sha256"],
        "provider_calls_made": False,
        "aspect_set_agreement_count": report["aspect_set_agreement_count"],
        "mixed_flag_agreement_count": report["mixed_flag_agreement_count"],
        "representability_failure_count": report["representability_failure_count"],
        "validation_failure_count": report["validation_failure_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
