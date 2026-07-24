import collections
import datetime
import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import tempfile


REPO = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
RUNNER_PATH = pathlib.Path(
    os.environ.get(
        "GATE7_REPLAY_RUNNER_PATH",
        REPO / "src" / "amazon_v5_rebuild" / "26_gate7_19_v16_candidate_id_calibration_runner.py",
    )
)
SOURCE_RUN_DIR = (
    pathlib.Path.home()
    / "AppData"
    / "Local"
    / "Temp"
    / "MachineLearningNCKH"
    / "v13_calibration_workspace"
    / "v13_calibration_runs"
    / "V14CIDCALIBRATION_FRESH_V1_A023_20260713T112747Z_7e2e2d5141"
)


def load_v16():
    spec = importlib.util.spec_from_file_location("gate7_19_v16_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner: {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "configure_runner"):
        module.configure_runner()
        return module.runner
    module.POLICY_VERSION = "semantic_policy_v1_4_candidate_id_baseline_replay"
    module.CAMPAIGN_ID = "V14BASELINE_REPLAY"
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


def family(record: dict) -> str:
    model = {tuple(pair) for pair in record["model_aspect_polarity_multiset"]}
    human = {tuple(pair) for pair in record["human_aspect_polarity_multiset"]}
    model_aspects = {aspect for aspect, _ in model}
    human_aspects = {aspect for aspect, _ in human}
    if "Listing_Expectation_Compatibility" in model_aspects and "Listing_Expectation_Compatibility" not in human_aspects:
        return "listing_expectation_overuse"
    if "Listing_Expectation_Compatibility" in human_aspects and "Listing_Expectation_Compatibility" not in model_aspects:
        return "listing_expectation_underuse"
    if "Packaging_Presentation" in model_aspects and "Packaging_Presentation" not in human_aspects:
        return "packaging_presentation_overuse_or_condition_boundary"
    if "Packaging_Presentation" in human_aspects and "Packaging_Presentation" not in model_aspects:
        return "packaging_presentation_underuse"
    if "Product_Condition_Quality" in model_aspects and "Product_Condition_Quality" not in human_aspects:
        return "product_condition_quality_overuse"
    if "Product_Condition_Quality" in human_aspects and "Product_Condition_Quality" not in model_aspects:
        return "product_condition_quality_underuse"
    if "Price_Value" in model_aspects and "Price_Value" not in human_aspects:
        return "price_value_overuse"
    if "Price_Value" in human_aspects and "Price_Value" not in model_aspects:
        return "price_value_underuse"
    if "Other_Specific" in model_aspects and "Other_Specific" not in human_aspects:
        return "other_specific_overuse"
    if len(model) > len(human):
        return "extra_aspect_overcount"
    if len(model) < len(human):
        return "missing_second_aspect"
    return "other_semantic_boundary"


def main() -> None:
    runner = load_v16()
    manifest = runner.load_manifest(REPO)
    grouped = runner.grouped_provider_inputs(manifest)
    base_schema = runner.load_json(REPO / runner.CANONICAL_SCHEMA_PATH)
    replay_root = (
        pathlib.Path(tempfile.gettempdir())
        / "MachineLearningNCKH"
        / "v13_calibration_workspace"
        / "v16_replay_runs"
    )
    replay_label = os.environ.get("GATE7_REPLAY_LABEL", runner.CAMPAIGN_ID.replace("CALIBRATION_FRESH_V1", "REPLAY"))
    run_id = f"{replay_label}_FROM_V14_RAW_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    run_dir = replay_root / run_id
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite replay dir: {run_dir}")
    (run_dir / "batches").mkdir(parents=True)

    batch_results = []
    for batch_id in runner.EXPECTED_BATCHES:
        source_batch = SOURCE_RUN_DIR / "batches" / batch_id
        batch_dir = run_dir / "batches" / batch_id
        batch_dir.mkdir()
        raw_text = (source_batch / "raw_provider_response.txt").read_text(encoding="utf-8")
        provider_input = grouped[batch_id]
        raw, resolved, derived = runner.validate_provider_output(raw_text, provider_input, base_schema)
        (batch_dir / "raw_provider_response.txt").write_text(raw_text, encoding="utf-8")
        (batch_dir / "raw_provider_object.json").write_text(
            json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        (batch_dir / "resolved_output.json").write_text(
            json.dumps(resolved, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        (batch_dir / "derived_validation_output.json").write_text(
            json.dumps(derived, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        batch_results.append({"batch_id": batch_id, "derived_status": derived["status"], "derived_error": derived["error_code"]})
        if derived["status"] != "PASS":
            break

    comparison = runner.compare_against_human(REPO, run_dir) if all(r["derived_status"] == "PASS" for r in batch_results) else None
    if comparison:
        (run_dir / "v16_replay_comparison_report.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        records = comparison["records"]
        aspect_disagreements = [record for record in records if not record["aspect_set_agreement"]]
        mixed_disagreements = [record for record in records if not record["mixed_flag_agreement"]]
        family_counts = dict(collections.Counter(family(record) for record in aspect_disagreements))
        status = (
            "GATE7_19_V16_REPLAY_SEMANTIC_PREFLIGHT_PASS"
            if comparison["status"].endswith("_PASS")
            else "GATE7_19_V16_REPLAY_SEMANTIC_PREFLIGHT_HOLD"
        )
    else:
        records = []
        aspect_disagreements = []
        mixed_disagreements = []
        family_counts = {}
        status = "GATE7_19_V16_REPLAY_STRUCTURAL_FAIL"

    report = {
        "gate": os.environ.get("GATE7_REPLAY_GATE", "Gate 7.19"),
        "status": status,
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "policy_version": runner.POLICY_VERSION,
        "source_run_dir": str(SOURCE_RUN_DIR),
        "source_raw_run_preserved": True,
        "replay_run_dir": str(run_dir),
        "provider_calls_made": False,
        "replay_uses_existing_raw_provider_outputs_only": True,
        "batch_results": batch_results,
        "item_count": len(records),
        "aspect_set_agreement_count": sum(record["aspect_set_agreement"] for record in records),
        "aspect_set_disagreement_count": len(aspect_disagreements),
        "mixed_flag_agreement_count": sum(record["mixed_flag_agreement"] for record in records),
        "mixed_flag_disagreement_count": len(mixed_disagreements),
        "semantic_disagreement_family_counts": family_counts,
        "comparison_status": comparison["status"] if comparison else None,
        "p000007_sentinel_record": comparison["p000007_sentinel_record"] if comparison else None,
        "creative_style_double_count_error_count": comparison["creative_style_double_count_error_count"] if comparison else None,
        "terminal_decision": "DO_NOT_RUN_LIVE_UNTIL_REPLAY_PASS" if status.endswith("_HOLD") else "READY_FOR_LIVE_WHEN_QUOTA_AVAILABLE",
        "comparison_report_path": str(run_dir / "v16_replay_comparison_report.json") if comparison else None,
        "comparison_report_sha256": sha256_file(run_dir / "v16_replay_comparison_report.json") if comparison else None,
        "disagreement_records": [
            record for record in records if not (record["aspect_set_agreement"] and record["mixed_flag_agreement"])
        ],
    }
    safe_label = replay_label.lower()
    out_path = OUT_DIR / f"gate7_replay_{safe_label}_from_v14_raw_report_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite report: {out_path}")
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "report_path": str(out_path),
                "report_sha256": sha256_file(out_path),
                "replay_run_dir": str(run_dir),
                "provider_calls_made": False,
                "aspect_disagreements": len(aspect_disagreements),
                "mixed_disagreements": len(mixed_disagreements),
                "family_counts": family_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
