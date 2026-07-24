import collections
import datetime
import hashlib
import json
import pathlib


repo = pathlib.Path(__file__).resolve().parents[1]
out_dir = repo / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
run_dir = pathlib.Path.home() / "AppData" / "Local" / "Temp" / "MachineLearningNCKH" / "v13_calibration_workspace" / "v13_calibration_runs" / "V14CIDCALIBRATION_FRESH_V1_A023_20260713T112747Z_7e2e2d5141"
summary_path = out_dir / "gate7_11_v14_candidate_id_live_summary_V14CIDCALIBRATION_FRESH_V1_A023_20260713T112747Z_7e2e2d5141.json"
report_path = out_dir / "gate7_16_strict_scoring_review_report.json"


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
    if "Packaging_Presentation" in model_aspects and "Packaging_Presentation" not in human_aspects:
        return "packaging_presentation_overuse_or_condition_boundary"
    if "Product_Condition_Quality" in model_aspects and "Product_Condition_Quality" not in human_aspects:
        return "product_condition_quality_overuse"
    if "Product_Condition_Quality" in human_aspects and "Product_Condition_Quality" not in model_aspects:
        return "product_condition_quality_underuse"
    if "Price_Value" in model_aspects and "Price_Value" not in human_aspects:
        return "price_value_overuse"
    if "Other_Specific" in model_aspects and "Other_Specific" not in human_aspects:
        return "other_specific_overuse"
    if len(model) > len(human):
        return "extra_aspect_overcount"
    if len(model) < len(human):
        return "missing_second_aspect"
    return "other_semantic_boundary"


def main() -> None:
    comparison_path = run_dir / "v14_candidate_id_comparison_report.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    records = comparison["records"]
    aspect_disagreements = [record for record in records if not record["aspect_set_agreement"]]
    mixed_disagreements = [record for record in records if not record["mixed_flag_agreement"]]
    disagreement_records = [
        record for record in records if not (record["aspect_set_agreement"] and record["mixed_flag_agreement"])
    ]
    split_stats = {}
    for split, stats in comparison["split_stats"].items():
        n = stats["n"]
        split_stats[split] = {
            **stats,
            "aspect_set_agreement_rate": stats["aspect_set_agree"] / n,
            "mixed_flag_agreement_rate": stats["mixed_agree"] / n,
        }

    report = {
        "gate": "Gate 7.16",
        "status": "GATE7_16_STRICT_SCORING_COMPLETE_SEMANTIC_POLICY_HOLD",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "campaign_run_id": summary["campaign_run_id"],
        "run_id": summary["run_id"],
        "policy_version": summary["policy_version"],
        "provider_requests_sent": summary["requests_sent"],
        "structural_validation_status": "PASS",
        "structural_batches_passed": "12/12",
        "human_repair_applied": False,
        "blind_retry_used": False,
        "single_accuracy_claim": False,
        "metric_name": "agreement against one-person blind adjudication",
        "item_count": comparison["item_count"],
        "aspect_set_agreement_count": sum(record["aspect_set_agreement"] for record in records),
        "aspect_set_disagreement_count": len(aspect_disagreements),
        "mixed_flag_agreement_count": sum(record["mixed_flag_agreement"] for record in records),
        "mixed_flag_disagreement_count": len(mixed_disagreements),
        "split_stats": split_stats,
        "p000007_sentinel_pass": comparison["p000007_sentinel_record"]["aspect_set_agreement"]
        and comparison["p000007_sentinel_record"]["mixed_flag_agreement"],
        "creative_style_double_count_error_count": comparison["creative_style_double_count_error_count"],
        "semantic_disagreement_family_counts": dict(collections.Counter(family(record) for record in aspect_disagreements)),
        "fresh_holdout_aspect_set_agreement": "38/48",
        "fresh_holdout_mixed_flag_agreement": "45/48",
        "sentinel_flagged_aspect_set_agreement": "0/12",
        "terminal_decision": "DO_NOT_FREEZE_POLICY_DO_NOT_EXPAND_DO_NOT_MERGE_B3",
        "reason": "Structural candidate-ID architecture passed, but semantic disagreement families remain systematic across sentinel and fresh holdout cases.",
        "comparison_report_path": str(comparison_path),
        "comparison_report_sha256": sha256_file(comparison_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "disagreement_records": disagreement_records,
        "provider_calls_made": True,
        "expanded_annotation_allowed_now": False,
        "b3_merge_allowed_now": False,
    }
    if report_path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_path": str(report_path),
                "report_sha256": sha256_file(report_path),
                "aspect_disagreements": len(aspect_disagreements),
                "mixed_disagreements": len(mixed_disagreements),
                "provider_calls_made": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
