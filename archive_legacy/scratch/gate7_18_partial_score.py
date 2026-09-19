import csv
import hashlib
import json
import pathlib
from collections import Counter


repo = pathlib.Path(__file__).resolve().parents[1]
out_dir = repo / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
run_dir = pathlib.Path.home() / "AppData" / "Local" / "Temp" / "MachineLearningNCKH" / "v13_calibration_workspace" / "v13_calibration_runs" / "V151CIDCALIBRATION_FRESH_V1_A002_20260713T120121Z_7c83eb2eff"
manifest_path = out_dir / "gate7_16_v14_candidate_id_execution_manifest_96.json"
human_path = out_dir / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
summary_path = out_dir / "gate7_11_v14_candidate_id_live_summary_V151CIDCALIBRATION_FRESH_V1_A002_20260713T120121Z_7c83eb2eff.json"
report_path = out_dir / "gate7_18_v151_partial_score_quota_hold_report.json"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pairs_from_model(item):
    pairs = []
    if item["Aspect1"] != "None":
        pairs.append((item["Aspect1"], item["Polarity1"]))
    if item["Aspect2"] is not None:
        pairs.append((item["Aspect2"], item["Polarity2"]))
    return sorted(pairs)


def pairs_from_human(row):
    pairs = []
    if row["human_Aspect1"] != "None":
        pairs.append((row["human_Aspect1"], row["human_Polarity1"]))
    if row["human_Aspect2"]:
        pairs.append((row["human_Aspect2"], row["human_Polarity2"]))
    return sorted(pairs)


manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
split_by_id = {item["annotation_item_id"]: item["selection_split"] for item in manifest["items"]}
human = {}
with human_path.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        human[row["annotation_item_id"]] = row

records = []
batch_status = []
for batch in manifest["batches"]:
    batch_id = batch["calibration_batch_id"]
    batch_dir = run_dir / "batches" / batch_id
    if not (batch_dir / "resolved_output.json").exists():
        batch_status.append({"batch_id": batch_id, "status": "NOT_EXECUTED_OR_NO_VALIDATED_OUTPUT"})
        continue
    derived = json.loads((batch_dir / "derived_validation_output.json").read_text(encoding="utf-8"))
    batch_status.append({"batch_id": batch_id, "status": derived["status"], "layer": derived["layer"], "error_code": derived["error_code"]})
    if derived["status"] != "PASS":
        continue
    resolved = json.loads((batch_dir / "resolved_output.json").read_text(encoding="utf-8"))
    for item in resolved["items"]:
        aid = item["annotation_item_id"]
        model_pairs = pairs_from_model(item)
        human_pairs = pairs_from_human(human[aid])
        mixed_human = human[aid]["human_Review_Mixed_Flag"].lower() == "true"
        records.append(
            {
                "annotation_item_id": aid,
                "selection_split": split_by_id[aid],
                "model_aspect_polarity_multiset": model_pairs,
                "human_aspect_polarity_multiset": human_pairs,
                "aspect_set_agreement": model_pairs == human_pairs,
                "mixed_flag_agreement": item["Review_Mixed_Flag"] == mixed_human,
            }
        )

split_stats = {}
for record in records:
    stats = split_stats.setdefault(record["selection_split"], {"n": 0, "aspect_set_agree": 0, "mixed_agree": 0})
    stats["n"] += 1
    stats["aspect_set_agree"] += int(record["aspect_set_agreement"])
    stats["mixed_agree"] += int(record["mixed_flag_agreement"])

report = {
    "gate": "Gate 7.18",
    "status": "GATE7_18_PARTIAL_SCORE_QUOTA_HOLD",
    "campaign_run_id": "V151CIDCALIBRATION_FRESH_V1",
    "run_id": "V151CIDCALIBRATION_FRESH_V1_A002_20260713T120121Z_7c83eb2eff",
    "provider_requests_sent": 8,
    "terminal_reason": "PROVIDER_REQUEST_FAILED:429 RESOURCE_EXHAUSTED free-tier daily quota",
    "completed_batches": sum(1 for b in batch_status if b.get("status") == "PASS"),
    "planned_batches": 12,
    "completed_items_scored": len(records),
    "aspect_set_agreement_count": sum(r["aspect_set_agreement"] for r in records),
    "aspect_set_disagreement_count": sum(not r["aspect_set_agreement"] for r in records),
    "mixed_flag_agreement_count": sum(r["mixed_flag_agreement"] for r in records),
    "mixed_flag_disagreement_count": sum(not r["mixed_flag_agreement"] for r in records),
    "split_stats": split_stats,
    "batch_status": batch_status,
    "disagreement_records": [r for r in records if not (r["aspect_set_agreement"] and r["mixed_flag_agreement"])],
    "summary_path": str(summary_path),
    "summary_sha256": sha256_file(summary_path),
    "provider_calls_made": True,
    "expanded_annotation_allowed_now": False,
    "b3_merge_allowed_now": False,
    "gate7_overall_status": "NOT_PASS_QUOTA_HOLD",
}

if report_path.exists():
    raise SystemExit(f"refusing overwrite {report_path}")
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": report["status"], "report_path": str(report_path), "report_sha256": sha256_file(report_path)}, indent=2))
