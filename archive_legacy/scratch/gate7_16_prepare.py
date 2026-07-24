import csv
import datetime
import hashlib
import json
import pathlib
from collections import Counter


repo = pathlib.Path(__file__).resolve().parents[1]
out_dir = repo / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
gate714_manifest = out_dir / "gate7_14_calibration_manifest_v1_4.jsonl"
gate715_human = out_dir / "gate7_15_fresh_holdout_blind_human_adjudication_completed.jsonl"
old_manifest = out_dir / "v13_calibration_manifest_48_v2.json"
old_human = out_dir / "v13_one_person_human_adjudication_completed_v2.csv"

manifest_out = out_dir / "gate7_16_v14_candidate_id_execution_manifest_96.json"
human_out = out_dir / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
report_out = out_dir / "gate7_16_v14_candidate_id_execution_manifest_report.json"
approval_out = out_dir / "v14_candidate_id_calibration_live_approval_A023_gate7_16.json"
ps_out = repo / "scripts" / "run_v14_candidate_id_calibration_live_A023_gate7_16.ps1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_new(path: pathlib.Path, data: bytes) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def load_json(path: pathlib.Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    gate714_rows = [json.loads(line) for line in gate714_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    old_manifest_obj = load_json(old_manifest)
    old_items = {item["annotation_item_id"]: item for item in old_manifest_obj["items"]}
    sentinel_ids = {
        row["annotation_item_id"]
        for row in gate714_rows
        if row["calibration_set"] == "B_12_disagreement_sentinel"
    }

    unique = {}
    for row in gate714_rows:
        aid = row["annotation_item_id"]
        if aid in unique:
            unique[aid]["gate7_14_calibration_sets"].append(row["calibration_set"])
            continue
        unique[aid] = dict(row)
        unique[aid]["gate7_14_calibration_sets"] = [row["calibration_set"]]

    if len(unique) != 96:
        raise SystemExit(f"expected 96 unique items, got {len(unique)}")

    ordered = []
    for row in gate714_rows:
        aid = row["annotation_item_id"]
        if aid in {item["annotation_item_id"] for item in ordered}:
            continue
        ordered.append(unique[aid])

    batch_size = 8
    if len(ordered) % batch_size:
        raise SystemExit("execution manifest item count must be divisible by 8")
    batch_ids = [f"V14CAL_B{i:03d}" for i in range(1, len(ordered) // batch_size + 1)]

    items = []
    batches = []
    for idx, row in enumerate(ordered, 1):
        batch_id = batch_ids[(idx - 1) // batch_size]
        position_in_batch = ((idx - 1) % batch_size) + 1
        aid = row["annotation_item_id"]
        provider_input = row["provider_input"]
        old_source = old_items.get(aid, {})
        selection_split = row["calibration_set"]
        if "B_12_disagreement_sentinel" in row["gate7_14_calibration_sets"]:
            selection_split = f"{selection_split}__sentinel_flagged"
        item = {
            "annotation_item_id": aid,
            "asin": old_source.get("asin"),
            "calibration_batch_id": batch_id,
            "calibration_item_id": f"V14CAL_{idx:03d}",
            "item_position": idx,
            "item_position_in_batch": position_in_batch,
            "provider_input": provider_input,
            "provider_input_sha256": sha256_bytes(canonical_json_bytes(provider_input)),
            "review_timestamp": old_source.get("review_timestamp"),
            "reviewerID": old_source.get("reviewerID"),
            "selection_reason": row["selection_reason"],
            "selection_split": selection_split,
            "gate7_14_calibration_sets": row["gate7_14_calibration_sets"],
            "is_gate7_12_sentinel": aid in sentinel_ids,
            "fresh_holdout": bool(row.get("fresh_holdout")),
            "used_to_draft_v1_4": bool(row.get("used_to_draft_v1_4")),
            "source": row.get("source"),
        }
        items.append(item)

    for i, batch_id in enumerate(batch_ids):
        batch_items = items[i * batch_size : (i + 1) * batch_size]
        provider_inputs = [item["provider_input"] for item in batch_items]
        batches.append(
            {
                "calibration_batch_id": batch_id,
                "annotation_item_ids": [item["annotation_item_id"] for item in batch_items],
                "item_count": batch_size,
                "planned_base_provider_requests": 1,
                "provider_input_sha256": sha256_bytes(canonical_json_bytes(provider_inputs)),
                "retry_policy": "no_blind_retry",
            }
        )

    manifest = {
        "gate": "Gate 7.16",
        "status": "EXECUTION_MANIFEST_READY_NO_PROVIDER_CALLS",
        "calibration_manifest_id": "GATE7_16_V14_CANDIDATE_ID_EXECUTION_MANIFEST_96",
        "manifest_version": "v1",
        "policy_version": "semantic_policy_v1_4_candidate_id",
        "batch_size": batch_size,
        "batch_count": len(batches),
        "item_count": len(items),
        "unique_annotation_item_count": len({item["annotation_item_id"] for item in items}),
        "batches": batches,
        "items": items,
        "selection_splits": dict(Counter(item["selection_split"] for item in items)),
        "sentinel_unique_count": len([item for item in items if item["is_gate7_12_sentinel"]]),
        "fresh_holdout_unique_count": len([item for item in items if item["fresh_holdout"]]),
        "provider_calls_allowed_now": False,
        "provider_calls_made": False,
        "human_repair_allowed_in_calibration": False,
        "blind_retry_allowed": False,
    }
    write_new(manifest_out, canonical_json_bytes(manifest))

    human_rows = {}
    with old_human.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            human_rows[row["annotation_item_id"]] = row
    for line in gate715_human.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        human_rows[row["annotation_item_id"]] = {
            "calibration_item_id": "",
            "calibration_batch_id": "",
            "annotation_item_id": row["annotation_item_id"],
            "selection_split": row["fresh_holdout_group"],
            "summary": row["summary"],
            "reviewText": row["reviewText"],
            "human_Aspect1": row["human_Aspect1"],
            "human_Polarity1": row["human_Polarity1"],
            "human_Evidence_Source1": row["human_Evidence_Source1"],
            "human_Evidence1": row["human_Evidence1"],
            "human_Aspect2": row["human_Aspect2"] or "",
            "human_Polarity2": row["human_Polarity2"] or "",
            "human_Evidence_Source2": row["human_Evidence_Source2"] or "",
            "human_Evidence2": row["human_Evidence2"] or "",
            "human_Review_Mixed_Flag": str(row["human_Review_Mixed_Flag"]).lower(),
            "human_rationale": row["human_rationale"],
        }

    fieldnames = [
        "calibration_item_id",
        "calibration_batch_id",
        "annotation_item_id",
        "selection_split",
        "summary",
        "reviewText",
        "human_Aspect1",
        "human_Polarity1",
        "human_Evidence_Source1",
        "human_Evidence1",
        "human_Aspect2",
        "human_Polarity2",
        "human_Evidence_Source2",
        "human_Evidence2",
        "human_Review_Mixed_Flag",
        "human_rationale",
    ]
    if len(set(human_rows).intersection({item["annotation_item_id"] for item in items})) != 96:
        missing = sorted({item["annotation_item_id"] for item in items} - set(human_rows))
        raise SystemExit(f"missing human rows: {missing}")
    with human_out.open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            row = dict(human_rows[item["annotation_item_id"]])
            row["calibration_item_id"] = item["calibration_item_id"]
            row["calibration_batch_id"] = item["calibration_batch_id"]
            row["selection_split"] = item["selection_split"]
            writer.writerow(row)

    report = {
        "gate": "Gate 7.16",
        "status": "GATE7_16_EXECUTION_MANIFEST_AND_APPROVAL_READY_NO_PROVIDER_CALLS",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_path": str(manifest_out),
        "manifest_sha256": sha256_file(manifest_out),
        "human_adjudication_path": str(human_out),
        "human_adjudication_sha256": sha256_file(human_out),
        "item_count": 96,
        "batch_count": 12,
        "batch_size": 8,
        "sentinel_unique_count": manifest["sentinel_unique_count"],
        "fresh_holdout_unique_count": manifest["fresh_holdout_unique_count"],
        "queue_resolution": {
            "A33GGROUQRQZS_3078550": "included in old human adjudication; final strict label retained",
            "A19T4IHJU5CZOE_3164728": "included in old human adjudication; final strict label retained",
        },
        "provider_calls_made": False,
        "expanded_annotation_allowed_now": False,
        "b3_merge_allowed_now": False,
    }
    write_new(report_out, canonical_json_bytes(report))

    approval = {
        "approval_id": "GATE7_16_V14CIDCALIBRATION_A023_OPERATOR_APPROVAL",
        "attempt_id": "A023",
        "execution_mode": "LIVE",
        "campaign_run_id": "V14CIDCALIBRATION_FRESH_V1",
        "policy_version": "semantic_policy_v1_4_candidate_id",
        "batch_start": batch_ids[0],
        "batch_end": batch_ids[-1],
        "ack_live_provider": True,
        "max_provider_requests": len(batch_ids),
        "max_concurrency": 1,
        "no_blind_retry": True,
        "human_repair_allowed": False,
        "scope": "Gate 7.16 v1.4 candidate-ID calibration over 96 unique cases: old regression, sentinel-flagged cases, and fresh holdout.",
        "manifest_sha256": sha256_file(manifest_out),
        "human_adjudication_sha256": sha256_file(human_out),
        "operator_note": "User requested to proceed. Provider output must be strict-scored; no human repair may make calibration pass.",
    }
    write_new(approval_out, canonical_json_bytes(approval))

    ps = f'''$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Repo ".venv\\Scripts\\python.exe"
$WorkspaceRoot = Join-Path ([System.IO.Path]::GetTempPath()) "MachineLearningNCKH\\v13_calibration_workspace"
$Approval = Join-Path $Repo "outputs\\amazon_v5_rebuild\\annotation\\v1_3_calibration_v2\\{approval_out.name}"

Set-Location $Repo
& $Python "src\\amazon_v5_rebuild\\23_gate7_16_v14_candidate_id_calibration_runner.py" `
  --execute-live `
  --workspace-root $WorkspaceRoot `
  --approval-file $Approval
'''
    write_new(ps_out, ps.encode("utf-8"))

    print(
        json.dumps(
            {
                "status": "GATE7_16_PREPARED",
                "manifest_sha256": sha256_file(manifest_out),
                "human_adjudication_sha256": sha256_file(human_out),
                "report_sha256": sha256_file(report_out),
                "approval_sha256": sha256_file(approval_out),
                "script_sha256": sha256_file(ps_out),
                "provider_calls_made": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
