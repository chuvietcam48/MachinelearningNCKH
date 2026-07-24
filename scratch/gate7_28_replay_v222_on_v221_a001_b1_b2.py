import csv
import importlib.util
import json
import pathlib
import tempfile


REPO = pathlib.Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO / "src" / "amazon_v5_rebuild" / "33_gate7_28_v222_aspect_only_calibration_runner.py"
SOURCE_RUN = pathlib.Path(r"C:\Users\Admin\AppData\Local\Temp\MachineLearningNCKH\v221_aspect_only_workspace\v22_aspect_only_runs\V221ASPECTONLYCALIBRATION_FRESH_V1_A001_20260714T061204Z_9c4f04d9f5")


def load_runner():
    spec = importlib.util.spec_from_file_location("v222_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pairs_from_item(item):
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


def main():
    runner = load_runner()
    manifest = runner.load_manifest(REPO)
    grouped = runner.grouped_provider_inputs(manifest)
    base_schema = runner.load_json(REPO / runner.CANONICAL_SCHEMA_PATH)
    human_path = REPO / runner.HUMAN_ADJ_PATH
    with human_path.open("r", encoding="utf-8-sig", newline="") as f:
        human = {row["annotation_item_id"]: row for row in csv.DictReader(f)}
    out_dir = pathlib.Path(tempfile.gettempdir()) / "MachineLearningNCKH" / "v222_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    batch_status = []
    for batch_id in ("V14CAL_B001", "V14CAL_B002"):
        raw_text = (SOURCE_RUN / "batches" / batch_id / "raw_provider_response.txt").read_text(encoding="utf-8")
        raw, resolved, derived = runner.validate_provider_output(raw_text, grouped[batch_id], base_schema)
        batch_status.append({"batch_id": batch_id, "status": derived["status"], "layer": derived["layer"], "error": derived["error_code"]})
        for item in resolved.get("items", []):
            h = human[item["annotation_item_id"]]
            mp = pairs_from_item(item)
            hp = pairs_from_human(h)
            mixed_h = h["human_Review_Mixed_Flag"].lower() == "true"
            records.append({
                "annotation_item_id": item["annotation_item_id"],
                "model": mp,
                "human": hp,
                "aspect_agree": mp == hp,
                "mixed_agree": item["Review_Mixed_Flag"] == mixed_h,
            })
    print(json.dumps({
        "batch_status": batch_status,
        "n": len(records),
        "aspect_agree": sum(r["aspect_agree"] for r in records),
        "mixed_agree": sum(r["mixed_agree"] for r in records),
        "disagreements": [r for r in records if not (r["aspect_agree"] and r["mixed_agree"])],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
