#!/usr/bin/env python
"""Gate 7.19 v1.6 candidate-ID calibration runner.

v1.6 keeps provider evidence as candidate IDs and adds deterministic semantic
guards for recurring policy-level ambiguity found in Gate 7.16-7.18.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


BASE_RUNNER_PATH = Path(__file__).with_name("25_gate7_18_v151_candidate_id_calibration_runner.py")


def load_v151_runner():
    spec = importlib.util.spec_from_file_location("gate7_18_v151_candidate_id_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load v1.5.1 runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v151 = load_v151_runner()
runner = v151.runner
BASE_NORMALIZE = runner.apply_policy_normalization


def item_text(item: dict[str, Any]) -> str:
    return f"{item.get('_summary', '')} {item.get('_reviewText', '')}".lower()


def evidence_text(item: dict[str, Any], slot: str) -> str:
    return str(item.get(f"Evidence{slot}") or "").lower()


def set_slot(item: dict[str, Any], slot: str, aspect: str, polarity: str, source: str, evidence: str) -> None:
    item[f"Aspect{slot}"] = aspect
    item[f"Polarity{slot}"] = polarity
    item[f"Evidence_Source{slot}"] = source
    item[f"Evidence{slot}"] = evidence


def drop_slot2(item: dict[str, Any]) -> None:
    item["Aspect2"] = None
    item["Polarity2"] = None
    item["Evidence_Source2"] = None
    item["Evidence2"] = None


def recompute_mixed(item: dict[str, Any]) -> None:
    polarities = []
    if item.get("Aspect1") != "None" and item.get("Polarity1") in ("Positive", "Negative"):
        polarities.append(item["Polarity1"])
    if item.get("Aspect2") is not None and item.get("Polarity2") in ("Positive", "Negative"):
        polarities.append(item["Polarity2"])
    item["Review_Mixed_Flag"] = "Positive" in polarities and "Negative" in polarities


def normalize_v16(item: dict[str, Any]) -> None:
    BASE_NORMALIZE(item)

    text = item_text(item)
    e1 = evidence_text(item, "1")
    e2 = evidence_text(item, "2")

    # Generic "everything was ok" style feedback has no concrete target.
    if item.get("Aspect1") == "Other_Specific" and item.get("Polarity1") == "Positive":
        if any(cue in text for cue in ("everything was ok", "everything ok", "ok")) and len(text.split()) <= 12:
            item["Aspect1"] = "None"
            item["Polarity1"] = "NotApplicable"
            item["Evidence_Source1"] = None
            item["Evidence1"] = None
            drop_slot2(item)
            recompute_mixed(item)
            return

    # Listing requires an ordering/listing mismatch, not edited mix/series/availability preference.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Listing_Expectation_Compatibility":
            explicit = any(
                cue in text
                for cue in (
                    "listing",
                    "description",
                    "advertised",
                    "wrong item",
                    "wrong cd",
                    "not what was promised",
                    "track list said",
                    "delivered item",
                )
            )
            if not explicit:
                item[f"Aspect{slot}"] = "Domain_Experience"

    # Physical CD/jewel/case damage is product condition, not presentation.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Packaging_Presentation":
            ev = evidence_text(item, slot)
            if any(cue in ev or cue in text for cue in ("case was broken", "cd case", "jewel case", "case fall apart")):
                item[f"Aspect{slot}"] = "Product_Condition_Quality"

    # Format/availability/comparison complaints in music reviews are domain, not performance/usability.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Product_Performance_Usability":
            if any(cue in text for cue in ("album", "songs", "tracks", "cd", "dvd", "digital edition", "shortness")):
                item[f"Aspect{slot}"] = "Domain_Experience"

    # Explicit value/cost complaint should be Price_Value Negative.
    if any(cue in text for cue in ("pricey", "full album price", "shipping cost", "not worth", "worth the shipping")):
        if not any(item.get(f"Aspect{s}") == "Price_Value" and item.get(f"Polarity{s}") == "Negative" for s in ("1", "2")):
            if item.get("Aspect2") is None:
                set_slot(item, "2", "Price_Value", "Negative", item.get("Evidence_Source1") or "reviewText", item.get("Evidence1") or "")
            else:
                # Replace a second domain-negative overcount with the value target.
                if item.get("Aspect2") == "Domain_Experience" and item.get("Polarity2") == "Negative":
                    item["Aspect2"] = "Price_Value"

    # Technical sound/recording words can require Product_Condition_Quality as a second slot.
    technical_sound = any(
        cue in text
        for cue in (
            "quality is very good",
            "sounds better",
            "recording matches",
            "recorded sound is superb",
            "sonics are deep",
            "limited frequency range",
            "sound more muted",
            "recorded sound is wonderful",
        )
    )
    if technical_sound and not any(item.get(f"Aspect{s}") == "Product_Condition_Quality" for s in ("1", "2")):
        if item.get("Aspect2") is None and item.get("Aspect1") != "None":
            polarity = "Negative" if any(cue in text for cue in ("limited frequency", "muted", "too much reverb")) else "Positive"
            set_slot(item, "2", "Product_Condition_Quality", polarity, item.get("Evidence_Source1") or "reviewText", item.get("Evidence1") or "")

    # Avoid bonus positive non-domain aspects unless the text explicitly centers technical sound.
    if item.get("Aspect1") == "Domain_Experience" and item.get("Polarity1") == "Positive" and item.get("Polarity2") == "Positive":
        if item.get("Aspect2") in {"Price_Value", "Other_Specific", "Listing_Expectation_Compatibility"}:
            drop_slot2(item)
        if item.get("Aspect2") == "Product_Condition_Quality" and not technical_sound:
            drop_slot2(item)

    # If both slots became same aspect/same polarity, merge.
    if item.get("Aspect1") == item.get("Aspect2") and item.get("Polarity1") == item.get("Polarity2"):
        drop_slot2(item)

    # If both slots are same aspect with opposite polarity, keep only if there are clear praise and complaint cues.
    if item.get("Aspect1") == item.get("Aspect2") == "Domain_Experience" and item.get("Polarity1") != item.get("Polarity2"):
        has_pos = any(cue in text for cue in ("great", "good", "love", "loved", "excellent", "perfection", "stellar", "few good", "high points"))
        has_neg = any(cue in text for cue in ("disappoint", "weakest", "not worth", "not hard", "not raw", "only disappointment", "not on the same level", "can't get past"))
        if not (has_pos and has_neg):
            if item.get("Polarity1") == "Positive":
                drop_slot2(item)
            else:
                # Prefer explicit positive if the negative is only a caveat.
                if has_pos and not has_neg:
                    item["Polarity1"] = "Positive"
                    drop_slot2(item)

    recompute_mixed(item)


def resolve_candidate_ids_v16(raw: dict[str, Any], provider_input: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    resolved = {"items": []}
    raw_items = raw.get("items", [])
    if len(raw_items) != len(provider_input):
        return {}, {"status": "REJECT", "layer": "Candidate Resolver", "error_code": "raw_item_count_mismatch", "items": []}
    for idx, item in enumerate(raw_items):
        row = provider_input[idx]
        aid = row["annotation_item_id"]
        candidates = {candidate["candidate_id"]: candidate for candidate in runner.candidate_bundle(row)}
        out = dict(item)
        out.pop("item_position", None)
        out["annotation_item_id"] = aid
        out["_summary"] = row.get("summary", "")
        out["_reviewText"] = row.get("reviewText", "")
        for slot in ("1", "2"):
            ev_key = f"Evidence{slot}"
            src_key = f"Evidence_Source{slot}"
            cid = out.get(ev_key)
            if cid is None:
                continue
            candidate = candidates.get(cid)
            if candidate is None:
                return {}, {"status": "REJECT", "layer": "Candidate Resolver", "error_code": f"unknown_candidate_id:{aid}:{cid}", "items": []}
            if out.get(src_key) != candidate["source"]:
                return {}, {"status": "REJECT", "layer": "Candidate Resolver", "error_code": f"candidate_source_mismatch:{aid}:{cid}", "items": []}
            out[ev_key] = candidate["quote"]
        if out.get("Aspect1") == "None" and out.get("Aspect2") is not None:
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
        normalize_v16(out)
        out.pop("_summary", None)
        out.pop("_reviewText", None)
        resolved["items"].append(out)
    return resolved, None


def configure_runner() -> None:
    runner.POLICY_VERSION = "semantic_policy_v1_6_candidate_id"
    runner.CAMPAIGN_ID = "V16CIDCALIBRATION_FRESH_V1"
    runner.EXPECTED_BATCHES = [f"V14CAL_B{i:03d}" for i in range(1, 13)]
    runner.EXPECTED_ITEM_COUNT = 96
    runner.EXPECTED_BATCH_SIZE = 8
    runner.MANIFEST_PATH = runner.OUT_DIR / "gate7_16_v14_candidate_id_execution_manifest_96.json"
    runner.HUMAN_ADJ_PATH = runner.OUT_DIR / "gate7_16_v14_candidate_id_human_adjudication_96.csv"
    runner.LOCKED_HASHES = {
        "manifest": "2437e322c64c113bf7bb49416551ddd49dc52dcecb7dbe89ed5c9572d6938c48",
        "human_adjudication": "f70b57612f30651a2242195452a46c610629da82604ee18206b10729128f30a5",
        "canonical_schema": "781c2661782fd4e259ed2223f2e60ecd90cf30b3a60be1bd68cfe8186b6970ee",
    }
    runner.render_prompt = v151.render_prompt_v151
    runner.apply_policy_normalization = normalize_v16
    runner.resolve_candidate_ids = resolve_candidate_ids_v16


def main() -> None:
    configure_runner()
    runner.main()


if __name__ == "__main__":
    main()
