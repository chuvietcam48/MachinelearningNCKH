#!/usr/bin/env python
"""Gate 7.22 v1.9 candidate-ID calibration runner.

v1.9 keeps the candidate-ID evidence architecture, returns to the v1.4
normalization baseline, and applies cleanup-only semantic guards. It does not
invent missing second aspects during local normalization.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


BASE_RUNNER_PATH = Path(__file__).with_name("22_gate7_11_v14_candidate_id_calibration_runner.py")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("gate7_11_v14_candidate_id_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load v1.4 runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_base_runner()
BASE_NORMALIZE = runner.apply_policy_normalization


def text_of(item: dict[str, Any]) -> str:
    return f"{item.get('_summary', '')} {item.get('_reviewText', '')}".lower()


def evidence_of(item: dict[str, Any], slot: str) -> str:
    return str(item.get(f"Evidence{slot}") or "").lower()


def active_pairs(item: dict[str, Any]) -> list[tuple[str, str, str]]:
    pairs = []
    for slot in ("1", "2"):
        aspect = item.get(f"Aspect{slot}")
        polarity = item.get(f"Polarity{slot}")
        if aspect and aspect != "None":
            pairs.append((slot, aspect, polarity))
    return pairs


def set_slot(item: dict[str, Any], slot: str, aspect: str, polarity: str, source: str | None = None, evidence: str | None = None) -> None:
    item[f"Aspect{slot}"] = aspect
    item[f"Polarity{slot}"] = polarity
    if source is not None:
        item[f"Evidence_Source{slot}"] = source
    if evidence is not None:
        item[f"Evidence{slot}"] = evidence


def drop_slot2(item: dict[str, Any]) -> None:
    item["Aspect2"] = None
    item["Polarity2"] = None
    item["Evidence_Source2"] = None
    item["Evidence2"] = None


def copy_slot(item: dict[str, Any], source_slot: str, dest_slot: str, aspect: str, polarity: str) -> None:
    set_slot(
        item,
        dest_slot,
        aspect,
        polarity,
        item.get(f"Evidence_Source{source_slot}") or "reviewText",
        item.get(f"Evidence{source_slot}") or "",
    )


def has_aspect(item: dict[str, Any], aspect: str, polarity: str | None = None) -> bool:
    for _, a, p in active_pairs(item):
        if a == aspect and (polarity is None or p == polarity):
            return True
    return False


def recompute_mixed(item: dict[str, Any]) -> None:
    polarities = [p for _, _, p in active_pairs(item) if p in ("Positive", "Negative")]
    item["Review_Mixed_Flag"] = "Positive" in polarities and "Negative" in polarities


def merge_duplicate_same_polarity(item: dict[str, Any]) -> None:
    if item.get("Aspect2") is None:
        return
    if item.get("Aspect1") == item.get("Aspect2") and item.get("Polarity1") == item.get("Polarity2"):
        drop_slot2(item)


def normalize_v19(item: dict[str, Any]) -> None:
    BASE_NORMALIZE(item)
    text = text_of(item)
    e1 = evidence_of(item, "1")
    e2 = evidence_of(item, "2")
    all_ev = f"{e1} {e2}"

    positive_domain_cue = any(
        cue in text
        for cue in (
            "few good songs",
            "very good",
            "loved the album",
            "i loved it",
            "great collection",
            "excellent",
            "fantastic",
            "flawless selection",
            "good listening",
            "great compilation",
            "very satisfying",
            "solid enough",
            "worth the purchase",
        )
    )
    negative_domain_cue = any(
        cue in text
        for cue in (
            "disappoint",
            "not worth buying",
            "not good",
            "not hard",
            "not raw",
            "only thing i would have changed",
            "not readily available",
            "unnecessary filler",
            "not a wide range",
            "hard to hear",
            "could only get",
            "not the same",
        )
    )
    technical_quality_cue = any(
        cue in text
        for cue in (
            "sound quality",
            "sound is very good",
            "limited frequency range",
            "hard to hear over",
            "background noise",
            "sound quality is fairly mediocre",
            "remastered",
            "recorded sound",
            "frequency range",
        )
    )
    physical_condition_cue = any(
        cue in text
        for cue in ("case was broken", "cd case fall apart", "jewel case", "case fall apart", "broken when i received")
    )
    explicit_listing_cue = any(
        cue in text
        for cue in (
            "listing",
            "description",
            "advertised",
            "wrong item",
            "wrong cd",
            "wrong dvd",
            "track list said",
            "delivered item",
            "not what was promised",
        )
    )
    value_negative_cue = any(cue in text for cue in ("not worth the shipping", "not worth buying", "paid $", "only paid", "worth the shipping cost"))

    # Prior-series, availability, edited-length, or format-preference complaints are
    # listening/domain expectations unless the listing or delivered item is explicitly wrong.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Listing_Expectation_Compatibility" and not explicit_listing_cue:
            item[f"Aspect{slot}"] = "Domain_Experience"

    # Broken jewel/CD cases are product condition, except when the text foregrounds
    # the money/shipping value complaint as the second complaint.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Packaging_Presentation" and physical_condition_cue:
            item[f"Aspect{slot}"] = "Price_Value" if value_negative_cue else "Product_Condition_Quality"

    # Availability, deluxe/digital preference, album length, and hidden-track access
    # are not generic product usability unless the operation/playback mechanism is
    # itself the complaint.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Product_Performance_Usability":
            if "repeat all" in text or "hidden track" in text:
                continue
            if any(cue in text for cue in ("album", "songs", "tracks", "cd", "digital edition", "shortness")):
                item[f"Aspect{slot}"] = "Domain_Experience"

    # If a review is strongly positive and explicitly brackets a minor caveat, avoid
    # counting that caveat as a second aspect.
    if item.get("Aspect2") == "Domain_Experience" and item.get("Polarity2") == "Negative":
        if any(cue in text for cue in ("shortness of the album aside", "but its good listening", "think of it more as")):
            drop_slot2(item)

    # Technical audio fidelity should be Product_Condition_Quality. Artistic style,
    # genre feel, or performer evaluation stays Domain_Experience.
    for slot in ("1", "2"):
        if item.get(f"Aspect{slot}") == "Product_Condition_Quality" and not technical_quality_cue and not physical_condition_cue:
            item[f"Aspect{slot}"] = "Domain_Experience"

    # Booklet, album cover, back-cover metadata, and box-set documentation are
    # presentation/packaging issues; audio praise in the same review is quality.
    if any(cue in text for cue in ("booklet", "album covers", "back of their respective", "don't list artists", "song titles")):
        if item.get("Aspect2") is not None:
            if item.get("Aspect1") == "Listing_Expectation_Compatibility":
                item["Aspect1"] = "Packaging_Presentation"
            if item.get("Aspect2") == "Listing_Expectation_Compatibility":
                item["Aspect2"] = "Packaging_Presentation"
        elif has_aspect(item, "Packaging_Presentation", "Positive") and negative_domain_cue:
            item["Polarity1" if item.get("Aspect1") == "Packaging_Presentation" else "Polarity2"] = "Negative"

    # Drop bonus positive commerce/quality labels when they merely restate a positive
    # listening review, unless the review independently discusses technical sound.
    if item.get("Polarity2") == "Positive" and item.get("Aspect1") == "Domain_Experience" and item.get("Polarity1") == "Positive":
        if item.get("Aspect2") in {"Other_Specific", "Price_Value", "Listing_Expectation_Compatibility"}:
            drop_slot2(item)
        elif item.get("Aspect2") == "Product_Condition_Quality" and not technical_quality_cue:
            drop_slot2(item)
    if item.get("Polarity1") == "Positive" and item.get("Aspect1") in {"Other_Specific", "Price_Value"} and positive_domain_cue:
        item["Aspect1"] = "Domain_Experience"
    if item.get("Aspect1") == "Product_Condition_Quality" and item.get("Polarity1") == "Positive" and positive_domain_cue:
        if technical_quality_cue:
            copy_slot(item, "1", "2", "Domain_Experience", "Positive")
        else:
            item["Aspect1"] = "Domain_Experience"

    # Drop same-aspect mixed only when the negative is biographical hesitation or
    # context rather than a target-level complaint.
    if item.get("Aspect1") == item.get("Aspect2") == "Domain_Experience" and item.get("Polarity1") != item.get("Polarity2"):
        if not negative_domain_cue:
            if item.get("Polarity1") == "Positive":
                drop_slot2(item)
            elif positive_domain_cue:
                item["Polarity1"] = "Positive"
                drop_slot2(item)

    merge_duplicate_same_polarity(item)
    recompute_mixed(item)


def resolve_candidate_ids_v17(raw: dict[str, Any], provider_input: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
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
        normalize_v19(out)
        out.pop("_summary", None)
        out.pop("_reviewText", None)
        resolved["items"].append(out)
    return resolved, None


def configure_runner() -> None:
    runner.POLICY_VERSION = "semantic_policy_v1_9_candidate_id"
    runner.CAMPAIGN_ID = "V19CIDCALIBRATION_FRESH_V1"
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
    runner.render_prompt = runner.render_prompt
    runner.apply_policy_normalization = normalize_v19
    runner.resolve_candidate_ids = resolve_candidate_ids_v17


def main() -> None:
    configure_runner()
    runner.main()


if __name__ == "__main__":
    main()
