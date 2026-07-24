#!/usr/bin/env python
"""Gate 7.17 v1.5 candidate-ID calibration runner."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


BASE_RUNNER_PATH = Path(__file__).with_name("22_gate7_11_v14_candidate_id_calibration_runner.py")


def load_base_runner():
    spec = importlib.util.spec_from_file_location("gate7_11_v14_candidate_id_base", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load_base_runner()
BASE_POLICY_NORMALIZATION = runner.apply_policy_normalization


def render_prompt_v15(items: list[dict[str, str]]) -> str:
    prompt = f"""You are an expert e-commerce review annotator.

Return exactly one JSON object with one key, items, containing exactly {len(items)} objects.

Each output object must contain exactly these 10 keys:
item_position, Aspect1, Polarity1, Evidence_Source1, Evidence1, Aspect2, Polarity2, Evidence_Source2, Evidence2, Review_Mixed_Flag.

IMPORTANT EVIDENCE CONTRACT:
- Evidence1 and Evidence2 must be candidate IDs, not quotes.
- Copy candidate IDs exactly from candidate_evidence_spans.
- Evidence_Source must match the source shown on that candidate.
- Return items in input order.
- Do not invent quotes or candidate IDs.

V1.5 DECISION CONTRACT:
- SINGLE-ASPECT DEFAULT. Use Aspect2 only for a second central target that changes downstream interpretation.
- Do not use Aspect2 for a minor caveat, title wording, context, generic praise, or a second label for the same target.
- If unsure between one aspect and two aspects, choose one aspect.
- If one evidence span supports both labels, choose the more specific policy label and do not duplicate it.
- Review_Mixed_Flag=true only when the reviewer personally expresses both positive and negative sentiment about central concrete targets.

Taxonomy:
- Domain_Experience: music/content/style/selection/curation/performance/artist/album/listening experience/creative production style/genre feel/atmosphere/musical energy/artistic direction.
- Product_Condition_Quality: physical item defect OR technical fidelity as product quality: mastering, recorded sound, mix clarity, balance, distortion, surface noise, muddy/thin audio, dynamic range, pressing, playback defect, damaged media.
- Product_Performance_Usability: non-content product operation failure, download/playback function, hidden-track access, usability. Never use for album length, song count, musical performance, track quality, or the fact a CD/DVD is less satisfying than another format.
- Delivery_Fulfillment: shipping speed, missing items, wrong items received, delivery.
- Packaging_Presentation: outer packaging, box/cardboard/booklet/liner/back-cover/visual presentation. Damaged/cracked jewel cases, discs, vinyl, and playable media are Product_Condition_Quality.
- Price_Value: explicit cost/worth/value-for-money judgment. Positive Price_Value is rare; use it only when value/price is central, not when the review merely says good purchase, bargain, budget price, worth hearing, or highly recommended.
- Listing_Expectation_Compatibility: only explicit mismatch with listing/delivered item/edition/format/description/pictures/track list/promised content at ordering time.
- Customer_Service_Returns: explicit refund, return, contact, support, or service interaction.
- Other_Specific: avoid unless no defined aspect fits.
- None: no concrete target.

Hard boundary rules:
- Prior album/series/artist expectations, edited mixes, shortened versions, track length preferences, availability wishes, import-only/digital-only access, and wishing for DVD/CD are Domain_Experience unless the review says the listing or delivered item was wrong.
- A physical CD/jewel case that is broken, cracked, falls apart, or is defective is Product_Condition_Quality, not Packaging_Presentation.
- Booklet/cardboard/cover/back-cover/liner-note presentation problems are Packaging_Presentation, not Listing_Expectation_Compatibility.
- Short album length, repeated tracks, song count, or an intro track are Domain_Experience caveats, not Product_Performance_Usability.
- Musical performance, recorded performance, composition, performer skill, interpretation, song quality, compilation quality, and repertoire are Domain_Experience, even when the words performance or recording appear.
- Use Product_Condition_Quality for sound only when the reviewer evaluates fidelity as sound quality. Do not add it from format words like SACD/CD, summary title only, or generic "recording" meaning the album/performance.
- Do not create positive Price_Value, Product_Condition_Quality, Packaging_Presentation, Listing_Expectation_Compatibility, or Other_Specific as a bonus second aspect when Domain_Experience already captures the review's main praise.
- Do not create negative sentiment from "may not appeal", "not for everyone", "I was hesitant", "I expected...", or "I thought..." if the reviewer later resolves the concern positively.
- If the review says "as promised" but is disappointed by the experience, do not use Listing_Expectation_Compatibility.

Input items:
"""
    for item in items:
        prompt += (
            "\n--- ITEM ---\n"
            f"item_position: {item['item_position']}\n"
            f"summary: {item['summary']}\n"
            f"reviewText: {item['reviewText']}\n"
            f"candidate_evidence_spans: {json.dumps(runner.candidate_bundle(item), ensure_ascii=False)}\n"
        )
    return prompt


def normalize_v15(item: dict[str, Any]) -> None:
    BASE_POLICY_NORMALIZATION(item)

    e1 = str(item.get("Evidence1") or "").lower()
    e2 = str(item.get("Evidence2") or "").lower()

    def slot(slot_id: str) -> tuple[str, str, str]:
        return (
            str(item.get(f"Aspect{slot_id}") or ""),
            str(item.get(f"Polarity{slot_id}") or ""),
            str(item.get(f"Evidence{slot_id}") or "").lower(),
        )

    def set_aspect(slot_id: str, aspect: str) -> None:
        item[f"Aspect{slot_id}"] = aspect

    def drop_slot2() -> None:
        item["Aspect2"] = None
        item["Polarity2"] = None
        item["Evidence_Source2"] = None
        item["Evidence2"] = None

    for slot_id in ("1", "2"):
        aspect, polarity, evidence = slot(slot_id)
        if aspect == "Listing_Expectation_Compatibility":
            explicit_listing = any(
                cue in evidence
                for cue in (
                    "listing",
                    "description",
                    "delivered",
                    "wrong item",
                    "wrong cd",
                    "track list",
                    "edition",
                    "promised",
                    "not as advertised",
                )
            )
            if not explicit_listing:
                set_aspect(slot_id, "Domain_Experience")

        if aspect == "Packaging_Presentation" and any(cue in evidence for cue in ("case", "jewel", "disc", "cd case")):
            set_aspect(slot_id, "Product_Condition_Quality")

        if aspect == "Product_Performance_Usability" and any(
            cue in evidence for cue in ("songs", "tracks", "album", "same tracks", "only 9", "shortness")
        ):
            set_aspect(slot_id, "Domain_Experience")

    # Bonus-positive aspects are usually overcounts in music reviews.
    if item.get("Aspect1") == "Domain_Experience" and item.get("Polarity1") == "Positive" and item.get("Polarity2") == "Positive":
        if item.get("Aspect2") in {"Price_Value", "Packaging_Presentation", "Listing_Expectation_Compatibility", "Other_Specific"}:
            drop_slot2()
        elif item.get("Aspect2") == "Product_Condition_Quality" and not any(
            cue in e2
            for cue in (
                "sound quality",
                "recorded sound",
                "sonics",
                "frequency",
                "distortion",
                "noise",
                "master",
                "remaster",
                "mix",
                "fidelity",
            )
        ):
            drop_slot2()

    # If slot2 is a negative caveat explicitly set aside by the positive review, drop it.
    if item.get("Aspect1") == "Domain_Experience" and item.get("Polarity1") == "Positive" and item.get("Polarity2") == "Negative":
        caveat = any(cue in e2 for cue in ("may not appeal", "not for everyone", "shortness", "hesitant", "expected"))
        if caveat:
            drop_slot2()

    # Duplicate same-aspect/same-polarity after remapping.
    if item.get("Aspect1") == item.get("Aspect2") and item.get("Polarity1") == item.get("Polarity2"):
        drop_slot2()

    polarities = []
    if item.get("Aspect1") != "None" and item.get("Polarity1") in ("Positive", "Negative"):
        polarities.append(item["Polarity1"])
    if item.get("Aspect2") is not None and item.get("Polarity2") in ("Positive", "Negative"):
        polarities.append(item["Polarity2"])
    item["Review_Mixed_Flag"] = "Positive" in polarities and "Negative" in polarities


def main() -> None:
    runner.POLICY_VERSION = "semantic_policy_v1_5_candidate_id"
    runner.CAMPAIGN_ID = "V15CIDCALIBRATION_FRESH_V1"
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
    runner.render_prompt = render_prompt_v15
    runner.apply_policy_normalization = normalize_v15
    runner.main()


if __name__ == "__main__":
    main()
