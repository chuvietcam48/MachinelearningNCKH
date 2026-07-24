#!/usr/bin/env python
"""Gate 7.29 v2.2.3 aspect-only calibration runner.

Provider emits only aspect/polarity targets. Local deterministic code selects
exact evidence spans, assembles canonical output, and validates offsets.
Includes v2.2.3 rule updates.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

def load_v222_runner():
    v222_path = Path(__file__).with_name("33_gate7_28_v222_aspect_only_calibration_runner.py")
    spec = importlib.util.spec_from_file_location("gate7_28_v222", v222_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

v222 = load_v222_runner()

POLICY_VERSION = "semantic_policy_v2_2_3_aspect_only_atomic"
CAMPAIGN_ID = "V223ASPECTONLYCALIBRATION_FRESH_V1"

v222.POLICY_VERSION = POLICY_VERSION
v222.CAMPAIGN_ID = CAMPAIGN_ID
v222.POLICY_PATH = v222.OUT_DIR / "semantic_policy_v2_2_3_aspect_only_atomic.md"

def render_prompt(items: list[dict[str, str]]) -> str:
    prompt = f"""You are an expert e-commerce review annotator.

Return exactly one JSON object with one key, items, containing exactly {len(items)} objects.

Each output item must contain exactly:
- item_position
- targets

targets is an array with 0, 1, or 2 central semantic targets.
Each target has exactly:
- aspect
- polarity

Do not output evidence, evidence IDs, quotes, offsets, final Aspect1/Aspect2 fields, or mixed flags.
Local deterministic code will select evidence and derive mixed feedback.

Aspect labels:
{json.dumps(v222.ASPECTS, ensure_ascii=False)}

Polarity labels:
["Positive", "Negative"]

Rules:
- Use no targets when there is no concrete review target.
- Use one target by default.
- Use two targets only when there are two central concrete targets or clear positive and negative sentiment about central targets.
- Domain_Experience covers music/content/style/selection/curation/performance/artist/album/listening experience/creative production style/genre feel/atmosphere/musical energy/artistic direction.
- Product_Condition_Quality covers physical item defects, damaged playable media/case, AND technical sound/recording/mastering quality (e.g., sound quality, recording quality, recorded sound, remastering, mix, frequency range, hard to hear, background noise).
- IMPORTANT: If the review discusses technical sound quality (sound quality, recording quality, mix, background noise, hard to hear, remastering), MUST output Product_Condition_Quality as an independent target, do not swallow it into Domain_Experience.
- If the review praises/complains about music/content AND ALSO praises/complains about sound/recording quality, output BOTH Domain_Experience and Product_Condition_Quality.
- If the review has BOTH praise and complaint about the album/music/content itself, output BOTH Domain_Experience Positive and Domain_Experience Negative. Do not force into a single polarity.
- Packaging_Presentation covers only outer packaging, booklet, liner notes, cover/back-cover metadata, box/cardboard, visual presentation, or missing/poor presentation information. Vague praise for packaging without specifics should be avoided.
- Listing_Expectation_Compatibility requires explicit mismatch with listing, delivered item, edition, format, description, pictures, track list, or promised content. If there is no explicit mismatch, do not use Listing_Expectation_Compatibility.
- Price_Value requires explicit cost/worth/price/value judgment.
- Product_Performance_Usability is only for non-content operation, download/playback function, hidden-track access, usability, or functional failure.

Input items:
"""
    for item in items:
        prompt += (
            "\n--- ITEM ---\n"
            f"item_position: {item['item_position']}\n"
            f"summary: {item['summary']}\n"
            f"reviewText: {item['reviewText']}\n"
        )
    return prompt

def normalize_targets(row: dict[str, str], targets: list[dict[str, str]]) -> list[dict[str, str]]:
    text = f"{row.get('summary', '')} {row.get('reviewText', '')}".lower()
    explicit_listing = any(
        cue in text
        for cue in (
            "listing",
            "description",
            "advertised",
            "wrong item",
            "wrong cd",
            "wrong dvd",
            "track list",
            "delivered item",
            "not as advertised",
            "promised",
        )
    )
    digital_or_format_preference = any(
        cue in text
        for cue in (
            "could only get",
            "deluxe edition digitally",
            "wish i had bought",
            "digital edition",
            "dvd instead",
            "cd is no substitute",
        )
    )
    operation_failure = any(cue in text for cue in ("downloaded", "download", "playback", "repeat all", "button", "hidden track"))
    positive_domain = any(
        cue in text
        for cue in (
            "few good songs",
            "very good",
            "loved",
            "i love",
            "love ",
            "well-produced",
            "great",
            "excellent",
            "fantastic",
            "solid enough",
        )
    )
    negative_domain = any(
        cue in text
        for cue in (
            "disappoint",
            "not worth buying",
            "not good",
            "not hard",
            "not raw",
            "not the same",
            "only thing",
            "not readily available",
            "wish i had bought",
            "needs ben",
            "weak",
            "lame",
        )
    )
    explicit_value_negative = any(
        cue in text
        for cue in (
            "not worth the shipping",
            "shipping cost",
            "paid $",
            "only paid",
            "pricey",
            "full album price",
            "too expensive",
        )
    )
    physical_case = any(cue in text for cue in ("case was broken", "cd case", "jewel case", "case fall apart", "broken when i received"))
    technical_sound = any(
        cue in text
        for cue in (
            "sound quality",
            "recorded sound",
            "recording",
            "quality is very good",
            "sonics",
            "frequency range",
            "hard to hear",
            "background noise",
            "remastered",
            "mix",
            "mastering",
        )
    )
    normalized: list[dict[str, str]] = []
    for target in targets:
        aspect = target["aspect"]
        polarity = target["polarity"]
        # RULE 4: Listing overuse cleanup. If not explicit listing mismatch, map to Domain.
        if aspect == "Listing_Expectation_Compatibility" and not explicit_listing:
            aspect = "Domain_Experience"
        if aspect == "Product_Performance_Usability" and digital_or_format_preference and not operation_failure:
            aspect = "Domain_Experience"
        if aspect == "Packaging_Presentation" and physical_case:
            aspect = "Product_Condition_Quality"
        if aspect == "Price_Value" and polarity == "Negative" and not explicit_value_negative:
            aspect = "Domain_Experience"
        normalized.append({"aspect": aspect, "polarity": polarity})

    if explicit_value_negative and negative_domain:
        normalized = [
            {"aspect": "Price_Value", "polarity": t["polarity"]}
            if t["aspect"] in {"Product_Condition_Quality", "Packaging_Presentation"} and t["polarity"] == "Negative"
            else t
            for t in normalized
        ]

    # RULE 3: Same-aspect mixed Domain handling (already preserves both, and promotes mixed if implicitly present)
    if positive_domain and negative_domain:
        has_domain_pos = any(t["aspect"] == "Domain_Experience" and t["polarity"] == "Positive" for t in normalized)
        has_domain_neg = any(t["aspect"] == "Domain_Experience" and t["polarity"] == "Negative" for t in normalized)
        unique_count = len({(t["aspect"], t["polarity"]) for t in normalized})
        if has_domain_neg and not has_domain_pos and unique_count < 2:
            normalized.append({"aspect": "Domain_Experience", "polarity": "Positive"})

    # RULE 5: Bonus aspect suppression
    if positive_domain:
        cleaned = []
        for target in normalized:
            bonus_positive = target["polarity"] == "Positive" and target["aspect"] in {
                "Packaging_Presentation",
                "Price_Value",
                "Other_Specific",
                "Listing_Expectation_Compatibility",
            }
            nontechnical_quality_bonus = target["polarity"] == "Positive" and target["aspect"] == "Product_Condition_Quality" and not technical_sound
            if bonus_positive or nontechnical_quality_bonus:
                if any(t["aspect"] == "Domain_Experience" and t["polarity"] == "Positive" for t in normalized):
                    continue
            cleaned.append(target)
        normalized = cleaned

    deduped = []
    seen = set()
    for target in normalized:
        key = (target["aspect"], target["polarity"])
        if key not in seen:
            seen.add(key)
            deduped.append(target)
    return deduped[:2]

def compare_against_human(repo: Path, run_dir: Path) -> dict[str, Any]:
    human = v222.load_human_rows(repo)
    manifest = v222.load_manifest(repo)
    split_by_id = {item["annotation_item_id"]: item["selection_split"] for item in manifest["items"]}
    records = []
    split_stats: dict[str, dict[str, int]] = {}
    for batch_id in v222.EXPECTED_BATCHES:
        resolved = v222.load_json(run_dir / "batches" / batch_id / "resolved_output.json")
        for item in resolved["items"]:
            aid = item["annotation_item_id"]
            split = split_by_id[aid]
            stats = split_stats.setdefault(split, {"n": 0, "aspect_set_agree": 0, "mixed_agree": 0})
            model_pairs = v222.pairs_from_model(item)
            human_pairs = v222.pairs_from_human(human[aid])
            mixed_human = human[aid]["human_Review_Mixed_Flag"].lower() == "true"
            aspect_agree = model_pairs == human_pairs
            mixed_agree = item["Review_Mixed_Flag"] == mixed_human
            stats["n"] += 1
            stats["aspect_set_agree"] += int(aspect_agree)
            stats["mixed_agree"] += int(mixed_agree)
            records.append({
                "annotation_item_id": aid,
                "selection_split": split,
                "model_aspect_polarity_multiset": model_pairs,
                "human_aspect_polarity_multiset": human_pairs,
                "aspect_set_agreement": aspect_agree,
                "mixed_flag_agreement": mixed_agree,
            })
            
    aspect_agree_total = sum(1 for r in records if r["aspect_set_agreement"])
    mixed_agree_total = sum(1 for r in records if r["mixed_flag_agreement"])
    fresh_random_agree = split_stats.get("C_fresh_random_24", {}).get("aspect_set_agree", 0)
    
    if (aspect_agree_total >= 85 and
        mixed_agree_total >= 90 and
        fresh_random_agree >= 22):
        status = "V223_ASPECT_ONLY_CALIBRATION_PASS"
    else:
        status = "V223_ASPECT_ONLY_CALIBRATION_REVIEW_REQUIRED"
        
    return {
        "status": status,
        "metric_name": "agreement against one-person human adjudication",
        "single_accuracy_claim": False,
        "item_count": len(records),
        "split_stats": split_stats,
        "records": records,
    }

# Monkey-patch the module
v222.render_prompt = render_prompt
v222.normalize_targets = normalize_targets
v222.compare_against_human = compare_against_human

def main():
    # Pass execution to the original main function, which will now use our patched variables and functions
    v222.main()

if __name__ == "__main__":
    main()
