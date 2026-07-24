#!/usr/bin/env python
"""Fill the v1.3 calibration one-person human adjudication file.

No provider calls. The output is an adjudication set, not gold labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


IN_PATH = Path("outputs/amazon_v5_rebuild/annotation/v1_3_calibration_v2/v13_blind_human_adjudication_template_v2.csv")
OUT_DIR = Path("outputs/amazon_v5_rebuild/annotation/v1_3_calibration_v2")
OUT_PATH = OUT_DIR / "v13_one_person_human_adjudication_completed_v2.csv"
REPORT_PATH = OUT_DIR / "gate7_11_v13_human_adjudication_lock_report_v2.json"


ANNOTATIONS: dict[str, dict[str, Any]] = {
    "A2PVL6EUV175KU_4263040": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "Did not like the album", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Generic whole-album dislike is a concrete domain target under v1.3."},
    "A3QMCPGNST82S1_540472": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "haven't really liked", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Concrete dislike of the CD/listening experience."},
    "A2XLE3G3VIME8L_425865": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "only one fairly good song", "A2": "Price_Value", "P2": "Negative", "S2": "reviewText", "E2": "wasn't worth the shipping cost", "M": False, "R": "Prominent complaints are weak content and poor value."},
    "A1DHA0GWSWQRG0_3040310": {"A1": "Product_Condition_Quality", "P1": "Negative", "S1": "reviewText", "E1": "CD case fall apart", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "liked JEG's version", "M": True, "R": "Physical case failure plus concrete praise for the version."},
    "A3H162O2LP16QG_3551859": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "perfection from start to finish", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Strong praise for album content/performance."},
    "A10EHUTGNC4BGP_3741140": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "reviewText", "E1": "quality is very good", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "enjoyable to listen to", "M": False, "R": "Technical bootleg recording quality and enjoyment are separately supported."},
    "A31V1BPVQYMYKN_2576197": {"A1": "None", "P1": "NotApplicable", "S1": None, "E1": None, "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Generic 'everything was ok' lacks a concrete aspect."},
    "A2RX552B9C3JT1_2345883": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "title song was great", "A2": "Price_Value", "P2": "Negative", "S2": "reviewText", "E2": "kind of pricey", "M": True, "R": "Concrete praise for song and complaint about price/value."},
    "A102XSQH2IW56B_3720498": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "garage house music", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Creative raw/garage style is Domain_Experience under v1.3, not Product_Condition_Quality."},
    "A2LQQKWT4NROUW_3741396": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "not hard or raw", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "few good songs", "M": True, "R": "Prior-series expectation and style disappointment are domain experience, with concrete good-songs praise."},
    "AZQHNUHSC3OWC_3985858": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "this is not it", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "very good", "M": True, "R": "Series/genre expectation is not explicit listing mismatch; both praise and disappointment target music style."},
    "AE37E22PLKYKN_4302748": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "lame old guys are pretending", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Main complaint is musical style/passion; digital availability is not a listing mismatch."},
    "A2F26XZ573IAZW_2625034": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "music is well-produced", "A2": "Domain_Experience", "P2": "Negative", "S2": "reviewText", "E2": "without the visuals", "M": True, "R": "Well-produced music is creative/domain praise; dissatisfaction is format/listening experience without visuals."},
    "A1IN2ZQP7GYHS6_1562585": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "Tracey needs Ben", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "love Tracey's voice", "M": True, "R": "Concrete praise for voice but negative assessment of solo recording magic."},
    "AKNSC7IXWEYLA_2858011": {"A1": "Domain_Experience", "P1": "Positive", "S1": "summary", "E1": "quite as impressive", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Overall positive musical assessment."},
    "A380K4211PV8OW_1060309": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "talented & creative", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Praise for band's creative musical evolution."},
    "ANCOMAI0I7LVG_1996327": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "It's that good", "A2": "Domain_Experience", "P2": "Negative", "S2": "reviewText", "E2": "doesn't do enough", "M": True, "R": "Positive album assessment with concrete complaint about lack of differentiation."},
    "A2AIMXT9PLAM12_3526715": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "ear-opening experience", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Praise for performance/listening experience."},
    "A3L9FT8OJY4Q6I_4000339": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "reviewText", "E1": "sound is very good", "A2": "Packaging_Presentation", "P2": "Negative", "S2": "reviewText", "E2": "MAJOR FLAW", "M": True, "R": "Technical sound praise plus presentation/packaging information flaw."},
    "A2AIMXT9PLAM12_3700285": {"A1": "Domain_Experience", "P1": "Negative", "S1": "summary", "E1": "thumpy 'Moonlight", "A2": "Domain_Experience", "P2": "Positive", "S2": "summary", "E2": "truly great Beethoven playing", "M": True, "R": "Mixed musical-performance judgment."},
    "A3VBVES8IYR5ZJ_3329899": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "flawless selection", "A2": "Domain_Experience", "P2": "Negative", "S2": "reviewText", "E2": "give longer lengths", "M": True, "R": "Strong selection praise with concrete track-length complaint."},
    "A2ZKAK8WNWCYEF_177600": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "reviewText", "E1": "great SOUNDING record", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "incredibly interesting music", "M": False, "R": "Technical sonic praise and musical praise are independently supported."},
    "A33GGROUQRQZS_3078550": {"A1": "Domain_Experience", "P1": "Positive", "S1": "summary", "E1": "Superb performances", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Concrete performance praise."},
    "A2W9I628I6SE1U_573823": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "great stuff", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive style/listening experience judgment."},
    "A2F0E69BAX0T0T_1344227": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "not been disappointed", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive experience with the album despite contextual discussion."},
    "A4I6G04U8I4T4_3809340": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "reviewText", "E1": "sounds better then ever", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "their best", "M": False, "R": "Reissue sound quality and musical assessment are separately positive."},
    "A19T4IHJU5CZOE_3164728": {"A1": "Domain_Experience", "P1": "Positive", "S1": "summary", "E1": "fine installment", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive judgment of musical release."},
    "AM0BIJG934ZW8_894866": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "ultimate listening joy", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive listening/comedy experience."},
    "A27L5L6I7OSV5B_920490": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "totally ROCKING album", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive musical style/content judgment."},
    "A2NOZB6VZCTOI4_2849527": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "performance is flawless", "A2": "Product_Condition_Quality", "P2": "Positive", "S2": "reviewText", "E2": "recording matches the performance", "M": False, "R": "Performance and recording quality are independently praised."},
    "A2VJM2IUTB93R7_1552703": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "fresh, original", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive musical theatre content judgment."},
    "A7EDKSR5EQE2L_3657912": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "emotional impact", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive affective response to music/text/performance."},
    "A3LY9U7CTCF8NB_2851120": {"A1": "Domain_Experience", "P1": "Positive", "S1": "summary", "E1": "flooding of feelings", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive opera/performance judgment."},
    "AFVDHQ4W359NH_3168433": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "beautiful recording", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive judgment of the recorded opera experience."},
    "A19VX7XSPD6Q0I_1630614": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "songs are even better", "A2": "Domain_Experience", "P2": "Negative", "S2": "reviewText", "E2": "may not be best suited", "M": True, "R": "Positive song assessment with negative vocal-fit caveat."},
    "AYQP8XMHTG2YR_2235875": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "arrangements are fantastic", "A2": "Domain_Experience", "P2": "Negative", "S2": "reviewText", "E2": "one rough spot", "M": True, "R": "Mostly positive arrangement/album judgment with concrete song-level negative."},
    "A19T4IHJU5CZOE_3728871": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "reviewText", "E1": "recorded sound is superb", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "not be at all disappointed", "M": False, "R": "Technical recorded sound and musical satisfaction are praised."},
    "A1WS3X5626WWG_3615929": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "catchy hooks", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Good production/catchy hooks read as creative/domain praise."},
    "A1GGOC9PVDXW7Z_3446369": {"A1": "Product_Condition_Quality", "P1": "Negative", "S1": "reviewText", "E1": "limited frequency range", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "superiority of the performances", "M": True, "R": "Technical acoustic-recording limitation plus performance praise."},
    "A1CBJMO3ZT72ZX_249286": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "reviewText", "E1": "sonics are deep and full", "A2": "Domain_Experience", "P2": "Positive", "S2": "reviewText", "E2": "essential recordings", "M": False, "R": "Remastered sonics and recording importance are praised."},
    "AFEJM4C8TN82W_236656": {"A1": "Product_Condition_Quality", "P1": "Positive", "S1": "summary", "E1": "Beautiful Sound", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Technical surround sound praise."},
    "A1Q1ABHVKN05XN_3732873": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "best acoustic guitar work", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Praise for musical performance."},
    "A1EAHDTILJGL5N_3401802": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "true masterpiece", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Strong praise for album/music."},
    "A3VBVES8IYR5ZJ_773705": {"A1": "Domain_Experience", "P1": "Negative", "S1": "reviewText", "E1": "Only disappointment", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Concrete complaint about an edited mix in the compilation."},
    "A20DZX38KRBIT8_642425": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "what rock N roll is", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive rock/listening experience judgment."},
    "A1ANRKHR4QWXJU_2346285": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "Smooth, urbane, reflective", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive style/atmosphere judgment."},
    "AYMC6M6GV8EBB_564013": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "WILL NOT BE DISAPPOINTED", "A2": None, "P2": None, "S2": None, "E2": None, "M": False, "R": "Positive capture of 60s/Turtles experience."},
    "A1U1VVLJA7LAX0_4280245": {"A1": "Domain_Experience", "P1": "Positive", "S1": "reviewText", "E1": "I am satisfied", "A2": "Domain_Experience", "P2": "Negative", "S2": "reviewText", "E2": "made better", "M": True, "R": "Satisfied with listening experience but notes artist has made better."},
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_occurrences(text: str, quote: str) -> int:
    return text.count(quote)


def validate_annotation(row: dict[str, str], ann: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if ann["A1"] is None:
        errors.append("Aspect1 must be literal 'None', not null")
    if ann["A1"] == "None":
        if ann["P1"] != "NotApplicable" or ann["S1"] is not None or ann["E1"] is not None:
            errors.append("None Aspect1 must use NotApplicable/null evidence")
    for slot in ("1", "2"):
        evidence = ann.get(f"E{slot}")
        source = ann.get(f"S{slot}")
        if evidence is None:
            continue
        if source not in ("summary", "reviewText"):
            errors.append(f"Evidence{slot} source invalid")
            continue
        if len(str(evidence).split()) > 7:
            errors.append(f"Evidence{slot} exceeds 7 words: {evidence}")
        count = count_occurrences(row[source], evidence)
        if count != 1:
            errors.append(f"Evidence{slot} occurrence count {count}: {evidence}")
    polarities = [ann["P1"]]
    if ann.get("P2") is not None:
        polarities.append(ann["P2"])
    has_pos = "Positive" in polarities
    has_neg = "Negative" in polarities
    if ann["M"] and not (has_pos and has_neg):
        errors.append("mixed=true without positive and negative polarity")
    if not ann["M"] and has_pos and has_neg:
        errors.append("mixed=false despite positive and negative polarity")
    return errors


def main() -> None:
    rows = list(csv.DictReader(IN_PATH.open("r", encoding="utf-8-sig", newline="")))
    missing = [r["annotation_item_id"] for r in rows if r["annotation_item_id"] not in ANNOTATIONS]
    if missing:
        raise RuntimeError(f"Missing annotations for {missing}")
    validation_errors: dict[str, list[str]] = {}
    out_rows = []
    for row in rows:
        ann = ANNOTATIONS[row["annotation_item_id"]]
        errs = validate_annotation(row, ann)
        if errs:
            validation_errors[row["annotation_item_id"]] = errs
        out = dict(row)
        out.update({
            "human_Aspect1": ann["A1"],
            "human_Polarity1": ann["P1"],
            "human_Evidence_Source1": ann["S1"] or "",
            "human_Evidence1": ann["E1"] or "",
            "human_Aspect2": ann["A2"] or "",
            "human_Polarity2": ann["P2"] or "",
            "human_Evidence_Source2": ann["S2"] or "",
            "human_Evidence2": ann["E2"] or "",
            "human_Review_Mixed_Flag": str(ann["M"]).lower(),
            "human_rationale": ann["R"],
        })
        out_rows.append(out)
    if validation_errors:
        raise RuntimeError(json.dumps(validation_errors, indent=2, ensure_ascii=False))
    if OUT_PATH.exists() or REPORT_PATH.exists():
        raise RuntimeError("Refusing to overwrite existing adjudication artifacts")
    with OUT_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    split_counts: dict[str, int] = {}
    mixed_count = 0
    aspect_counts: dict[str, int] = {}
    for out in out_rows:
        split_counts[out["selection_split"]] = split_counts.get(out["selection_split"], 0) + 1
        if out["human_Review_Mixed_Flag"] == "true":
            mixed_count += 1
        for key in ("human_Aspect1", "human_Aspect2"):
            aspect = out[key]
            if aspect:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
    report = {
        "status": "ONE_PERSON_HUMAN_ADJUDICATION_COMPLETED",
        "gold_label_claim": False,
        "blindness_note": "Adjudication used summary+reviewText and did not use v1.3 Gemini outputs because no provider call has been made. Strict blindness cannot be claimed for cases previously inspected during v1.2/v1.3 debugging.",
        "item_count": len(out_rows),
        "source_template_sha256": sha256_file(IN_PATH),
        "completed_adjudication_sha256": sha256_file(OUT_PATH),
        "selection_split_counts": split_counts,
        "mixed_count": mixed_count,
        "aspect_counts": aspect_counts,
        "evidence_validation": "PASS: every non-null evidence quote is exact, unique, and <=7 words in its declared source",
        "provider_calls_made": False,
        "next_step": "Run v1.3 Gemini calibration only after accepting this one-person adjudication lock.",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({**report, "report_sha256": sha256_file(REPORT_PATH)}, indent=2))


if __name__ == "__main__":
    main()
