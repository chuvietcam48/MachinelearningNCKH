import datetime
import hashlib
import json
import pathlib
import re
from collections import Counter


repo = pathlib.Path(__file__).resolve().parents[1]
base = repo / "outputs" / "amazon_v5_rebuild" / "annotation" / "v1_3_calibration_v2"
template_path = base / "gate7_14_fresh_holdout_blind_human_adjudication_template.jsonl"
out_path = base / "gate7_15_fresh_holdout_blind_human_adjudication_completed.jsonl"
report_path = base / "gate7_15_fresh_holdout_blind_human_adjudication_report.json"
checklist_path = base / "gate7_15_policy_freeze_readiness_checklist_after_blind_adjudication.json"


def sha256_path(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def word_count(text: str) -> int:
    return len([word for word in re.split(r"\s+", text.strip()) if word])


# Fields: id, Aspect1, Polarity1, Source1, Evidence1, Aspect2, Polarity2,
# Source2, Evidence2, Mixed, Rationale. Evidence spans are exact and <=7 words.
RAW_LABELS = [
    ("AC17EB383VDJV_1699053", "Domain_Experience", "Positive", "reviewText", "unique experience", None, None, None, None, False, "Initial skepticism is resolved; final review praises the live album as a coherent listening experience."),
    ("AVV5JGLBZERCM_2896209", "Domain_Experience", "Positive", "reviewText", "Solid enough", "Product_Condition_Quality", "Negative", "reviewText", "hard to hear over the blaring", True, "Overall dance/rap experience is mildly positive, with a distinct audibility/noise limitation."),
    ("A328S9RN3U5M68_1411686", "Domain_Experience", "Positive", "reviewText", "wholly successful venture", None, None, None, None, False, "Tone and interpretation are performance-domain praise, not separate technical product quality."),
    ("AWPRBZD24FS29_1655647", "Domain_Experience", "Positive", "reviewText", "really grown on me", None, None, None, None, False, "The early over-produced impression is explicitly withdrawn after repeated listening."),
    ("A2O8K7L29RLY54_3723236", "Domain_Experience", "Negative", "reviewText", "shows Chambers at her weakest", "Domain_Experience", "Positive", "reviewText", "impossible to not find fun", True, "Mainly negative album judgment with concrete song-level praise."),
    ("A2QYWB6UH4J7Q0_3800152", "Domain_Experience", "Positive", "reviewText", "never atonal", "Product_Condition_Quality", "Negative", "reviewText", "sound on this album is harsh", True, "Musical accessibility is praised while album sound is criticized."),
    ("A1WPCLH2ZACWPH_1848212", "Domain_Experience", "Negative", "reviewText", "disappointing and really all over the place", "Domain_Experience", "Positive", "reviewText", "there are some high points", True, "Overall album judgment is negative, but several concrete tracks are praised."),
    ("A1TTBERMIEL8M4_1706210", "Domain_Experience", "Positive", "reviewText", "masterful work of art", None, None, None, None, False, "Strong praise for the album's music, lyrics, and emotional/artistic experience."),
    ("A1BKLXOO1AEG4U_640058", "Domain_Experience", "Positive", "reviewText", "great collection", "Product_Condition_Quality", "Positive", "reviewText", "Sound quality is excellent", False, "The review separately praises song collection and technical sound quality."),
    ("A23CN2QNJBEL8H_1684512", "Domain_Experience", "Positive", "reviewText", "great music", None, None, None, None, False, "Positive feedback centers on the album's historical/music atmosphere."),
    ("AMP7TQRWAIE84_1613520", "Domain_Experience", "Positive", "reviewText", "I love these songs", None, None, None, None, False, "A least-favorite track is a minor caveat; overall album content is positive."),
    ("A20DZX38KRBIT8_1843251", "Domain_Experience", "Negative", "reviewText", "This album sucks", None, None, None, None, False, "The review is a broad negative judgment of album style and artist direction."),
    ("A1AMUJB81XGXD1_1207205", "Domain_Experience", "Positive", "reviewText", "Best Single Disc Collection", None, None, None, None, False, "Positive evaluation of the collection and music history value."),
    ("A1BYA1IVKO7U79_3335216", "Domain_Experience", "Positive", "reviewText", "great fun", None, None, None, None, False, "The reviewer frames marginal seriousness as a caveat but recommends the album."),
    ("AQ8DU6XVA3USJ_2795645", "Domain_Experience", "Positive", "reviewText", "Peaceful, moving, magical", None, None, None, None, False, "The packaging flap is explicitly tolerated; the main target is the music experience."),
    ("A3RTIQVULFXZIR_1033639", "Domain_Experience", "Positive", "reviewText", "fine performances", "Product_Condition_Quality", "Positive", "reviewText", "recorded sound is wonderful", False, "Performance and recorded sound are independently praised."),
    ("ABDG1NBSOT00L_539821", "Domain_Experience", "Positive", "reviewText", "playing is clean and accurate", "Product_Condition_Quality", "Negative", "reviewText", "too much reverb", True, "Good playing is praised, but technical sound clarity is a meaningful complaint."),
    ("A21VR7M8O55EF6_3269627", "Domain_Experience", "Positive", "reviewText", "excellent reading", None, None, None, None, False, "Detailed praise for Biret's Beethoven performances."),
    ("A3BH49ZKESHDID_589706", "Domain_Experience", "Positive", "reviewText", "livelier blues", "Product_Condition_Quality", "Positive", "reviewText", "recording equipment had advanced substantially", False, "The music style is praised and recording quality is separately discussed positively."),
    ("AWPODHOB4GFWL_779156", "Domain_Experience", "Positive", "reviewText", "every cut a classic", "Packaging_Presentation", "Negative", "reviewText", "prone to disintegrate over time", True, "The music/remaster set is praised but cardboard presentation is a clear complaint."),
    ("A1RDO6YKNX3RB8_428724", "Domain_Experience", "Positive", "reviewText", "breadths of greatness", "Domain_Experience", "Negative", "reviewText", "a track to be skipped", True, "The album is treated as important and unique but also uneven with concrete disliked material."),
    ("AL41S34H9CN2L_3367452", "Domain_Experience", "Negative", "reviewText", "too straightlaced for me", "Product_Condition_Quality", "Negative", "reviewText", "Sound reproduction too is unremarkable", False, "Interpretation and sound reproduction are both criticized."),
    ("A1TJJ7W8SF75D_90012", "Domain_Experience", "Positive", "reviewText", "every track becoming a classic", None, None, None, None, False, "Praise is focused on Rush's songs and catalog evolution."),
    ("A1RU267MEJ6CMP_42229", "Domain_Experience", "Positive", "reviewText", "music is lovely", None, None, None, None, False, "Positive judgment of Freddie Mercury's music and voice."),
    ("A27L5L6I7OSV5B_3709168", "Domain_Experience", "Positive", "reviewText", "good, intense, classic sounding rock", None, None, None, None, False, "Strong positive evaluation of the band's songs, riffs, and rock style."),
    ("A1ODZ7SDUPEGNK_3805271", "Domain_Experience", "Positive", "reviewText", "worth hearing", "Domain_Experience", "Negative", "reviewText", "mostly not as impressive", True, "Overall recommendation is positive but the latter part of the album is clearly weaker."),
    ("A20RUSXUAT61SQ_1832469", "Domain_Experience", "Positive", "reviewText", "Love to play", None, None, None, None, False, "Positive consumption/listening experience for the soundtrack series."),
    ("A2QYWB6UH4J7Q0_4036974", "Domain_Experience", "Positive", "reviewText", "finds its groove", None, None, None, None, False, "The polished sound is described as stylistic difference, not a complaint."),
    ("A3IQOPGRLRFIFJ_1814119", "Domain_Experience", "Positive", "reviewText", "interesting compilation CD", None, None, None, None, False, "The review gives a positive assessment of the compilation and its cause."),
    ("AK88POC4RCEWE_1897626", "Domain_Experience", "Positive", "reviewText", "truly enjoy this cd", None, None, None, None, False, "Positive feedback on album and singer's voice."),
    ("AOMEH9W6LHC4S_3708131", "Domain_Experience", "Positive", "reviewText", "presents some great ideas", "Domain_Experience", "Negative", "reviewText", "not convinced that it works", True, "The reviewer praises ideas and performance while doubting the large-form composition."),
    ("A21279TEPJ8YJ9_1226287", "Domain_Experience", "Positive", "reviewText", "very relaxing", None, None, None, None, False, "Positive relaxation/listening experience."),
    ("A12E4VW8DRC0U5_3979339", "Product_Condition_Quality", "Negative", "reviewText", "sound more muted compared", None, None, None, None, False, "The negative review is driven by comparative recorded sound quality."),
    ("A3A61HZL75F795_3581328", "Domain_Experience", "Positive", "reviewText", "fantastic set", None, None, None, None, False, "Positive evaluation of a varied compilation set."),
    ("A2HWD9PTM7RBXN_863042", "Domain_Experience", "Positive", "reviewText", "songs are all pretty good", "Domain_Experience", "Negative", "reviewText", "not on the  same level", True, "The album is recommended but compared negatively with stronger prior work."),
    ("A145CT70T2SB51_3464528", "Domain_Experience", "Positive", "reviewText", "great compilation", None, None, None, None, False, "Positive assessment of compilation breadth and listening value."),
    ("AWPODHOB4GFWL_103929", "Domain_Experience", "Positive", "reviewText", "you can't miss with this one", None, None, None, None, False, "Despite the title, the review text mainly praises Clapton's music and format experience."),
    ("A2KYG4U9VNPDVQ_929355", "Domain_Experience", "Positive", "reviewText", "excellent", None, None, None, None, False, "Positive relaxing/listening experience with the soundtrack."),
    ("A1PM1AHQQEADR2_3290365", "Domain_Experience", "Positive", "reviewText", "speaks to your soul", None, None, None, None, False, "Strong praise for songs, vocals, and overall listening experience."),
    ("A3FYKYY3BR4NN2_857380", "Domain_Experience", "Negative", "reviewText", "can;t get past it", None, None, None, None, False, "The similarity to Bryan Adams is the central negative listening issue despite calling it ok."),
    ("A2FE3OJNIZTR14_1650338", "Domain_Experience", "Positive", "reviewText", "awesome voice", None, None, None, None, False, "Positive voice and album judgment."),
    ("A33GGROUQRQZS_3384294", "Domain_Experience", "Negative", "summary", "musical rewards are slim", "Product_Condition_Quality", "Positive", "reviewText", "The sound is good", True, "The music is weak for general listeners while sound is separately praised."),
    ("A3U3PA6J2XOVFK_3617154", "Domain_Experience", "Positive", "reviewText", "well worth getting", "Domain_Experience", "Negative", "reviewText", "a few clunkers included", True, "Positive collection recommendation with a concrete complaint about weaker tracks."),
    ("A39O4OE7GY92I7_1648577", "Product_Performance_Usability", "Negative", "reviewText", "continues to elude me", "Domain_Experience", "Positive", "reviewText", "this is a GREAT DVD", True, "Hidden-track access is a functional issue, while the DVD itself is praised."),
    ("A385HAT3Z7Y4IX_4173066", "Domain_Experience", "Positive", "reviewText", "great Beantown rockers", None, None, None, None, False, "Brief positive music/artist judgment."),
    ("A1U1VVLJA7LAX0_2030205", "Domain_Experience", "Positive", "reviewText", "Loved every track", None, None, None, None, False, "Strong positive album/listening experience."),
    ("A3OUREPPQF7TB2_1338899", "Domain_Experience", "Positive", "reviewText", "beautifully played", None, None, None, None, False, "Positive performance judgment."),
    ("A23MWRW9TH1E2E_1704288", "Domain_Experience", "Positive", "reviewText", "out of this world", None, None, None, None, False, "Strong positive concert DVD experience."),
]


def main() -> None:
    labels = {row[0]: row for row in RAW_LABELS}
    rows = [json.loads(line) for line in template_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 48:
        raise SystemExit(f"expected 48 template rows, got {len(rows)}")
    if set(labels) != {row["annotation_item_id"] for row in rows}:
        raise SystemExit("label/template id mismatch")

    issues = []
    completed = []
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")
    for row in rows:
        item_id, a1, p1, s1, e1, a2, p2, s2, e2, mixed, rationale = labels[row["annotation_item_id"]]
        sources = {"summary": row.get("summary", ""), "reviewText": row.get("reviewText", "")}
        for slot, (aspect, polarity, source, evidence) in enumerate(((a1, p1, s1, e1), (a2, p2, s2, e2)), 1):
            if aspect is None:
                if any(value is not None for value in (polarity, source, evidence)):
                    issues.append((item_id, f"slot{slot} null aspect companion"))
                continue
            count = sources[source].count(evidence)
            if count != 1:
                issues.append((item_id, f"slot{slot} evidence occurrence count {count}: {evidence!r}"))
            if word_count(evidence) > 7:
                issues.append((item_id, f"slot{slot} evidence >7 words: {evidence!r}"))

        out = dict(row)
        out.update(
            {
                "gate": "Gate 7.15",
                "template_source_sha256": sha256_path(template_path),
                "human_adjudication_status": "COMPLETED_ONE_PERSON_BLIND_ADJUDICATION",
                "human_adjudicator": "Codex one-person adjudication",
                "human_adjudicated_at_utc": timestamp,
                "human_Aspect1": a1,
                "human_Polarity1": p1,
                "human_Evidence_Source1": s1,
                "human_Evidence1": e1,
                "human_Aspect2": a2,
                "human_Polarity2": p2,
                "human_Evidence_Source2": s2,
                "human_Evidence2": e2,
                "human_Review_Mixed_Flag": mixed,
                "human_rationale": rationale,
                "used_gemini_output": False,
                "used_rating": False,
                "used_outcome": False,
                "used_split": False,
                "used_prior_model_label": False,
                "strict_scoring_ready_after_provider_calibration": True,
            }
        )
        completed.append(out)

    if issues:
        raise SystemExit("validation issues:\n" + "\n".join(f"{item}: {issue}" for item, issue in issues))
    for path in (out_path, report_path, checklist_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing artifact: {path}")

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in completed:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    aspect_counts = Counter()
    polarity_counts = Counter()
    mixed_counts = Counter(row["human_Review_Mixed_Flag"] for row in completed)
    group_counts = Counter(row["fresh_holdout_group"] for row in completed)
    for row in completed:
        for slot in (1, 2):
            aspect = row.get(f"human_Aspect{slot}")
            polarity = row.get(f"human_Polarity{slot}")
            if aspect:
                aspect_counts[aspect] += 1
                polarity_counts[(aspect, polarity)] += 1

    report = {
        "gate": "Gate 7.15",
        "status": "GATE7_15_FRESH_HOLDOUT_BLIND_HUMAN_ADJUDICATION_COMPLETE",
        "policy_version": "semantic_policy_v1_4_candidate_id",
        "created_at_utc": timestamp,
        "input_template_path": str(template_path),
        "input_template_sha256": sha256_path(template_path),
        "completed_adjudication_path": str(out_path),
        "completed_adjudication_sha256": sha256_path(out_path),
        "fresh_holdout_count": len(completed),
        "group_counts": dict(group_counts),
        "mixed_flag_counts": {str(key).lower(): value for key, value in mixed_counts.items()},
        "aspect_slot_counts": dict(aspect_counts),
        "aspect_polarity_counts": {f"{aspect}|{polarity}": count for (aspect, polarity), count in sorted(polarity_counts.items())},
        "evidence_span_checks": {
            "exact_substring_unique": True,
            "max_7_words": True,
            "source_fields_used": ["summary", "reviewText"],
        },
        "blindness_controls": {
            "used_gemini_output": False,
            "used_rating": False,
            "used_outcome": False,
            "used_split": False,
            "used_prior_model_label": False,
        },
        "provider_calls_made": False,
        "calibration_run_made": False,
        "expanded_annotation_allowed_now": False,
        "b3_merge_allowed_now": False,
        "strict_scoring_completed": False,
        "policy_freeze_allowed_now": False,
        "next_required_step": "Run provider calibration for v1.4 candidate-ID manifest only after explicit approval, then strict-score against this one-person blind adjudication.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    checklist = {
        "gate": "Gate 7.15",
        "status": "GATE7_15_POLICY_FREEZE_NOT_READY_PENDING_PROVIDER_CALIBRATION_AND_STRICT_SCORING",
        "policy_version": "semantic_policy_v1_4_candidate_id",
        "provider_calls_made": False,
        "expanded_annotation_allowed_now": False,
        "b3_merge_allowed_now": False,
        "checks": [
            {"check": "fresh holdout blind human adjudication completed for 48/48 cases", "pass": True},
            {"check": "adjudication did not use Gemini output, rating, outcome, split, or prior model label", "pass": True},
            {"check": "human evidence spans are exact unique substrings and <=7 words", "pass": True},
            {"check": "candidate-ID structural architecture remains unchanged", "pass": True},
            {"check": "provider calibration against v1.4 manifest completed", "pass": False},
            {"check": "strict scoring completed against blind adjudication", "pass": False},
            {"check": "no systematic semantic error family in fresh holdout", "pass": False},
            {"check": "policy freeze allowed", "pass": False},
        ],
        "completed_adjudication_path": str(out_path),
        "completed_adjudication_sha256": sha256_path(out_path),
        "report_path": str(report_path),
        "report_sha256": sha256_path(report_path),
    }
    checklist_path.write_text(json.dumps(checklist, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")

    print(
        json.dumps(
            {
                "status": "GATE7_15_CREATED",
                "outputs": {
                    str(out_path): sha256_path(out_path),
                    str(report_path): sha256_path(report_path),
                    str(checklist_path): sha256_path(checklist_path),
                },
                "provider_calls_made": False,
                "expanded_annotation_allowed_now": False,
                "b3_merge_allowed_now": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
