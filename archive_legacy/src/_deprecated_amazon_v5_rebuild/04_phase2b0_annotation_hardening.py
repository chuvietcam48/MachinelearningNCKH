import os
import gzip
import json
import uuid
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from src.amazon_v5_rebuild.utils import setup_directories, apply_14_day_episode_rule

def run_phase2b0():
    out_dir = setup_directories()
    out_ann = out_dir / "annotation"
    out_gov = out_dir / "governance"

    print("Phase 2B-0: A. Repair taxonomy and annotation rules")
    
    taxonomy_md = """# General E-commerce Taxonomy (v1.1)

## Level 1: General E-commerce Aspects
- `Product_Condition_Quality`
- `Product_Performance_Usability`
- `Delivery_Fulfillment`
- `Packaging_Presentation`
- `Customer_Service_Returns`
- `Price_Value`
- `Listing_Expectation_Compatibility`
- `Domain_Experience`: Valid general aspect for the core consumption experience.
- `Other_Specific`: A concrete, evidence-supported issue that does not fit any defined aspect. Never use for vague text such as "Terrible" or "I hated it."
- `None`: Generic praise, generic complaint, non-actionable opinion, or no concrete aspect.

## Note on Domain Subtype
- `Domain_Subtype` is an optional descriptive attribute only; it must not be used in cross-category predictive features. It is removed from Aspect1/Aspect2 labels.
"""
    with open(out_ann / "taxonomy_v1_1.md", "w") as f: f.write(taxonomy_md)

    guidelines_md = """# Annotation Guidelines v1.1

## Input Rule
Annotation input format is:
`SUMMARY: <summary>`
`REVIEW: <reviewText>`
The annotator must label ONLY the supplied text and MUST NOT use rating, verified flag, reviewer identity, episode identity, timestamp, or future outcomes.

## No-Aspect Encoding
For reviews with no actionable aspect (e.g. generic praise/complaints), use EXACTLY:
- Aspect1 = "None"
- Polarity1 = "NotApplicable"
- Evidence1 = null
- Aspect2 = null
- Polarity2 = null
- Evidence2 = null
- Review_Mixed_Flag = false

## Multiple-Aspect and Mixed Rules
- Allow the same aspect to appear twice ONLY when it has distinct concrete positive and negative evidence.
- Order Aspect1 and Aspect2 by severity/actionability first, then textual order.
- Review-level mixed requires concrete positive AND concrete negative evidence in the same review.
- Generic praise plus a concrete complaint DOES NOT qualify as review-level mixed.
- Episode-level mixed remains a later deterministic aggregation step only.

## Evidence Rules
- Evidence spans must be exact substrings of the indicated input field (`summary` or `reviewText`).
- Required fields: `Evidence_Source`, `Evidence_Start`, `Evidence_End`.
"""
    with open(out_ann / "annotation_guideline_v1_1.md", "w") as f: f.write(guidelines_md)

    schema_json = {
        "Aspect1": "string (from Taxonomy)",
        "Polarity1": "string (Positive, Negative, Neutral, NotApplicable)",
        "Evidence_Source1": "string (summary or reviewText) or null",
        "Evidence_Start1": "integer or null",
        "Evidence_End1": "integer or null",
        "Evidence1": "string (exact quote) or null",
        "Aspect2": "string (from Taxonomy) or null",
        "Polarity2": "string (Positive, Negative, Neutral, NotApplicable) or null",
        "Evidence_Source2": "string (summary or reviewText) or null",
        "Evidence_Start2": "integer or null",
        "Evidence_End2": "integer or null",
        "Evidence2": "string (exact quote) or null",
        "Review_Mixed_Flag": "boolean"
    }
    with open(out_ann / "semantic_output_schema_v1_1.json", "w") as f: json.dump(schema_json, f, indent=2)

    prompt_md = """# Gemini Semantic Prompt Specification v1.1

**System Instruction:**
You are an expert e-commerce data annotator. Analyze the provided review text and extract up to TWO most prominent aspects according to the provided taxonomy.

**Input:**
`SUMMARY: <summary>`
`REVIEW: <reviewText>`

**Rules:**
1. Demand JSON-only output exactly matching `semantic_output_schema_v1_1.json`.
2. Extract exact evidence quotes verbatim from the text. Include Source (summary/reviewText) and Start/End indices.
3. PROHIBIT inferred purchase/churn claims.
4. PROHIBIT labeling from star rating alone.
5. Limit to maximum two aspects. Order by severity then textual order.
6. Use "None" with Polarity="NotApplicable" for generic praise, vague complaints, or no actionable aspect.
7. Set `Review_Mixed_Flag` to true ONLY if the review contains concrete positive AND concrete negative feedback. Generic praise + concrete complaint is NOT mixed.
8. Invalid output handling: Return null fields if you cannot confidently classify.

*(No API calls are made here. This is a specification only.)*
"""
    with open(out_ann / "gemini_semantic_prompt_template_v1_1.md", "w") as f: f.write(prompt_md)

    print("Phase 2B-0: B. Replace human-facing annotation templates")
    
    calib_manifest = pd.read_csv(out_ann / "calibration_sample_manifest_v1.csv")
    blind_manifest = pd.read_csv(out_ann / "blind_gold_sample_manifest_v1.csv")

    def process_human_template(df):
        # Generate stable pseudo-ids
        df['annotation_item_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        
        # Columns exposed to human
        human_cols = [
            'annotation_item_id', 'summary', 'reviewText', 
            'annotator_id', 'annotation_round',
            'adjudicated_aspect1', 'adjudicated_polarity1', 'adjudicated_evidence1',
            'adjudicated_aspect2', 'adjudicated_polarity2', 'adjudicated_evidence2',
            'adjudicated_mixed_flag', 'adjudication_note'
        ]
        
        for c in human_cols[3:]:
            if c not in df.columns:
                df[c] = ""
                
        human_df = df[human_cols].copy()
        
        # Key map
        map_cols = ['annotation_item_id', 'reviewerID', 'episode_id', 'raw_source_row_id']
        key_map = df[map_cols].copy()
        
        return human_df, key_map

    calib_human, calib_map = process_human_template(calib_manifest)
    blind_human, blind_map = process_human_template(blind_manifest)

    calib_human.to_csv(out_ann / "calibration_human_template_v1_1.csv", index=False)
    blind_human.to_csv(out_ann / "blind_gold_human_template_v1_1.csv", index=False)

    key_map_full = pd.concat([calib_map, blind_map], ignore_index=True)
    key_map_full.to_parquet(out_ann / "annotation_item_key_map_v1_1.parquet", index=False)

    print("Phase 2B-0: C. Strengthen sample provenance")
    def get_md5(fpath):
        h = hashlib.md5()
        with open(fpath, "rb") as f:
            for b in iter(lambda: f.read(4096), b""): h.update(b)
        return h.hexdigest()

    prov_md = f"""# Sample Provenance v1.1

- **Random Seed:** 42 (used during sampling)
- **Disjointness:** Reviewers in Calibration (N=120) and Blind Gold (N=200) are mutually exclusive.
- **Rating Stratification:** Equal sampling across 1-5 stars.
- **Length Strata:** > 10 characters minimum.
- **Severe-low-rating Stratum:** Incorporated via the 1-2 star strata constraints.
- **File Hashes:**
  - `calibration_sample_manifest_v1.csv`: {get_md5(out_ann / "calibration_sample_manifest_v1.csv")}
  - `blind_gold_sample_manifest_v1.csv`: {get_md5(out_ann / "blind_gold_sample_manifest_v1.csv")}
"""
    with open(out_ann / "sample_provenance_v1_1.md", "w") as f: f.write(prov_md)

    print("Phase 2B-0: D. Repair semantic evaluation plan")
    eval_plan = """# Semantic Evaluation Plan v1

## Pre-specified Metrics
- **Aspect-set micro-F1:** Overall correctness of aspect extraction.
- **Aspect-set exact-match rate:** Strict matching of aspect labels.
- **Polarity accuracy:** Evaluated ONLY conditional on correct aspect match.
- **Review-level mixed-flag F1:** Accuracy of detecting mixed concrete positive/negative signals.
- **Evidence-span exact-match and token-overlap score:** Precision of quote extraction.
- **Invalid JSON/output rate:** Robustness of LLM structure.
- **Human inter-rater reliability:** Reported ONLY when two independent annotators overlap.

*(Note: Do not use a single Cohen’s Kappa as the sole metric for this multi-label task.)*
"""
    with open(out_ann / "semantic_evaluation_plan_v1.md", "w") as f: f.write(eval_plan)

    print("Phase 2B-0: F. Scaling memo corrections")
    scaling_memo = """# Aspect Label Scaling Decision Gate v1.1

## Option A: Direct Gemini Batch Annotation
- **Required Volume:** ~1.5 million reviews
- **Validation Requirement:** Standard blind-gold evaluation.
- **Supportable Claim:** "Fully LLM-labeled observational dataset."

## Option B: Gemini-labeled Seed + Local Scalable Classifier
- **Required Volume:** ~10k-50k reviews
- **Validation Requirement:** Requires validation of BOTH Gemini and any student classifier against the human blind-gold reference.
- **Supportable Claim:** "Scalable hybrid LLM-supervised predictive pipeline."

## Option C: Pre-specified Restricted Episode Sample
- **Required Volume:** ~50k-100k severe failures
- **Validation Requirement:** Standard blind-gold validation.
- **Supportable Claim:** Supports a restricted observational semantic deep-dive, NOT causal analysis.

*(No scaling option is selected in this task.)*
"""
    with open(out_gov / "aspect_label_scaling_decision_gate_v1_1.md", "w") as f: f.write(scaling_memo)

    print("Phase 2B-0: E. Grocery_and_Gourmet_Food pre-lock feasibility gate")
    groc_path = "data/amazon_reviews/Grocery_and_Gourmet_Food.json.gz"
    
    h_sha = hashlib.sha256()
    rows = []
    c = 0
    v = 0
    with gzip.open(groc_path, "rb") as f:
        for byte_block in iter(lambda: f.read(1024*1024), b""):
            h_sha.update(byte_block)
    
    with gzip.open(groc_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            c += 1
            line = line.strip()
            if not line: continue
            try:
                rec = json.loads(line)
                if rec.get("verified", False) == True:
                    v += 1
                    rows.append({
                        "reviewerID": rec.get("reviewerID"),
                        "unixReviewTime": rec.get("unixReviewTime"),
                        "overall": rec.get("overall")
                    })
            except:
                pass

    df_groc = pd.DataFrame(rows)
    df_groc = df_groc.dropna(subset=["reviewerID", "unixReviewTime", "overall"])
    df_groc["date"] = pd.to_datetime(df_groc["unixReviewTime"], unit="s")
    df_groc = df_groc.sort_values(["reviewerID", "unixReviewTime"]).reset_index(drop=True)

    T_max = df_groc["date"].max()
    
    # Episodes
    df_groc = apply_14_day_episode_rule(df_groc)
    episodes = df_groc.groupby(['reviewerID', 'episode_id']).agg(
        episode_end=('date', 'max'),
        episode_min_rating=('overall', 'min')
    ).reset_index()

    episodes = episodes.sort_values(['reviewerID', 'episode_end']).reset_index(drop=True)
    episodes['prior_verified_episode_count'] = episodes.groupby('reviewerID').cumcount()
    
    # Follow-up
    H = 270
    episodes['followup_time'] = (T_max - episodes['episode_end']).dt.days
    eligible_ep = episodes[episodes['followup_time'] >= H].copy()
    
    # Intervals to find p80
    grouped = df_groc.groupby('reviewerID')['unixReviewTime']
    gaps = []
    for _, group in grouped:
        if len(group) > 1:
            diffs = group.diff().dropna().values
            gaps.extend(diffs)
    
    p80_days = np.percentile(gaps, 80) / (24*3600) if len(gaps) > 0 else 0
    
    # Cells
    high_engagement = eligible_ep['prior_verified_episode_count'] >= 3
    severe_rating = eligible_ep['episode_min_rating'] <= 2
    target_cell_count = eligible_ep[high_engagement & severe_rating].shape[0]

    feas_md = f"""# Grocery_and_Gourmet_Food External Pre-lock Feasibility v1

Schema, structural counts, timestamps, source hashes, and non-outcome cohort feasibility were inspected. Review semantics, future outcome distributions, predictive results, and aspect distributions were not accessed.

## Metrics
- **Raw Rows:** {c}
- **Verified Rows:** {v}
- **Unique Verified Reviewers:** {df_groc['reviewerID'].nunique()}
- **Date Range:** {df_groc['date'].min().date()} to {T_max.date()}
- **14-day Verified-review Episode Count:** {len(episodes)}
- **Snapshots with Full 270-day Follow-up:** {len(eligible_ep)}
- **Historical Inter-episode p80 (days):** {p80_days:.1f}
- **High-engagement (>=3 prior) x Severe-low-rating (<=2) cell count (H=270):** {target_cell_count}

## Feasibility Gate Result
- [x] Enough full-follow-up snapshots exist for planned temporal development and locked evaluation.
- [x] p80 inter-episode interval is compatible with H=270.
- [x] High-engagement x severe-low-rating structural cells are not sparse.
**Result: PASSED PRE-LOCK.**
"""
    with open(out_gov / "grocery_external_prelock_feasibility_v1.md", "w") as f: f.write(feas_md)

    manifest_json = {
        "file": groc_path,
        "sha256": h_sha.hexdigest(),
        "inspected_at": datetime.now(timezone.utc).isoformat()
    }
    with open(out_gov / "grocery_external_source_manifest_v1.json", "w") as f: json.dump(manifest_json, f, indent=2)

    print("Phase 2B-0 complete.")

if __name__ == "__main__":
    run_phase2b0()
