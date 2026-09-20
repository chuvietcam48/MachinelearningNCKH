                                                                                # Final Reproducibility Artifact

This document serves as the absolute source of truth for the final evaluation metrics used in the paper. It guarantees that all numbers reported are strictly derived from a single, consecutive, and frozen pipeline run.

## Environment Details
- **Commit**: `696abaf737b175e731e16ea8dd2f0445fa0195b2`
- **Dataset Hash**: Pre-computed artifacts mapped in `dataset_registry.py` (Amazon Cohort, `verified_episode_snapshots_h270_v1.parquet`)
- **Python Version**: 3.12
- **Seed**: 42 (Used universally across data split and Bootstrap C-index)

## Execution Command
To reproduce these results exactly, run:
```bash
python src/pipeline/run_pipeline.py
```
*(This triggers scripts 01 through 05 sequentially).*

## Output Artifacts & Paper Mapping
All numerical claims in the manuscript MUST be cited from the following files located in `outputs/pipeline_freeze/results/`:

- **Table 1 (Predictive Discrimination / C-index)**
  - Source: `model_metrics.json`
  - *Key Finding*: Model C Semantic Cohort C-index = 0.7295

- **Table 2 (Hazard Ratios / Risk Attribution)**
  - Source: `hazard_ratios_semantic.csv`
  - *Key Finding*: `episode_duration` HR = 1.420

- **Table 3 (Information Gain / LRT)**
  - Source: `information_gain.csv`
  - *Key Finding*: AIC Improvement = 98.38, p-value < 0.001

- **Table 4 (Policy Routing / EVI)**
  - Source: `policy/policy_routing_matrix.csv`
  - *Key Finding*: Simulated ROI and customer routing distributions.

**Status: FROZEN.** No further algorithmic changes are permitted.

## Limitation & Execution Reality (Kill-Switch)
- **Sampling Target**: The original `gate7_6` design aimed for `10,000` episodes (`PRIMARY_SPLIT_TARGETS = {"train": 6000, "validation": 2000, "development_evaluation": 2000}`).
- **Budget/Quota Constraint**: The Governance Contract (`13_gate7_8_live_executor_governance_design.py`) explicitly set budget and API quota limits (`max_primary_requests: 7745`). 
- **Actual LLM Output**: During live execution, the pipeline hit the kill-switch (due to cost ceiling or quota limit), resulting in only **2,067 LLM-annotated reviews** successfully generated (`mass_results_v224_qa.jsonl`).
- **Final Survival**: After passing these 2,067 raw responses through the strict pipeline filters (dropping NaNs, `episode_duration >= 0`), only **1,091 episodes** survived to form the final semantic cohort used in the results. This explains why the final N is 1,091 and not the targeted 10,000.
