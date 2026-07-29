# E1, E2, E3 Protocol: Universal Prediction Phase

## 1. Definition
These experiments test the predictive limits of the new Universal Framework on the Amazon dataset. They intentionally bypass the Decision Engine to isolate prediction capabilities.

## 2. Input
- E1: `master_test_full.parquet` (Behavioral columns only)
- E2: `master_test_semantic.parquet` (Behavioral + VADER columns)
- E3: `master_test_semantic.parquet` (Behavioral + LLM Semantic Trajectory columns)

## 3. Pipeline Constraints
- **Survival Engine**: Must use the new `SurvivalEngine` plugin with L2 regularization and Variance Dropping.
- **Evaluation**: 1000 Bootstrap iterations for `concordance_index_censored`.
- **Decision Engine**: **BYPASSED**. Do not route policies.

## 4. Output Checklist
- E1: `experiments/E1_behavior/e1_model_metrics.json`
- E2: `experiments/E2_vader/e2_model_metrics.json`
- E3: `experiments/E3_llm/e3_model_metrics.json`, `e3_hazard_ratios.csv`, `e3_lrt.csv`

## 5. Acceptance Criteria
- C-index confidence intervals do not overlap.
- Likelihood Ratio Test successfully proves information gain.
