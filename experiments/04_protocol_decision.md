# E4 Protocol: Universal Decision Phase

## 1. Definition
This experiment tests the business impact of the Universal Framework by injecting the most powerful predictions (from E3) into the advanced Expected Value of Intervention (EVI) Decision Support System.

## 2. Input
- `e3_predictions.parquet` (Predicted hazard scores and survival probabilities from E3).
- `master_test_semantic.parquet` (For CLV proxy and semantic reason attribution).

## 3. Pipeline Constraints
- **Prediction Phase**: **BYPASSED**. Do not train new models.
- **Decision Engine**: Must execute the full EVI Monte Carlo simulation. Must map LLM aspect conflicts into actionable intervention strategies (e.g., "Human Escalation").

## 4. Output Checklist (Saved to `experiments/E4_dss/`)
- `e4_policy_routing_matrix.csv`
- `e4_scenario_evaluation.csv`
- `e4_priority_list_top100.csv`

## 5. Acceptance Criteria
- DSS successfully maps semantic reasons to specific policies.
- Simulated ROI is demonstrably higher than the legacy DSS (E0).
