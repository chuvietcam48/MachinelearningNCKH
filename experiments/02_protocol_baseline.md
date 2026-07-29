# E0 Protocol: Original Framework (Baseline)

## 1. Definition
This experiment executes the archived implementation of the original framework on the current Amazon dataset. It establishes the ultimate controlled baseline: **Original Framework on the Amazon Dataset**.

## 2. Input
- `master_train_full.parquet` (Behavioral features only, identical 70/15/15 split).

## 3. Pipeline Constraints
- **Data Construction**: Use basic behavioral features. 
- **Survival Engine**: Execute the archived survival engine as implemented in the prior work. No advanced L2 regularization or Feature Fusion plugins.
- **Decision Engine**: Execute the archived DSS exactly as implemented, without modification.

## 4. Output Checklist (Saved to `experiments/E0_original_framework/`)
- `e0_c_index.json`
- `e0_hazard_ratios.csv`
- `e0_policy_distribution.csv`
- `e0_roi_simulation.csv`

## 5. Acceptance Criteria
- Code matches the architectural descriptions from the old paper.
- Output metrics successfully generated without crashing.
