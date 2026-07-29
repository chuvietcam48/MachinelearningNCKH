# Master Execution Plan

## 1. Objective
To conclusively prove that the **Universal Customer Churn Decision Framework** outperforms the **Original Framework** by executing both architectures on the exact same dataset (Amazon). This controls for dataset variation and isolates the architectural and semantic improvements as the true sources of information gain.

## 2. Core Contributions Mapped
- **Contribution 1 (Architecture)**: `Original Framework` → `Universal Framework`
- **Contribution 2 (Semantic)**: `Behavior` → `Behavior + VADER` → `Behavior + LLM` (Within Universal Framework)

## 3. Philosophy: Separation of Layers
The experiments strictly separate the evaluation of prediction capabilities from the evaluation of business decisions:
- **Prediction Layer Experiments**: E0, E1, E2, E3 (Evaluates how accurately we predict churn).
- **Decision Layer Experiments**: E4 (Evaluates how effectively we retain customers and maximize ROI based on those predictions).

## 4. Execution Policy
- The current Universal Framework (`src/pipeline/`) is frozen. No further modifications will be made to its algorithms.
- The archived implementation of the Original Framework will be executed specifically for E0 to ensure an unbiased baseline.
