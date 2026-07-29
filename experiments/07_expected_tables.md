# Planned Output Tables and Figures

## Table 1: Dataset Demographics
- **Source**: `02_dataset_construction.py` outputs.
- **Content**: Summary statistics of the Amazon cohort (Total users, events, censoring rate, median reviews per user).

## Table 2: Model Comparison (Ablation Study)
- **Source**: E0, E1, E2, E3 artifacts (`c_index.json`, `lrt.csv`).
- **Content**:
    - **Original Framework** (Baseline)
    - **Universal Architecture** (Behavior Only)
    - **Universal Architecture** (+ VADER)
    - **Universal Architecture** (+ LLM Trajectories)
- **Metrics**: C-index, 95% CI, AIC, LRT p-value.

## Table 3: Semantic Hazard Ratios
- **Source**: E3 (`e3_hazard_ratios.csv`).
- **Content**: Top 5 most significant behavioral features and top 10 semantic trajectory features, their HR, Confidence Intervals, and p-values.

## Table 4: Policy Routing Distribution
- **Source**: E0 and E4 (`policy_routing_matrix.csv`).
- **Content**: Side-by-side comparison of how many customers are routed to specific interventions (e.g., Replacement, Discount) under the Legacy DSS vs. the Semantic DSS.

## Table 5: Business ROI Simulation
- **Source**: E0 and E4 (`scenario_evaluation.csv`).
- **Content**: Total intervention budget, expected retained customers, and Net ROI under Low, Medium, and High effectiveness scenarios.

---

## Planned Figures
- **Figure 1**: The Universal Framework Architecture (Flowchart).
- **Figure 2**: Kaplan-Meier survival curves stratifying users with high vs. low Semantic Conflict.
- **Figure 3**: Forest plot of the Hazard Ratios from Table 3.
