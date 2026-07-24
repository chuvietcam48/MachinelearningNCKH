# Phase A: Evidence Freeze Dossier

This dossier establishes the undeniable foundation of the experimental setup before any modeling or statistical validation occurs. It covers Dataset descriptions, the exact Mapping of Experiments, and the Temporal Integrity (Leakage) Audit.

## A1. Dataset Audit

| Dataset | Total Rows (Episodes) | Unique Customers | Snapshot Date | Observation Window | Prediction Window | Semantic Coverage | Churn Definition |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CDNOW Music** | 69,579 | 23,502 | 1998-07-01 | 18 months | Time-to-event (T) | N/A (Baseline) | Censoring at Snapshot |
| **Amazon Reviews** | 1,733,777 | ~346K (Train split) | 2018-01-06 | > 10 years | Time-to-event (T) | 100% frozen labels | Censoring at Snapshot |

*Note: Train/Test splits are standardized dynamically via `run_framework.py` (80/20 random split seeded at 42) to ensure unbiased hazard estimation.*

## A2. Experiment Traceability Audit

To prevent "ghost results", every outcome must map perfectly from the script to the output to the paper's table.

| Exp ID | Config / Features | Executing Script | Intermediate Output | Mapped Figure/Table in Paper |
| :--- | :--- | :--- | :--- | :--- |
| **Exp01** | Baseline Behavior (CDNOW) | `run_framework.py --dataset cdnow` | CLI Logs (AIC: 275518) | Table 1 (Baseline Results) |
| **Exp02** | Behavior + Semantic (Amazon) | `run_framework.py --dataset amazon --enable-semantic` | CLI Logs (AIC: 6326947) | Table 2 (Semantic Extension) |
| **Exp03** | Survival HR Evaluation | `survival_engine.py` (cph.summary) | Internal CoxPH Model | Table 3 (Hazard Ratios) |
| **Exp04** | Decision Support Routing | `decision_support_engine.py` | Target Policies (e.g., 260K No Action) | Table 5 (Intervention Policies) |

## A3. Temporal Integrity (Leakage) Audit

The most critical step in survival modeling is ensuring no future information leaks into the features at time `t`.

| Feature | Uses Future Data? | Build Constraint | Leakage Risk | Status |
| :--- | :--- | :--- | :--- | :--- |
| `days_since_previous_episode` | No | Only looks at $(t-1)$ relative to $t$. | None | **PASS** |
| `customer_lifetime` | No | Bounded strictly by `episode_start`. | None | **PASS** |
| `rolling_negative_ratio` | No | Computed iteratively up to episode `k`. | None | **PASS** |
| `negative_streak` | No | Accumulates consecutively until reset. | None | **PASS** |
| `aspect_entropy` | No | Derived exclusively from frozen annotations of current/past episodes. | None | **PASS** |
| `cross_aspect_conflict` | No | Evaluated within the boundary of episode $k$. | None | **PASS** |

**Conclusion**: All generated features are 100% temporally safe. The framework adheres strictly to right-censoring survival principles.
