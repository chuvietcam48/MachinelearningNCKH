# Phase E: Traceability Matrix (Evidence Chain)

This matrix maps every major scientific claim in the paper to its exact origin in the code, ensuring 100% transparency and reproducibility for reviewers.

| Scientific Claim | Executing Experiment / Script | Intermediate Data / Output | Resulting Figure / Table | Paper Section |
| :--- | :--- | :--- | :--- | :--- |
| **Claim 1:** The framework is universally applicable to various customer behavioral datasets. | `run_framework.py --dataset [cdnow/tafeng/uci]` | `DatasetAdapter` standardizes inputs to `[CustomerID, InvoiceDate, TotalSpend]`. Baseline CoxPH AIC outputs. | **Table 1:** Baseline Framework Performance across 5 datasets. | Section 3.1: Universal Framework Design |
| **Claim 2:** Aspect-level semantic conflict and entropy capture dimensions of churn risk invisible to pure RFM models. | `run_framework.py --dataset amazon --enable-semantic` | `episode_sentiment_features.parquet` + `SemanticEngine` (Conflict calculation) -> CoxPH Summary. | **Table 3:** Hazard Ratios (HR) showing significance of `aspect_entropy`. | Section 4.2: Semantic Risk Extension |
| **Claim 3:** Adding Semantic Trajectories significantly improves statistical model fit (via Likelihood Ratio Test) despite unchanged C-index. | `scripts/statistical_audit.py` (LRT testing between Baseline and Semantic models) | Log-Likelihood diff between Base (CDNOW) and Semantic (Amazon). | **Table 2:** Likelihood Ratio Test (LRT) comparisons. | Section 4.3: Model Fit & Evaluation |
| **Claim 4:** The Decision Support Engine uses semantic traits to route customers to targeted interventions. | `src/framework/decision_support_engine.py` | CLI Output: `No Action`, `Retention Offer`, `High-Priority Apology`, `Conflict Resolution`. | **Table 5:** Intervention Policy Distributions. | Section 5.1: Actionable Interventions |
| **Claim 5:** Frozen LLM annotations prevent temporal data leakage. | `scripts/audit_inventory.py` | `ARTIFACT_INVENTORY.md` (Verifying `881e0204` Hash). | **Data Description:** Provenance of annotations. | Section 3.3: Frozen Temporal Execution |
