# Project Artifact Inventory

This document catalogs all critical research artifacts (datasets, scripts, outputs, and models) used in the Universal Customer Churn Decision Framework project. This ensures full reproducibility and traceablity for reviewers.

## 1. Dataset Inventory

All raw and frozen datasets are versioned with SHA-256 hashes to guarantee data integrity across experiments.

| Dataset | SHA-256 (8) | Size | Rows | Purpose |
| ------- | ----------- | ---- | ---- | ------- |
| CDNOW Music | `0829e4bc` | 2.00MB | 69,659 | Raw behavior dataset (Baseline Validation) |
| Ta Feng Grocery | `1d575e5d` | 60.69MB | 817,741 | Raw behavior dataset (Validation) |
| UCI Online Retail | `43465a06` | 22.62MB | 541,909 | Raw behavior dataset (Validation) |
| X5 RetailHero | `a9e132e3` | 4256.99MB | 45,786,568 | Raw behavior dataset (Validation) |
| Amazon Snapshots | `881e0204` | 34.90MB | 1,733,777 | Frozen semantic episode snapshots (Amazon) |
| Amazon Sentiments | `0b1ba975` | 24.56MB | 1,794,750 | Frozen episode-level sentiment (Amazon) |
| Amazon Mass Results| `49206603` | 0.99MB | 2,067 | Frozen raw LLM aspect classification output |

> **Note**: The X5 RetailHero dataset is significantly larger (45M+ rows) and demonstrates the framework's scalability in handling massive raw behavioral logs.

## 2. Core Script Inventory

| Script Path | Purpose |
| ----------- | ------- |
| `run_framework.py` | Universal Orchestrator. Executes the entire pipeline dynamically. |
| `src/dataset_registry.py` | Schema Mapper & Dataset Registry. Unifies schema dynamically. |
| `src/framework/plugin_manager.py` | Implements the Plugin Pattern to selectively inject Semantic capabilities. |
| `src/framework/dataset_adapter.py` | Standardizes raw input datasets (Universal IO Layer). |
| `src/framework/behavior_engine.py` | Extracts standard RFM and Recency-frequency trajectory signals. |
| `src/framework/semantic_engine.py` | Computes dynamic conflict, entropy, and temporal semantic trajectories. |
| `src/framework/feature_fusion.py` | Merges Behavioral and Semantic features via Feature Registry Pattern. |
| `src/framework/survival_engine.py` | Fits and evaluates the Cox Proportional Hazards model. |
| `src/framework/decision_support_engine.py` | Maps risk scores and semantic traits to interpretable intervention policies. |

## 3. Key Outputs & Metric Paths

| Artifact Type | Expected Output Location | Description |
| ------------- | ------------------------ | ----------- |
| Survival Metrics | Console / Logs | C-index, AIC, log-likelihood, Concordance. |
| Feature Weights | Console / Logs | Hazard Ratios (HR) for each feature. |
| Intervention Policy | Console / Logs | Distribution of recommended retention actions per customer. |
