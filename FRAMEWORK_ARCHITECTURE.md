# Universal Customer Churn Decision Framework: Architecture & Plugin Flow

This document outlines the execution and data flow of the framework. It demonstrates how the system maintains a strict separation of concerns, decoupling orchestration, feature engineering, and optional extensions.

## 1. Overall Framework Architecture
The core pipeline guarantees that all standard behavioral datasets follow the same path, while optional data is injected seamlessly via the Feature Registry.

```text
       [Raw Dataset (CSV/Parquet)]
                   │
                   ▼
         ┌───────────────────┐
         │  Dataset Adapter  │ (Standardizes IDs, dates, and schema)
         └───────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Behavior Engine   │ (Computes RFM, Interpurchase gaps, T, E)
        └─────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Plugin Manager    │ (Conditionally injects Semantic extensions)
        └─────────────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │  Feature Fusion Engine  │ (Merges features via dynamic Feature Registry)
      └─────────────────────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │   Survival Engine   │ (Trains dynamic CoxPH on non-zero variance features)
        └─────────────────────┘
                   │
                   ▼
   ┌───────────────────────────────┐
   │    Decision Support Engine    │ (Outputs Policy Routing & Counterfactuals)
   └───────────────────────────────┘
```

## 2. Plugin Flow (Methodological Extension)
When `enable-semantic` is triggered, the **Plugin Manager** coordinates with specific Data Providers to fetch derived artifacts without polluting the main orchestration logic.

```text
[Orchestration Request: --enable-semantic]
                   │
                   ▼
         ┌───────────────────┐
         │  Plugin Manager   │ (Checks dataset configuration and active plugins)
         └───────────────────┘
                   │
                   ▼
       ┌───────────────────────┐
       │ AmazonSemanticProvider│ (Loads Frozen LLM Annotations from data/artifacts/)
       └───────────────────────┘
                   │
                   ▼
         ┌───────────────────┐
         │  Semantic Engine  │ (Pure functional module: input DF -> output DF)
         └───────────────────┘
                   │
                   ▼
     (Returns Semantic DataFrame to Orchestration for Fusion)
```

## 3. Case Study Execution Examples

The framework operates gracefully regardless of dataset complexity. Below is the contrast between a traditional transactional dataset (CDNOW) and a complex e-commerce dataset (Amazon).

### Example A: Traditional Retail (CDNOW)
```text
Dataset (CDNOW)
      │
      ▼
Dataset Adapter (customer_id -> CustomerID)
      │
      ▼
Behavior Engine (Standard RFM & Survival Targets)
      │
      ▼
Plugin Manager (No semantic provider available)
      │
      ▼
Feature Fusion (Returns pure Behavior features)
      │
      ▼
Survival Engine & DSS (Outputs baseline policies)
```

### Example B: Semantic E-commerce (Amazon)
```text
Dataset (Amazon Episode Snapshots)
      │
      ▼
Dataset Adapter (reviewerID -> CustomerID)
      │
      ▼
Behavior Engine (Advanced Episode Trajectory Features)
      │
      ▼
Plugin Manager (Activates AmazonSemanticProvider)
      │
      ▼
Semantic Engine (Builds Aspect Streaks, Conflicts, Entropy)
      │
      ▼
Feature Fusion (Combines Behavior + Semantic Trajectories)
      │
      ▼
Survival Engine & DSS (Outputs Semantic-aware interventions)
```
