# Phase C: Novelty & Contribution Audit

This matrix explicitly defines the academic contributions of this paper compared to existing Information Systems (IS) and Decision Support Systems (DSS) literature.

## Contribution Matrix

| Core Area | Existing Work | Our Work (Novelty) | Evidence in Framework |
| :--- | :--- | :--- | :--- |
| **Framework Universality** | Churn models are typically evaluated on a single domain or proprietary dataset. | Proposes a **Universal Framework** validated across 5 distinct datasets (Music, Grocery, General Retail, E-commerce). | `run_framework.py` seamlessly executes CDNOW, TaFeng, UCI, X5, and Amazon. |
| **Semantic Integration** | Semantic traits (NLP) are used as static snapshot features (e.g., average review score). | Introduces **History-Aware Semantic Trajectories** (entropy, cross-aspect conflict, sentiment flip rates) over time. | `SemanticEngine` computes temporal semantic paths before fusion. |
| **Architectural Modularity** | Textual data is tightly coupled to the pipeline, breaking when text is unavailable. | Implements a **Plugin Manager Pattern**. Semantic data is treated as an optional extension plugin. | `PluginManager` dynamically injects `AmazonSemanticProvider` only when requested. |
| **Output Utility** | Models predict raw churn probabilities (predictive focus). | Focuses on **Decision Support and Prescriptive Policy**. Routes customers to specific interventions (e.g., Apology vs Discount). | `DecisionSupportEngine` translates Hazard Risk + Semantic Traits into Actionable Policies. |
| **Evaluation Strategy** | Models evaluated solely on AUC or C-index. | Proposes **Coverage-Aware Evaluation** (Dual-cohort analysis) to demonstrate the specific uplift of the semantic cohort. | LRT and Hazard Ratios validate the significant contribution of semantic traits over pure behavior. |

## Defense of "Unchanged C-Index"
While traditional ML papers rely on absolute C-index improvements, our contribution focuses on **Interpretability, Risk Attribution, and Intervention Prioritization**. By integrating Aspect Trajectories (like Conflict and Entropy), the model fit improves significantly (Likelihood Ratio Test p < 0.05), providing granular insights for the *Decision Support Engine* to recommend targeted actions (e.g., "Immediate Service Recovery" for high-conflict users) rather than just generating a generic churn score.
