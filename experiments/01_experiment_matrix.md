# Experimental Matrix

This matrix defines the exact configurations for all experiments, separating Prediction evaluations from Decision evaluations. All experiments operate strictly on the **Amazon Dataset**.

| Exp | Dataset | Survival Model | Sentiment Level | Decision Engine | Purpose | Output Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **E0** | Amazon | Original Pipeline | ✗ (None) | Old DSS (Legacy) | True Baseline (Re-executing prior paper's architecture on Amazon data). | `experiments/E0_original_framework/` |
| **E1** | Amazon | Universal Framework | ✗ (None) | ✗ (N/A) | Evaluate pure behavioral capability of the new Universal Architecture. | `experiments/E1_behavior/` |
| **E2** | Amazon | Universal Framework | VADER (Lexicon) | ✗ (N/A) | Evaluate intermediate NLP contribution (Binary Sentiment). | `experiments/E2_vader/` |
| **E3** | Amazon | Universal Framework | LLM Trajectories | ✗ (N/A) | Evaluate advanced NLP contribution (State-aware Semantics). | `experiments/E3_llm/` |
| **E4** | Amazon | Universal Framework | LLM Trajectories | New DSS (EVI) | Evaluate the financial impact of Semantic EVI routing against E0's Old DSS. | `experiments/E4_dss/` |

> **Note**: E1, E2, and E3 do not test the Decision Engine. They purely measure predictive performance (C-index, HR). E4 introduces the Decision Engine to measure business ROI.
