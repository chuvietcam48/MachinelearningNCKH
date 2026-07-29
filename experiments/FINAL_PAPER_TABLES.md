# Final ACMLC Result Tables

## Table 1: Predictive Performance ($\Delta$ C-index) across Frameworks

| Framework | Model | C-index (95% CI) | $\Delta$ C-index (95% CI) |
| :--- | :--- | :--- | :--- |
| Historical | Behavior Only | 0.8099 [0.788, 0.832] | - |
| Historical | + VADER | 0.8261 [0.806, 0.847] | - |
| Historical | + LLM Semantics | 0.8073 [0.785, 0.828] | -0.0025 [-0.009, 0.004] |
| Universal | Behavior Only | 0.5568 [0.503, 0.615] | - |
| Universal | + VADER | 0.5964 [0.539, 0.648] | - |
| Universal | + LLM Semantics | 0.7158 [0.659, 0.774] | +0.1590 [0.083, 0.241] |

## Table 2: Likelihood Ratio Test (Behavior vs +LLM)

| Framework | $\chi^2$ Statistic | p-value | Significance |
| :--- | :--- | :--- | :--- |
| Historical | 11.58 | 1.1507e-01 | ns |
| Universal | 64.85 | 1.6139e-11 | *** |

## Table 3: Semantic Variable Interpretability (Universal Framework)

*Only variables remaining statistically significant after multivariable adjustment are shown.*

| Semantic Feature | Hazard Ratio ($\exp(\beta)$) | 95% CI | z-score | p-value |
| :--- | :--- | :--- | :--- | :--- |
| `net_sentiment` | 1.369 | [1.107, 1.692] | 2.90 | 0.0038 |
| `positive_minus_negative_ratio` | 1.419 | [1.148, 1.753] | 3.24 | 0.0012 |
| `semantic_variance` | 1.402 | [1.181, 1.665] | 3.86 | 0.0001 |

## Table 4: Decision Support System (DSS) Business Value

| Metric | Historical Framework DSS | Universal Framework DSS |
| :--- | :--- | :--- |
| Targeting Precision | 1.00 | 0.42 |
| False Incentive Rate | 0.00 | 0.57 |
| Expected Saved Customers | 60.0 | 25.5 |
| Expected Net Benefit ($) | 4000 | 550 |
| ROI | 2.00 | 0.28 |
| Resource Allocation Efficiency | 3.00 | 1.27 |

## EXP 4: Representative Case Analysis

**Case 1: The Sleeping Dog**
- **Profile**: High historical frequency/monetary value, but recent sentiment is heavily negative (e.g., `net_sentiment` drops sharply, `sentiment_variance` spikes).
- **Historical DSS Action**: Intervenes aggressively due to high baseline behavioral risk.
- **Universal DSS Action**: Correctly identifies the recent trajectory and prioritizes intervention *before* the structural behavior degrades.
