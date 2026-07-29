# Statistical Analysis Methodology

To ensure robust academic validity, the following statistical tests will be executed and reported in the paper.

## 1. Predictive Performance Significance (C-index)
- **Method**: 1,000 Bootstrap Iterations on the Test Set.
- **Metric**: Concordance Index for Censored Data (`concordance_index_censored`).
- **Reporting**: Report the mean C-index alongside the 95% Confidence Interval (e.g., $0.729 \pm [0.720, 0.738]$).
- **Success Criteria**: The lower bound of E3's CI must be strictly greater than the upper bound of E1's CI.

## 2. Model Fit and Information Gain (LRT)
- **Method**: Likelihood Ratio Test (LRT) comparing the baseline nested model (E1) against the full model (E3).
- **Metric**: LRT Chi-Square Statistic, degrees of freedom, and p-value.
- **Success Criteria**: $p < 0.001$.

## 3. Variable Risk Attribution (Hazard Ratios)
- **Method**: Wald Test per coefficient in the CoxPH model.
- **Metric**: Hazard Ratio ($\exp(\beta)$) and standard errors.
- **Success Criteria**: At least 3 semantic trajectory features must maintain statistical significance ($p < 0.05$) after controlling for behavioral variables.
