# Note on Legacy Model Artifacts in gate10_baseline/

The serialized files `model_a.pkl`, `model_b.pkl`, and `model_c.pkl` located in this directory are historical checkpoints produced during exploratory testing using scikit-survival's `CoxnetSurvivalAnalysis` (ElasticNet with $L_1$ ratio = 0.9). 

### Discrepancy Note:
- **Active Paper Models**: The official manuscript results, ablation studies, and bootstrap confidence intervals reported in the paper (Table II, Table IV, Section IV) are computed using `lifelines.CoxPHFitter` with an explicit small ridge penalty ($\lambda = 0.001$, or $\lambda = 0.1$ for small cohort regularization) without ElasticNet feature zeroing.
- These `.pkl` files are retained solely for internal checkpoint audit trail and should **not** be confused with the unpenalized/ridge-regularized CoxPH models evaluated in the reported ablation and TOST tests.
