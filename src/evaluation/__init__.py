# Re-export everything from the original evaluation module (now in core.py)
from .core import (
    load_config_with_overrides,
    bootstrap_c_index,
    compute_c_index,
    compute_integrated_brier_score,
    compute_time_dependent_auc,
    cross_validate_survival_model,
    compute_outreach_efficiency,
    compute_revenue_lift,
    print_full_report,
)

# Extended: counterfactual policy evaluation
from .counterfactual_evaluator import PolicyEvaluator, run_counterfactual_evaluation

__all__ = [
    # core evaluation functions
    "load_config_with_overrides",
    "bootstrap_c_index",
    "compute_c_index",
    "compute_integrated_brier_score",
    "compute_time_dependent_auc",
    "cross_validate_survival_model",
    "compute_outreach_efficiency",
    "compute_revenue_lift",
    "print_full_report",
    # counterfactual
    "PolicyEvaluator",
    "run_counterfactual_evaluation",
]
