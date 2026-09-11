import numpy as np

def compute_percentile_ci(bootstrap_samples, alpha=0.05):
    """
    Computes the standard Percentile Confidence Interval from bootstrap samples.
    
    Args:
        bootstrap_samples (list or np.ndarray): Array of metric differences/values from bootstrap iterations.
        alpha (float): Significance level (default 0.05 for 95% CI).
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if bootstrap_samples is None or len(bootstrap_samples) == 0:
        return (np.nan, np.nan)
        
    lower = np.percentile(bootstrap_samples, (alpha / 2) * 100)
    upper = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)
    return (lower, upper)
