"""
Statistical Rigor Utilities.

Provides proper statistical testing for experimental validation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from scipy import stats


@dataclass
class ExperimentStats:
    """Statistical summary of an experiment."""
    # Basic statistics
    mean: float
    std: float
    n: int

    # Confidence interval
    ci_low: float
    ci_high: float
    confidence: float = 0.95

    # Additional metrics
    median: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    # Comparison stats (when comparing to baseline)
    p_value: Optional[float] = None
    effect_size: Optional[float] = None  # Cohen's d
    power: Optional[float] = None

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        confidence: float = 0.95,
    ) -> 'ExperimentStats':
        """Compute statistics from samples."""
        n = len(samples)
        mean = np.mean(samples)
        std = np.std(samples, ddof=1) if n > 1 else 0.0

        # Confidence interval using t-distribution
        if n > 1:
            t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin = t_crit * std / np.sqrt(n)
            ci_low = mean - margin
            ci_high = mean + margin
        else:
            ci_low = mean
            ci_high = mean

        return cls(
            mean=mean,
            std=std,
            n=n,
            ci_low=ci_low,
            ci_high=ci_high,
            confidence=confidence,
            median=np.median(samples),
            min_val=np.min(samples),
            max_val=np.max(samples),
        )


def paired_significance_test(
    baseline: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    correction: str = 'none',
    num_comparisons: int = 1,
) -> Dict[str, float]:
    """
    Perform paired significance test.

    Args:
        baseline: Baseline measurements
        treatment: Treatment measurements
        alpha: Significance level
        correction: Multiple testing correction ('none', 'bonferroni', 'holm')
        num_comparisons: Number of comparisons for correction

    Returns:
        Dictionary with test results.
    """
    if len(baseline) != len(treatment):
        raise ValueError("Baseline and treatment must have same length")

    n = len(baseline)

    # Paired t-test
    t_stat, p_value_t = stats.ttest_rel(baseline, treatment)

    # Wilcoxon signed-rank (non-parametric alternative)
    try:
        w_stat, p_value_w = stats.wilcoxon(baseline, treatment)
    except ValueError:
        # All differences are zero
        w_stat, p_value_w = 0.0, 1.0

    # Apply multiple testing correction
    corrected_alpha = alpha
    if correction == 'bonferroni':
        corrected_alpha = alpha / num_comparisons
    elif correction == 'holm':
        # For Holm correction, would need all p-values
        # Approximate with Bonferroni
        corrected_alpha = alpha / num_comparisons

    # Effect size (Cohen's d)
    diff = baseline - treatment
    effect_size = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    # Compute statistical power (post-hoc)
    # Using effect size and sample size
    noncentrality = abs(effect_size) * np.sqrt(n)
    critical_t = stats.t.ppf(1 - corrected_alpha/2, n-1)
    power = 1 - stats.nct.cdf(critical_t, n-1, noncentrality)
    power += stats.nct.cdf(-critical_t, n-1, noncentrality)

    return {
        't_statistic': t_stat,
        'p_value_ttest': p_value_t,
        'wilcoxon_statistic': w_stat,
        'p_value_wilcoxon': p_value_w,
        'effect_size_d': effect_size,
        'power': power,
        'significant_ttest': p_value_t < corrected_alpha,
        'significant_wilcoxon': p_value_w < corrected_alpha,
        'corrected_alpha': corrected_alpha,
        'n': n,
    }


def compute_effect_size(
    group1: np.ndarray,
    group2: np.ndarray,
    method: str = 'cohens_d',
) -> float:
    """
    Compute effect size between two groups.

    Args:
        group1: First group samples
        group2: Second group samples
        method: 'cohens_d', 'hedges_g', 'glass_delta'

    Returns:
        Effect size value.
    """
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    if method == 'cohens_d':
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        d = (mean1 - mean2) / (pooled_std + 1e-10)

    elif method == 'hedges_g':
        # Small sample correction
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        d = (mean1 - mean2) / (pooled_std + 1e-10)
        # Correction factor
        correction = 1 - 3 / (4*(n1+n2) - 9)
        d *= correction

    elif method == 'glass_delta':
        # Use only control group std
        d = (mean1 - mean2) / (np.sqrt(var2) + 1e-10)

    else:
        raise ValueError(f"Unknown method: {method}")

    return d


def confidence_interval(
    samples: np.ndarray,
    confidence: float = 0.95,
    method: str = 'percentile',
    num_bootstrap: int = 1000,
) -> Tuple[float, float]:
    """
    Compute confidence interval.

    Args:
        samples: Sample data
        confidence: Confidence level
        method: 'percentile', 't', 'bootstrap'
        num_bootstrap: Number of bootstrap samples

    Returns:
        (lower, upper) bounds.
    """
    n = len(samples)
    mean = np.mean(samples)
    std = np.std(samples, ddof=1)

    if method == 't':
        # t-distribution based
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_crit * std / np.sqrt(n)
        return (mean - margin, mean + margin)

    elif method == 'percentile':
        # Simple percentile method
        alpha = 1 - confidence
        lower = np.percentile(samples, 100 * alpha / 2)
        upper = np.percentile(samples, 100 * (1 - alpha / 2))
        return (lower, upper)

    elif method == 'bootstrap':
        # Bootstrap percentile
        boot_means = []
        for _ in range(num_bootstrap):
            boot_sample = np.random.choice(samples, size=n, replace=True)
            boot_means.append(np.mean(boot_sample))

        alpha = 1 - confidence
        lower = np.percentile(boot_means, 100 * alpha / 2)
        upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
        return (lower, upper)

    else:
        raise ValueError(f"Unknown method: {method}")


def report_results(
    experiment_name: str,
    baseline_samples: np.ndarray,
    treatment_samples: np.ndarray,
    metric_name: str = 'Performance',
    alpha: float = 0.05,
) -> str:
    """
    Generate a formatted report of experimental results.

    Returns:
        Formatted string report.
    """
    lines = [
        f"\n{'='*60}",
        f"EXPERIMENTAL RESULTS: {experiment_name}",
        f"{'='*60}",
        "",
    ]

    # Basic statistics
    base_stats = ExperimentStats.from_samples(baseline_samples)
    treat_stats = ExperimentStats.from_samples(treatment_samples)

    lines.extend([
        f"Metric: {metric_name}",
        f"Sample size: n = {base_stats.n}",
        "",
        f"Baseline:",
        f"  Mean ± SD: {base_stats.mean:.4f} ± {base_stats.std:.4f}",
        f"  95% CI: [{base_stats.ci_low:.4f}, {base_stats.ci_high:.4f}]",
        f"  Range: [{base_stats.min_val:.4f}, {base_stats.max_val:.4f}]",
        "",
        f"Treatment:",
        f"  Mean ± SD: {treat_stats.mean:.4f} ± {treat_stats.std:.4f}",
        f"  95% CI: [{treat_stats.ci_low:.4f}, {treat_stats.ci_high:.4f}]",
        f"  Range: [{treat_stats.min_val:.4f}, {treat_stats.max_val:.4f}]",
        "",
    ])

    # Comparison
    comparison = paired_significance_test(baseline_samples, treatment_samples, alpha)

    lines.extend([
        "Comparison:",
        f"  Difference: {treat_stats.mean - base_stats.mean:+.4f}",
        f"  Effect size (Cohen's d): {comparison['effect_size_d']:.3f}",
        f"  t-statistic: {comparison['t_statistic']:.3f}",
        f"  p-value (t-test): {comparison['p_value_ttest']:.4f}",
        f"  p-value (Wilcoxon): {comparison['p_value_wilcoxon']:.4f}",
        f"  Statistical power: {comparison['power']:.3f}",
        "",
    ])

    # Interpretation
    sig_t = comparison['significant_ttest']
    sig_w = comparison['significant_wilcoxon']
    d = comparison['effect_size_d']

    if sig_t and sig_w:
        if d > 0.8:
            interpretation = "Large significant improvement"
        elif d > 0.5:
            interpretation = "Medium significant improvement"
        elif d > 0.2:
            interpretation = "Small significant improvement"
        elif d < -0.8:
            interpretation = "Large significant decline"
        elif d < -0.5:
            interpretation = "Medium significant decline"
        elif d < -0.2:
            interpretation = "Small significant decline"
        else:
            interpretation = "Statistically significant but negligible effect"
    else:
        interpretation = "No statistically significant difference"

    lines.extend([
        f"Interpretation: {interpretation}",
        f"{'='*60}",
    ])

    return "\n".join(lines)


def multiple_comparison_correction(
    p_values: List[float],
    method: str = 'holm',
    alpha: float = 0.05,
) -> Tuple[List[float], List[bool]]:
    """
    Apply multiple comparison correction.

    Args:
        p_values: List of p-values
        method: 'bonferroni', 'holm', 'fdr' (Benjamini-Hochberg)
        alpha: Significance level

    Returns:
        (corrected_p_values, significant_flags)
    """
    n = len(p_values)
    p_values = np.array(p_values)

    if method == 'bonferroni':
        corrected = np.minimum(p_values * n, 1.0)
        significant = corrected < alpha

    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros(n)
        significant = np.zeros(n, dtype=bool)

        cummax = 0
        for i, idx in enumerate(sorted_idx):
            adj_p = p_values[idx] * (n - i)
            cummax = max(cummax, adj_p)
            corrected[idx] = min(cummax, 1.0)
            significant[idx] = corrected[idx] < alpha

    elif method == 'fdr':
        # Benjamini-Hochberg FDR
        sorted_idx = np.argsort(p_values)
        corrected = np.zeros(n)
        significant = np.zeros(n, dtype=bool)

        cummin = 1.0
        for i in range(n-1, -1, -1):
            idx = sorted_idx[i]
            adj_p = p_values[idx] * n / (i + 1)
            cummin = min(cummin, adj_p)
            corrected[idx] = min(cummin, 1.0)
            significant[idx] = corrected[idx] < alpha

    else:
        raise ValueError(f"Unknown method: {method}")

    return corrected.tolist(), significant.tolist()


def minimum_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test: str = 'paired',
) -> int:
    """
    Calculate minimum sample size needed.

    Args:
        effect_size: Expected effect size (Cohen's d)
        power: Desired statistical power
        alpha: Significance level
        test: 'paired' or 'independent'

    Returns:
        Minimum sample size per group.
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)

    if test == 'paired':
        n = ((z_alpha + z_beta) / effect_size) ** 2
    else:  # independent
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2

    return int(np.ceil(n))
