"""
Rigorous statistical evaluation routines:
- Video-level paired bootstrap confidence intervals (10,000 resamples)
- Sign-flip permutation testing (100,000 permutations)
- Exact McNemar tests for binary accuracy
- Clopper-Pearson exact binomial confidence intervals
- Stay-put baseline and moved-away breakdown diagnostics
"""

import numpy as np
from typing import List, Tuple, Dict, Any

def clopper_pearson_ci(x: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Clopper-Pearson confidence interval for binomial proportions."""
    from scipy.stats import beta
    lo = 0.0 if x == 0 else beta.ppf(alpha / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - alpha / 2, x + 1, n - x)
    return float(lo), float(hi)

def bootstrap_ci(v: np.ndarray, num_resamples: int = 10000, seed: int = 1) -> Tuple[float, float]:
    """Bootstrap confidence interval for means."""
    rng = np.random.default_rng(seed)
    v = np.asarray(v, float)
    m = v[rng.integers(0, len(v), (num_resamples, len(v)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

def signflip_permutation_p(diffs: np.ndarray, num_permutations: int = 100000, seed: int = 2) -> float:
    """Paired sign-flip permutation test p-value."""
    d = np.asarray(diffs, float)
    if np.all(d == 0):
        return 1.0
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=(num_permutations, len(d)))
    obs = abs(d.mean())
    return float((np.sum(np.abs((s * d).mean(axis=1)) >= obs - 1e-12) + 1) / (num_permutations + 1))

def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    """Exact McNemar test on paired binary observations."""
    from scipy.stats import binomtest
    a, b = np.asarray(a), np.asarray(b)
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    p_val = 1.0 if n10 + n01 == 0 else float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return n10, n01, p_val

def compute_video_level_metrics(parsed_rounds: Dict[int, str], parsed_final: str, gt_rounds: List[str], gt_final: str) -> Dict[str, Any]:
    """Compute per-video accuracy metrics."""
    n = len(gt_rounds)
    round_matches = sum(1 for i, g in enumerate(gt_rounds, 1) if parsed_rounds.get(i) == g)
    is_final_correct = int(parsed_final == gt_final) if parsed_final else 0
    is_exact_match = int(round_matches == n)
    return {
        "final": is_final_correct,
        "per_round": round_matches / n,
        "exact": is_exact_match
    }
