"""
Package initialization for visual state tracking research.
"""

from src.dataset import load_ground_truth, build_stratified_folds
from src.prompts import PROMPTS, prompt_final, prompt_trajectory, render_trajectory
from src.parsing import parse_model_output
from src.corruption import build_b_records, build_bprime_v3
from src.metrics import (
    clopper_pearson_ci,
    bootstrap_ci,
    signflip_permutation_p,
    mcnemar_exact,
    compute_video_level_metrics
)

__all__ = [
    "load_ground_truth",
    "build_stratified_folds",
    "PROMPTS",
    "prompt_final",
    "prompt_trajectory",
    "render_trajectory",
    "parse_model_output",
    "build_b_records",
    "build_bprime_v3",
    "clopper_pearson_ci",
    "bootstrap_ci",
    "signflip_permutation_p",
    "mcnemar_exact",
    "compute_video_level_metrics"
]
