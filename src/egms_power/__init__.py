"""Prospective protocol and power-analysis utilities for EGMS-Drive."""

from .statistics import (
    exact_mcnemar_power,
    find_minimum_exact_mcnemar_n,
    paired_joint_probabilities,
    required_n_two_independent_proportions,
    two_independent_proportions_power,
)

__all__ = [
    "exact_mcnemar_power",
    "find_minimum_exact_mcnemar_n",
    "paired_joint_probabilities",
    "required_n_two_independent_proportions",
    "two_independent_proportions_power",
]

__version__ = "1.1.0"
