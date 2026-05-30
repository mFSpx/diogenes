# DARWIN HAMMER — match 2407, survivor 0
# gen: 5
# parent_a: hybrid_omni_chaotic_sprint_jepa_energy_m80_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py (gen4)
# born: 2026-05-29T23:42:18Z

import numpy as np
import math
import random
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Module docstring
# ----------------------------------------------------------------------
"""
Hybrid Algorithm: Chaotic Sprint Fusion

This module fuses the graph-based representation learning and chaotic sprint 
algorithm of omni_chaotic_sprint.py with the Fisher information-based angle 
selection and contextual multi-armed bandit of hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s1.py.

The mathematical bridge lies in the interpretation of the bandit-produced 
`propensity` as a confidence scalar that modulates the chaotic sprint process, 
resulting in a confidence-weighted Fisher information computation.

"""

# ----------------------------------------------------------------------
# Function 1: compute_confidence_weighted_fisher_information
# ----------------------------------------------------------------------
def compute_confidence_weighted_fisher_information(theta, mu, sigma, v, propensity):
    """
    Compute the confidence-weighted Fisher information.

    Parameters:
    theta (float): the angle
    mu (float): the mean angle
    sigma (float): the standard deviation of the angle
    v (float): the intensity
    propensity (float): the confidence scalar

    Returns:
    confidence_weighted_intensity (float): the intensity weighted by the confidence
    confidence_weighted_fisher_information (float): the Fisher information weighted by the confidence
    """
    intensity, fisher_information = compute_fisher_information(theta, mu, sigma, v)
    confidence_weighted_intensity = intensity * propensity
    confidence_weighted_fisher_information = fisher_information * propensity
    return confidence_weighted_intensity, confidence_weighted_fisher_information

# ----------------------------------------------------------------------
# Function 2: select_confidence_weighted_action
# ----------------------------------------------------------------------
def select_confidence_weighted_action(context, actions, propensity, algorithm="linucb", epsilon=0.1, seed=7):
    """
    Choose an action and return its BanditAction descriptor with confidence weighting.

    Parameters:
    context (Dict[str, float]): the context
    actions (List[str]): the actions
    propensity (float): the confidence scalar
    algorithm (str): the algorithm (default: "linucb")
    epsilon (float): the epsilon value (default: 0.1)
    seed (int | str | None): the seed (default: 7)

    Returns:
    BanditAction: the BanditAction descriptor with confidence weighting
    """
    if not actions:
        raise ValueError("No actions provided")
    action = random.choice(actions)
    return BanditAction(action, propensity, compute_fisher_information(action, context["mean_angle"], context["std_dev"], context["intensity"])[1], math.sqrt(context["std_dev"]), algorithm)

# ----------------------------------------------------------------------
# Function 3: chaotic_sprint_with_confidence_weighting
# ----------------------------------------------------------------------
def chaotic_sprint_with_confidence_weighting(theta, mu, sigma, v, propensity):
    """
    Perform chaotic sprint with confidence weighting.

    Parameters:
    theta (float): the angle
    mu (float): the mean angle
    sigma (float): the standard deviation of the angle
    v (float): the intensity
    propensity (float): the confidence scalar

    Returns:
    float: the confidence-weighted chaotic sprint value
    """
    return propensity * compute_fisher_information(theta, mu, sigma, v)[0]

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    theta = 0.5
    mu = 0.7
    sigma = 0.3
    v = 1.0
    propensity = 0.8
    context = {"mean_angle": mu, "std_dev": sigma, "intensity": v}
    actions = ["action1", "action2", "action3"]
    print(compute_confidence_weighted_fisher_information(theta, mu, sigma, v, propensity))
    print(select_confidence_weighted_action(context, actions, propensity))
    print(chaotic_sprint_with_confidence_weighting(theta, mu, sigma, v, propensity))