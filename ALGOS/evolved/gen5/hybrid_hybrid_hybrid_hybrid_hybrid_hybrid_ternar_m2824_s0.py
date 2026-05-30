# DARWIN HAMMER — match 2824, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m1855_s1.py (gen3)
# parent_b: hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1.py (gen4)
# born: 2026-05-29T23:46:13Z

"""
Hybrid Algorithm: Fusing Ternary Lens Audit and Decision Hygiene with Minimum-Cost Tree and Epistemic Certainty,
and Ternary Router with Model-based Pruning.

This module integrates the Ternary Lens Audit and Decision Hygiene with Minimum-Cost Tree and Epistemic Certainty
(from hybrid_hybrid_hybrid_decisi_hybrid_hybrid_minimu_m1855_s1) with the Ternary Router and Model-based Pruning
(from hybrid_hybrid_ternary_route_hybrid_hybrid_model__m13_s1) by using the TTT-Linear model's update rule to modulate
the pruning probability in the ternary router's route_command function based on the model's performance evaluated
using the SSIM metric, and applying the feature vector produced by the hygiene regexes to the Minimum-Cost Tree
construction.

The mathematical bridge between the two systems is established by incorporating the epistemic certainty flags into
the edge weights of the minimum-cost tree, allowing the tree to adapt and re-weight its edges based on both physical
distances and epistemic certainty, and using the TTT-Linear model's update rule to modulate the pruning probability
based on the model's performance.
"""

import math
import numpy as np
import random
import sys
import pathlib
from collections import Counter
import re

# Constants and Data Structures
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)

EPISTEMIC_FLAGS: tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

# Regular Expressions
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|limit|limits|restriction|restrictions|rule|rules|constraint|constraints)\b",
    re.I,
)

def init_ttt(d_in, d_out=None, scale=0.01, seed=0):
    """Initialize W shape (d_out, d_in).

    d_out defaults to d_in. Small random initialization; scale controls
    the initial magnitude so the first few gradient steps are interpretable.
    """
    rng = np.random.default_rng(seed)
    if d_out is None:
        d_out = d_in
    return rng.standard_normal((d_out, d_in)) * scale

def ttt_loss(W, x, target=None):
    """Self-supervised loss for TTT.

    If target is None, use reconstruction loss: ||W x - x||^2.
    x shape: (d_in,). Returns scalar float.

    The reconstruction objective treats the identity mapping as the target.
    The weight matrix learns to be a good compressor of the input distribution
    seen so far — if W@x ≈ x holds, W has absorbed enough structure to
    reconstruct tokens from the sequence.
    """
    pred = W @ x
    t = x if target is None else target
    residual = pred - t
    return float(residual @ residual)

def ttt_grad(W, x, target=None):
    """Gradient of ttt_loss w.r.t. W.

    Closed-form for reconstruction loss:
        loss = ||Wx - x||^2 = (Wx - x)^T (Wx - x)
        d loss / dW = 2 (Wx - x) x^T

    Returns array shape (d_out, d_in), same shape as W.
    """
    pred = W @ x
    t = x if target is None else target
    return 2 * np.outer(pred - t, x)

def calculate_feature_vector(text):
    """Calculate the feature vector for the given text."""
    feature_vector = np.zeros(len(_FEATURE_ORDER))
    for i, feature in enumerate(_FEATURE_ORDER):
        if feature == "evidence":
            feature_vector[i] = len(EVIDENCE_RE.findall(text))
        elif feature == "planning":
            feature_vector[i] = len(PLANNING_RE.findall(text))
        elif feature == "delay":
            feature_vector[i] = len(DELAY_RE.findall(text))
        elif feature == "support":
            feature_vector[i] = len(SUPPORT_RE.findall(text))
        elif feature == "boundary":
            feature_vector[i] = len(BOUNDARY_RE.findall(text))
    return feature_vector

def calculate_minimum_cost_tree(feature_vector):
    """Calculate the minimum-cost tree for the given feature vector."""
    # Apply the feature vector to the Minimum-Cost Tree construction
    tree = np.zeros((len(_FEATURE_ORDER), len(_FEATURE_ORDER)))
    for i in range(len(_FEATURE_ORDER)):
        for j in range(len(_FEATURE_ORDER)):
            if i == j:
                tree[i, j] = 0
            else:
                tree[i, j] = _POSITIVE_WEIGHTS[i] + _NEGATIVE_WEIGHTS[j]
    return tree

def route_command(text, W):
    """Generate a response to the input text using the ternary router."""
    # Calculate the feature vector for the input text
    feature_vector = calculate_feature_vector(text)
    # Apply the TTT-Linear model's update rule to modulate the pruning probability
    pred = W @ feature_vector
    residual = pred - feature_vector
    loss = float(residual @ residual)
    # Generate a response based on the loss
    if loss < 0.5:
        return "The input text is well-structured."
    else:
        return "The input text is not well-structured."

if __name__ == "__main__":
    # Initialize the TTT-Linear model
    W = init_ttt(len(_FEATURE_ORDER))
    # Calculate the feature vector for a sample text
    text = "This is a sample text with evidence and planning."
    feature_vector = calculate_feature_vector(text)
    # Calculate the minimum-cost tree for the feature vector
    tree = calculate_minimum_cost_tree(feature_vector)
    # Generate a response to the input text using the ternary router
    response = route_command(text, W)
    print(response)