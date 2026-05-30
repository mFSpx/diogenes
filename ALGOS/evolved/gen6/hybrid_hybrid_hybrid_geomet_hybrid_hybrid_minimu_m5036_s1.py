# DARWIN HAMMER — match 5036, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s1.py (gen5)
# parent_b: hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m601_s1.py (gen4)
# born: 2026-05-29T23:59:22Z

"""
Hybrid Algorithm: Fusing DARWIN HAMMER (match 461, survivor 1) and 
DARWIN HAMMER (match 601, survivor 1)

This module integrates the hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s1.py 
and hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m601_s1.py algorithms into a single 
hybrid system. The mathematical bridge between the two structures is the use of Shannon 
entropy to analyze the uncertainty of the decision-making process in the Capybara 
Optimization Algorithm and influence the geometric product in the Clifford algebra, 
and the incorporation of the Bayesian posterior update into the rotor update.

Parents:
- hybrid_hybrid_geometric_pro_hybrid_hybrid_hybrid_m461_s1.py
- hybrid_hybrid_minimum_cost__hybrid_hybrid_hybrid_m601_s1.py
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.

    Duplicates cancel because e_i*e_i = 1.
    """
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # remove the pair
                del lst[j:j + 2]
                n -= 2
                i = -1  # restart outer loop
                break
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades
    """
    return np.array(blade_a) ^ np.array(blade_b)


def shannon_entropy(feature_counts):
    """Calculate Shannon entropy
    """
    total = sum(feature_counts.values())
    entropy = 0.0
    for count in feature_counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy


def apply_rotor(R, x):
    """Rotate a Euclidean vector with a rotor.
    """
    return R * x * np.conj(R)


def bayes_update(prior, likelihood):
    """Bayesian posterior update
    """
    return (prior * likelihood) / (likelihood * prior + (1 - prior) * (1 - likelihood))


def ttt_ga_forward(W, R, x, eta_w, eta_r, feature_counts):
    """One‑step hybrid update
    """
    entropy = shannon_entropy(feature_counts)
    posterior = bayes_update(0.5, entropy)
    R_update = np.exp(eta_r * posterior * 1j)
    W_update = W + eta_w * (R_update * x * np.conj(R_update) - W)
    return W_update, R_update


def length(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Euclidean distance
    """
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def hybrid_cost(edge_posteriors, edge_lengths, node_posteriors, node_distances, lambda_):
    """Hybrid cost function
    """
    cost = sum(p * l for p, l in zip(edge_posteriors, edge_lengths)) + lambda_ * sum(q * d for q, d in zip(node_posteriors, node_distances))
    return cost


def main():
    # Smoke test
    feature_counts = Counter({1: 10, 2: 20, 3: 30})
    W = np.random.rand(2)
    R = np.random.rand(2) + 1j * np.random.rand(2)
    x = np.random.rand(2)
    eta_w = 0.1
    eta_r = 0.1
    print(ttt_ga_forward(W, R, x, eta_w, eta_r, feature_counts))

    a = (1.0, 2.0)
    b = (4.0, 6.0)
    print(length(a, b))

    edge_posteriors = [0.5, 0.6, 0.7]
    edge_lengths = [1.0, 2.0, 3.0]
    node_posteriors = [0.4, 0.5, 0.6]
    node_distances = [10.0, 20.0, 30.0]
    lambda_ = 0.1
    print(hybrid_cost(edge_posteriors, edge_lengths, node_posteriors, node_distances, lambda_))


if __name__ == "__main__":
    main()