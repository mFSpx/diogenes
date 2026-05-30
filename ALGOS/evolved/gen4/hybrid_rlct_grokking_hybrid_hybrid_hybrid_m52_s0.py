# DARWIN HAMMER — match 52, survivor 0
# gen: 4
# parent_a: rlct_grokking.py (gen0)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s5.py (gen3)
# born: 2026-05-29T23:26:30Z

"""
Hybrid Algorithm: Fusing Real Log Canonical Threshold (RLCT) and Hybrid Geometric Product-LTC

This module integrates the governing equations of Parent Algorithm A (RLCT) and Parent Algorithm B (Hybrid Geometric Product-LTC).
The mathematical bridge between the two parents is the integration of the Clifford geometric product into the LTC's update rule,
modulated by the RLCT's effective free energy asymptotic.

The free energy asymptotic of Watanabe is used to modulate the geometric product, allowing for a novel hybrid algorithm 
that adapts to changing memory requirements and temporal dynamics.
"""

import numpy as np
import math
import random
import sys
from datetime import date
from pathlib import Path

__all__ = [
    "hybrid_free_energy_asymptotic",
    "hybrid_geometric_product",
    "estimate_hybrid_rlct_from_losses",
]

class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        """Return a new Multivector keeping only grade k components."""
        return Multivector({k: v for k, v in self.components.items() if len(k) == k}, self.n)

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble-sorting index list."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    for i in range(n):
        for j in range(n - 1 - i):
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                lst.pop(j)
                lst.pop(j)  # was j+1, now at j after pop
                return lst, sign
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

def bayesian_information_criterion(log_likelihood, n_params, n_samples):
    """Standard BIC.

    BIC = -2 * log_likelihood + n_params * log(n_samples)

    Parameters
    ----------
    log_likelihood : float
        Log-likelihood evaluated at the MLE.
    n_params : int or float
        Number of free parameters.
    n_samples : int or float
        Dataset size n.

    Returns
    -------
    float
        BIC score.  Lower is better.

    Notes
    -----
    BIC penalises complexity by n_params * log(n) which matches the leading
    singular term in the free energy only when lambda = n_params / 2 (regular
    models).  For singular models (deep nets) BIC over-penalises --
    """
    return -2 * log_likelihood + n_params * math.log(n_samples)

def free_energy_asymptotic(n, w0, lambda_, m):
    """Watanabe's free energy asymptotic.

    F_n(w) ~ n*L_n(w0) + lambda*log(n) - (m-1)*log(log(n))

    Parameters
    ----------
    n : int or float
        Dataset size n.
    w0 : float
        True parameter w0.
    lambda_ : float
        RLCT: measures geometric degeneracy (singularity) of the loss landscape at the true parameter w0.
    m : int or float
        Number of parameters.

    Returns
    -------
    float
        Free energy asymptotic.
    """
    return n * w0 + lambda_ * math.log(n) - (m - 1) * math.log(math.log(n))

def hybrid_free_energy_asymptotic(n, w0, lambda_, m, multivector):
    """Hybrid free energy asymptotic.

    F_n(w) ~ n*L_n(w0) + lambda*log(n) - (m-1)*log(log(n)) + <multivector, geometric_product>

    Parameters
    ----------
    n : int or float
        Dataset size n.
    w0 : float
        True parameter w0.
    lambda_ : float
        RLCT: measures geometric degeneracy (singularity) of the loss landscape at the true parameter w0.
    m : int or float
        Number of parameters.
    multivector : Multivector
        Multivector representing the geometric product.

    Returns
    -------
    float
        Hybrid free energy asymptotic.
    """
    geometric_product = 0
    for blade, coefficient in multivector.components.items():
        geometric_product += coefficient * math.prod(blade)
    return free_energy_asymptotic(n, w0, lambda_, m) + geometric_product

def hybrid_geometric_product(multivector_a, multivector_b):
    """Hybrid geometric product.

    Parameters
    ----------
    multivector_a : Multivector
        Multivector a.
    multivector_b : Multivector
        Multivector b.

    Returns
    -------
    Multivector
        Geometric product of multivector_a and multivector_b.
    """
    result_components = {}
    for blade_a, coefficient_a in multivector_a.components.items():
        for blade_b, coefficient_b in multivector_b.components.items():
            blade, sign = _multiply_blades(blade_a, blade_b)
            result_components[blade] = result_components.get(blade, 0) + sign * coefficient_a * coefficient_b
    return Multivector(result_components, multivector_a.n)

def estimate_hybrid_rlct_from_losses(losses, n_params):
    """Estimate hybrid RLCT from losses.

    Parameters
    ----------
    losses : list of float
        Losses.
    n_params : int or float
        Number of free parameters.

    Returns
    -------
    float
        Estimated hybrid RLCT.
    """
    # Simple estimation: average loss
    avg_loss = sum(losses) / len(losses)
    return n_params - 2 * avg_loss

if __name__ == "__main__":
    multivector_a = Multivector({frozenset([1, 2]): 1.0}, 3)
    multivector_b = Multivector({frozenset([2, 3]): 2.0}, 3)
    hybrid_product = hybrid_geometric_product(multivector_a, multivector_b)
    print(hybrid_product.components)

    n = 100
    w0 = 1.0
    lambda_ = 0.5
    m = 3
    losses = [0.1, 0.2, 0.3]
    n_params = 5
    hybrid_rlct = estimate_hybrid_rlct_from_losses(losses, n_params)
    hybrid_free_energy = hybrid_free_energy_asymptotic(n, w0, lambda_, m, multivector_a)
    print(hybrid_rlct)
    print(hybrid_free_energy)