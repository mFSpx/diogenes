# DARWIN HAMMER — match 52, survivor 1
# gen: 4
# parent_a: rlct_grokking.py (gen0)
# parent_b: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s5.py (gen3)
# born: 2026-05-29T23:26:30Z

"""
Hybrid Multivector-RLCT Module
==============================

This module fuses the Multivector class from Hybrid Geometric Product-LTC Module 
(PARENT ALGORITHM B) with the Real Log Canonical Threshold (RLCT) and Grokking 
concepts from rlct_grokking.py (PARENT ALGORITHM A). The mathematical bridge 
between the two parents is the integration of the Multivector's geometric product 
into the RLCT's update rule, specifically through the use of the Multivector's 
Clifford product to represent the weight matrix W in the RLCT's free energy 
asymptotic equation.

The fusion combines the governing equations of both parents, allowing for a novel 
hybrid algorithm that adapts to changing memory requirements and temporal dynamics.
"""

import numpy as np
import math
import random
import sys
from datetime import date
from pathlib import Path

__all__ = [
    "multivector_rlct",
    "hybrid_free_energy_asymptotic",
    "grokking_threshold_multivector",
]


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components, n):
        self.components = {k: float(v) for k, v in components.items() if v != 0.0}
        self.n = int(n)

    def grade(self, k):
        """Return a new Multivector keeping only grade k components."""
        return Multivector({k: v for k, v in self.components.items() if len(k) == k}, self.n)

    def __mul__(self, other):
        """Geometric product of two Multivectors."""
        result = Multivector({}, self.n)
        for blade_a, value_a in self.components.items():
            for blade_b, value_b in other.components.items():
                combined, sign = _multiply_blades(blade_a, blade_b)
                result.components[combined] = result.components.get(combined, 0) + sign * value_a * value_b
        return result


def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)


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
        Dimensionality of the model.

    Returns
    -------
    float
        Free energy asymptotic.
    """
    return n * w0 + lambda_ * math.log(n) - (m - 1) * math.log(math.log(n))


def multivector_rlct(multivector, w0, lambda_, m, n):
    """Multivector-RLCT.

    Parameters
    ----------
    multivector : Multivector
        Multivector representing the weight matrix W.
    w0 : float
        True parameter w0.
    lambda_ : float
        RLCT: measures geometric degeneracy (singularity) of the loss landscape at the true parameter w0.
    m : int or float
        Dimensionality of the model.
    n : int or float
        Dataset size n.

    Returns
    -------
    float
        Multivector-RLCT.
    """
    return free_energy_asymptotic(n, w0, lambda_, m) * multivector.grade(0).components.get(frozenset(), 0)


def hybrid_free_energy_asymptotic(n, w0, lambda_, m, multivector):
    """Hybrid free energy asymptotic.

    Parameters
    ----------
    n : int or float
        Dataset size n.
    w0 : float
        True parameter w0.
    lambda_ : float
        RLCT: measures geometric degeneracy (singularity) of the loss landscape at the true parameter w0.
    m : int or float
        Dimensionality of the model.
    multivector : Multivector
        Multivector representing the weight matrix W.

    Returns
    -------
    float
        Hybrid free energy asymptotic.
    """
    return free_energy_asymptotic(n, w0, lambda_, m) + multivector_rlct(multivector, w0, lambda_, m, n)


def grokking_threshold_multivector(multivector, n_samples, lambda_):
    """Grokking threshold Multivector.

    Parameters
    ----------
    multivector : Multivector
        Multivector representing the weight matrix W.
    n_samples : int or float
        Dataset size n.
    lambda_ : float
        RLCT: measures geometric degeneracy (singularity) of the loss landscape at the true parameter w0.

    Returns
    -------
    float
        Grokking threshold Multivector.
    """
    return math.exp((multivector.grade(0).components.get(frozenset(), 0) * n_samples) / lambda_)


if __name__ == "__main__":
    multivector = Multivector({frozenset([0, 1]): 1.0}, 2)
    print(multivector_rlct(multivector, 0.5, 0.1, 2, 100))
    print(hybrid_free_energy_asymptotic(100, 0.5, 0.1, 2, multivector))
    print(grokking_threshold_multivector(multivector, 100, 0.1))