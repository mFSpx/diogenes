# DARWIN HAMMER — match 1134, survivor 3
# gen: 4
# parent_a: hybrid_physarum_network_hybrid_hybrid_bandit_m11_s2.py (gen3)
# parent_b: hybrid_hybrid_cockpit_metri_hard_truth_math_m27_s0.py (gen2)
# born: 2026-05-29T23:33:06Z

import math
import random
import sys
from pathlib import Path
from typing import Tuple, Dict

import numpy as np

def flux(conductance: float, edge_length: float, pressure_a: float, pressure_b: float, eps: float = 1e-12) -> float:
    """Flux through an edge given its conductance, length and endpoint pressures."""
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float, dt: float = 1.0,
                       gain: float = 1.0, decay: float = 0.05) -> float:
    """Physarum conductance ODE discretisation."""
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    """Ratio of supported claims, clamped to [0, 1]."""
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0,
                claims_with_evidence / total_claims_emitted))


def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    """Fraction of displayed items that are known‑good, clamped to [0, 1]."""
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))


def stylometry_vector(text: str) -> np.ndarray:
    """
    Very lightweight stylometry: returns a 3‑dimensional feature vector
    [character count, vowel ratio, average word length].
    """
    if not text:
        return np.zeros(3)
    chars = len(text)
    vowels = sum(c.lower() in "aeiou" for c in text)
    words = text.split()
    avg_word_len = sum(len(w) for w in words) / len(words) if words else 0.0
    return np.array([chars, vowels / max(chars, 1), avg_word_len])


def trust_from_metrics(displayed_ok: int, unknown_displayed_as_ok: int,
                       claims_with_evidence: int, total_claims_emitted: int,
                       sample_text: str) -> float:
    """
    Combine cockpit honesty, anti‑slop ratio and a simple stylometry‑derived signal
    into a single trust scalar in [0, 1].
    """
    honesty = cockpit_honesty(displayed_ok, unknown_displayed_as_ok)
    evidence = anti_slop_ratio(claims_with_evidence, total_claims_emitted)
    style_feat = stylometry_vector(sample_text)
    w = np.array([0.001, 5.0, 0.05])  
    style_score = 1.0 / (1.0 + math.exp(-np.dot(w, style_feat)))
    return honesty * evidence * style_score


class HybridBanditPhysarumCockpit:
    """
    Core object holding:
    - conductance matrix (physarum)
    - propensity matrix (bandit)
    - trust scalar (cockpit + stylometry)
    The trust scalar modulates both propensity and conductance gain.
    """

    def __init__(self,
                 d_in: int,
                 d_out: int,
                 seed: int = 0,
                 base_eta: float = 0.01,
                 alpha: float = 1.0,
                 beta: float = 0.05,
                 dt: float = 1.0,
                 store_decay: float = 0.99) -> None:
        self.rng = np.random.default_rng(seed)
        self.base_eta = base_eta          
        self.alpha = alpha                
        self.beta = beta                  
        self.dt = dt
        self.store_decay = store_decay
        self.d_in = d_in
        self.d_out = d_out
        self.conductance = np.ones((d_in, d_out))
        self.propensity = np.ones((d_in, d_out)) / (d_out)  
        self.trust = 1.0

    def compute_trust(self,
                      displayed_ok: int,
                      unknown_displayed_as_ok: int,
                      claims_with_evidence: int,
                      total_claims_emitted: int,
                      sample_text: str) -> None:
        """Update internal trust scalar."""
        self.trust = trust_from_metrics(displayed_ok,
                                        unknown_displayed_as_ok,
                                        claims_with_evidence,
                                        total_claims_emitted,
                                        sample_text)

    def select_action(self) -> Tuple[int, int]:
        """
        Stochastic selection of an (input, output) edge proportional to
        propensity * conductance (i.e. the effective flow capacity).
        Returns the flat index (i, j).
        """
        weights = self.propensity * self.conductance
        flat_weights = weights.ravel()
        flat_weights_sum = flat_weights.sum()
        if flat_weights_sum == 0:
            probs = np.full_like(flat_weights, 1 / flat_weights.size)
        else:
            probs = flat_weights / flat_weights_sum
        choice = self.rng.choice(flat_weights.size, p=probs)
        i, j = divmod(choice, self.d_out)
        return i, j

    def hybrid_update(self,
                      action: Tuple[int, int],
                      reward: float,
                      edge_length: float,
                      pressure_a: float,
                      pressure_b: float) -> None:
        """
        Perform a single hybrid update:
        1. Compute flux on the selected edge.
        2. Update conductance using trust‑scaled gain.
        3. Update propensity using a trust‑weighted exponential moving average.
        """
        i, j = action
        q = flux(self.conductance[i, j], edge_length, pressure_a, pressure_b)

        gain = self.alpha * self.trust
        self.conductance[i, j] = update_conductance(
            self.conductance[i, j], q, dt=self.dt, gain=gain, decay=self.beta)

        eta = self.base_eta * self.trust
        self.propensity[i, j] = (1 - eta) * self.propensity[i, j] + eta * reward

        row_sum = self.propensity[i].sum()
        if row_sum > 0:
            self.propensity[i] /= row_sum

    def normalize_propensity(self) -> None:
        """Normalize propensity rows to keep them a probability distribution."""
        for i in range(self.d_in):
            row_sum = self.propensity[i].sum()
            if row_sum > 0:
                self.propensity[i] /= row_sum