# DARWIN HAMMER — match 420, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_krampus_brain_hybrid_indy_learning_m273_s0.py (gen4)
# parent_b: hybrid_fisher_localization_hybrid_ternary_route_m40_s2.py (gen2)
# born: 2026-05-29T23:28:53Z

"""Hybrid Fisher‑KL‑SSIM Algorithm
=================================

This module fuses the core topologies of two parent algorithms:

* **Parent A** – *hybrid_hybrid_krampus_brain_hybrid_indy_learning_m273_s0.py*  
  Provides a pheromone‑based probabilistic representation (via
  ``PheromoneEntry``/``PheromoneStore``) and a geometric vector that can be
  interpreted as a probability distribution.

* **Parent B** – *hybrid_fisher_localization_hybrid_ternary_route_m40_s2.py*  
  Supplies a Gaussian‑beam Fisher information score ``F(θ)`` and a
  1‑D Structural Similarity Index ``SSIM`` between two textual signals.

**Mathematical bridge**

Both parents produce scalar quality measures.  The bridge is built on the
information‑theoretic Kullback‑Leibler (KL) divergence between the
probability distribution derived from the pheromone store (Parent A) and the
soft‑max of the geometric vector (also Parent A).  The KL divergence is used
as a *compatibility weight* for the product of the Fisher score and the SSIM
from Parent B:


H(θ, txt) =  F(θ) · SSIM(txt, ref) · ϕ(KL(p‖q))


where ``p`` is the normalized pheromone distribution, ``q`` the soft‑max of the
geometric vector, and ``ϕ`` is a monotone decreasing mapper (here
``exp(-KL)``).  The resulting hybrid metric simultaneously evaluates
*parameter sharpness* (Fisher), *contextual similarity* (SSIM) and *topological
agreement* (KL).

The module implements the necessary building blocks and demonstrates three
public functions that compose the hybrid operation.
"""

import math
import random
import sys
import pathlib
import uuid
from datetime import datetime, timezone
from typing import List, Sequence, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Pheromone infrastructure and geometric vector handling
# ----------------------------------------------------------------------


class PheromoneEntry:
    """A single pheromone signal attached to a surface key."""
    __slots__ = ("uuid", "surface_key", "signal_kind", "signal_value",
                 "half_life_seconds", "created_at", "last_decay")

    def __init__(self, surface_key: str, signal_kind: str,
                 signal_value: float, half_life_seconds: int):
        self.uuid = str(uuid.uuid1())
        self.surface_key = surface_key
        self.signal_kind = signal_kind
        self.signal_value = signal_value
        self.half_life_seconds = half_life_seconds
        now = datetime.now(timezone.utc)
        self.created_at = now
        self.last_decay = now

    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.last_decay).total_seconds()

    def decay_factor(self) -> float:
        """Multiplicative decay factor since ``last_decay``."""
        if self.half_life_seconds <= 0:
            return 0.0
        return 0.5 ** (self.age_seconds() / self.half_life_seconds)

    def apply_decay(self) -> None:
        factor = self.decay_factor()
        self.signal_value *= factor
        self.last_decay = datetime.now(timezone.utc)


class PheromoneStore:
    """Container for multiple ``PheromoneEntry`` objects."""
    def __init__(self):
        self._entries: List[PheromoneEntry] = []

    def add(self, entry: PheromoneEntry) -> None:
        self._entries.append(entry)

    def decay_all(self) -> None:
        for e in self._entries:
            e.apply_decay()

    def distribution(self, surface_keys: Sequence[str]) -> np.ndarray:
        """
        Return a normalized probability vector ``p`` over ``surface_keys``.
        The probability mass of a key is the sum of its (decayed) signal values.
        """
        self.decay_all()
        mass = np.zeros(len(surface_keys), dtype=float)
        key_to_idx = {k: i for i, k in enumerate(surface_keys)}
        for e in self._entries:
            idx = key_to_idx.get(e.surface_key)
            if idx is not None:
                mass[idx] += max(e.signal_value, 0.0)   # ignore negative values
        total = mass.sum()
        if total <= 0.0:
            # Uniform fallback to avoid division by zero
            return np.full_like(mass, 1.0 / len(mass))
        return mass / total


def softmax_vector(v: np.ndarray) -> np.ndarray:
    """Convert an arbitrary vector to a probability distribution via soft‑max."""
    if v.ndim != 1:
        raise ValueError("softmax_vector expects a 1‑D array")
    max_val = np.max(v)
    exp_shifted = np.exp(v - max_val)          # shift for numerical stability
    sum_exp = exp_shifted.sum()
    if sum_exp == 0.0:
        return np.full_like(v, 1.0 / v.size)
    return exp_shifted / sum_exp


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute the Kullback‑Leibler divergence KL(p‖q).

    Both ``p`` and ``q`` must be valid probability vectors (sum to 1, non‑negative).
    """
    if p.shape != q.shape:
        raise ValueError("p and q must have the same shape")
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


# ----------------------------------------------------------------------
# Parent B – Fisher information and SSIM
# ----------------------------------------------------------------------


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float,
                 eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ.

    F(θ) = (∂I/∂θ)² / I   where I = Gaussian beam intensity.
    """
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def _to_numeric_signal(txt: str) -> np.ndarray:
    """Map a Unicode string to a 1‑D numeric signal (code‑point values)."""
    return np.fromiter((ord(ch) for ch in txt), dtype=float)


def ssim(x: Sequence[float], y: Sequence[float],
         dynamic_range: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    """Structural Similarity Index (SSIM) between two 1‑D signals."""
    if len(x) != len(y):
        raise ValueError("signals must have equal length")
    if not x:
        raise ValueError("signals must not be empty")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    mu_x = x_arr.mean()
    mu_y = y_arr.mean()
    sigma_x2 = x_arr.var()
    sigma_y2 = y_arr.var()
    sigma_xy = ((x_arr - mu_x) * (y_arr - mu_y)).mean()

    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x2 + sigma_y2 + c2)

    return float(numerator / denominator)


# ----------------------------------------------------------------------
# Hybrid core – combining Fisher, SSIM and KL divergence
# ----------------------------------------------------------------------


def hybrid_metric(theta: float, center: float, width: float,
                  txt: str, reference_txt: str,
                  geom_vector: np.ndarray,
                  pheromone_store: PheromoneStore,
                  surface_keys: Sequence[str]) -> float:
    """
    Compute the fused hybrid metric:

        H = F(θ) · SSIM(txt, reference) · exp(-KL(p‖q))

    where:
        * F(θ) – Fisher information (Parent B)
        * SSIM – structural similarity between the Unicode code‑point series of
          ``txt`` and ``reference_txt`` (Parent B)
        * p – pheromone probability distribution over ``surface_keys`` (Parent A)
        * q – soft‑max of ``geom_vector`` (Parent A)
        * KL – Kullback‑Leibler divergence (information‑theoretic bridge)

    The exponential map ``exp(-KL)`` penalises large topological disagreement,
    yielding a scalar that respects the ordering of all three components.
    """
    # Fisher component
    f = fisher_score(theta, center, width)

    # SSIM component (convert strings → numeric signals)
    x_sig = _to_numeric_signal(txt)
    y_sig = _to_numeric_signal(reference_txt)

    # Pad the shorter signal with zeros to match lengths (required by SSIM)
    max_len = max(len(x_sig), len(y_sig))
    if len(x_sig) < max_len:
        x_sig = np.pad(x_sig, (0, max_len - len(x_sig)), constant_values=0)
    if len(y_sig) < max_len:
        y_sig = np.pad(y_sig, (0, max_len - len(y_sig)), constant_values=0)

    s = ssim(x_sig, y_sig)

    # KL‑divergence component
    p = pheromone_store.distribution(surface_keys)
    q = softmax_vector(geom_vector)
    kl = kl_divergence(p, q)

    weight = math.exp(-kl)

    return f * s * weight


def hybrid_fisher(txt: str, reference_txt: str,
                  pheromone_store: PheromoneStore,
                  surface_keys: Sequence[str],
                  geom_vector: np.ndarray) -> float:
    """
    Hybrid variant of the Fisher score that incorporates SSIM and KL weighting.
    The angle θ is *implicitly* taken as the location of the maximum of the
    Gaussian beam (i.e., ``center``).  This helper illustrates how a pure
    Fisher evaluation can be enriched without explicitly passing θ.
    """
    # Use the centre of the Gaussian as the “optimal” θ (arbitrary choice)
    centre = 0.0
    width = 1.0
    theta = centre

    return hybrid_metric(theta, centre, width,
                         txt, reference_txt,
                         geom_vector, pheromone_store, surface_keys)


def hybrid_ssim(theta: float, center: float, width: float,
                txt: str, reference_txt: str,
                pheromone_store: PheromoneStore,
                surface_keys: Sequence[str]) -> float:
    """
    Hybrid variant of SSIM that is modulated by Fisher information and KL weight.
    Demonstrates the opposite ordering of the components compared to
    ``hybrid_fisher``.
    """
    # Compute KL weight first (independent of θ)
    p = pheromone_store.distribution(surface_keys)
    q = softmax_vector(np.ones(len(surface_keys)))  # uniform geometric vector
    kl = kl_divergence(p, q)
    weight = math.exp(-kl)

    # Fisher component
    f = fisher_score(theta, center, width)

    # SSIM component
    x_sig = _to_numeric_signal(txt)
    y_sig = _to_numeric_signal(reference_txt)

    # Align lengths
    max_len = max(len(x_sig), len(y_sig))
    if len(x_sig) < max_len:
        x_sig = np.pad(x_sig, (0, max_len - len(x_sig)), constant_values=0)
    if len(y_sig) < max_len:
        y_sig = np.pad(y_sig, (0, max_len - len(y_sig)), constant_values=0)

    s = ssim(x_sig, y_sig)

    return f * s * weight


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a minimal set of surface keys for the pheromone store
    keys = ["alpha", "beta", "gamma"]

    # Populate the pheromone store with synthetic entries
    store = PheromoneStore()
    for k in keys:
        entry = PheromoneEntry(
            surface_key=k,
            signal_kind="signal",
            signal_value=random.uniform(0.5, 2.0),
            half_life_seconds=30
        )
        store.add(entry)

    # Geometric vector (could be any real-valued vector)
    geom_vec = np.array([0.2, -0.1, 0.5])

    # Example textual data
    txt = "Hello, world!"
    ref = "Hello, Earth!"

    # Parameters for the Gaussian beam
    theta_val = 0.3
    centre = 0.0
    beam_width = 1.0

    # Compute the three hybrid demonstrations
    h_metric = hybrid_metric(theta_val, centre, beam_width,
                             txt, ref, geom_vec, store, keys)
    h_fisher = hybrid_fisher(txt, ref, store, keys, geom_vec)
    h_ssim = hybrid_ssim(theta_val, centre, beam_width,
                         txt, ref, store, keys)

    print(f"Hybrid metric (full): {h_metric:.6f}")
    print(f"Hybrid Fisher‑augmented: {h_fisher:.6f}")
    print(f"Hybrid SSIM‑augmented: {h_ssim:.6f}")