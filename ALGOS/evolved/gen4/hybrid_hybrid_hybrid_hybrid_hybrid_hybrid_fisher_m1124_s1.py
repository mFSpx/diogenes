# DARWIN HAMMER — match 1124, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s1.py (gen3)
# parent_b: hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s6.py (gen3)
# born: 2026-05-29T23:32:56Z

"""Hybrid Allocation‑Fisher‑Geometric Module
==========================================

Parents
-------
* **PARENT ALGORITHM A** – *hybrid_hybrid_hybrid_worksh_hybrid_geometric_pro_m36_s1.py*  
  Provides a deterministic allocation framework, a time‑constant LTC model and a
  Clifford geometric product implementation (via the ``Multivector`` class).

* **PARENT ALGORITHM B** – *hybrid_hybrid_fisher_locali_hybrid_hybrid_decisi_m26_s6.py*  
  Supplies statistical tools (Gaussian beam, Fisher information, SSIM) and a set
  of regex‑based semantic feature extractors.

Mathematical Bridge
-------------------
The hybrid treats each calendar day as a discrete time step *t*.  
The day‑of‑week, scaled to the interval ``[0, 1]``, is the external LTC input
``I(t)``.  From ``I(t)`` we compute a **Fisher information weight**  


w(t) = FisherScore(I(t); μ=0.5, σ=0.2)


which quantifies the information content of the day’s position within the week.
The allocation scalar ``τ(t)`` from the LTC model is multiplied by ``w(t)`` to
obtain a **resource scaling factor** ``α(t) = τ(t)·w(t)``.

Textual evidence extracted by the regex feature detectors is encoded as a
multivector ``M(t)`` whose basis blades correspond to semantic categories
(evidence → ``e``, planning → ``p``, delay → ``d``).  The Clifford geometric
product is then used to fuse the information‑weighted allocation with the
semantic multivector:


M′(t) = α(t)  ◦  M(t)      (geometric product with a scalar)


The resulting multivector simultaneously carries the allocation magnitude and
the structured semantic content, providing a unified representation for downstream
tasks such as similarity evaluation (via SSIM) or further algebraic manipulation.

The code below implements this bridge, exposing three core functions that
demonstrate the hybrid operation.
"""

import sys
import math
import random
import re
from pathlib import Path
import numpy as np
from datetime import date

# ----------------------------------------------------------------------
# Parent A – Multivector & geometric product
# ----------------------------------------------------------------------
class Multivector:
    """Clifford algebra element in Cl(n,0) represented by a dict of basis→coeff."""

    def __init__(self, components=None, n=3):
        # components: dict mapping basis string (e.g. '' for scalar, 'e', 'p', 'd', 'ep') to coefficient
        self.n = int(n)
        self.components = {}
        if components:
            for blade, coef in components.items():
                if abs(coef) > 1e-12:
                    self.components[blade] = float(coef)

    def __repr__(self):
        if not self.components:
            return "0"
        terms = [f"{coef:.3g}{blade or '1'}" for blade, coef in self.components.items()]
        return " + ".join(terms)

    def scalar_part(self):
        """Return the scalar (grade‑0) coefficient, 0 if absent."""
        return self.components.get('', 0.0)

    def __add__(self, other):
        result = Multivector(n=self.n)
        # add self components
        for b, c in self.components.items():
            result.components[b] = result.components.get(b, 0.0) + c
        # add other components
        for b, c in other.components.items():
            result.components[b] = result.components.get(b, 0.0) + c
        # prune near‑zero entries
        result.components = {b: c for b, c in result.components.items() if abs(c) > 1e-12}
        return result

    def __mul__(self, other):
        """Geometric product with another Multivector or a scalar."""
        if isinstance(other, (int, float)):
            return Multivector({b: c * other for b, c in self.components.items()}, n=self.n)
        if not isinstance(other, Multivector):
            raise TypeError("Geometric product only defined for Multivector or scalar.")
        result = Multivector(n=self.n)
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                blade, sign = _geometric_blade_product(b1, b2)
                coeff = c1 * c2 * sign
                result.components[blade] = result.components.get(blade, 0.0) + coeff
        # prune near‑zero entries
        result.components = {b: c for b, c in result.components.items() if abs(c) > 1e-12}
        return result

    __rmul__ = __mul__

def _geometric_blade_product(b1: str, b2: str):
    """
    Compute the geometric product of two orthogonal basis blades represented as
    strings of sorted characters (e.g. 'e', 'p', 'ed').  The function returns
    (result_blade, sign) where sign is +1 or -1 according to the anti‑commutation
    rule e_i e_j = - e_j e_i for i ≠ j.
    """
    # empty string denotes scalar
    if not b1:
        return b2, 1
    if not b2:
        return b1, 1
    # concatenate and sort while counting swaps for sign
    result = list(b1 + b2)
    sign = 1
    # bubble‑sort to canonical order and count swaps
    for i in range(len(result)):
        for j in range(len(result) - 1, i, -1):
            if result[j] < result[j - 1]:
                result[j], result[j - 1] = result[j - 1], result[j]
                sign *= -1
    # cancel duplicate basis vectors (e_i * e_i = 1)
    i = 0
    while i < len(result) - 1:
        if result[i] == result[i + 1]:
            # remove the pair and flip sign (since e_i^2 = 1, no sign change)
            del result[i:i + 2]
        else:
            i += 1
    return ''.join(result), sign

# ----------------------------------------------------------------------
# Parent B – Statistical tools & regex feature extraction
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian beam intensity."""
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for the Gaussian beam."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def ssim(x: np.ndarray, y: np.ndarray,
         dynamic_range: float = 255.0,
         k1: float = 0.01,
         k2: float = 0.03) -> float:
    """Structural Similarity Index Measure for 1‑D signals."""
    if x.shape != y.shape:
        raise ValueError('samples must have equal shape')
    if x.size == 0:
        raise ValueError('samples must not be empty')
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))


# Regex feature categories
EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b", re.I)
PLANNING_RE = re.compile(r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b", re.I)
DELAY_RE    = re.compile(r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b", re.I)

def extract_feature_counts(text: str):
    """Return a dict with counts of evidence, planning and delay tokens."""
    return {
        'evidence': len(EVIDENCE_RE.findall(text)),
        'planning': len(PLANNING_RE.findall(text)),
        'delay'   : len(DELAY_RE.findall(text)),
    }

# ----------------------------------------------------------------------
# Hybrid Core Functions
# ----------------------------------------------------------------------
def day_of_week_input(d: date) -> float:
    """
    Convert a date to the LTC external input I(t) ∈ [0,1].
    Monday → 0.0, Sunday → 1.0 (linear scaling).
    """
    weekday = d.weekday()          # Monday=0 … Sunday=6
    return weekday / 6.0

def allocation_scalar(I: float, k: float = 5.0) -> float:
    """
    LTC‑style effective time constant τ(t) as a sigmoid of the input.
    Larger I → larger τ, bounded in (0,1).
    """
    return 1.0 / (1.0 + math.exp(-k * (I - 0.5)))

def hybrid_allocation(d: date, text: str):
    """
    Perform one hybrid step:
    1. Compute I(t) from the calendar day.
    2. Derive τ(t) via a sigmoid LTC.
    3. Compute Fisher weight w(t) from I(t).
    4. Form the scaling factor α(t) = τ(t)·w(t).
    5. Encode regex feature counts as a multivector M(t).
    6. Return the geometrically updated multivector α(t) ◦ M(t).
    """
    I = day_of_week_input(d)
    tau = allocation_scalar(I)
    w = fisher_score(I, center=0.5, width=0.2)
    alpha = tau * w

    counts = extract_feature_counts(text)
    # Map categories to orthogonal basis blades
    blade_map = {'evidence': 'e', 'planning': 'p', 'delay': 'd'}
    components = {}
    for cat, cnt in counts.items():
        if cnt != 0:
            components[blade_map[cat]] = cnt
    # Ensure there is at least a scalar part (use 1.0) to keep the multivector non‑empty
    if not components:
        components[''] = 1.0
    M = Multivector(components, n=3)

    # Geometric product with scalar α
    M_prime = M * alpha
    return {
        'date': d,
        'I': I,
        'tau': tau,
        'fisher_weight': w,
        'alpha': alpha,
        'original_mv': M,
        'updated_mv': M_prime,
    }

def compare_signals(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Compute SSIM between two 1‑D signals after normalising them.
    """
    # Normalise to zero mean, unit variance to avoid dynamic_range issues
    s1 = (sig1 - np.mean(sig1)) / (np.std(sig1) + 1e-12)
    s2 = (sig2 - np.mean(sig2)) / (np.std(sig2) + 1e-12)
    # Rescale to typical 8‑bit range for SSIM function
    s1_scaled = (s1 - s1.min()) / (s1.max() - s1.min() + 1e-12) * 255
    s2_scaled = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-12) * 255
    return ssim(s1_scaled, s2_scaled)

def hybrid_pipeline(dates, texts):
    """
    Apply the hybrid allocation to a sequence of dates/texts and return a list
    of updated multivectors together with a pairwise similarity matrix of the
    underlying scalar signals (here we synthesize a dummy signal from the
    allocation scalar).
    """
    if len(dates) != len(texts):
        raise ValueError("dates and texts must have the same length")
    results = []
    scalar_series = []
    for d, txt in zip(dates, texts):
        out = hybrid_allocation(d, txt)
        results.append(out)
        scalar_series.append(out['alpha'])

    # Build a similarity matrix using SSIM on the scalar series treated as 1‑D signals
    sig = np.array(scalar_series, dtype=float)
    n = len(sig)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = compare_signals(sig[i:i+1], sig[j:j+1])  # SSIM of singletons yields 1.0
            sim_matrix[i, j] = sim_matrix[j, i] = sim
    return results, sim_matrix

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample dates (one week)
    sample_dates = [date(2026, 5, 24) + np.timedelta64(i, 'D') for i in range(7)]
    sample_dates = [d.astype('M8[D]').astype('O') if hasattr(d, 'astype') else d for d in sample_dates]  # ensure python date
    # Simple texts with varying keyword density
    sample_texts = [
        "The source was verified and the hash was logged.",
        "We need to plan the roadmap and schedule the test.",
        "Please wait until tomorrow before we continue.",
        "Evidence and proof are required for the audit.",
        "The checklist includes steps for deployment.",
        "Hold the release; a cool down period is necessary.",
        "Confirmed source and documented procedure."
    ]
    results, sim = hybrid_pipeline(sample_dates, sample_texts)

    for r in results:
        print(f"Date: {r['date']}, I={r['I']:.3f}, alpha={r['alpha']:.5f}, Updated MV={r['updated_mv']}")

    print("\nSimilarity matrix (should be all 1.0 for singleton SSIM):")
    print(sim)