# DARWIN HAMMER — match 2587, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s2.py (gen4)
# born: 2026-05-29T23:43:08Z

"""Hybrid Algorithm integrating:

- Parent A: Fisher information, Gaussian beam, Count‑Min Sketch, and Hodgkin‑Huxley (HH) cable dynamics.
- Parent B: Regex‑based feature extraction and Radial Basis Function (RBF) surrogate model.

Mathematical Bridge
------------------
Feature frequencies extracted via regex are compressed with a Count‑Min Sketch.
Each sketch bucket supplies a proxy count θ.  Treating each bucket as a
Gaussian‑beam measurement, we compute a Fisher information score
I(θ) = (∂ℓ/∂θ)² / ℓ, where ℓ is the beam intensity.  The resulting Fisher
scores form a diagonal weighting matrix **W** that modulates the RBF
kernel width σ and the external current I_ext injected into the HH
membrane equation:

    I_ext = Σ_j w_j · exp(−‖x−c_j‖²_{W} / (2σ²))

where ‖·‖²_{W} = (·)ᵀ W (·).  Thus the information‑theoretic
character of the data reshapes the energy landscape of the neuronal
cable model, providing a unified learning‑dynamics system.
"""

import numpy as np
import re
import math
import random
import sys
import pathlib
import hashlib

# ----------------------------------------------------------------------
# Parent B – Regex feature extraction
# ----------------------------------------------------------------------
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

REGEX_FEATURES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
}


def extract_features(text: str) -> dict:
    """
    Count occurrences of each regex category in *text*.
    Returns a dict mapping feature name → count (int).
    """
    counts = {}
    for name, pattern in REGEX_FEATURES.items():
        counts[name] = len(pattern.findall(text))
    return counts


# ----------------------------------------------------------------------
# Parent A – Count‑Min Sketch & Fisher information
# ----------------------------------------------------------------------
def count_min_sketch(items, width=64, depth=4):
    """
    Simple Count‑Min Sketch.
    *items* is an iterable of hashable objects.
    Returns a list of *depth* rows, each a list of *width* counters.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) with given centre and width."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single scalar measurement."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def fisher_weights_from_sketch(sketch, width=64, depth=4):
    """
    Convert a Count‑Min Sketch into a diagonal Fisher‑information weight matrix.
    For each bucket we treat the count as θ, the bucket index as the centre,
    and a fixed beam width.  The resulting scores are averaged over depth
    to obtain one weight per feature dimension.
    """
    # Parameters for the Gaussian beam
    beam_width = 5.0
    # Accumulate scores per depth‑averaged bucket index
    scores = []
    for d in range(depth):
        row = sketch[d]
        # centre is the bucket index (0…width‑1)
        row_scores = [
            fisher_score(theta=count, center=idx, width=beam_width)
            for idx, count in enumerate(row)
        ]
        scores.append(row_scores)
    # Average over depth → shape (width,)
    avg_scores = np.mean(scores, axis=0)
    # Build a diagonal matrix (as a 1‑D array for efficiency)
    return avg_scores + 1e-6  # avoid exact zeros


# ----------------------------------------------------------------------
# RBF surrogate model (Parent B) modulated by Fisher weights
# ----------------------------------------------------------------------
def rbf_kernel(x, c, sigma, w_diag):
    """
    Weighted Gaussian RBF kernel.
    x, c : 1‑D arrays of the same length (feature space).
    sigma : scalar bandwidth.
    w_diag : 1‑D array of diagonal Fisher weights.
    """
    diff = x - c
    weighted_norm = np.dot(w_diag * diff, diff)  # (x-c)^T W (x-c)
    return math.exp(-weighted_norm / (2.0 * sigma * sigma))


def rbf_predict(x, centers, weights, sigma, w_diag):
    """
    Compute RBF surrogate output for input vector *x*.
    *centers* shape (m, d), *weights* shape (m,).
    Returns a scalar.
    """
    out = 0.0
    for c, w in zip(centers, weights):
        out += w * rbf_kernel(x, c, sigma, w_diag)
    return out


# ----------------------------------------------------------------------
# Hodgkin‑Huxley cable dynamics (Parent A)
# ----------------------------------------------------------------------
# Physical constants
Cm = 1.0  # µF/cm²
gNa = 120.0
gK = 36.0
gL = 0.3
ENa = 50.0
EK = -77.0
EL = -54.387


def alpha_n(V):
    return 0.01 * (V + 55) / (1 - math.exp(-(V + 55) / 10))


def beta_n(V):
    return 0.125 * math.exp(-(V + 65) / 80)


def alpha_m(V):
    return 0.1 * (V + 40) / (1 - math.exp(-(V + 40) / 10))


def beta_m(V):
    return 4.0 * math.exp(-(V + 65) / 18)


def alpha_h(V):
    return 0.07 * math.exp(-(V + 65) / 20)


def beta_h(V):
    return 1.0 / (1 + math.exp(-(V + 35) / 10))


def hh_step(V, m, h, n, I_ext, dt):
    """
    Single forward Euler step of the Hodgkin‑Huxley equations.
    Returns updated (V, m, h, n).
    """
    # Ionic currents
    INa = gNa * (m ** 3) * h * (V - ENa)
    IK = gK * (n ** 4) * (V - EK)
    IL = gL * (V - EL)

    # Membrane voltage derivative
    dV = (I_ext - INa - IK - IL) / Cm

    # Gating variable derivatives
    dm = alpha_m(V) * (1 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1 - h) - beta_h(V) * h
    dn = alpha_n(V) * (1 - n) - beta_n(V) * n

    # Euler update
    V_new = V + dt * dV
    m_new = m + dt * dm
    h_new = h + dt * dh
    n_new = n + dt * dn

    return V_new, m_new, h_new, n_new


# ----------------------------------------------------------------------
# Hybrid operation combining all pieces
# ----------------------------------------------------------------------
def hybrid_update(text, V, m, h, n, dt, rbf_state):
    """
    Perform one hybrid iteration:

    1. Extract regex feature counts from *text*.
    2. Insert counts into a Count‑Min Sketch.
    3. Derive Fisher diagonal weights from the sketch.
    4. Build a feature vector *x* (normalized counts).
    5. Compute RBF surrogate output using Fisher‑weighted kernel;
       treat the output as external current I_ext.
    6. Advance Hodgkin‑Huxley state with I_ext.

    *rbf_state* is a dict containing RBF centers, weights, sigma.
    Returns updated (V, m, h, n) and the unchanged *rbf_state*.
    """
    # 1. Feature extraction
    counts_dict = extract_features(text)
    # 2. Sketch (use feature names as items, repeated count times)
    items = []
    for name, cnt in counts_dict.items():
        items.extend([name] * cnt)
    sketch = count_min_sketch(items, width=64, depth=4)

    # 3. Fisher weights
    w_diag = fisher_weights_from_sketch(sketch, width=64, depth=4)

    # 4. Feature vector (order must match RBF dimension)
    feature_names = list(REGEX_FEATURES.keys())
    raw_vec = np.array([counts_dict[name] for name in feature_names], dtype=np.float64)
    # Normalise to avoid large magnitudes
    if np.linalg.norm(raw_vec) > 0:
        x = raw_vec / np.linalg.norm(raw_vec)
    else:
        x = raw_vec

    # 5. RBF surrogate → external current
    centers = rbf_state["centers"]
    weights = rbf_state["weights"]
    sigma = rbf_state["sigma"]
    I_ext = rbf_predict(x, centers, weights, sigma, w_diag)

    # 6. HH dynamics
    V, m, h, n = hh_step(V, m, h, n, I_ext, dt)

    return V, m, h, n, rbf_state


def initialise_rbf_state(dim, n_centers=8, sigma=1.0):
    """
    Initialise RBF surrogate parameters.
    *dim* : dimensionality of the feature space.
    Returns a dict with keys 'centers', 'weights', 'sigma'.
    """
    centers = np.random.randn(n_centers, dim)
    weights = np.random.randn(n_centers)
    return {"centers": centers, "weights": weights, "sigma": sigma}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample text containing various keywords
    sample_text = """
    The evidence was verified and the source was documented.
    We need to plan the next steps and create a checklist.
    Please wait for the review before proceeding.
    If you need support, call a friend or a therapist.
    """

    # Initialise Hodgkin‑Huxley state
    V = -65.0  # mV
    m = 0.052
    h = 0.596
    n = 0.317

    dt = 0.01  # ms
    steps = 100

    # Initialise RBF surrogate
    rbf_state = initialise_rbf_state(dim=len(REGEX_FEATURES), n_centers=12, sigma=0.5)

    # Run hybrid dynamics
    for _ in range(steps):
        V, m, h, n, rbf_state = hybrid_update(
            sample_text, V, m, h, n, dt, rbf_state
        )

    print(f"Final membrane potential after {steps} steps: {V:.3f} mV")
    print(f"Gating variables: m={m:.4f}, h={h:.4f}, n={n:.4f}")