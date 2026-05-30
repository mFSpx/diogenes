# DARWIN HAMMER — match 2587, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_perceptual_de_m264_s2.py (gen4)
# born: 2026-05-29T23:43:08Z

"""Hybrid Algorithm: Fisher‑Sketch‑HH‑RBF Fusion

Parents:
- PARENT ALGORITHM A: Fisher information & Count‑Min Sketch (with Hodgkin‑Huxley energy perspective)
- PARENT ALGORITHM B: Regex feature extraction & Radial Basis Function surrogate model

Mathematical Bridge:
The regex feature counts are compressed with a Count‑Min Sketch.  
Fisher information is computed on the sketch buckets (treating each bucket index as a
“θ” in a Gaussian beam).  The aggregated Fisher score forms an external current
I_ext that drives the Hodgkin‑Huxley cable dynamics.  The resulting membrane
potential V(t) is fed to an RBF surrogate whose centers and weights are derived
from the same sketch, thus closing a loop where information (Fisher) shapes energy
(HH) which in turn modulates perception (RBF)."""

import math
import random
import sys
import hashlib
import pathlib
import re
import numpy as np

# ----------------------------------------------------------------------
# Regex feature extraction (Parent B)
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
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)

_REGEX_PATTERNS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
}


def extract_regex_counts(texts):
    """
    Count occurrences of each regex pattern across a list of strings.
    Returns a dict mapping pattern name → integer count.
    """
    counts = {name: 0 for name in _REGEX_PATTERNS}
    for txt in texts:
        for name, pat in _REGEX_PATTERNS.items():
            counts[name] += len(pat.findall(txt))
    return counts


# ----------------------------------------------------------------------
# Count‑Min Sketch (Parent A)
# ----------------------------------------------------------------------
def count_min_sketch(items, width=64, depth=4):
    """
    Simple Count‑Min Sketch.
    `items` should be an iterable of hashable objects.
    Returns a 2‑D list [depth][width] of integer counters.
    """
    table = [[0] * width for _ in range(depth)]
    for item in items:
        for d in range(depth):
            h = hashlib.sha256(f"{d}:{item}".encode()).hexdigest()
            idx = int(h, 16) % width
            table[d][idx] += 1
    return table


def sketch_from_counts(counts, width=64, depth=4):
    """
    Convert a dict of feature counts into a Count‑Min Sketch.
    Each (feature, count) pair is inserted `count` times.
    """
    items = []
    for feat, cnt in counts.items():
        items.extend([feat] * cnt)
    return count_min_sketch(items, width, depth)


# ----------------------------------------------------------------------
# Fisher information on sketch buckets (Parent A)
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    """Gaussian intensity I(θ) of a beam centred at *center* with *width*."""
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)


def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    """Fisher information for a single angle θ."""
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity


def fisher_from_sketch(sketch, width=64, depth=4):
    """
    Compute a Fisher information matrix of shape (depth, width) from a sketch.
    The bucket index plays the role of θ; the mean index of the sketch is used
    as the Gaussian centre, and a fixed width (≈σ) is chosen from the spread.
    """
    # Flatten to numpy for convenience
    arr = np.array(sketch, dtype=np.float64)
    # Global centre and width (simple statistics)
    flat = arr.ravel()
    center = flat.mean()
    std = flat.std() if flat.std() > 0 else 1.0
    fisher_mat = np.empty_like(arr)
    for d in range(depth):
        for w in range(width):
            fisher_mat[d, w] = fisher_score(w, center, std)
    return fisher_mat


# ----------------------------------------------------------------------
# Hodgkin‑Huxley cable model (energy side of Parent A)
# ----------------------------------------------------------------------
def init_hh_state(V_rest=-65.0):
    """Return a dict with initial HH state variables."""
    return {
        "V": V_rest,          # membrane potential (mV)
        "m": 0.05,            # Na activation
        "h": 0.6,             # Na inactivation
        "n": 0.32,            # K activation
    }


def hh_step(state, I_ext, dt=0.01):
    """
    Perform one integration step of the Hodgkin‑Huxley equations.
    `I_ext` is an external current (µA/cm²) derived from Fisher information.
    Returns the updated state dict.
    """
    V = state["V"]
    m = state["m"]
    h = state["h"]
    n = state["n"]

    # Constants
    C_m = 1.0  # µF/cm²
    g_Na = 120.0
    g_K = 36.0
    g_L = 0.3
    E_Na = 50.0
    E_K = -77.0
    E_L = -54.387

    # Alpha/Beta functions
    def alpha_n(V): return 0.01 * (V + 55.0) / (1.0 - math.exp(-(V + 55.0) / 10.0))
    def beta_n(V):  return 0.125 * math.exp(-(V + 65.0) / 80.0)
    def alpha_m(V): return 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))
    def beta_m(V):  return 4.0 * math.exp(-(V + 65.0) / 18.0)
    def alpha_h(V): return 0.07 * math.exp(-(V + 65.0) / 20.0)
    def beta_h(V):  return 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))

    # Gating variable derivatives
    dn = alpha_n(V) * (1.0 - n) - beta_n(V) * n
    dm = alpha_m(V) * (1.0 - m) - beta_m(V) * m
    dh = alpha_h(V) * (1.0 - h) - beta_h(V) * h

    # Currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)

    # Voltage derivative
    dV = (I_ext - I_Na - I_K - I_L) / C_m

    # Euler update
    state["V"] = V + dt * dV
    state["m"] = m + dt * dm
    state["h"] = h + dt * dh
    state["n"] = n + dt * dn
    return state


def fisher_to_current(fisher_mat, scale=0.1):
    """
    Reduce the Fisher matrix to a scalar external current.
    Simple approach: sum all entries, scale, and add a small random jitter.
    """
    total = fisher_mat.sum()
    jitter = random.uniform(-0.01, 0.01)
    return scale * total + jitter


# ----------------------------------------------------------------------
# Radial Basis Function surrogate (Parent B)
# ----------------------------------------------------------------------
def rbf_surrogate(V, sketch, sigma=5.0):
    """
    Compute an RBF output using the membrane potential V as the query point.
    Each sketch bucket defines a centre c = bucket index and a weight proportional
    to its count.  The result is a scalar that can be interpreted as a perceptual
    prediction.
    """
    sketch_arr = np.array(sketch, dtype=np.float64)
    # Normalise counts to obtain weights
    total_counts = sketch_arr.sum()
    if total_counts == 0:
        return 0.0
    weights = sketch_arr / total_counts

    depth, width = sketch_arr.shape
    # Create centre matrix: each bucket index repeated across depth
    centres = np.arange(width, dtype=np.float64)  # shape (width,)
    centres = np.tile(centres, (depth, 1))        # shape (depth, width)

    # RBF kernel
    diff = V - centres
    kernel = np.exp(-0.5 * (diff ** 2) / (sigma ** 2))

    # Weighted sum over all buckets
    output = np.sum(weights * kernel)
    return float(output)


# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_iteration(texts, state, dt=0.01):
    """
    One hybrid iteration:
      1. Extract regex feature counts from `texts`.
      2. Build a Count‑Min Sketch of those counts.
      3. Compute Fisher information on the sketch.
      4. Convert Fisher to an external current and step HH dynamics.
      5. Feed the new membrane potential into an RBF surrogate.
    Returns (new_state, rbf_output, fisher_matrix).
    """
    # 1. Feature extraction
    counts = extract_regex_counts(texts)

    # 2. Sketch
    sketch = sketch_from_counts(counts, width=64, depth=4)

    # 3. Fisher
    fisher_mat = fisher_from_sketch(sketch, width=64, depth=4)

    # 4. External current & HH update
    I_ext = fisher_to_current(fisher_mat, scale=0.05)
    new_state = hh_step(state.copy(), I_ext, dt)

    # 5. RBF surrogate
    rbf_out = rbf_surrogate(new_state["V"], sketch, sigma=10.0)

    return new_state, rbf_out, fisher_mat


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The evidence was verified and the source was logged.",
        "We need a plan and a checklist for the next phase.",
        "Please wait until tomorrow before we proceed.",
        "Contact the doctor for support and advice.",
        "Maintain the boundary and do not share private data."
    ]

    # Initialise Hodgkin‑Huxley state
    hh_state = init_hh_state()

    # Run a few hybrid iterations
    for step in range(5):
        hh_state, rbf_val, fisher = hybrid_iteration(sample_texts, hh_state, dt=0.01)
        print(f"Step {step+1:02d} | V={hh_state['V']:+.2f} mV | RBF={rbf_val:.4f}")

    sys.exit(0)