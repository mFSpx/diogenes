# DARWIN HAMMER — match 3263, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2541_s4.py (gen6)
# parent_b: hybrid_hybrid_hybrid_rbf_su_hybrid_hybrid_caputo_m1013_s0.py (gen4)
# born: 2026-05-29T23:48:49Z

"""Hybrid Algorithm Fusion of Parent A and Parent B
===================================================

This module fuses the core topologies of the two parent algorithms:

* **Parent A** – provides a feature‑extraction pipeline for linguistic
  function‑category frequencies, a temperature‑dependent developmental rate
  based on the Schoolfield‑Rollinson poikilotherm model, and a NLMS‑style
  adaptive weight update.

* **Parent B** – supplies a Radial Basis Function (RBF) surrogate that maps a
  feature vector to a scalar, and a Caputo fractional derivative that
  weights the influence of a geometric (rotor‑like) update.

**Mathematical Bridge**

The bridge is the *feature vector* produced by Parent A.  
It is fed to the RBF surrogate (Parent B) to obtain a scalar ``γ`` that
modulates the temperature ``T`` used in the Schoolfield rate.  The same
scalar ``γ`` also scales the order ``α`` of the Caputo fractional derivative,
which in turn weights the NLMS learning‑rate term.  Consequently the weight
update simultaneously respects:

* temperature‑dependent developmental dynamics,
* long‑range memory via the fractional derivative, and
* data‑driven modulation through the RBF surrogate.

The three key functions below illustrate this hybrid operation.
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Set, Hashable, Mapping

import numpy as np

# ----------------------------------------------------------------------
# Parent A – linguistic function‑category extraction
# ----------------------------------------------------------------------
FUNCTION_CATS: Dict[str, Set[str]] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot cant wont dont didnt isnt arent wasnt werent".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even".split()
    ),
}

def extract_function_category_frequencies(tokens: List[str]) -> Dict[str, float]:
    """Return normalized frequencies of each function category in *tokens*."""
    counts = {cat: 0 for cat in FUNCTION_CATS}
    total = 0
    for token in tokens:
        token_l = token.lower()
        matched = False
        for cat, vocab in FUNCTION_CATS.items():
            if token_l in vocab:
                counts[cat] += 1
                matched = True
                break
        if matched:
            total += 1
    if total == 0:
        return {cat: 0.0 for cat in FUNCTION_CATS}
    return {cat: cnt / total for cat, cnt in counts.items()}

def text_to_feature_vector(text: str) -> np.ndarray:
    """Simple tokenisation + category frequencies → 1‑D feature vector."""
    tokens = text.split()
    freq_dict = extract_function_category_frequencies(tokens)
    # Preserve a deterministic order
    ordered = [freq_dict[cat] for cat in sorted(FUNCTION_CATS.keys())]
    return np.array(ordered, dtype=float)

# ----------------------------------------------------------------------
# Parent B – RBF surrogate
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must have same dimension")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

@dataclass(frozen=True)
class RBFSurrogate:
    """RBF surrogate with fixed centers, weights and shape parameter."""
    centers: List[Tuple[float, ...]]
    weights: List[float]
    epsilon: float = 1.0

    def predict(self, x: Sequence[float]) -> float:
        if len(self.centers) != len(self.weights):
            raise ValueError("centers and weights length mismatch")
        phi = 0.0
        for c, w in zip(self.centers, self.weights):
            r = euclidean(x, c)
            phi += w * gaussian(r, self.epsilon)
        return phi

# ----------------------------------------------------------------------
# Temperature‑dependent developmental rate (Schoolfield‑Rollinson)
# ----------------------------------------------------------------------
def schoolfield_rate(T: float, params: Tuple[float, float, float, float, float]) -> float:
    """
    Compute the developmental rate at temperature *T*.

    Parameters
    ----------
    T : float
        Absolute temperature (Kelvin).
    params : (E, Eh, Th, El, Tl)
        *E*  – activation energy,
        *Eh* – high‑temperature deactivation energy,
        *Th* – high‑temperature midpoint,
        *El* – low‑temperature deactivation energy,
        *Tl* – low‑temperature midpoint.

    Returns
    -------
    float
        Rate value.
    """
    E, Eh, Th, El, Tl = params
    k = 8.617e-5  # Boltzmann constant (eV·K⁻¹)
    num = math.exp(-E / (k * T))
    den = 1 + math.exp(Eh / k * (1 / Th - 1 / T)) + math.exp(El / k * (1 / T - 1 / Tl))
    return num / den

# ----------------------------------------------------------------------
# Caputo fractional derivative (Grünwald‑Letnikov approximation)
# ----------------------------------------------------------------------
def caputo_fractional_derivative(
    f: Sequence[float], dt: float, alpha: float
) -> np.ndarray:
    """
    Approximate the Caputo fractional derivative of order *alpha* for a
    discrete signal *f* using the Grünwald‑Letnikov scheme.

    Parameters
    ----------
    f : sequence of float
        Signal values sampled uniformly with spacing *dt*.
    dt : float
        Sampling interval.
    alpha : float
        Fractional order (0 < alpha < 1).

    Returns
    -------
    np.ndarray
        Approximation of D^alpha f at each sample (same length as *f*).
    """
    n = len(f)
    coeffs = np.zeros(n)
    coeffs[0] = 1.0
    for k in range(1, n):
        coeffs[k] = coeffs[k - 1] * (alpha - k + 1) / k
    coeffs = coeffs * ((-1) ** np.arange(n))
    # Convolution implements the sum ∑_{k=0}^{i} coeff_k * f_{i-k}
    conv = np.convolve(f, coeffs)[:n]
    return conv / (dt ** alpha)

# ----------------------------------------------------------------------
# Hybrid NLMS‑style adaptive update that incorporates the three bridges
# ----------------------------------------------------------------------
def hybrid_nlms_update(
    w: np.ndarray,
    x: np.ndarray,
    d: float,
    T: float,
    surrogate: RBFSurrogate,
    school_params: Tuple[float, float, float, float, float],
    dt: float,
    past_errors: List[float],
) -> Tuple[np.ndarray, float]:
    """
    Perform a single hybrid NLMS weight update.

    *w*          – current weight vector (shape (d,))
    *x*          – input feature vector (shape (d,))
    *d*          – desired scalar output.
    *T*          – ambient temperature (Kelvin).
    *surrogate*  – RBF surrogate that maps *x* → scalar γ.
    *school_params* – parameters for the Schoolfield rate.
    *dt*         – time step for the fractional derivative.
    *past_errors*– list of previous errors (used for fractional derivative).

    Returns
    -------
    (w_new, e) – updated weight vector and current error.
    """
    # 1. RBF surrogate provides a scaling factor γ
    gamma = surrogate.predict(x.tolist())

    # 2. Temperature‑dependent learning‑rate factor μ_T
    mu_T = schoolfield_rate(T, school_params) * gamma

    # 3. Compute current error
    y = np.dot(w, x)
    e = d - y

    # 4. Fractional derivative weighting of the error history
    err_series = np.array(past_errors + [e])
    D_alpha_err = caputo_fractional_derivative(err_series, dt, alpha=0.5 * gamma)  # α scaled by γ
    weight_factor = D_alpha_err[-1]  # most recent fractional derivative value

    # 5. NLMS update with combined scaling
    eps = 1e-8
    norm_x2 = np.dot(x, x) + eps
    mu = mu_T * weight_factor
    w_new = w + (mu * e * x) / norm_x2

    return w_new, e

# ----------------------------------------------------------------------
# High‑level hybrid model encapsulation
# ----------------------------------------------------------------------
class HybridModel:
    """
    End‑to‑end hybrid system:

    1. Convert raw text to a feature vector.
    2. Use the RBF surrogate to obtain γ.
    3. Compute a temperature‑dependent rate.
    4. Update adaptive weights with NLMS, fractional‑derivative‑weighted learning.
    """

    def __init__(
        self,
        surrogate: RBFSurrogate,
        school_params: Tuple[float, float, float, float, float],
        init_temp: float = 298.15,
        dt: float = 0.1,
    ):
        self.surrogate = surrogate
        self.school_params = school_params
        self.T = init_temp
        self.dt = dt
        self.w = np.zeros(len(FUNCTION_CATS))  # weight vector size matches feature dim
        self.past_errors: List[float] = []

    def step(self, text: str, desired: float) -> float:
        """
        Process *text*, perform a hybrid update, and return the prediction error.
        """
        x = text_to_feature_vector(text)
        self.w, e = hybrid_nlms_update(
            w=self.w,
            x=x,
            d=desired,
            T=self.T,
            surrogate=self.surrogate,
            school_params=self.school_params,
            dt=self.dt,
            past_errors=self.past_errors,
        )
        # Update temperature (simple exponential drift toward a reference)
        self.T = 0.99 * self.T + 0.01 * 310.0  # drift toward 310 K
        self.past_errors.append(e)
        if len(self.past_errors) > 50:
            self.past_errors.pop(0)
        return e

    def predict(self, text: str) -> float:
        """Return the current model output for *text*."""
        x = text_to_feature_vector(text)
        return float(np.dot(self.w, x))

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Construct a tiny RBF surrogate with random centers/weights
    dim = len(FUNCTION_CATS)
    rng = np.random.default_rng(42)
    centers = [tuple(rng.random(dim)) for _ in range(5)]
    weights = list(rng.random(5) - 0.5)  # allow negative influence
    surrogate = RBFSurrogate(centers=centers, weights=weights, epsilon=2.0)

    # Schoolfield parameters (E, Eh, Th, El, Tl) – arbitrary plausible values
    school_params = (0.65, 2.5, 310.0, 1.5, 280.0)

    model = HybridModel(surrogate=surrogate, school_params=school_params)

    # Dummy training loop
    samples = [
        ("I love the quick brown fox", 0.8),
        ("You should not go there", -0.3),
        ("They are running quickly", 0.5),
        ("No one can stop the rain", -0.6),
    ]

    for epoch in range(3):
        total_err = 0.0
        for txt, target in samples:
            err = model.step(txt, target)
            total_err += err ** 2
        print(f"Epoch {epoch+1}, MSE={total_err/len(samples):.4f}")

    # Final prediction demo
    test_txt = "She will not be late"
    pred = model.predict(test_txt)
    print(f"Prediction for \"{test_txt}\": {pred:.4f}")