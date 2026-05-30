# DARWIN HAMMER — match 3523, survivor 1
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s1.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1330_s0.py (gen6)
# born: 2026-05-29T23:50:37Z

import numpy as np
import re
from collections import Counter
from datetime import date
from typing import List, Tuple, Dict

# ----------------------------------------------------------------------
# Global constants – tuned for the hybrid dynamics
# ----------------------------------------------------------------------
ALPHA = 0.6          # Store inflow threshold for success
BETA = 0.4           # Store outflow coefficient
DT = 1.0             # Time step for store dynamics
ETA0 = 0.01          # Base learning rate for NLMS
MU = 0.1             # Normalized step size for NLMS (0 < MU <= 1)
EPS = 1e-8           # Regularisation term for NLMS denominator
K1 = 0.01            # SSIM constant (luminance)
K2 = 0.03            # SSIM constant (contrast)
L = 255.0            # Dynamic range for SSIM (assume 8‑bit)

# ----------------------------------------------------------------------
# Feature handling – the “semantic fingerprint” of a text
# ----------------------------------------------------------------------
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

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.float64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.float64)

# Compile a regex that matches any of the feature keywords (case‑insensitive)
_EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|"
    r"hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)

def _extract_feature_vector(text: str) -> np.ndarray:
    """
    Produce a 9‑dimensional feature vector for *text*.
    Each dimension corresponds to the count of the associated keyword list
    (case‑insensitive) in the text.
    """
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    counter = Counter(tokens)

    vec = np.zeros(len(_FEATURE_ORDER), dtype=np.float64)
    for idx, feat in enumerate(_FEATURE_ORDER):
        vec[idx] = counter[feat]
    return vec

# ----------------------------------------------------------------------
# SSIM – structural similarity index for 1‑D feature vectors
# ----------------------------------------------------------------------
def _ssim_1d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the SSIM index between two 1‑D signals *x* and *y*.
    The implementation follows the standard SSIM definition but
    operates on a single dimension (the feature vector).
    """
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    sigma_x = np.var(x)
    sigma_y = np.var(y)
    sigma_xy = np.mean((x - mu_x) * (y - mu_y))

    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)

    return float(numerator / (denominator + EPS))

def hybrid_ssim_decision_hygiene(text: str, reference_texts: List[str]) -> float:
    """
    Return the average SSIM similarity between *text* and each reference.
    The similarity is in [0, 1] where 1 means identical feature composition.
    """
    if not reference_texts:
        return 0.0

    vec_main = _extract_feature_vector(text)
    scores = [_ssim_1d(vec_main, _extract_feature_vector(ref)) for ref in reference_texts]
    return float(np.mean(scores))

# ----------------------------------------------------------------------
# NLMS adaptive filter – learns a graph weight from similarity history
# ----------------------------------------------------------------------
def nlms_adaptive_filtering(
    similarity: float,
    prev_weight: float,
    prev_input: float,
    learning_rate: float = ETA0,
) -> Tuple[float, float]:
    """
    Perform one NLMS update.

    Parameters
    ----------
    similarity : float
        Current similarity (treated as the input sample *x[n]*).
    prev_weight : float
        Weight estimate from the previous iteration (*w[n‑1]*).
    prev_input : float
        Input sample used in the previous iteration (*x[n‑1]*). Needed for the
        denominator *x[n]·x[n] + ε*.
    learning_rate : float
        Base learning rate (μ). The effective step size is μ / (‖x‖² + ε).

    Returns
    -------
    new_weight : float
        Updated weight estimate.
    new_input : float
        The current input (to be stored for the next iteration).
    """
    # Desired response – we want the weight to grow when similarity is high
    d = similarity  # treat similarity itself as the desired output

    # NLMS update
    x = similarity
    denom = x * x + EPS
    error = d - prev_weight * x
    new_weight = prev_weight + (learning_rate / denom) * x * error

    # Clip weight to a reasonable range to avoid divergence
    new_weight = max(0.0, min(new_weight, 10.0))

    return new_weight, x

# ----------------------------------------------------------------------
# Core objects – store, endpoint, morphology
# ----------------------------------------------------------------------
class EndpointCircuitBreaker:
    """Tracks consecutive failures and opens/closes the circuit."""
    def __init__(self, failure_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.failures = 0
        self.open = False

    def record_success(self) -> None:
        self.failures = 0
        self.open = False

    def record_failure(self) -> None:
        self.failures += 1
        self.open = self.failures >= self.failure_threshold

    def allow(self) -> bool:
        """True if the circuit is closed (endpoint may receive work)."""
        return not self.open

    def failure_rate(self) -> float:
        """Normalized failure rate in [0, 1]."""
        return min(self.failures / self.failure_threshold, 1.0)


class Morphology:
    """Geometric description of an endpoint – influences store dynamics."""
    def __init__(self, length: float, width: float, height: float, mass: float):
        self.length = length
        self.width = width
        self.height = height
        self.mass = mass

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def density(self) -> float:
        return self.mass / max(self.volume, EPS)


class StoreState:
    """Simple accumulator representing the “health” of an endpoint."""
    def __init__(self, level: float = 0.0):
        self.level = level


class HybridStore:
    """
    Combines SSIM‑based similarity, NLMS‑learned graph weight and morphology
    into a single store dynamics model.
    """
    def __init__(
        self,
        store: StoreState,
        endpoint: EndpointCircuitBreaker,
        morphology: Morphology,
    ):
        self.store = store
        self.endpoint = endpoint
        self.morphology = morphology
        self._prev_weight = 0.5          # initial graph weight
        self._prev_input = 0.0           # previous similarity (for NLMS denominator)

    def update(self, similarity: float) -> None:
        """
        Perform one hybrid update step:

        1. Adapt the graph weight with NLMS using the current similarity.
        2. Compute a morphology‑scaled inflow term.
        3. Update the store level.
        4. Adjust the circuit‑breaker based on the new level.
        """
        # 1️⃣ NLMS weight adaptation
        self._prev_weight, self._prev_input = nlms_adaptive_filtering(
            similarity,
            self._prev_weight,
            self._prev_input,
            learning_rate=MU,
        )

        # 2️⃣ Morphology‑aware inflow – denser (heavier) endpoints receive
        #    proportionally less inflow, mimicking “inertia”.
        inflow = self._prev_weight * similarity * (1.0 / (1.0 + self.morphology.density))

        # 3️⃣ Store dynamics (Euler integration)
        delta = (inflow - BETA) * DT
        self.store.level = max(0.0, self.store.level + delta)

        # 4️⃣ Circuit‑breaker logic
        if self.store.level > ALPHA:
            self.endpoint.record_success()
        else:
            self.endpoint.record_failure()


def hybrid_store_update(
    similarity: float,
    store: StoreState,
    endpoint: EndpointCircuitBreaker,
    morphology: Morphology,
) -> None:
    """
    Convenience wrapper that creates a temporary HybridStore instance,
    performs the update, and discards the object.  This mirrors the original
    API while keeping the internal state encapsulated.
    """
    hs = HybridStore(store, endpoint, morphology)
    hs.update(similarity)


# ----------------------------------------------------------------------
# Utility – day‑of‑week cyclic index (kept from the original code)
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return a cyclic day index in [0, 6] (0 = Monday, 6 = Sunday)."""
    return (date(year, month, day).weekday() + 1) % 7


# ----------------------------------------------------------------------
# Smoke test – demonstrates the improved pipeline
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example texts
    text = "Evidence of planning and support was recorded in the outcome."
    reference_texts = [
        "The evidence confirms the planning and support for the outcome.",
        "No evidence of risk or scarcity was found in the report.",
    ]

    # Compute similarity via the refined SSIM pipeline
    sim = hybrid_ssim_decision_hygiene(text, reference_texts)

    # Initialise system components
    store = StoreState()
    endpoint = EndpointCircuitBreaker()
    morphology = Morphology(length=10.0, width=5.0, height=2.0, mass=1.0)

    # Perform a single hybrid update
    hybrid_store_update(sim, store, endpoint, morphology)

    # Output a concise status report
    print(f"Similarity: {sim:.4f}")
    print(f"Store level: {store.level:.4f}")
    print(f"Circuit open? {'No' if endpoint.allow() else 'Yes'}")
    print(f"Failure rate: {endpoint.failure_rate():.2f}")