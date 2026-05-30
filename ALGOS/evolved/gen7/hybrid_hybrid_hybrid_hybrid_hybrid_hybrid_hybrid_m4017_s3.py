# DARWIN HAMMER — match 4017, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_pherom_m96_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1471_s1.py (gen6)
# born: 2026-05-29T23:53:04Z

"""Hybrid Algorithm combining:
- Parent A: geometric algebra with Koopman operator, Shannon entropy, pheromone probabilities.
- Parent B: stylometry feature extraction and Hoeffding bound uncertainty evaluation.

Mathematical Bridge:
The multivector coefficients are evolved by a data‑driven Koopman operator, producing a
real‑valued state vector. This state is turned into a probability distribution whose
Shannon entropy quantifies uncertainty. The same entropy term modulates the influence
of stylometry‑derived feature vectors (computed from text corpora) when scoring
candidates. Finally, a Hoeffding bound is applied to the entropy‑weighted scores to
select the most reliable candidate. This creates a single pipeline that fuses linear
dynamics (Koopman), information theory (entropy), linguistic stylometry, and statistical
confidence (Hoeffding)."""

import math
import random
import sys
import pathlib
import numpy as np
from collections import Counter

# ----------------------------------------------------------------------
# Parent A – Geometric Algebra & Koopman
# ----------------------------------------------------------------------
def _blade_sign(indices: list) -> tuple:
    """Return (sorted_blade, sign) after bubble‑sorting index list."""
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
                del lst[j : j + 2]
                n -= 2
                continue
            j += 1
        i += 1
    return lst, sign


def _multiply_blades(blade_a: frozenset, blade_b: frozenset) -> tuple:
    """Multiply two basis blades, returning (result_blade, sign)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign


class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""

    def __init__(self, components: dict, n: int):
        # discard near‑zero components for stability
        self.components = {k: v for k, v in components.items() if abs(v) > 1e-15}
        self.n = int(n)

    def grade(self, k: int) -> "Multivector":
        return Multivector(
            {blade: coef for blade, coef in self.components.items() if len(blade) == k},
            self.n,
        )

    def scalar_part(self) -> float:
        return self.components.get(frozenset(), 0.0)

    def as_vector(self) -> np.ndarray:
        """Return a dense vector of length 2**n ordered by binary blade index."""
        size = 1 << self.n
        vec = np.zeros(size)
        for blade, coef in self.components.items():
            # binary index: sum 2**i for each basis index i
            idx = sum(1 << i for i in blade)
            vec[idx] = coef
        return vec

    @staticmethod
    def from_vector(vec: np.ndarray) -> "Multivector":
        n = int(math.log2(vec.size))
        comps = {}
        for idx, val in enumerate(vec):
            if abs(val) > 1e-15:
                # decode blade from binary index
                blade = frozenset(i for i in range(n) if (idx >> i) & 1)
                comps[blade] = val
        return Multivector(comps, n)


def koopman_matrix(X: np.ndarray, X_prime: np.ndarray) -> np.ndarray:
    """Least‑squares estimate of the Koopman operator K such that X_prime ≈ K @ X."""
    # Use Moore‑Penrose pseudoinverse for stability
    pinv = np.linalg.pinv(X)
    K = X_prime @ pinv
    return K


def apply_koopman(multivector: Multivector, K: np.ndarray) -> Multivector:
    """Evolve a multivector by the linear Koopman operator K."""
    vec = multivector.as_vector()
    evolved = K @ vec
    return Multivector.from_vector(evolved)


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability vector (base‑e)."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


# ----------------------------------------------------------------------
# Parent B – Stylometry & Hoeffding bound
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
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
        "no not never none neither cannot cant wont dont didnt isnt arent was wasnt werent".split()
    ),
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only".split()
    ),
}


def stylometry_features(text: str) -> np.ndarray:
    """Return a normalized feature vector derived from FUNCTION_CATS."""
    tokens = [tok.strip(".,!?;:()[]\"'").lower() for tok in text.split()]
    total = max(len(tokens), 1)
    counts = []
    for cat in sorted(FUNCTION_CATS.keys()):
        cat_words = FUNCTION_CATS[cat]
        cnt = sum(1 for t in tokens if t in cat_words)
        counts.append(cnt / total)
    return np.array(counts)


def hoeffding_bound(epsilon: float, n: int, R: float = 1.0) -> float:
    """Hoeffding bound for bounded random variable in [0,R]."""
    if n <= 0:
        return float("inf")
    return math.sqrt((R * R * math.log(1.0 / epsilon)) / (2.0 * n))


# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def hybrid_evolve_multivector(multivector: Multivector, X: np.ndarray, X_prime: np.ndarray) -> Multivector:
    """Evolve a multivector using a data‑driven Koopman operator."""
    K = koopman_matrix(X, X_prime)
    return apply_koopman(multivector, K)


def hybrid_entropy_from_state(multivector: Multivector) -> float:
    """Map multivector coefficients to a probability distribution and compute entropy."""
    vec = np.abs(multivector.as_vector())
    total = np.sum(vec) + 1e-12
    probs = vec / total
    return shannon_entropy(probs)


def hybrid_candidate_score(
    multivector_entropy: float,
    stylometry_vec: np.ndarray,
    n_samples: int,
    epsilon: float = 0.05,
) -> tuple[float, float]:
    """
    Combine entropy with stylometry to produce a score and its Hoeffding confidence.

    Returns (score, bound). Lower entropy (more certainty) amplifies the stylometry
    similarity score; the Hoeffding bound quantifies uncertainty given n_samples.
    """
    # Simple similarity: dot product with a uniform reference vector
    reference = np.ones_like(stylometry_vec) / stylometry_vec.size
    similarity = np.dot(stylometry_vec, reference)

    # Entropy weighting: higher certainty (lower entropy) -> larger weight
    weight = math.exp(-multivector_entropy)  # in (0,1]
    score = weight * similarity

    bound = hoeffding_bound(epsilon, n_samples)
    return score, bound


def hybrid_select_best_candidate(
    multivector: Multivector,
    X: np.ndarray,
    X_prime: np.ndarray,
    corpus: list[str],
) -> dict:
    """
    Full pipeline:
    1. Evolve the multivector via Koopman.
    2. Compute entropy of the evolved state.
    3. Extract stylometry vectors for each text.
    4. Score each candidate with entropy‑weighted similarity.
    5. Return the candidate with the highest lower‑confidence bound (score - bound).
    """
    evolved = hybrid_evolve_multivector(multivector, X, X_prime)
    entropy = hybrid_entropy_from_state(evolved)

    results = []
    for txt in corpus:
        feats = stylometry_features(txt)
        score, bound = hybrid_candidate_score(entropy, feats, n_samples=len(txt.split()))
        lower_conf = score - bound
        results.append(
            {
                "text": txt,
                "score": score,
                "bound": bound,
                "lower_confidence": lower_conf,
            }
        )

    best = max(results, key=lambda d: d["lower_confidence"])
    return {"best_candidate": best, "entropy": entropy, "evolved_multivector": evolved}


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Simple 2‑dimensional geometric algebra (Cl(2,0))
    mv = Multivector(
        {
            frozenset(): 1.0,               # scalar
            frozenset({0}): 0.5,            # e1
            frozenset({1}): -0.3,           # e2
            frozenset({0, 1}): 0.2,         # e1^e2
        },
        n=2,
    )

    # Random data matrices for Koopman estimation
    np.random.seed(42)
    X = np.random.randn(4, 100)          # 2**n = 4 basis components, 100 snapshots
    X_prime = np.random.randn(4, 100)

    corpus = [
        "I think therefore I am. The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore, and the shells she sells are surely seashells.",
        "Data science combines statistics, computer science, and domain expertise."
    ]

    result = hybrid_select_best_candidate(mv, X, X_prime, corpus)

    print("Entropy of evolved state:", result["entropy"])
    print("Best candidate text:", result["best_candidate"]["text"])
    print("Score:", result["best_candidate"]["score"])
    print("Hoeffding bound:", result["best_candidate"]["bound"])
    print("Lower‑confidence estimate:", result["best_candidate"]["lower_confidence"])