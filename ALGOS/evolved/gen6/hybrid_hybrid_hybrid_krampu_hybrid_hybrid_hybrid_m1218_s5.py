# DARWIN HAMMER — match 1218, survivor 5
# gen: 6
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m932_s2.py (gen5)
# born: 2026-05-29T23:34:29Z

"""
HybridBanditRBF – Fusion of:
    * hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m524_s0.py (bandit router + Gaussian kernel)
    * hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m932_s2.py (geometric‑algebra Multivector)

Mathematical bridge
-------------------
The Gaussian kernel in the original bandit algorithm depends on the Euclidean distance 
‖x‑y‖² between two context vectors *x* and *y*.  In the geometric‑algebra parent the 
context is naturally represented as a **Multivector** *X*.  The squared norm of a 
multivector is  

    ‖X‖² = ⟨X·X̃⟩₀ ,

where *̃* denotes reversion and ⟨·⟩₀ extracts the scalar part.  Because the basis
vectors are orthonormal, this scalar equals the sum of squared coefficients – i.e.
the ordinary Euclidean norm of the coefficient vector.  Therefore

    ‖x‑y‖²  ≡  ‖X‑Y‖² .

We replace the plain NumPy context vectors by multivectors, compute the Gaussian
kernel with the multivector norm, and use the resulting similarity matrix as a
weighting factor for reward aggregation in the contextual bandit.  This yields a
single unified system where geometric‑algebraic structure informs stochastic
pruning and action selection.
"""

import math
import random
import sys
import pathlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

# ----------------------------------------------------------------------
# Data structures (from Parent A)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str


@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float


# ----------------------------------------------------------------------
# Multivector utilities (from Parent B)
# ----------------------------------------------------------------------
class Multivector:
    """
    Sparse multivector for a Euclidean Clifford algebra G(n).

    * ``components`` maps a frozenset of basis indices to a scalar coefficient.
    * The empty frozenset represents the scalar (grade‑0) part.
    """

    def __init__(self, components: Dict[frozenset, float], n: int):
        self.n = int(n)
        # discard near‑zero entries to keep the representation sparse
        self.components = {
            k: float(v) for k, v in components.items() if abs(v) > 1e-15
        }

    # ------------------------------------------------------------------
    # Algebraic helpers
    # ------------------------------------------------------------------
    def __add__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) + coef
        return Multivector(result, self.n)

    def __sub__(self, other: "Multivector") -> "Multivector":
        result = dict(self.components)
        for blade, coef in other.components.items():
            result[blade] = result.get(blade, 0.0) - coef
        return Multivector(result, self.n)

    def __mul__(self, scalar: float) -> "Multivector":
        return Multivector({b: c * scalar for b, c in self.components.items()}, self.n)

    __rmul__ = __mul__

    def reverse(self) -> "Multivector":
        """Reversion flips the sign of blades of grade 2,3,6,7,… (grade k → (-1)^{k(k‑1)/2})"""
        result = {}
        for blade, coef in self.components.items():
            k = len(blade)
            sign = -1 if (k * (k - 1) // 2) % 2 else 1
            result[blade] = coef * sign
        return Multivector(result, self.n)

    def geometric_product(self, other: "Multivector") -> "Multivector":
        """Very naive implementation – sufficient for norm calculations."""
        result = {}
        for b1, c1 in self.components.items():
            for b2, c2 in other.components.items():
                # concatenate basis lists and sort, counting swaps for sign
                combined = list(b1) + list(b2)
                sign = 1
                # bubble sort to canonical order while counting swaps
                for i in range(len(combined)):
                    for j in range(i + 1, len(combined)):
                        if combined[i] > combined[j]:
                            combined[i], combined[j] = combined[j], combined[i]
                            sign *= -1
                # cancel duplicate indices (e_i * e_i = 1)
                i = 0
                while i < len(combined) - 1:
                    if combined[i] == combined[i + 1]:
                        del combined[i : i + 2]
                        # two equal basis vectors square to +1, no sign change
                        i = max(i - 1, 0)
                    else:
                        i += 1
                blade = frozenset(combined)
                result[blade] = result.get(blade, 0.0) + sign * c1 * c2
        return Multivector(result, self.n)

    def scalar_part(self) -> float:
        """Extract grade‑0 (scalar) coefficient."""
        return self.components.get(frozenset(), 0.0)

    def norm_squared(self) -> float:
        """‖X‖² = ⟨X·X̃⟩₀ . For Euclidean metric this equals sum of squares of coefficients."""
        # Using the definition directly is cheap because the basis is orthonormal.
        return sum(c * c for c in self.components.values())

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        if not self.components:
            return "Multivector(0)"
        terms = []
        for blade, coef in sorted(self.components.items()):
            label = "1" if not blade else "e" + "".join(str(i) for i in sorted(blade))
            terms.append(f"{coef:+.6g}*{label}")
        return "Multivector(" + " ".join(terms) + ")"


def context_dict_to_mv(context: Dict[int, float], n: int) -> Multivector:
    """
    Convert a plain dictionary ``{basis_index: value}`` into a multivector.
    The basis indices must be in ``[0, n-1]``.
    """
    comps = {frozenset({i}): float(v) for i, v in context.items() if v != 0.0}
    return Multivector(comps, n)


# ----------------------------------------------------------------------
# Kernel utilities (from Parent A, rewritten for multivectors)
# ----------------------------------------------------------------------
def gaussian_kernel(r2: float, epsilon: float = 1.0) -> float:
    """
    Gaussian kernel based on squared distance r2.
    """
    return math.exp(-((epsilon ** 2) * r2))


def rbf_similarity(mv1: Multivector, mv2: Multivector, epsilon: float = 1.0) -> float:
    """
    Compute Gaussian RBF similarity between two multivectors using their Euclidean
    norm (the sum of squared coefficients).
    """
    diff = mv1 - mv2
    r2 = diff.norm_squared()
    return gaussian_kernel(r2, epsilon)


# ----------------------------------------------------------------------
# Hybrid bandit that uses multivector‑based RBF weighting
# ----------------------------------------------------------------------
class HybridBanditRBF:
    """
    Contextual bandit where each stored experience consists of:
        (action_id, context_multivector, reward)

    The expected reward for a new context is estimated as a kernel‑weighted
    average of past rewards for the same action.
    """

    def __init__(self, epsilon: float = 1.0, confidence_alpha: float = 1.0):
        self.epsilon = epsilon
        self.confidence_alpha = confidence_alpha
        # Mapping: action_id -> list of (Multivector, reward)
        self._history: Dict[str, List[Tuple[Multivector, float]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(self, updates: List[BanditUpdate], context_store: Dict[str, Multivector]) -> None:
        """
        Incorporate a batch of BanditUpdate objects.
        ``context_store`` maps ``context_id`` to its multivector representation.
        """
        for u in updates:
            ctx_mv = context_store.get(u.context_id)
            if ctx_mv is None:
                continue  # silently ignore unknown contexts
            self._history.setdefault(u.action_id, []).append((ctx_mv, float(u.reward)))

    def estimate(self, action_id: str, query_mv: Multivector) -> Tuple[float, float]:
        """
        Return a tuple ``(estimated_reward, confidence)`` for ``action_id`` given
        the query context ``query_mv``.
        """
        records = self._history.get(action_id, [])
        if not records:
            return 0.0, float("inf")  # no data → maximal uncertainty

        # Compute kernel weights
        weights = np.array([rbf_similarity(query_mv, ctx_mv, self.epsilon) for ctx_mv, _ in records])
        rewards = np.array([rew for _, rew in records])

        # Normalised weighted average
        weight_sum = weights.sum()
        if weight_sum == 0.0:
            avg = 0.0
        else:
            avg = np.dot(weights, rewards) / weight_sum

        # Confidence bound (inverse‑sqrt of effective sample size)
        effective_n = weight_sum  # because each weight can be interpreted as a fractional count
        confidence = self.confidence_alpha / math.sqrt(effective_n) if effective_n > 0 else float("inf")
        return avg, confidence

    def select_action(self, query_mv: Multivector, candidate_actions: Iterable[BanditAction]) -> BanditAction:
        """
        Choose the action with the highest Upper Confidence Bound (UCB):
            UCB = estimated_reward + confidence_bound
        """
        best_action: Optional[BanditAction] = None
        best_score = -float("inf")
        for a in candidate_actions:
            est, conf = self.estimate(a.action_id, query_mv)
            ucb = est + conf
            if ucb > best_score:
                best_score = ucb
                best_action = BanditAction(
                    action_id=a.action_id,
                    propensity=1.0,  # placeholder – could be derived from softmax of UCBs
                    expected_reward=est,
                    confidence_bound=conf,
                    algorithm="HybridBanditRBF",
                )
        if best_action is None:
            raise RuntimeError("No candidate actions provided")
        return best_action


# ----------------------------------------------------------------------
# Helper functions demonstrating the hybrid operation
# ----------------------------------------------------------------------
def gaussian(r: float, epsilon: float = 1.0) -> float:
    """Legacy wrapper kept for compatibility with Parent A code."""
    return math.exp(-((epsilon * r) ** 2))


def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length


def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)


def hybrid_compute_features(morph: Morphology) -> Dict[int, float]:
    """
    Produce a deterministic feature vector (as a dict) from a Morphology instance.
    The keys are treated as basis indices for the multivector representation.
    """
    # Simple engineered mapping – in practice this could be learned.
    return {
        0: morph.length,
        1: morph.width,
        2: morph.height,
        3: morph.mass,
        4: sphericity_index(morph.length, morph.width, morph.height),
        5: flatness_index(morph.length, morph.width, morph.height),
    }


def demo_hybrid_flow() -> None:
    """
    End‑to‑end demonstration:
        1. Create a few Morphology objects → contexts.
        2. Store them as multivectors.
        3. Generate synthetic BanditUpdates.
        4. Update the HybridBanditRBF.
        5. Query with a new context and obtain a selected action.
    """
    # 1. Context generation
    morphs = [
        Morphology(2.0, 1.0, 0.5, 3.0),
        Morphology(1.5, 1.2, 0.7, 2.5),
        Morphology(2.2, 0.9, 0.6, 3.2),
    ]
    context_store: Dict[str, Multivector] = {}
    for i, m in enumerate(morphs):
        ctx_id = f"ctx_{i}"
        feats = hybrid_compute_features(m)
        context_store[ctx_id] = context_dict_to_mv(feats, n=6)

    # 2. Define two possible actions
    actions = [
        BanditAction(action_id="A", propensity=0.0, expected_reward=0.0, confidence_bound=0.0, algorithm="Hybrid"),
        BanditAction(action_id="B", propensity=0.0, expected_reward=0.0, confidence_bound=0.0, algorithm="Hybrid"),
    ]

    # 3. Synthetic updates (reward = random + bias per action)
    updates: List[BanditUpdate] = []
    random.seed(42)
    for ctx_id in context_store.keys():
        for act in actions:
            reward = random.gauss(0.0, 1.0) + (1.0 if act.action_id == "A" else -0.5)
            updates.append(
                BanditUpdate(
                    context_id=ctx_id,
                    action_id=act.action_id,
                    reward=reward,
                    propensity=1.0,
                )
            )

    # 4. Initialise and train the hybrid bandit
    bandit = HybridBanditRBF(epsilon=0.8, confidence_alpha=0.5)
    bandit.update(updates, context_store)

    # 5. Query with a fresh morphology
    query_morph = Morphology(2.1, 1.0, 0.55, 3.1)
    query_mv = context_dict_to_mv(hybrid_compute_features(query_morph), n=6)
    chosen = bandit.select_action(query_mv, actions)

    print("Chosen action:", chosen.action_id)
    print("Estimated reward:", chosen.expected_reward)
    print("Confidence bound:", chosen.confidence_bound)


if __name__ == "__main__":
    demo_hybrid_flow()