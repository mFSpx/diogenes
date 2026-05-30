# DARWIN HAMMER — match 3947, survivor 2
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s0.py (gen4)
# parent_b: hybrid_hybrid_perceptual_de_hybrid_hybrid_hybrid_m1188_s5.py (gen4)
# born: 2026-05-29T23:52:52Z

import numpy as np
import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Dict

# ----------------------------------------------------------------------
# Utility hashing – deterministic 64‑bit MinHash primitives
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1


def _hash(seed: int, token: str) -> int:
    """
    Deterministic 64‑bit hash of *token* with a *seed*.
    Uses Blake2b (8‑byte digest) for speed and uniformity.
    """
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], n_hashes: int = 128) -> np.ndarray:
    """
    Compute a MinHash signature for a set of *tokens*.
    Returns a 1‑D ``np.ndarray`` of shape ``(n_hashes,)`` with dtype ``uint64``.
    """
    token_set = {t for t in tokens if t}
    if n_hashes <= 0:
        raise ValueError("n_hashes must be positive")
    if not token_set:
        # Empty set → maximal hash values (worst‑case Jaccard similarity 0)
        return np.full(n_hashes, MAX64, dtype=np.uint64)

    sig = np.empty(n_hashes, dtype=np.uint64)
    for i in range(n_hashes):
        # min over all tokens for the i‑th hash function
        sig[i] = min(_hash(i, t) for t in token_set)
    return sig


def minhash_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """
    Estimate Jaccard similarity from two MinHash signatures.
    """
    if sig_a.shape != sig_b.shape:
        raise ValueError("signatures must have the same shape")
    return float(np.count_nonzero(sig_a == sig_b)) / sig_a.shape[0]


# ----------------------------------------------------------------------
# Pheromone model
# ----------------------------------------------------------------------
@dataclass
class PheromoneSignal:
    """
    A single pheromone emission.
    ``created`` – timestamp (float, seconds since start)
    ``value``   – raw intensity (float)
    ``half_life`` – exponential decay half‑life (float, same time units as ``created``)
    """
    created: float
    value: float
    half_life: float

    def decay_factor(self, now: float) -> float:
        """
        Exponential decay factor at time *now*.
        """
        if self.half_life <= 0:
            return 0.0
        elapsed = max(0.0, now - self.created)
        return 0.5 ** (elapsed / self.half_life)

    def decayed_value(self, now: float) -> float:
        """Current intensity after decay."""
        return self.value * self.decay_factor(now)


# ----------------------------------------------------------------------
# Tokenisation of pheromones for MinHash
# ----------------------------------------------------------------------
def _quantise(x: float, bins: int = 16) -> str:
    """
    Simple quantisation: map a float to a bucket index string.
    """
    if np.isfinite(x):
        idx = int(np.clip(np.floor(x * bins), 0, bins - 1))
        return f"b{idx}"
    return "nan"


def pheromone_tokens(signal: PheromoneSignal, now: float, *,
                     time_bins: int = 8,
                     value_bins: int = 16) -> List[str]:
    """
    Convert a pheromone signal into a richer token list suitable for MinHash.
    Tokens encode:
      * discretised creation time (relative to *now*)
      * discretised raw value
      * discretised half‑life
      * decayed value
    This yields a multi‑token representation, allowing the MinHash estimator
    to capture partial overlaps between similar pheromones.
    """
    rel_time = now - signal.created
    tokens = [
        f"t{_quantise(rel_time, time_bins)}",
        f"v{_quantise(signal.value, value_bins)}",
        f"h{_quantise(signal.half_life, time_bins)}",
        f"d{_quantise(signal.decayed_value(now), value_bins)}",
    ]
    return tokens


def pheromone_signature(signal: PheromoneSignal, now: float,
                        n_hashes: int = 128) -> np.ndarray:
    """
    MinHash signature of a pheromone signal at time *now*.
    """
    toks = pheromone_tokens(signal, now)
    return minhash_signature(toks, n_hashes)


# ----------------------------------------------------------------------
# Radial‑Basis Function surrogate
# ----------------------------------------------------------------------
def gaussian_rbf(r: float, epsilon: float = 1.0) -> float:
    """Isotropic Gaussian RBF."""
    return math.exp(-((epsilon * r) ** 2))


def rbf_matrix(centers: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
    """
    Compute the symmetric RBF kernel matrix K_{ij}=exp(-ε²‖c_i-c_j‖²)
    for a set of *centers* (shape ``(m, d)``).
    """
    diff = centers[:, None, :] - centers[None, :, :]   # (m, m, d)
    sqdist = np.sum(diff ** 2, axis=2)                # (m, m)
    return np.exp(- (epsilon ** 2) * sqdist)


# ----------------------------------------------------------------------
# Hybrid system – deep integration of pheromones and RBF surrogate
# ----------------------------------------------------------------------
class HybridPheromoneRBFSystem:
    """
    A bandit‑style decision engine that fuses:
      * Decayed pheromone intensities (as *priors*),
      * A Gaussian RBF surrogate built on the same pheromone centres,
    to produce a hybrid score for each arm.
    The system maintains a dynamic set of pheromone signals, updates their
    decay, and recomputes the surrogate on demand.
    """

    def __init__(
        self,
        n_arms: int,
        epsilon: float = 1.0,
        n_hashes: int = 128,
        seed: int | None = None,
    ):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.n_hashes = n_hashes
        self._rng = np.random.default_rng(seed)

        # storage for active pheromones
        self._signals: List[PheromoneSignal] = []

        # per‑arm statistics for classic UCB1 (counts & rewards)
        self._counts = np.zeros(n_arms, dtype=int)
        self._rewards = np.zeros(n_arms, dtype=float)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def emit(self, signal: PheromoneSignal) -> None:
        """Add a new pheromone emission to the system."""
        self._signals.append(signal)

    def step(self, now: float, arm_features: np.ndarray) -> int:
        """
        Choose an arm at time *now* given a matrix of arm feature vectors
        (shape ``(n_arms, d)``). Returns the selected arm index.
        """
        if arm_features.shape[0] != self.n_arms:
            raise ValueError("arm_features must have a row for each arm")
        hybrid_scores = self._hybrid_scores(now, arm_features)
        ucb = self._ucb1(hybrid_scores)
        return int(np.argmax(ucb))

    def update(self, arm_index: int, reward: float) -> None:
        """Incorporate observed *reward* for *arm_index*."""
        self._counts[arm_index] += 1
        self._rewards[arm_index] += reward

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------
    def _active_signals(self, now: float) -> List[Tuple[PheromoneSignal, float]]:
        """
        Return a list of (signal, decayed_value) for signals whose
        decayed value is still non‑negligible.
        """
        active = []
        for sig in self._signals:
            decayed = sig.decayed_value(now)
            if decayed > 1e-6:          # threshold to prune near‑zero pheromones
                active.append((sig, decayed))
        return active

    def _centers_and_weights(self, now: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build RBF centres from active pheromones and compute their weights.
        Each centre is a 2‑D vector: (time_bucket, value_bucket) in the
        quantised space used for MinHash tokens. The weight is the
        decayed intensity, acting as a prior for the surrogate.
        """
        active = self._active_signals(now)
        if not active:
            # No active pheromones → empty surrogate
            return np.empty((0, 2)), np.empty(0)

        centres = []
        weights = []
        for sig, decayed in active:
            # Use the same quantisation scheme as the tokeniser
            time_tok = _quantise(now - sig.created, time_bins=8)
            value_tok = _quantise(sig.value, value_bins=16)
            # map token strings to integer coordinates for the surrogate
            time_coord = int(time_tok[1:])          # strip leading 'b'
            value_coord = int(value_tok[1:])
            centres.append([time_coord, value_coord])
            weights.append(decayed)

        return np.asarray(centres, dtype=float), np.asarray(weights, dtype=float)

    def _rbf_surrogate(self, now: float, arm_features: np.ndarray) -> np.ndarray:
        """
        Evaluate the RBF surrogate for each arm.
        Returns a 1‑D array of shape ``(n_arms,)``.
        """
        centres, weights = self._centers_and_weights(now)
        if centres.shape[0] == 0:
            # No pheromone information → neutral surrogate (zero)
            return np.zeros(self.n_arms, dtype=float)

        # Kernel between centres and arms
        diff = centres[:, None, :] - arm_features[None, :, :]   # (m, n_arms, d)
        sqdist = np.sum(diff ** 2, axis=2)                     # (m, n_arms)
        K = np.exp(- (self.epsilon ** 2) * sqdist)            # (m, n_arms)

        # Weighted sum of RBF contributions
        surrogate = np.dot(weights, K)                         # (n_arms,)
        # Normalise by total weight to keep scale comparable across steps
        total_w = np.sum(weights)
        if total_w > 0:
            surrogate /= total_w
        return surrogate

    def _minhash_similarity(self, now: float, arm_features: np.ndarray) -> np.ndarray:
        """
        Compute a MinHash‑based similarity between each arm and the *aggregate*
        pheromone signature at time *now*. The aggregate signature is the
        element‑wise minimum of all active pheromone signatures (a conservative
        estimate of common tokens).
        Returns a vector of similarities in ``[0, 1]``.
        """
        active = self._active_signals(now)
        if not active:
            return np.zeros(self.n_arms, dtype=float)

        # Aggregate pheromone signature (element‑wise min)
        agg_sig = np.full(self.n_hashes, MAX64, dtype=np.uint64)
        for sig, _ in active:
            agg_sig = np.minimum(agg_sig, pheromone_signature(sig, now, n_hashes=self.n_hashes))

        # Build a temporary MinHash signature for each arm using the same tokeniser
        arm_sigs = np.empty((self.n_arms, self.n_hashes), dtype=np.uint64)
        for i, feat in enumerate(arm_features):
            # Treat the arm feature vector as a pseudo‑pheromone for tokenisation
            pseudo_sig = PheromoneSignal(created=now, value=feat[0], half_life=1.0)
            arm_sigs[i] = pheromone_signature(pseudo_sig, now, n_hashes=self.n_hashes)

        # Pairwise Jaccard‑like similarity
        matches = np.count_nonzero(arm_sigs == agg_sig, axis=1)
        return matches.astype(float) / self.n_hashes

    def _hybrid_scores(self, now: float, arm_features: np.ndarray) -> np.ndarray:
        """
        Combine three ingredients into a single scalar per arm:
          1. RBF surrogate (continuous, captures smooth similarity)
          2. MinHash similarity (discrete, captures exact token overlap)
          3. Decayed pheromone intensity (global prior)
        The combination uses a weighted geometric mean to keep the scale
        bounded in ``[0, 1]``.
        """
        rbf = self._rbf_surrogate(now, arm_features)          # ∈ [0, 1] after normalisation
        mh = self._minhash_similarity(now, arm_features)     # ∈ [0, 1]

        # Global prior = total decayed pheromone mass (normalised)
        _, weights = self._centers_and_weights(now)
        total_mass = np.sum(weights)
        prior = total_mass / (total_mass + 1.0)                # maps (0, ∞) → (0, 1)

        # Avoid zeros that would kill the geometric mean
        eps = 1e-9
        hybrid = (np.maximum(rbf, eps) *
                  np.maximum(mh, eps) *
                  np.maximum(prior, eps)) ** (1.0 / 3.0)
        return hybrid

    def _ucb1(self, hybrid_scores: np.ndarray) -> np.ndarray:
        """
        Classic UCB1 confidence bound, but replace the empirical mean
        with the *hybrid* surrogate score.
        """
        total_counts = np.sum(self._counts) + 1  # avoid division by zero
        # Exploration term – same as standard UCB1
        exploration = np.sqrt(2 * np.log(total_counts) / (self._counts + 1))
        return hybrid_scores + exploration


# ----------------------------------------------------------------------
# Simple smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a system with 3 arms and 2‑dimensional arm features
    sys = HybridPheromoneRBFSystem(n_arms=3, epsilon=0.5, seed=42)

    # Emit a few pheromones at different times
    now = 0.0
    sys.emit(PheromoneSignal(created=now, value=5.0, half_life=2.0))
    sys.emit(PheromoneSignal(created=now + 1.0, value=3.0, half_life=1.5))

    # Define arm feature vectors (e.g., [time_bucket, value_bucket])
    arm_features = np.array([
        [0, 5],   # arm 0
        [2, 3],   # arm 1
        [4, 1],   # arm 2
    ], dtype=float)

    # Choose an arm
    chosen = sys.step(now + 2.0, arm_features)
    print(f"Chosen arm at t=2.0: {chosen}")

    # Simulate a reward and update
    reward = random.random()
    sys.update(chosen, reward)
    print(f"Reward {reward:.3f} added to arm {chosen}")