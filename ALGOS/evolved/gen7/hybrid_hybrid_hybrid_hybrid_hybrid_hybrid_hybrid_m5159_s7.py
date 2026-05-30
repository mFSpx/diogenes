# DARWIN HAMMER — match 5159, survivor 7
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2321_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m2511_s0.py (gen6)
# born: 2026-05-30T00:00:28Z

import math
import random
import hashlib
from collections import Counter, defaultdict
from typing import List, Tuple, Sequence, Dict, Any

import numpy as np


# ----------------------------------------------------------------------
# Sketch primitives
# ----------------------------------------------------------------------
class CountMinSketch:
    """Count‑Min sketch with pairwise‑independent hash functions."""

    def __init__(self, width: int = 1024, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item: Any, seed: int) -> int:
        return (hash((item, seed)) % self.width)

    def add(self, item: Any) -> None:
        for d, seed in enumerate(self.seeds):
            self.table[d, self._hash(item, seed)] += 1

    def estimate(self, item: Any) -> int:
        return min(self.table[d, self._hash(item, seed)] for d, seed in enumerate(self.seeds))

    def total(self) -> int:
        """Return an estimate of the total number of items added."""
        return int(self.table.sum() / self.depth)


class MinHashSketch:
    """Lightweight MinHash sketch for Jaccard‑like similarity."""

    def __init__(self, num_perm: int = 64):
        self.num_perm = num_perm
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self.minhash = [2**63 - 1] * num_perm

    def _hash(self, item: Any, seed: int) -> int:
        h = hashlib.sha256(f"{item}_{seed}".encode()).hexdigest()
        return int(h, 16) & ((1 << 63) - 1)

    def add(self, item: Any) -> None:
        for i, seed in enumerate(self.seeds):
            h = self._hash(item, seed)
            if h < self.minhash[i]:
                self.minhash[i] = h

    def signature(self) -> Tuple[int, ...]:
        return tuple(self.minhash)

    @staticmethod
    def jaccard(sig1: Tuple[int, ...], sig2: Tuple[int, ...]) -> float:
        """Estimate Jaccard similarity from two MinHash signatures."""
        if len(sig1) != len(sig2):
            raise ValueError("Signatures must have equal length")
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)


# ----------------------------------------------------------------------
# RLCT estimation (Parent A)
# ----------------------------------------------------------------------
def estimate_rlct(losses: Sequence[float], sample_sizes: Sequence[int]) -> float:
    """
    Ordinary‑least‑squares fit of log(L) = λ·log(n) + c.
    Returns λ̂ (the slope).  If the regression is ill‑conditioned,
    falls back to 0.0.
    """
    if len(losses) < 2 or len(losses) != len(sample_sizes):
        raise ValueError("Need at least two (loss, n) pairs of equal length")
    log_n = np.log(np.array(sample_sizes, dtype=np.float64) + 1e-12)
    log_L = np.log(np.array(losses, dtype=np.float64) + 1e-12)
    A = np.vstack([log_n, np.ones_like(log_n)]).T
    try:
        λ, _ = np.linalg.lstsq(A, log_L, rcond=None)[0]
    except np.linalg.LinAlgError:
        λ = 0.0
    return float(λ)


# ----------------------------------------------------------------------
# Path signature and entropy (Parent B)
# ----------------------------------------------------------------------
def compute_path_signature(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Level‑1 (mean) and level‑2 (empirical covariance) signatures.
    Path shape: (T, d) where T is time steps.
    """
    if path.ndim != 2:
        raise ValueError("Path must be a 2‑D array")
    level1 = np.mean(path, axis=0, keepdims=True)          # (1, d)
    level2 = (path.T @ path) / max(path.shape[0], 1)       # (d, d)
    return level1, level2


def signature_entropy(level2_sig: np.ndarray) -> float:
    """
    Shannon entropy of the normalized eigen‑spectrum of a symmetric matrix.
    """
    eigvals = np.linalg.eigvalsh(level2_sig)
    eigvals = np.clip(eigvals, 1e-12, None)
    probs = eigvals / eigvals.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


# ----------------------------------------------------------------------
# Force‑series utilities (Parent B)
# ----------------------------------------------------------------------
def minhash_force_series(data: Sequence[float]) -> List[int]:
    """Hash each scalar into a 31‑bit integer; returns a discrete series."""
    series = []
    for x in data:
        h = hashlib.sha256(str(x).encode()).hexdigest()
        series.append(int(h, 16) % (2**31 - 1))
    return series


def peak_velocity(force_series: Sequence[int]) -> float:
    """Peak absolute first‑difference of the cumulative sum."""
    if not force_series:
        return 0.0
    cumulative = np.cumsum(force_series, dtype=np.float64)
    diffs = np.diff(cumulative, prepend=0.0)
    return float(np.max(np.abs(diffs)))


# ----------------------------------------------------------------------
# Hybrid bandit core
# ----------------------------------------------------------------------
class HybridBandit:
    """
    Contextual bandit that fuses RLCT‑driven exploration with
    signature‑entropy‑scaled surrogate modelling.
    """

    def __init__(
        self,
        actions: List[str],
        base_ucb_coef: float = 1.0,
        base_rbf_width: float = 1.0,
        base_lr: float = 0.01,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.actions = actions
        self.base_ucb_coef = base_ucb_coef
        self.base_rbf_width = base_rbf_width
        self.base_lr = base_lr
        self.alpha = alpha          # weight of λ̂ in scaling
        self.beta = beta            # weight of entropy in scaling

        # per‑action statistics
        self.counts = defaultdict(int)          # int
        self.reward_sums = defaultdict(float)   # Σ r
        self.reward_sq_sums = defaultdict(float)  # Σ r²

        # global sketches
        self.reward_sketch = CountMinSketch()
        self.context_sketch = MinHashSketch()

        # history for RLCT
        self.loss_history: List[float] = []
        self.sample_size_history: List[int] = []

        # cache for λ̂ (updated lazily)
        self._lambda_hat: float = 0.0

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def _extract_features(
        self,
        path: np.ndarray,
        ternary_labels: Sequence[int],
        text_tokens: Sequence[str],
        force_data: Sequence[float],
    ) -> Tuple[np.ndarray, float]:
        """
        Returns a concatenated feature vector Φ and the entropy H.
        """
        # signature
        lvl1, lvl2 = compute_path_signature(path)
        H = signature_entropy(lvl2)

        # ternary label histogram
        label_counts = Counter(ternary_labels)
        label_vec = np.array(
            [
                label_counts.get(-1, 0),
                label_counts.get(0, 0),
                label_counts.get(1, 0),
            ],
            dtype=np.float64,
        )

        # bag‑of‑words (top‑k)
        token_counts = Counter(text_tokens)
        top_k = 20
        most_common = token_counts.most_common(top_k)
        token_vec = np.array([cnt for _, cnt in most_common], dtype=np.float64)
        if token_vec.size < top_k:
            token_vec = np.pad(token_vec, (0, top_k - token_vec.size), constant_values=0.0)

        # force series peak velocity
        fv = peak_velocity(minhash_force_series(force_data))

        # assemble
        Φ = np.concatenate(
            [
                lvl1.ravel(),
                lvl2.ravel(),
                label_vec,
                token_vec,
                np.array([fv], dtype=np.float64),
            ]
        )
        return Φ, H

    # ------------------------------------------------------------------
    # Scaling factor computation
    # ------------------------------------------------------------------
    def _update_lambda_hat(self) -> None:
        """Re‑estimate λ̂ from accumulated loss history."""
        if len(self.loss_history) >= 2:
            n_est = self.reward_sketch.total()
            sample_sizes = [n_est] * len(self.loss_history)
            self._lambda_hat = estimate_rlct(self.loss_history, sample_sizes)
        else:
            self._lambda_hat = 0.0

    def _scaling_factors(self, entropy: float) -> Tuple[float, float, float]:
        """
        Returns (exploration_bonus, rbf_width, learning_rate) derived from
        λ̂ and entropy using a deeper, non‑linear fusion.
        """
        self._update_lambda_hat()
        λ = self._lambda_hat
        # fused exponentiated factor δ ∈ (0, ∞)
        δ = (1.0 + λ) ** self.alpha * (1.0 + entropy) ** self.beta
        # exploration bonus grows with δ, but we cap to avoid runaway
        exploration = self.base_ucb_coef * min(δ, 10.0)
        # RBF width shrinks with δ (more confident → narrower kernel)
        rbf_width = max(self.base_rbf_width / max(δ, 0.1), 1e-3)
        # learning rate scales linearly with δ, bounded
        lr = min(self.base_lr * δ, 1.0)
        return exploration, rbf_width, lr

    # ------------------------------------------------------------------
    # RBF surrogate
    # ------------------------------------------------------------------
    @staticmethod
    def _rbf(x: np.ndarray, y: np.ndarray, width: float) -> float:
        diff = x - y
        return math.exp(-np.dot(diff, diff) / (2.0 * width ** 2))

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(
        self,
        context: Dict[str, Any],
        path: np.ndarray,
        ternary_labels: Sequence[int],
        text_tokens: Sequence[str],
        force_data: Sequence[float],
    ) -> str:
        """
        Choose an action using γ‑augmented UCB and an RBF surrogate.
        """
        Φ, entropy = self._extract_features(path, ternary_labels, text_tokens, force_data)
        exploration, rbf_width, _ = self._scaling_factors(entropy)

        # update context sketch for similarity reuse
        for token in text_tokens:
            self.context_sketch.add(token)

        # compute surrogate predictions for each action
        best_score = -float("inf")
        chosen = None
        t = sum(self.counts.values()) + 1
        for a in self.actions:
            # empirical mean reward
            n_a = self.counts[a]
            mean_a = self.reward_sums[a] / max(n_a, 1)

            # UCB term
            bonus = exploration * math.sqrt(math.log(t + 1) / (n_a + 1))

            # RBF surrogate: compare current Φ to a running prototype per action
            # (simple exponential moving average of past Φ)
            proto = getattr(self, f"_proto_{a}", None)
            if proto is None:
                # first encounter → no surrogate, rely on UCB only
                surrogate = 0.0
            else:
                surrogate = self._rbf(Φ, proto, rbf_width)

            score = mean_a + bonus + surrogate
            if score > best_score:
                best_score = score
                chosen = a

        # store prototype for the selected action (EMA with decay 0.9)
        if chosen is not None:
            proto_attr = f"_proto_{chosen}"
            old = getattr(self, proto_attr, None)
            if old is None:
                setattr(self, proto_attr, Φ.copy())
            else:
                setattr(self, proto_attr, 0.9 * old + 0.1 * Φ)

        return chosen

    # ------------------------------------------------------------------
    # Update after observing reward
    # ------------------------------------------------------------------
    def update(
        self,
        action: str,
        reward: float,
        loss: float,
        path: np.ndarray,
        ternary_labels: Sequence[int],
        text_tokens: Sequence[str],
        force_data: Sequence[float],
    ) -> None:
        """
        Incorporate observed reward and loss, update sketches and statistics.
        """
        # basic statistics
        self.counts[action] += 1
        self.reward_sums[action] += reward
        self.reward_sq_sums[action] += reward ** 2

        # sketch the reward (used for RLCT sample‑size estimate)
        self.reward_sketch.add((action, reward))

        # record loss for RLCT regression
        self.loss_history.append(loss)
        # sample size estimate via sketch total count (global)
        self.sample_size_history.append(self.reward_sketch.total())

        # optionally update context sketch with new tokens
        for token in text_tokens:
            self.context_sketch.add(token)

        # recompute scaling factors to keep them fresh for next round
        _, _, lr = self._scaling_factors(entropy=0.0)  # entropy not needed for lr alone

        # simple NLMS‑style linear predictor per action (optional extension)
        # Here we maintain a weight vector per action that is updated online.
        weight_attr = f"_w_{action}"
        Φ, _ = self._extract_features(path, ternary_labels, text_tokens, force_data)
        w = getattr(self, weight_attr, np.zeros_like(Φ))
        pred = np.dot(w, Φ)
        error = reward - pred
        w_new = w + lr * error * Φ / (np.linalg.norm(Φ) ** 2 + 1e-12)
        setattr(self, weight_attr, w_new)

    # ------------------------------------------------------------------
    # Utility: Jaccard similarity between current context and stored prototype
    # ------------------------------------------------------------------
    def context_similarity(self, other_tokens: Sequence[str]) -> float:
        """
        Estimate similarity of the current textual context to a previously
        observed token set using MinHash signatures.
        """
        tmp_sketch = MinHashSketch(num_perm=self.context_sketch.num_perm)
        for token in other_tokens:
            tmp_sketch.add(token)
        return MinHashSketch.jaccard(self.context_sketch.signature(), tmp_sketch.signature())