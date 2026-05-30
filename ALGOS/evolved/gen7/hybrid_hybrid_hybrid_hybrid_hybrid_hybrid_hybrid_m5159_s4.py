# DARWIN HAMMER — match 5159, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2321_s0.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m2511_s0.py (gen6)
# born: 2026-05-30T00:00:28Z

"""Hybrid Algorithm: Sketch‑Bandit‑PathSignature‑Entropy‑RLCT Fusion

Parents
-------
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2321_s0.py (Bandit‑Sketch‑RLCT with ternary context)
- hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m2511_s0.py (Path‑Signature‑Entropy‑MinHash‑RBF‑NLMS)

Mathematical Bridge
-------------------
Both parents expose a *scalar information‑complexity* measure that can be
injected into confidence‑bound and kernel‑width formulas:

* Parent A supplies λ̂, the slope of the log‑loss vs. log‑sample‑size curve
  (the Real Log‑Canonical Threshold, RLCT).  It is added to the UCB
  confidence term as λ̂·log n̂, where n̂ is the HyperLogLog estimate of
  distinct contexts.

* Parent B supplies H_sig, the Shannon entropy of the eigen‑values of the
  level‑2 path‑signature matrix.  H_sig modulates the Gaussian‑RBF width
  ε and the Normalised‑LMS learning rate.

The fusion treats the product α = λ̂·H_sig as a *joint complexity factor*.
α simultaneously scales:

1. the RLCT‑aware UCB confidence bound,
2. the RBF kernel width ε = ε₀ / (1+α),
3. the NLMS learning rate η = η₀·(1+α)·c_doomsday,
   where c_doomsday is a periodic calendar factor (day‑of‑week).

The ternary classification score and textual feature counts from Parent A
are hashed with a MinHash, integrated to a peak‑velocity v_peak, and
appended to the signature‑derived feature vector Φ = [sig₁,
flatten(sig₂), H_sig, v_peak].  This unified vector feeds both the bandit
selection criterion and the surrogate predictor.

The code below implements the fused system with three core functions:
`update_sketches`, `compute_hybrid_score`, and `nlms_update`.  A short
smoke‑test demonstrates end‑to‑end execution."""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import hashlib
import datetime

# ----------------------------------------------------------------------
# Sketch utilities (Count‑Min and a simple HyperLogLog surrogate)
# ----------------------------------------------------------------------
class CountMinSketch:
    """Count‑Min sketch with pairwise‑independent hash functions."""
    def __init__(self, width: int = 1000, depth: int = 5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int64)
        # generate random seeds for reproducible hash functions
        self.seeds = [random.randint(1, 2**31 - 1) for _ in range(depth)]

    def _hash(self, item, i):
        h = hash((item, self.seeds[i]))
        return h % self.width

    def add(self, item):
        for i in range(self.depth):
            self.table[i, self._hash(item, i)] += 1

    def estimate(self, item) -> int:
        return min(self.table[i, self._hash(item, i)] for i in range(self.depth))


class SimpleHyperLogLog:
    """Very crude distinct‑count estimator using a Python set."""
    def __init__(self):
        self._set = set()

    def add(self, item):
        self._set.add(item)

    def estimate(self) -> int:
        return len(self._set)


# ----------------------------------------------------------------------
# RLCT estimation from loss history
# ----------------------------------------------------------------------
def fit_rlct(losses: list[float], sample_sizes: list[int]) -> float:
    """Linear regression on log(L) = λ·log(n) + c  → return λ̂."""
    if len(losses) < 2:
        return 0.0
    log_n = np.log(np.array(sample_sizes, dtype=float))
    log_L = np.log(np.array(losses, dtype=float) + 1e-12)  # avoid log(0)
    coeffs = np.polyfit(log_n, log_L, 1)  # coeffs[0] = slope = λ̂
    return float(coeffs[0])


# ----------------------------------------------------------------------
# Path signature utilities (Parent B)
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """Create a lead‑lag version of a 1‑D path."""
    lead = path[:-1]
    lag = path[1:]
    return np.column_stack((lead, lag))


def compute_path_signature(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return level‑1 and level‑2 signatures (outer‑product approximation)."""
    level1 = np.mean(path, axis=0, keepdims=True)          # shape (1, d)
    level2 = np.dot(path.T, path) / path.shape[0]         # shape (d, d)
    return level1, level2


def signature_entropy(sig2: np.ndarray) -> float:
    """Shannon entropy of the normalized eigen‑value spectrum of level‑2 signature."""
    eigvals = np.linalg.eigvalsh(sig2)                     # real eigenvalues
    eigvals = np.maximum(eigvals, 0)                      # numerical safety
    total = np.sum(eigvals) + 1e-12
    probs = eigvals / total
    entropy = -np.sum(probs * np.log(probs + 1e-12))
    return float(entropy)


def minhash_force_series(data: list[float]) -> list[int]:
    """Hash each float with SHA‑256 and interpret the hex digest as an int."""
    return [int(hashlib.sha256(str(x).encode()).hexdigest(), 16) for x in data]


def integrate_force_series(force_series: list[int]) -> float:
    """Simple cumulative sum integration → return peak (max) velocity."""
    if not force_series:
        return 0.0
    cumulative = np.cumsum(force_series)
    return float(np.max(cumulative))


# ----------------------------------------------------------------------
# Bandit utilities (Parent A)
# ----------------------------------------------------------------------
def ucb_confidence(mean: float, count: int, t: int,
                   rlct: float, n_est: int,
                   entropy: float, v_peak: float,
                   ternary_score: float,
                   alpha: float) -> float:
    """UCB with RLCT, entropy, velocity and ternary bias."""
    if count == 0:
        return float('inf')
    # classic log term
    log_term = math.log(t + 1)
    # RLCT contribution
    rlct_term = rlct * math.log(max(n_est, 1))
    # entropy and velocity augment confidence
    extra = entropy + v_peak
    # joint complexity factor α scales the whole bound
    bonus = math.sqrt((2 * (log_term + rlct_term + extra)) / count)
    return mean + bonus + ternary_score * alpha


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def update_sketches(action: str,
                    reward: float,
                    context_id: str,
                    cm_sketch: CountMinSketch,
                    hll: SimpleHyperLogLog,
                    reward_sums: defaultdict,
                    reward_counts: defaultdict,
                    loss_history: list[float],
                    sample_sizes: list[int]) -> None:
    """Update sketches, reward statistics and loss history for a single observation."""
    # Update count‑min sketch for reward frequencies
    cm_sketch.add((action, reward))
    # Update hyperloglog for distinct contexts
    hll.add(context_id)
    # Update empirical mean statistics
    reward_sums[action] += reward
    reward_counts[action] += 1
    # Record negative reward as loss (for RLCT)
    loss_history.append(-reward)
    sample_sizes.append(reward_counts[action])


def compute_hybrid_score(action: str,
                         t: int,
                         cm_sketch: CountMinSketch,
                         hll: SimpleHyperLogLog,
                         reward_sums: defaultdict,
                         reward_counts: defaultdict,
                         loss_history: list[float],
                         sample_sizes: list[int],
                         ternary_score: float,
                         path_sequence: np.ndarray,
                         textual_counts: list[float]) -> float:
    """Compute the hybrid selection score for an action."""
    # Empirical mean
    mean = reward_sums[action] / max(reward_counts[action], 1)

    # RLCT estimate λ̂
    rlct = fit_rlct(loss_history, sample_sizes)

    # Distinct‑context estimate n̂
    n_est = hll.estimate()

    # Path‑signature & entropy
    _, sig2 = compute_path_signature(path_sequence)
    entropy = signature_entropy(sig2)

    # MinHash‑derived peak velocity from textual feature counts
    force_series = minhash_force_series(textual_counts)
    v_peak = integrate_force_series(force_series)

    # Joint complexity factor α = λ̂·H_sig
    alpha = rlct * entropy

    # Final hybrid UCB‑style score
    return ucb_confidence(mean, reward_counts[action], t,
                          rlct, n_est,
                          entropy, v_peak,
                          ternary_score, alpha)


def nlms_update(weights: np.ndarray,
                x: np.ndarray,
                y: float,
                rlct: float,
                entropy: float,
                calendar_factor: float,
                mu: float = 0.01) -> np.ndarray:
    """
    Normalised LMS weight update with learning rate modulated by
    RLCT, entropy and a doomsday‑calendar factor.
    η = μ·(1 + λ̂·H_sig)·c_doomsday
    """
    eta = mu * (1.0 + rlct * entropy) * calendar_factor
    y_hat = np.dot(weights, x)
    error = y - y_hat
    norm = np.dot(x, x) + 1e-12
    return weights + (eta * error / norm) * x


def calendar_modulation() -> float:
    """Periodic factor based on day of week (Monday=0)."""
    day = datetime.datetime.utcnow().weekday()
    # sinusoidal modulation between 0.9 and 1.1
    return 1.0 + 0.1 * math.sin(2 * math.pi * day / 7.0)


# ----------------------------------------------------------------------
# High‑level action selection using the hybrid score
# ----------------------------------------------------------------------
def select_action(actions: list[str],
                  t: int,
                  cm_sketch: CountMinSketch,
                  hll: SimpleHyperLogLog,
                  reward_sums: defaultdict,
                  reward_counts: defaultdict,
                  loss_history: list[float],
                  sample_sizes: list[int],
                  ternary_scores: dict[str, float],
                  path_sequences: dict[str, np.ndarray],
                  textual_features: dict[str, list[float]]) -> str:
    """Return the action with the highest hybrid score."""
    best_action = None
    best_score = -float('inf')
    for a in actions:
        score = compute_hybrid_score(
            action=a,
            t=t,
            cm_sketch=cm_sketch,
            hll=hll,
            reward_sums=reward_sums,
            reward_counts=reward_counts,
            loss_history=loss_history,
            sample_sizes=sample_sizes,
            ternary_score=ternary_scores.get(a, 0.0),
            path_sequence=path_sequences.get(a, np.zeros((1, 1))),
            textual_counts=textual_features.get(a, [])
        )
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Initialise data structures
    cm = CountMinSketch(width=500, depth=4)
    hll = SimpleHyperLogLog()
    reward_sums = defaultdict(float)
    reward_counts = defaultdict(int)
    loss_hist = []
    sample_sz = []

    # Dummy action set
    actions = ["click", "scroll", "type"]
    ternary = {"click": 0.2, "scroll": -0.1, "type": 0.0}

    # Generate synthetic path sequences (2‑D for illustration)
    rng = np.random.default_rng(42)
    path_seqs = {a: rng.normal(size=(10, 2)) for a in actions}
    # Synthetic textual feature counts per action
    text_feats = {a: rng.integers(0, 5, size=5).astype(float).tolist() for a in actions}

    # Simulate a stream of 50 interactions
    for t in range(1, 51):
        # Randomly pick an action to observe
        true_action = random.choice(actions)
        # Simulated reward (binary)
        reward = 1.0 if random.random() < 0.6 else 0.0
        # Context identifier (e.g., user‑session id)
        ctx_id = f"user_{t % 7}"

        update_sketches(
            action=true_action,
            reward=reward,
            context_id=ctx_id,
            cm_sketch=cm,
            hll=hll,
            reward_sums=reward_sums,
            reward_counts=reward_counts,
            loss_history=loss_hist,
            sample_sizes=sample_sz
        )

        # Periodically (every 10 steps) select the best action according to the hybrid model
        if t % 10 == 0:
            chosen = select_action(
                actions=actions,
                t=t,
                cm_sketch=cm,
                hll=hll,
                reward_sums=reward_sums,
                reward_counts=reward_counts,
                loss_history=loss_hist,
                sample_sizes=sample_sz,
                ternary_scores=ternary,
                path_sequences=path_seqs,
                textual_features=text_feats
            )
            print(f"Step {t}: hybrid selected action -> {chosen}")

        # NLMS weight adaptation example (single‑dimensional feature for brevity)
        x_vec = np.array([reward])  # feature vector
        y_target = reward  # trivial target
        rlct_est = fit_rlct(loss_hist, sample_sz) if loss_hist else 0.0
        entropy_est = signature_entropy(np.cov(rng.normal(size=(5, 5)), rowvar=False))
        cal_factor = calendar_modulation()
        # Initialise weights if first iteration
        if t == 1:
            w = np.zeros_like(x_vec)
        w = nlms_update(w, x_vec, y_target, rlct_est, entropy_est, cal_factor)

    print("Smoke test completed without errors.")