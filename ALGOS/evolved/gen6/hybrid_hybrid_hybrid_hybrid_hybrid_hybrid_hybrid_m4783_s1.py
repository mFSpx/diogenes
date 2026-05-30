# DARWIN HAMMER — match 4783, survivor 1
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_doomsday_cale_m1280_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_korpus_text_h_m1000_s1.py (gen5)
# born: 2026-05-29T23:58:15Z

"""HybridBanditPheromoneEntropyNLMS
Combines:
- Parent A: HybridBanditRouterHoneybeeStore (bandit policy, NLMS weight adaptation)
- Parent B: pheromone probability calculation, decision hygiene scoring, Shannon entropy, and regret‑weighted similarity.

Mathematical bridge:
1. The expected reward for an action a is adjusted by a pheromone factor ϕ_a (probability from surface pheromones) and an entropy factor η derived from the Shannon entropy of the decision‑hygiene scores of the context text.
   \hat{r}_a = r_a * ϕ_a * (1 + η)

2. A LinUCB‑style confidence bound is multiplied by the same entropy factor, giving a tighter bound when the context is informationally rich.
   UCB_a = \hat{r}_a + α * η * sqrt( x_aᵀ A_a^{-1} x_a )

3. NLMS adapts a per‑action weight vector w_a to minimise the error between the adjusted reward and the linear prediction w_aᵀ x.
   w_a ← w_a + μ / (ε + ‖x‖²) * x * ( \hat{r}_a - w_aᵀ x )

Thus the three mathematical structures (pheromone weighting, information‑theoretic entropy, and adaptive NLMS) are fused into a single decision engine.
"""

import numpy as np
import math
import random
import sys
import pathlib
import re
import hashlib

# ----------------------------------------------------------------------
# Utility functions from Parent B (adapted to allowed imports)
# ----------------------------------------------------------------------
def calculate_pheromone_probabilities(surface_key: str, limit: int = 10) -> list[float]:
    """Generate mock pheromone probabilities for a given surface key.
    In the original algorithm these are read from a DB; here we simulate
    a Dirichlet distribution to obtain a valid probability simplex."""
    rng = np.random.default_rng(hash(surface_key) % (2**32 - 1))
    raw = rng.gamma(shape=1.0, scale=1.0, size=limit)
    total = raw.sum()
    return (raw / total).tolist()


def decision_hygiene_scores(text: str) -> dict[str, int]:
    """Very simple heuristic: count occurrences of evidence‑related keywords."""
    keywords = {
        "evidence": 3,
        "verify": 2,
        "verified": 2,
        "proof": 4,
        "test": 1,
        "experiment": 2,
    }
    scores: dict[str, int] = {}
    lowered = text.lower()
    for kw, weight in keywords.items():
        count = lowered.count(kw)
        if count:
            scores[kw] = count * weight
    return scores


def shannon_entropy(freqs: dict[str, int]) -> float:
    """Compute Shannon entropy of a discrete distribution given by integer frequencies."""
    total = sum(freqs.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for v in freqs.values():
        p = v / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def minhash_signature(text: str, num_perm: int = 64) -> np.ndarray:
    """Create a simple MinHash signature using deterministic hashing of n‑grams."""
    tokens = re.findall(r"\w+", text.lower())
    signature = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in tokens:
        token_hash = int(hashlib.sha1(token.encode()).hexdigest(), 16)
        for i in range(num_perm):
            combined = token_hash ^ (i * 0x9e3779b97f4a7c15)
            if combined < signature[i]:
                signature[i] = combined
    return signature


def jaccard_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signatures must have the same length")
    return np.mean(sig1 == sig2)


# ----------------------------------------------------------------------
# NLMS update (from Parent A)
# ----------------------------------------------------------------------
def nlms_update(weights: np.ndarray, x: np.ndarray, d: float, mu: float = 0.01, eps: float = 1e-8) -> np.ndarray:
    """Normalized Least‑Mean‑Squares weight update."""
    x = x.reshape(-1, 1)                     # column vector
    y = float(weights @ x)                   # current output
    e = d - y                                 # error
    norm_factor = eps + float(x.T @ x)        # ε + ‖x‖²
    delta = (mu / norm_factor) * e * x.ravel()
    return weights + delta


# ----------------------------------------------------------------------
# Core hybrid class
# ----------------------------------------------------------------------
class HybridBanditPheromoneEntropyNLMS:
    """Hybrid bandit that uses pheromone weighting, entropy‑modulated confidence,
    and NLMS adaptation for per‑action linear models."""
    def __init__(self, context_dim: int = 5, alpha: float = 0.1, mu: float = 0.01):
        self._policy: dict[str, list[float]] = {}               # action_id -> [cumulative_reward, count]
        self._weights: dict[str, np.ndarray] = {}               # action_id -> weight vector
        self.context_dim = context_dim
        self.alpha = alpha          # LinUCB exploration coefficient
        self.mu = mu                # NLMS step size
        self._A_inv: dict[str, np.ndarray] = {}                # For LinUCB, maintain A^{-1} per action
        self._init_default_matrices()

    def _init_default_matrices(self):
        """Initialize A^{-1} as identity for each action on first use."""
        pass  # lazy initialisation in _ensure_action

    def _ensure_action(self, action: str):
        if action not in self._weights:
            self._weights[action] = np.zeros(self.context_dim)
            self._A_inv[action] = np.eye(self.context_dim)

    def reset(self):
        self._policy.clear()
        self._weights.clear()
        self._A_inv.clear()

    # ------------------------------------------------------------------
    # Policy bookkeeping (inherits logic from Parent A)
    # ------------------------------------------------------------------
    def update_policy(self, action: str, reward: float):
        stats = self._policy.setdefault(action, [0.0, 0.0])
        stats[0] += float(reward)
        stats[1] += 1.0
        self._ensure_action(action)

    def _average_reward(self, action: str) -> float:
        total, n = self._policy.get(action, [0.0, 0.0])
        return total / n if n > 0 else 0.0

    # ------------------------------------------------------------------
    # Core hybrid computation
    # ------------------------------------------------------------------
    def _adjusted_reward(self, action: str, pheromone_factor: float, entropy_factor: float) -> float:
        """Base reward scaled by pheromone and entropy."""
        base = self._average_reward(action)
        return base * pheromone_factor * (1.0 + entropy_factor)

    def _confidence_bound(self, action: str, x: np.ndarray, entropy_factor: float) -> float:
        """LinUCB bound multiplied by entropy factor."""
        A_inv = self._A_inv[action]
        xb = x @ A_inv @ x.T
        return self.alpha * entropy_factor * math.sqrt(float(xb))

    def select_action(
        self,
        context: dict[str, float],
        actions: list[str],
        text_context: str,
        surface_key: str,
        algorithm: str = "linucb",
        epsilon: float = 0.1,
        seed: int | str | None = 7,
    ) -> dict:
        """Select an action using the hybrid scoring rule."""
        if not actions:
            raise ValueError("actions required")
        rng = random.Random(seed)

        # 1. Build context vector (fixed order of keys for reproducibility)
        ctx_keys = sorted(context.keys())
        x = np.array([context[k] for k in ctx_keys], dtype=float)
        if x.size != self.context_dim:
            # pad or truncate to required dimension
            if x.size < self.context_dim:
                x = np.pad(x, (0, self.context_dim - x.size), constant_values=0.0)
            else:
                x = x[: self.context_dim]

        # 2. Compute auxiliary factors
        pheromone_probs = calculate_pheromone_probabilities(surface_key, limit=len(actions))
        pheromone_map = dict(zip(actions, pheromone_probs))

        hygiene = decision_hygiene_scores(text_context)
        entropy = shannon_entropy(hygiene)  # η ∈ [0, log2|V|]
        entropy_factor = 1.0 + entropy  # ensure >1 for scaling

        # 3. Choose according to algorithm
        if algorithm == "epsilon_greedy" and rng.random() < epsilon:
            chosen = rng.choice(actions)
        else:
            scores = {}
            for a in actions:
                self._ensure_action(a)
                phi = pheromone_map.get(a, 1.0 / len(actions))
                adj_r = self._adjusted_reward(a, phi, entropy_factor)
                if algorithm == "linucb":
                    cb = self._confidence_bound(a, x, entropy_factor)
                    score = adj_r + cb
                else:  # default to greedy on adjusted reward
                    score = adj_r
                scores[a] = score
            chosen = max(scores, key=scores.get)

        # 4. Return detailed info
        propensity = 1.0 / len(actions) if algorithm != "epsilon_greedy" else epsilon / len(actions)
        return {
            "action_id": chosen,
            "propensity": propensity,
            "expected_reward": self._average_reward(chosen),
            "entropy": entropy,
            "pheromone_factor": pheromone_map.get(chosen, 1.0 / len(actions)),
            "confidence_bound": self._confidence_bound(chosen, x, entropy_factor) if algorithm == "linucb" else None,
        }

    # ------------------------------------------------------------------
    # NLMS weight adaptation after observing true reward
    # ------------------------------------------------------------------
    def adapt_weights(self, action: str, context: dict[str, float], observed_reward: float):
        """Update the linear model for *action* using NLMS."""
        self._ensure_action(action)
        ctx_keys = sorted(context.keys())
        x = np.array([context[k] for k in ctx_keys], dtype=float)
        if x.size != self.context_dim:
            if x.size < self.context_dim:
                x = np.pad(x, (0, self.context_dim - x.size), constant_values=0.0)
            else:
                x = x[: self.context_dim]

        # Compute adjusted target using pheromone and entropy (mirrors selection)
        pheromone_factor = 1.0  # during learning we use neutral factor
        entropy_factor = 1.0
        target = observed_reward * pheromone_factor * (1.0 + entropy_factor)

        # NLMS update
        w_old = self._weights[action]
        w_new = nlms_update(w_old, x, target, mu=self.mu)
        self._weights[action] = w_new

        # Update LinUCB matrix A^{-1} with Sherman‑Morrison
        A_inv = self._A_inv[action]
        x_col = x.reshape(-1, 1)
        numerator = A_inv @ x_col @ x_col.T @ A_inv
        denominator = 1.0 + float(x_col.T @ A_inv @ x_col)
        self._A_inv[action] = A_inv - numerator / denominator

    # ------------------------------------------------------------------
    # Helper to expose internal state for testing / introspection
    # ------------------------------------------------------------------
    def get_policy(self) -> dict[str, list[float]]:
        return self._policy.copy()

    def get_weights(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._weights.items()}


# ----------------------------------------------------------------------
# Stand‑alone demonstration functions (require at least three)
# ----------------------------------------------------------------------
def simulate_round(bandit: HybridBanditPheromoneEntropyNLMS, context: dict[str, float],
                   actions: list[str], text: str, surface_key: str) -> None:
    """Run a single interaction: select, receive mock reward, update."""
    decision = bandit.select_action(context, actions, text, surface_key, algorithm="linucb")
    chosen = decision["action_id"]
    # Mock reward: dot product of context with a hidden weight + noise
    hidden_w = np.arange(bandit.context_dim) * 0.5 + 1.0
    reward = float(np.dot(hidden_w, np.array([context[k] for k in sorted(context.keys())])) + random.gauss(0, 0.1))
    bandit.update_policy(chosen, reward)
    bandit.adapt_weights(chosen, context, reward)


def batch_train(bandit: HybridBanditPheromoneEntropyNLMS, n_rounds: int = 100) -> None:
    """Train the hybrid bandit on synthetic data."""
    actions = ["A", "B", "C"]
    for i in range(n_rounds):
        ctx = {f"f{j}": random.random() for j in range(bandit.context_dim)}
        txt = "Evidence suggests that the experiment is verified."
        surf = f"surface_{i % 5}"
        simulate_round(bandit, ctx, actions, txt, surf)


def evaluate_policy(bandit: HybridBanditPheromoneEntropyNLMS, n_trials: int = 20) -> float:
    """Return average reward of the current policy on fresh contexts."""
    total = 0.0
    actions = ["A", "B", "C"]
    for _ in range(n_trials):
        ctx = {f"f{j}": random.random() for j in range(bandit.context_dim)}
        txt = "Proof of concept with evidence."
        surf = "surface_eval"
        decision = bandit.select_action(ctx, actions, txt, surf, algorithm="greedy")
        chosen = decision["action_id"]
        hidden_w = np.arange(bandit.context_dim) * 0.5 + 1.0
        reward = float(np.dot(hidden_w, np.array([ctx[k] for k in sorted(ctx.keys())])))
        total += reward
    return total / n_trials


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    hb = HybridBanditPheromoneEntropyNLMS(context_dim=5, alpha=0.2, mu=0.05)
    batch_train(hb, n_rounds=200)
    avg_reward = evaluate_policy(hb, n_trials=30)
    print(f"Average reward after training: {avg_reward:.4f}")
    print("Final policy stats:", hb.get_policy())
    print("Sample weight vector for action 'A':", hb.get_weights().get("A"))