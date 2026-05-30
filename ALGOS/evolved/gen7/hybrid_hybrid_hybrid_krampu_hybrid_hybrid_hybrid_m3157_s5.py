# DARWIN HAMMER — match 3157, survivor 5
# gen: 7
# parent_a: hybrid_hybrid_krampus_brain_hybrid_hybrid_hybrid_m2239_s1.py (gen5)
# parent_b: hybrid_hybrid_hybrid_geomet_hybrid_hybrid_regret_m2442_s1.py (gen6)
# born: 2026-05-29T23:48:20Z

import hashlib
import math
import random
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def extract_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑feature extractor (Krampus stub)."""
    if not text.strip():
        return {}
    words = text.split()
    base = sum(hash(w) for w in words) % 1000
    return {
        "operator_visceral_ratio": (base % 10) / 10.0,
        "operator_tech_ratio": ((base // 10) % 10) / 10.0,
        "operator_legal_osint_ratio": ((base // 100) % 10) / 10.0,
        "operator_ledger_density": ((base // 1000) % 10) / 10.0,
        "operator_signal_strength": ((base // 2) % 10) / 10.0,
        "operator_noise_level": ((base // 3) % 10) / 10.0,
        "operator_latency": ((base // 5) % 10) / 10.0,
        "operator_bandwidth": ((base // 7) % 10) / 10.0,
        "operator_error_rate": ((base // 11) % 10) / 10.0,
        "operator_uptime": ((base // 13) % 10) / 10.0,
        "operator_load_factor": ((base // 17) % 10) / 10.0,
        "operator_security_score": ((base // 19) % 10) / 10.0,
    }


def dict_to_vector(feat_dict: Dict[str, float]) -> np.ndarray:
    """Convert a feature dict to a deterministic ordered NumPy vector."""
    if not feat_dict:
        return np.zeros(12)
    keys = sorted(feat_dict.keys())
    return np.array([feat_dict[k] for k in keys], dtype=float)


def minhash_curvature(text: str) -> float:
    """MinHash‑derived curvature in (0,1)."""
    h = hashlib.blake2b(text.encode(), digest_size=8).digest()
    i = int.from_bytes(h, byteorder="big", signed=False)
    raw = (i % 1000) / 100.0 - 5.0  # centre around 0
    return 1.0 / (1.0 + math.exp(-raw))


def shannon_entropy(text: str) -> float:
    """Shannon entropy of token frequencies."""
    tokens = text.split()
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def rotor_matrix_from_entropy(entropy: float, max_entropy: float = 5.0) -> np.ndarray:
    """
    Build a geometric rotor (2‑D rotation) from entropy.
    The rotation angle is θ = π·min(e, max_entropy) / max_entropy.
    The rotor acts on the first two dimensions; remaining dimensions are unchanged.
    """
    e = min(entropy, max_entropy)
    theta = math.pi * e / max_entropy
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]])
    R = np.identity(12)
    R[:2, :2] = rot
    return R


def entropy_scaling_factor(entropy: float, max_entropy: float = 5.0) -> float:
    """
    A mild positive scaling derived from entropy.
    Using a logarithmic map keeps values bounded:
        ρ(e) = 1 + log(1 + e / max_entropy)
    """
    e = min(entropy, max_entropy)
    return 1.0 + math.log1p(e / max_entropy)


class HybridLinUCB:
    """
    Hybrid LinUCB that fuses:
      • MinHash curvature (c) as a curvature‑aware regulariser,
      • Entropy‑driven geometric rotor (R) as a true linear map,
      • Entropy‑derived mild scaling (ρ) to temper magnitudes,
      • Regret‑weighted exploration (wₐ).
    """

    def __init__(self, actions: List[str], dim: int = 12, alpha: float = 1.0):
        self.actions = actions
        self.dim = dim
        self.alpha = alpha
        eps = 1e-6

        # Per‑action statistics
        self.A_inv: Dict[str, np.ndarray] = {}
        self.b: Dict[str, np.ndarray] = {}
        self.w: Dict[str, float] = {}
        self.last_reward: Dict[str, float] = {}

        for a in actions:
            # Initialise A as (I / c) once curvature is known; we start with identity
            self.A_inv[a] = np.identity(dim)
            self.b[a] = np.zeros(dim)
            self.w[a] = 1.0
            self.last_reward[a] = -float("inf")  # ensures first regret is zero

    def _theta(self, a: str) -> np.ndarray:
        """θₐ = Aₐ⁻¹ bₐ."""
        return self.A_inv[a] @ self.b[a]

    def _apply_transform(self, x: np.ndarray, entropy: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply rotor and scaling to the feature vector.
        Returns transformed vector and the rotor matrix (for possible introspection).
        """
        R = rotor_matrix_from_entropy(entropy)
        rho = entropy_scaling_factor(entropy)
        x_rot = R @ x
        return rho * x_rot, R

    def compute_ucb(self, a: str, x: np.ndarray, curvature: float, entropy: float) -> float:
        """
        Compute the hybrid UCB for action *a*.
        """
        x_tilde, _ = self._apply_transform(x, entropy)

        theta = self._theta(a)
        mean = float(theta @ x_tilde)

        # uncertainty term using the current inverse covariance
        uncertainty = math.sqrt(float(x_tilde @ self.A_inv[a] @ x_tilde))

        # blend curvature and entropy scaling into the exploration bonus
        bonus = self.alpha * (curvature + entropy_scaling_factor(entropy)) * self.w[a] * uncertainty
        return mean + bonus

    def select_action(self, text: str) -> Tuple[str, float, float]:
        """Select action with highest hybrid UCB for the given context."""
        feats = extract_features(text)
        x = dict_to_vector(feats)
        curvature = minhash_curvature(text)
        entropy = shannon_entropy(text)

        best_a = None
        best_val = -float("inf")
        for a in self.actions:
            val = self.compute_ucb(a, x, curvature, entropy)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a, curvature, entropy

    def _sherman_morrison_update(self, a: str, x: np.ndarray) -> None:
        """
        Update the inverse covariance matrix A⁻¹ using the Sherman‑Morrison formula:
            A⁻¹ ← A⁻¹ - (A⁻¹ x xᵀ A⁻¹) / (1 + xᵀ A⁻¹ x)
        """
        Ainv = self.A_inv[a]
        Ax = Ainv @ x
        denom = 1.0 + float(x @ Ax)
        if denom == 0.0:
            return  # numerical safeguard
        self.A_inv[a] = Ainv - np.outer(Ax, Ax) / denom

    def update(self, a: str, text: str, reward: float) -> None:
        """
        Update internal statistics after observing *reward* for action *a*.
        """
        feats = extract_features(text)
        x = dict_to_vector(feats)
        curvature = minhash_curvature(text)
        entropy = shannon_entropy(text)

        x_tilde, _ = self._apply_transform(x, entropy)

        # LinUCB parameter updates on the transformed vector
        self._sherman_morrison_update(a, x_tilde)
        self.b[a] += reward * x_tilde

        # Regret‑weighted exploration update
        max_reward_sofar = max(self.last_reward.values())
        regret = max(0.0, max_reward_sofar - reward)
        self.w[a] *= (1.0 + regret)

        # Record the reward for future regret calculations
        self.last_reward[a] = reward


# ----------------------------------------------------------------------
# Light‑weight demonstration (no I/O, deterministic)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    actions = ["allow", "block", "quarantine"]
    bandit = HybridLinUCB(actions, alpha=0.5)

    sample_texts = [
        "intrusion detected on node 7",
        "routine backup completed successfully",
        "unusual login attempt from unknown ip address",
        "system health check passed",
        "malware signature matched on file xyz",
    ]

    for _ in range(20):
        txt = random.choice(sample_texts)
        a, curv, ent = bandit.select_action(txt)
        # Simulated reward: higher for "block" on malicious texts, lower otherwise
        reward = 1.0 if ("intrusion" in txt or "malware" in txt) and a == "block" else 0.0
        bandit.update(a, txt, reward)