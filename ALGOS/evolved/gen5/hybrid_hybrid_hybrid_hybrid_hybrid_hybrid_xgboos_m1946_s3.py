# DARWIN HAMMER — match 1946, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_worksh_m1043_s0.py (gen4)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_regret_m283_s2.py (gen4)
# born: 2026-05-29T23:40:07Z

import math
import random
import hashlib
from datetime import date, datetime
from typing import Iterable, List, Tuple, Dict

import numpy as np

# ----------------------------------------------------------------------
# Parent A – pheromone utilities
# ----------------------------------------------------------------------
def _rng_from_text(text: str) -> random.Random:
    """Deterministic RNG seeded from a SHA‑256 hash of the text."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    return random.Random(seed)


def extract_full_features(text: str) -> Dict[str, float]:
    """Return a fixed dictionary of 13 pseudo‑features in [0,1)."""
    rnd = _rng_from_text(text)
    keys = [
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy",
        "psyche_dissociative_index",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
    ]
    return {k: rnd.random() for k in keys}


def doomsday_factor(year: int, month: int, day: int) -> float:
    """
    Map weekday to a scaling factor in (0,1].
    Mon → 1.0, Tue → 0.9, …, Sun → 0.4 (linear decay).
    """
    w = (date(year, month, day).weekday() + 1) % 7  # 1‑7, wrap Sun→0
    return 1.0 - 0.1 * ((w + 6) % 7)


def pheromone_signal(text: str, when: datetime) -> float:
    """Scalar pheromone signal P ∈ [0,1] for a given text and timestamp."""
    feats = extract_full_features(text)
    mean_feat = sum(feats.values()) / len(feats)
    day_factor = doomsday_factor(when.year, when.month, when.day)
    return mean_feat * day_factor


# ----------------------------------------------------------------------
# Parent B – MinHash, similarity & entropy utilities
# ----------------------------------------------------------------------
def _minhash_signature(tokens: Iterable[str], num_perm: int = 128) -> np.ndarray:
    """
    Lightweight MinHash: for each permutation a different seed is appended
    to the token bytes before hashing.
    """
    max_uint = np.iinfo(np.uint64).max
    sig = np.full(num_perm, max_uint, dtype=np.uint64)
    for token in tokens:
        token_bytes = token.encode("utf-8")
        for i in range(num_perm):
            h = hashlib.sha256(token_bytes + i.to_bytes(2, "little")).digest()
            hv = int.from_bytes(h[:8], "little")
            if hv < sig[i]:
                sig[i] = hv
    return sig


def minhash_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity via MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("Signature shapes must match")
    return float(np.mean(sig1 == sig2))


def shannon_entropy(counts: np.ndarray) -> float:
    """Natural‑log Shannon entropy of a discrete distribution given raw counts."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -float(np.sum(probs * np.log(probs)))


def combined_entropy(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Entropy of the multiset union of two MinHash signatures."""
    combined = np.concatenate([sig1, sig2])
    _, counts = np.unique(combined, return_counts=True)
    return shannon_entropy(counts.astype(float))


# ----------------------------------------------------------------------
# Hybrid loss utilities – deeper integration
# ----------------------------------------------------------------------
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable sigmoid."""
    x_arr = np.asarray(x, dtype=float)
    pos = x_arr >= 0
    out = np.empty_like(x_arr, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out.item() if np.isscalar(x) else out


def instance_weight(pheromone: float, similarity: float, beta: float = 0.5) -> float:
    """
    Weight applied to the logistic loss of a single instance.
    The weight grows with pheromone * similarity, encouraging
    the model to focus on “high‑signal” samples.
    """
    return 1.0 + beta * pheromone * similarity


def regularizer_term(pheromone: float, similarity: float, entropy: float, alpha: float = 0.1) -> float:
    """
    Additive regularisation term that does **not** depend on the model margin.
    """
    return alpha * pheromone * similarity * entropy


def adjusted_grad_hess(
    y_true: np.ndarray,
    margin: np.ndarray,
    weight: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient and Hessian of the *weighted* logistic loss.
    The regulariser contributes a constant to the objective but
    zero to the first‑ and second‑order derivatives w.r.t. the margin.
    """
    sigma = sigmoid(margin)
    grad = (sigma - y_true) * weight
    hess = sigma * (1.0 - sigma) * weight
    return grad, hess


def split_gain(
    G_left: float,
    H_left: float,
    G_right: float,
    H_right: float,
    G_total: float,
    H_total: float,
    lam: float = 1.0,
) -> float:
    """
    XGBoost split gain with L2 regularisation λ.
    Uses the standard formula (½)·(G²/(H+λ)).
    """
    def _gain(G, H):
        return (G ** 2) / (H + lam)

    return 0.5 * (_gain(G_left, H_left) + _gain(G_right, H_right) - _gain(G_total, H_total))


# ----------------------------------------------------------------------
# End‑to‑end statistics computation
# ----------------------------------------------------------------------
def compute_hybrid_statistics(
    texts: List[str],
    timestamps: List[datetime],
    tokens_a: List[List[str]],
    tokens_b: List[List[str]],
    y_true: np.ndarray,
    margin: np.ndarray,
    alpha: float = 0.1,
    beta: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each sample compute:
      * weighted gradient and hessian,
      * regulariser contribution,
    Returns three arrays: grads, hessians, reg_terms.
    """
    grads: List[float] = []
    hess: List[float] = []
    regs: List[float] = []

    for i, (txt, ts, t_a, t_b) in enumerate(zip(texts, timestamps, tokens_a, tokens_b)):
        P = pheromone_signal(txt, ts)
        sig_a = _minhash_signature(t_a)
        sig_b = _minhash_signature(t_b)

        s = minhash_similarity(sig_a, sig_b)
        H = combined_entropy(sig_a, sig_b)

        w = instance_weight(P, s, beta=beta)
        g, h = adjusted_grad_hess(
            y_true=np.array([y_true[i]]),
            margin=np.array([margin[i]]),
            weight=w,
        )
        grads.append(g.item())
        hess.append(h.item())
        regs.append(regularizer_term(P, s, H, alpha=alpha))

    return np.array(grads), np.array(hess), np.array(regs)


def compute_total_objective(
    y_true: np.ndarray,
    margin: np.ndarray,
    grads: np.ndarray,
    regs: np.ndarray,
) -> float:
    """
    Full objective value = Σ weighted logistic loss + Σ regulariser.
    The weighted logistic loss can be recovered from gradients:
        ℓ_i = (g_i + y_i)·m_i - log(1+exp(m_i))
    but we compute it directly for clarity.
    """
    sigma = sigmoid(margin)
    logistic = -np.mean(y_true * np.log(sigma + 1e-15) + (1 - y_true) * np.log(1 - sigma + 1e-15))
    return logistic + regs.mean()


# ----------------------------------------------------------------------
# Demonstration utilities
# ----------------------------------------------------------------------
def demo_split_gain():
    """Create a tiny synthetic split and compute its gain."""
    G_total, H_total = 3.0, 2.5
    G_left, H_left = 1.2, 0.9
    G_right, H_right = 1.8, 1.3
    gain = split_gain(G_left, H_left, G_right, H_right, G_total, H_total, lam=1.0)
    print(f"Demo split gain: {gain:.6f}")


def demo_hybrid_step():
    """Run a single hybrid gradient/hessian computation on dummy data."""
    texts = ["sample text one", "another example"]
    timestamps = [datetime(2023, 3, 14, 12, 0), datetime(2023, 3, 15, 9, 30)]
    tokens_a = [["sample", "text", "one"], ["another", "example"]]
    tokens_b = [["sample", "one"], ["another", "example", "extra"]]
    y_true = np.array([1, 0], dtype=float)
    margin = np.array([0.2, -0.4], dtype=float)

    grads, hess, regs = compute_hybrid_statistics(
        texts, timestamps, tokens_a, tokens_b, y_true, margin,
        alpha=0.1, beta=0.5,
    )
    obj = compute_total_objective(y_true, margin, grads, regs)

    print("Gradients:", grads)
    print("Hessians :", hess)
    print("Reg term :", regs)
    print("Objective:", obj)


if __name__ == "__main__":
    demo_split_gain()
    demo_hybrid_step()