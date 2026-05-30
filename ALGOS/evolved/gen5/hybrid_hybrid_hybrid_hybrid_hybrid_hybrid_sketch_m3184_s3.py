# DARWIN HAMMER — match 3184, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_hybrid_ternar_m241_s4.py (gen4)
# parent_b: hybrid_hybrid_sketches_rlct_hybrid_hybrid_distri_m194_s1.py (gen3)
# born: 2026-05-29T23:48:22Z

import numpy as np
import hashlib
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# --------------------------------------------------------------
# Utility functions – more mathematically sound and numerically stable
# --------------------------------------------------------------

def shannon_entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy with safe handling of zeros."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    mask = p > 0
    return -np.sum(p[mask] * np.log2(p[mask]))


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback–Leibler divergence D_KL(p‖q). Both vectors must sum to 1."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))


def regret_weighted_probs(p: np.ndarray, regret: np.ndarray, temperature: float = 0.1) -> np.ndarray:
    """
    Convert raw probabilities and regret values into a softmax‑weighted
    distribution. Higher regret lowers the weight.
    """
    p = np.asarray(p, dtype=np.float64)
    regret = np.asarray(regret, dtype=np.float64)
    # Prevent overflow / underflow
    logits = np.log(p + 1e-12) - regret / temperature
    logits -= np.max(logits)  # for numerical stability
    w = np.exp(logits)
    return w / w.sum()


def sign_quantisation(p: np.ndarray) -> np.ndarray:
    """
    Preserve magnitude information by mapping to {-1,0,1} while keeping
    the original scale as a weight vector.
    """
    signs = np.where(p > 0.5, 1, np.where(p < 0.5, -1, 0))
    magnitudes = np.abs(p - 0.5) * 2  # scale to [0,1]
    return signs * magnitudes


def path_signature(x: np.ndarray, t: int) -> np.ndarray:
    """
    A richer signature: mean, variance of increments and a weighted
    cumulative sum that respects a decreasing pruning schedule.
    """
    x = np.asarray(x, dtype=np.float64)[:t]
    if len(x) < 2:
        return np.zeros(3)
    increments = np.diff(x)
    schedule = decreasing_pruning_schedule(len(x))
    weighted_cumsum = np.cumsum(x * schedule)
    return np.array([
        np.mean(x),
        np.var(increments),
        weighted_cumsum[-1] / np.sum(schedule)
    ])


def decreasing_pruning_schedule(length: int, rate: float = 0.7) -> np.ndarray:
    """Geometric decay used to down‑weight older coordinates."""
    return rate ** np.arange(length)[::-1]  # newest gets highest weight


# --------------------------------------------------------------
# Sketching primitives – vectorised and integrated with regret
# --------------------------------------------------------------

def _hash(item: str, seed: int, width: int) -> int:
    """Deterministic hash based on SHA‑256 and a seed."""
    h = hashlib.sha256(f'{seed}:{item}'.encode()).hexdigest()
    return int(h, 16) % width


def count_min_sketch(items: List[str], width: int = 128, depth: int = 5,
                     weights: np.ndarray = None) -> np.ndarray:
    """
    Vectorised Count‑Min sketch. Optional `weights` (same length as items)
    allow regret‑weighted updates.
    """
    if weights is None:
        weights = np.ones(len(items), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    sketch = np.zeros((depth, width), dtype=np.float64)
    for d in range(depth):
        indices = np.fromiter((_hash(it, d, width) for it in items), dtype=np.int64, count=len(items))
        np.add.at(sketch[d], indices, weights)
    return sketch


def minhash_lsh_index(docs: Dict[int, List[str]], num_perm: int = 64) -> Dict[str, List[int]]:
    """
    Build an LSH index using MinHash signatures.
    Each document is represented by `num_perm` hash values; the smallest
    hash per permutation forms the signature.
    """
    signatures = {}
    for doc_id, shingles in docs.items():
        sig = []
        for perm in range(num_perm):
            min_hash = min(_hash(sh, perm, 2**32 - 1) for sh in shingles)
            sig.append(min_hash)
        signatures[doc_id] = tuple(sig)

    # Bucket by first k bits of the signature (here k=8)
    buckets = defaultdict(list)
    for doc_id, sig in signatures.items():
        key = ''.join(format(h, '08x')[:2] for h in sig[:4])  # simple 8‑bit key
        buckets[key].append(doc_id)
    return dict(buckets)


# --------------------------------------------------------------
# Core hybrid algorithm – deeper mathematical coupling
# --------------------------------------------------------------

def estimate_information_loss(original: np.ndarray,
                              reduced: np.ndarray) -> float:
    """
    Estimate loss as KL divergence between original distribution
    (e.g., empirical frequencies) and a reduced representation
    obtained from the sketch.
    """
    return kl_divergence(original, reduced)


def hybrid_operation(p: np.ndarray,
                     x: np.ndarray,
                     t: int,
                     items: List[str],
                     regrets: np.ndarray,
                     width: int = 128,
                     depth: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Integrated hybrid routine:
      1. Compute regret‑weighted probabilities.
      2. Derive a path signature that respects a pruning schedule.
      3. Build a regret‑weighted Count‑Min sketch.
      4. Estimate information loss via KL divergence between the
         empirical distribution of `p` and the normalized sketch frequencies.
    """
    # 1. Regret‑weighted probabilities (preserve magnitude)
    pw = regret_weighted_probs(p, regrets)

    # 2. Path signature with pruning
    signature = path_signature(x, t)

    # 3. Regret‑weighted sketch (weights = pw aligned to items if possible)
    # Align weights: if items length differs from p, broadcast or truncate.
    if len(items) != len(pw):
        # Simple repeat/truncate to match lengths
        repeats = int(np.ceil(len(items) / len(pw)))
        weights = np.tile(pw, repeats)[:len(items)]
    else:
        weights = pw
    sketch = count_min_sketch(items, width, depth, weights)

    # 4. Information loss: compare empirical distribution of p with sketch marginal
    # Collapse sketch to a single frequency vector (minimum across rows)
    sketch_counts = sketch.min(axis=0)  # conservative estimate
    if sketch_counts.sum() == 0:
        reduced_dist = np.full_like(pw, 1.0 / len(pw))
    else:
        reduced_dist = sketch_counts / sketch_counts.sum()
        # Pad/reduce to match length of pw
        if len(reduced_dist) != len(pw):
            repeats = int(np.ceil(len(pw) / len(reduced_dist)))
            reduced_dist = np.tile(reduced_dist, repeats)[:len(pw)]
            reduced_dist = reduced_dist / reduced_dist.sum()
    loss = estimate_information_loss(pw, reduced_dist)

    return pw, signature, sketch, loss


def tropical_broadcast(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Tropical (max‑plus) broadcast: each node receives the maximum
    incoming weight plus its own self‑weight.
    """
    adj = np.asarray(adjacency_matrix, dtype=np.float64)
    # Ensure diagonal contains self‑weights (could be zero)
    np.fill_diagonal(adj, np.maximum(np.diag(adj), 0.0))
    return np.max(adj, axis=0)


def hoeffding_split_test(loss: float, epsilon: float = 0.1, delta: float = 0.05) -> bool:
    """
    Hoeffding bound based split test.
    Accept split if loss is statistically significantly below epsilon.
    """
    bound = np.sqrt(np.log(2 / delta) / (2 * 1))  # n=1 for single aggregated loss
    return loss + bound < epsilon


# --------------------------------------------------------------
# Example driver – deterministic seed for reproducibility
# --------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Synthetic data
    p_raw = np.random.rand(12)
    regrets = np.random.exponential(scale=0.5, size=12)
    x_series = np.cumsum(np.random.randn(12))  # random walk
    t_cut = 8
    items = [f"item_{i}" for i in range(12)]
    adjacency = np.random.rand(12, 12)

    # Run hybrid routine
    probs, sig, sketch_mat, info_loss = hybrid_operation(
        p=p_raw,
        x=x_series,
        t=t_cut,
        items=items,
        regrets=regrets,
        width=256,
        depth=6
    )

    broadcast_vec = tropical_broadcast(adjacency)
    split_ok = hoeffding_split_test(info_loss)

    # Output
    print("Regret‑Weighted Probabilities:", probs)
    print("Path Signature:", sig)
    print("Count‑Min Sketch shape:", sketch_mat.shape)
    print("Estimated Information Loss (KL):", info_loss)
    print("Tropical Broadcast Vector:", broadcast_vec)
    print("Hoeffding Split Test Passed:", split_ok)