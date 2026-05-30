# DARWIN HAMMER — match 21, survivor 8
# gen: 4
# parent_a: korpus_text.py (gen0)
# parent_b: hybrid_hybrid_regret_engine_hybrid_bandit_router_m38_s5.py (gen3)
# born: 2026-05-29T23:26:34Z

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Deterministic utilities (replaces Parent A randomness)
# ----------------------------------------------------------------------
INT16_MAX = 2 ** 15 - 1
DEFAULT_MINHASH_K = 64
DEFAULT_SHINGLE_W = 5
DEFAULT_EMBED_DIM = 128
_FIXED_SEED = 0xC0FFEE  # deterministic seed for all pseudo‑random generators


def _stable_int_hash(data: bytes) -> int:
    """Stable 64‑bit integer hash using SHA‑256 (first 8 bytes)."""
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def _shingles(text: str, width: int = DEFAULT_SHINGLE_W) -> List[str]:
    """Return overlapping substrings (shingles) of length *width*."""
    cleaned = " ".join(text.split()).lower()
    return [cleaned[i: i + width] for i in range(len(cleaned) - width + 1)]


def _deterministic_seeds(k: int, base: int = _FIXED_SEED) -> List[int]:
    """Generate *k* deterministic 32‑bit seeds from a fixed base."""
    rng = random.Random(base)
    return [rng.randrange(0, 2 ** 32) for _ in range(k)]


def minhash_signature(
    tokens: Iterable[str],
    k: int = DEFAULT_MINHASH_K,
    width: int = DEFAULT_SHINGLE_W,
) -> List[int]:
    """
    Deterministic MinHash signature of length *k*.
    Tokens are first de‑duplicated; each seed yields the minimum hash value.
    """
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [0] * k

    seeds = _deterministic_seeds(k)
    signature = []
    for seed in seeds:
        # combine seed with token deterministically
        min_hash = min(
            _stable_int_hash(seed.to_bytes(4, "big") + t.encode("utf-8", "ignore"))
            for t in token_set
        )
        signature.append(min_hash)
    return signature


def minhash_for_text(text: str, k: int = DEFAULT_MINHASH_K) -> List[int]:
    """Convenient wrapper for raw text."""
    return minhash_signature(_shingles(text or ""), k=k)


def shannon_entropy(chars: List[str]) -> float:
    """Shannon entropy over a list of characters."""
    if not chars:
        return 0.0
    counts: Dict[str, int] = {}
    for c in chars:
        counts[c] = counts.get(c, 0) + 1
    total = len(chars)
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())


def entropy_for_text(text: str, max_len: int = 10_000) -> float:
    """Entropy of the first *max_len* characters."""
    return shannon_entropy(list((text or "")[:max_len])) if text else 0.0


def deterministic_embedding(text: str, dim: int = DEFAULT_EMBED_DIM) -> np.ndarray:
    """
    Fully deterministic pseudo‑embedding.
    Each dimension is derived from a stable hash of (text, index) → int16,
    then scaled to [-1, 1].
    """
    vec = np.empty(dim, dtype=np.float32)
    for i in range(dim):
        data = f"{text}|{i}".encode("utf-8", "ignore")
        h = _stable_int_hash(data)
        # map 0 … 2**64‑1 → -INT16_MAX … INT16_MAX
        signed = (h % (2 * INT16_MAX + 1)) - INT16_MAX
        vec[i] = signed / INT16_MAX
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity handling zero‑vectors gracefully."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ----------------------------------------------------------------------
# Core data structures (shared by both parents)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    """An action with a textual description and scalar attributes."""
    id: str
    description: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    """Counterfactual outcome for an action."""
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass
class StoreState:
    """Store providing the bounded *dance* signal."""
    dance: float = 1.0  # neutral scaling

    def update(self, reward: float) -> None:
        """Exponential moving average, clamped to [0.5, 1.5]."""
        alpha = 0.1
        self.dance = max(0.5, min(1.5, (1 - alpha) * self.dance + alpha * reward))


# ----------------------------------------------------------------------
# Enhanced hybrid mathematics (deeper integration)
# ----------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard similarity estimate from two MinHash signatures."""
    if len(sig_a) != len(sig_b) or not sig_a:
        return 0.0
    matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
    return matches / len(sig_a)


def linucb_confidence(A_inv: np.ndarray, x: np.ndarray, alpha: float = 1.0) -> float:
    """LinUCB confidence bound √(xᵀ A⁻¹ x) scaled by *alpha*."""
    return alpha * math.sqrt(float(x.T @ A_inv @ x))


def hybrid_score(
    action: MathAction,
    counterfactuals: List[MathCounterfactual],
    ref_signature: List[int],
    ref_embedding: np.ndarray,
    store: StoreState,
    A_inv: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> float:
    """
    Compute a richer hybrid score.

    1. Raw regret Rᵢ = EV − cost − risk + Σ cf contributions.
    2. Regret weighting g(Rᵢ) = sigmoid(Rᵢ).
    3. Textual similarity:
       • MinHash Jaccard similarity (sim_j).
       • Cosine similarity of deterministic embeddings (sim_c).
       Combined via weighted geometric mean: sim_text = (sim_j ** (1‑β)) * (sim_c ** β).
    4. Entropy boost: higher entropy of description → modest increase.
    5. LinUCB confidence bound (conf).
    6. Final score:
       Sᵢ = g(Rᵢ) · (1 + sim_text) · (1 + entropy_factor) · store.dance · (1 + conf)
    """
    # 1. Raw regret
    cf_sum = sum(
        cf.outcome_value * cf.probability
        for cf in counterfactuals
        if cf.action_id == action.id
    )
    R_i = action.expected_value - action.cost - action.risk + cf_sum

    # 2. Regret weighting
    g_R = _sigmoid(R_i)

    # 3. Textual similarities
    act_sig = minhash_signature(_shingles(action.description), k=len(ref_signature))
    sim_j = jaccard_similarity(act_sig, ref_signature)

    act_emb = deterministic_embedding(action.description, dim=ref_embedding.shape[0])
    sim_c = cosine_similarity(act_emb, ref_embedding)

    # Weighted geometric mean (avoid zero by adding eps)
    eps = 1e-12
    sim_text = (max(sim_j, eps) ** (1 - beta)) * (max(sim_c, eps) ** beta)

    # 4. Entropy factor (scaled to [0, 0.2] for stability)
    ent = entropy_for_text(action.description)
    # Normalise by theoretical max log2(|alphabet|) ≈ 8 for ASCII printable
    max_ent = math.log2(95)  # printable ASCII range
    entropy_factor = 0.2 * min(ent / max_ent, 1.0)

    # 5. Confidence bound
    conf = linucb_confidence(A_inv, act_emb, alpha=alpha)

    # 6. Combine
    score = (
        g_R
        * (1.0 + sim_text)
        * (1.0 + entropy_factor)
        * store.dance
        * (1.0 + conf)
    )
    return score


def softmax(scores: List[float]) -> List[float]:
    """Numerically stable softmax."""
    if not scores:
        return []
    max_s = max(scores)
    exps = [math.exp(s - max_s) for s in scores]
    sum_exps = sum(exps) or 1.0
    return [e / sum_exps for e in exps]


def select_action(actions: List[MathAction], scores: List[float]) -> MathAction:
    """Sample an action proportionally to softmax(scores)."""
    probs = softmax(scores)
    chosen_id = random.choices([a.id for a in actions], weights=probs, k=1)[0]
    return next(a for a in actions if a.id == chosen_id)


# ----------------------------------------------------------------------
# Demonstration / helper functions (≥3 functions)
# ----------------------------------------------------------------------
def compute_reference_signature(past_actions: List[MathAction]) -> List[int]:
    """
    Build a reference MinHash signature by aggregating shingles from all past actions.
    The aggregation is performed by taking the union of all shingles and then
    computing a deterministic MinHash signature of the resulting set.
    """
    all_shingles: set = set()
    for act in past_actions:
        all_shingles.update(_shingles(act.description))
    return minhash_signature(all_shingles, k=DEFAULT_MINHASH_K)


def compute_reference_embedding(past_actions: List[MathAction]) -> np.ndarray:
    """
    Compute a reference embedding as the (L2‑normalised) mean of deterministic
    embeddings of the provided actions. If no actions are supplied, returns a zero vector.
    """
    if not past_actions:
        return np.zeros(DEFAULT_EMBED_DIM, dtype=np.float32)
    embeddings = np.stack(
        [deterministic_embedding(act.description, dim=DEFAULT_EMBED_DIM) for act in past_actions]
    )
    mean_vec = embeddings.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec


def run_hybrid_bandit(
    actions: List[MathAction],
    counterfactuals: List[MathCounterfactual],
    past_actions: List[MathAction],
    store: StoreState,
    A_inv: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
) -> MathAction:
    """
    End‑to‑end demo: compute reference structures, score each candidate,
    select an action, update the store with a mock reward, and return the chosen action.
    """
    ref_sig = compute_reference_signature(past_actions)
    ref_emb = compute_reference_embedding(past_actions)

    scores = [
        hybrid_score(
            act,
            counterfactuals,
            ref_sig,
            ref_emb,
            store,
            A_inv,
            alpha=alpha,
            beta=beta,
        )
        for act in actions
    ]

    chosen = select_action(actions, scores)

    # Mock reward: use expected value as proxy (in real use‑case replace with true feedback)
    mock_reward = chosen.expected_value
    store.update(mock_reward)

    return chosen