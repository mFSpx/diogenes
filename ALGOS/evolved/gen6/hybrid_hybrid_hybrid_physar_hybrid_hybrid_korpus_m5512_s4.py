# DARWIN HAMMER — match 5512, survivor 4
# gen: 6
# parent_a: hybrid_hybrid_physarum_netw_hybrid_hybrid_hybrid_m373_s0.py (gen4)
# parent_b: hybrid_hybrid_korpus_text_h_hybrid_hybrid_hybrid_m1341_s3.py (gen5)
# born: 2026-05-30T00:02:27Z

"""Hybrid Physarum‑MinHash‑HDC Algorithm
Parents:
- hybrid_hybrid_physarum_netw_hybrid_hybrid_m373_s0.py (physarum flux + Fisher information)
- hybrid_hybrid_korpus_text_h_hybrid_hybrid_hybrid_m1341_s3.py (MinHash → hypervector, regret, free‑energy)

Mathematical bridge:
Both parents extract textual statistics.  Parent A counts pattern matches (Fisher information) and
uses the count to bias conductance updates in a Physarum network.  Parent B builds a MinHash
signature, expands each hash into a bipolar hypervector and combines them (binding) to obtain a
high‑dimensional reference vector.  The bridge is formed by feeding the Fisher count as a scalar
modulation of the hypervector‑based free‑energy term and by letting the hybrid regret‑weighted
score directly scale the Physarum conductance gain.  Consequently the conductance dynamics
are driven by a unified score

    g_i = σ(R_i)·(1+J_i)·exp(−F_i) ,

where σ is a sigmoid over the regret term R_i, J_i is the Jaccard similarity of MinHash
signatures, and F_i is a variational free‑energy computed from the action’s hypervector,
the reference hypervector and an observation vector whose amplitude is proportional to the
Fisher information of the underlying text.  This creates a single adaptive system that
simultaneously routes flow in the Physarum network and evaluates textual actions in the
HDC‑Regret space.
"""

import re
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Callable

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Physarum & Fisher utilities
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float            # interpreted as inflow rate
    expected_reward: float
    confidence_bound: float      # interpreted as outflow rate
    algorithm: str


def flux(conductance: float, edge_length: float,
         pressure_a: float, pressure_b: float,
         eps: float = 1e-12) -> float:
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)


def update_conductance(conductance: float, q: float,
                       dt: float = 1.0, gain: float = 1.0,
                       decay: float = 0.05) -> float:
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non‑negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))


def fisher_information(text: str, feature_regex: re.Pattern) -> float:
    """Simple Fisher information proxy: count of regex matches."""
    return float(len(feature_regex.findall(text)))


# ----------------------------------------------------------------------
# Parent B – MinHash / Hyperdimensional utilities
# ----------------------------------------------------------------------
INT16_MAX = 2**15 - 1
DEFAULT_DIM = 10000  # dimensionality for hypervectors


def shingles(text: str, width: int = 5) -> List[str]:
    txt = re.sub(r"\s+", " ", text or "").strip().lower()
    return [txt[i:i + width] for i in range(len(txt) - width + 1)]


def minhash(tokens: Iterable[str], k: int = 64) -> List[int]:
    """Return a list of k integer hashes (MinHash signature)."""
    token_set = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not token_set:
        return [0] * k

    signature = []
    for seed in range(k):
        min_hash = sys.maxsize
        for t in token_set:
            # deterministic hash using seed
            h = hash((t, seed))
            if h < min_hash:
                min_hash = h
        signature.append(min_hash & ((1 << 32) - 1))  # keep 32‑bit positive
    return signature


def _bipolar_vector(seed: int, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Generate a bipolar (+1 / -1) hypervector from an integer seed."""
    rng = np.random.RandomState(seed % (2**32))
    vec = rng.randint(0, 2, size=dim, dtype=np.int8)
    return np.where(vec == 0, -1, 1).astype(np.int8)


def hypervector_from_signature(sig: List[int],
                               dim: int = DEFAULT_DIM) -> np.ndarray:
    """Bind (element‑wise multiply) bipolar vectors generated from each hash."""
    hv = np.ones(dim, dtype=np.int8)
    for h in sig:
        hv = hv * _bipolar_vector(h, dim)
    return hv


def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
    """Jaccard similarity on sets of hash values."""
    set_a, set_b = set(sig_a), set(sig_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def variational_free_energy(hv_action: np.ndarray,
                            hv_ref: np.ndarray,
                            observation: np.ndarray) -> float:
    """
    Simple quadratic free‑energy:
        F = 0.5 * ||hv_action - hv_ref||² + λ * ||observation||²
    λ is fixed to 1e-3.
    """
    lam = 1e-3
    diff = hv_action.astype(np.float32) - hv_ref.astype(np.float32)
    energy = 0.5 * np.sum(diff * diff) + lam * np.sum(observation * observation)
    return float(energy)


# ----------------------------------------------------------------------
# Hybrid core functions
# ----------------------------------------------------------------------
def hybrid_regret_score(bandit: BanditAction,
                        cost: float,
                        risk: float) -> float:
    """Regret term R_i = expected_reward – cost – risk."""
    return bandit.expected_reward - cost - risk


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def hybrid_score(bandit: BanditAction,
                 cost: float,
                 risk: float,
                 sig_action: List[int],
                 sig_ref: List[int],
                 hv_action: np.ndarray,
                 hv_ref: np.ndarray,
                 fisher_cnt: float) -> float:
    """
    Compute the unified hybrid score

        S = σ(R) · (1 + J) · exp( –F ),

    where the observation vector amplitude is scaled by the Fisher count.
    """
    R = hybrid_regret_score(bandit, cost, risk)
    sigma_R = sigmoid(R)

    J = jaccard_similarity(sig_action, sig_ref)

    # Observation vector: a random bipolar vector scaled by Fisher count
    obs_seed = int(fisher_cnt) % (2**32)
    observation = _bipolar_vector(obs_seed, hv_action.shape[0]).astype(np.float32) * fisher_cnt

    F = variational_free_energy(hv_action, hv_ref, observation)

    return sigma_R * (1.0 + J) * math.exp(-F)


def physarum_hybrid_update(edge_length: float,
                           pressure_a: float,
                           pressure_b: float,
                           conductance: float,
                           bandit: BanditAction,
                           text: str,
                           feature_regex: re.Pattern,
                           ref_signature: List[int],
                           dim: int = DEFAULT_DIM,
                           dt: float = 1.0,
                           decay: float = 0.05) -> Tuple[float, float]:
    """
    Perform a single hybrid update:
    1. Compute flux q from current pressures.
    2. Derive Fisher information from the text.
    3. Build MinHash signature of the text and its hypervector.
    4. Evaluate hybrid score S.
    5. Use S as the gain factor in the conductance update.
    Returns (new_conductance, hybrid_score).
    """
    q = flux(conductance, edge_length, pressure_a, pressure_b)

    # Fisher information
    fisher_cnt = fisher_information(text, feature_regex)

    # MinHash and hypervector for the current action text
    tokens = shingles(text)
    sig_action = minhash(tokens, k=len(ref_signature))
    hv_action = hypervector_from_signature(sig_action, dim)

    # Hybrid score (acts as gain)
    # For demonstration we treat cost = bandit.propensity and risk = bandit.confidence_bound
    S = hybrid_score(bandit,
                     cost=bandit.propensity,
                     risk=bandit.confidence_bound,
                     sig_action=sig_action,
                     sig_ref=ref_signature,
                     hv_action=hv_action,
                     hv_ref=hypervector_from_signature(ref_signature, dim),
                     fisher_cnt=fisher_cnt)

    new_conductance = update_conductance(conductance, q,
                                         dt=dt,
                                         gain=S,
                                         decay=decay)
    return new_conductance, S


def build_reference_signature(corpus_texts: List[str],
                              k: int = 64) -> List[int]:
    """
    Build a reference MinHash signature from a list of corpus documents.
    The signature is the element‑wise minimum over all document signatures.
    """
    if not corpus_texts:
        return [0] * k
    min_sig = [sys.maxsize] * k
    for txt in corpus_texts:
        toks = shingles(txt)
        sig = minhash(toks, k)
        min_sig = [min(a, b) for a, b in zip(min_sig, sig)]
    return min_sig


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Sample data
    example_text = "The quick brown fox jumps over the lazy dog."
    feature_pat = re.compile(r"\b\w{4,}\b")          # words with 4+ letters

    # Create a dummy reference corpus
    corpus = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Pack my box with five dozen liquor jugs."
    ]
    ref_sig = build_reference_signature(corpus, k=32)

    # Initialise a BanditAction
    ba = BanditAction(
        action_id="act_1",
        propensity=0.3,
        expected_reward=1.2,
        confidence_bound=0.2,
        algorithm="demo"
    )

    # Physarum edge parameters
    edge_len = 1.5
    p_a = 1.0
    p_b = 0.0
    conduct = 0.8

    # Perform hybrid update
    new_cond, score = physarum_hybrid_update(
        edge_length=edge_len,
        pressure_a=p_a,
        pressure_b=p_b,
        conductance=conduct,
        bandit=ba,
        text=example_text,
        feature_regex=feature_pat,
        ref_signature=ref_sig,
        dim=DEFAULT_DIM,
        dt=1.0,
        decay=0.05
    )

    print(f"Hybrid score (gain): {score:.6e}")
    print(f"Conductance updated from {conduct:.4f} to {new_cond:.4f}")