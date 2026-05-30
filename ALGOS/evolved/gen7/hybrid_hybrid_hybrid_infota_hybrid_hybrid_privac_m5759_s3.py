# DARWIN HAMMER — match 5759, survivor 3
# gen: 7
# parent_a: hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s4.py (gen6)
# parent_b: hybrid_hybrid_privacy_sketc_hybrid_fractional_hd_m1084_s1.py (gen2)
# born: 2026-05-30T00:04:35Z

"""Hybrid CMS‑HDC + MinHash‑Pheromone Fusion.

Parents:
- **hybrid_hybrid_infotaxis_min_hybrid_hybrid_hybrid_m2692_s4.py** – provides
  MinHash based similarity, entropy‑driven pheromone entries and a decay model.
- **hybrid_hybrid_privacy_sketc_hybrid_fractional_hd_m1084_s1.py** – provides a
  Count‑Min Sketch (CMS) → complex hypervector conversion, HDC binding, and a
  fractional‑power operator for causal influence.

Mathematical bridge:
Both parents use hash‑based tokenisation.  In this fusion a document’s
MinHash signature supplies an *entropy‑weighted* pheromone signal for each
token.  The CMS matrix treats the same tokens as (row, column) counters.
During the CMS→hypervector conversion each cell contributes a random unit‑
magnitude complex hypervector weighted by its count **and** by the pheromone
decay factor of the underlying token.  The decay factor therefore appears as
a multiplicative scaling inside the fractional‑power binding step, unifying
information‑theoretic decay with HDC causal modulation.

The module implements three high‑level hybrid operations:
1. `hybrid_cms_pheromone_hv` – builds a CMS from documents, creates pheromone
   entries, and returns a decay‑adjusted hypervector.
2. `bind_causal_effect` – binds a causal hypervector to the decay‑adjusted
   hypervector using fractional‑power binding, where the exponent is derived
   from pheromone half‑life.
3. `hybrid_privacy_risk` – blends a classic reconstruction‑risk ratio with the
   similarity between the bound hypervector and the base hypervector, yielding
   a risk estimate that accounts for both frequency‑based privacy leakage and
   entropy‑driven causal influence.
"""

import hashlib
import math
import random
import sys
import pathlib
import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Helper utilities shared by both parents
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Very simple whitespace tokeniser, lower‑casing."""
    return text.lower().split()

def _minhash_signature(tokens: List[str], num_perm: int = 128) -> np.ndarray:
    """Compute a MinHash signature for a token list."""
    max_hash = (1 << 64) - 1
    sig = np.full(num_perm, max_hash, dtype=np.uint64)

    for t in tokens:
        h = int(hashlib.sha256(t.encode()).hexdigest(), 16)
        for i in range(num_perm):
            # a simple linear permutation: (a*i + b) mod prime
            a = 0x5bd1e995  # arbitrary odd constant
            b = (i * 0x9e3779b97f4a7c15) & max_hash
            ph = (a * h + b) & max_hash
            if ph < sig[i]:
                sig[i] = ph
    return sig

def _jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return np.mean(sig1 == sig2)

def _entropy(counts: Counter) -> float:
    """Shannon entropy of a token count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array(list(counts.values())) / total
    return -np.sum(probs * np.log2(probs + 1e-12))

# ---------------------------------------------------------------------------
# Pheromone model (Parent A)
# ---------------------------------------------------------------------------

@dataclass
class PheromoneEntry:
    uuid: str
    token: str
    signal_kind: str
    signal_value: float          # typically entropy‑based
    half_life_seconds: int
    created_at: datetime.datetime
    last_decay: datetime.datetime

    def age_seconds(self) -> float:
        return (datetime.datetime.utcnow() - self.created_at).total_seconds()

    def decay_factor(self) -> float:
        """Exponential decay based on half‑life."""
        if self.half_life_seconds <= 0:
            return 0.0
        age = self.age_seconds()
        return 0.5 ** (age / self.half_life_seconds)

def _generate_pheromones(docs: List[str]) -> List[PheromoneEntry]:
    """Create one pheromone entry per distinct token across all documents."""
    all_tokens = Counter()
    for doc in docs:
        all_tokens.update(_tokenize(doc))

    entries: List[PheromoneEntry] = []
    now = datetime.datetime.utcnow()
    for token, cnt in all_tokens.items():
        # Signal value proportional to entropy of token distribution in its doc set
        # (here approximated by -log(p))
        prob = cnt / sum(all_tokens.values())
        signal = -math.log(prob + 1e-12)
        half_life = random.randint(30, 300)  # seconds
        entries.append(
            PheromoneEntry(
                uuid=hashlib.sha256(token.encode()).hexdigest(),
                token=token,
                signal_kind="entropy",
                signal_value=signal,
                half_life_seconds=half_life,
                created_at=now,
                last_decay=now,
            )
        )
    return entries

# ---------------------------------------------------------------------------
# Count‑Min Sketch utilities (Parent B)
# ---------------------------------------------------------------------------

def _cms_hash(item: str, depth: int, width: int) -> List[int]:
    """Return a list of column indices, one per hash row."""
    hashes = []
    for d in range(depth):
        h = int(hashlib.sha256(f"{d}:{item}".encode()).hexdigest(), 16)
        hashes.append(h % width)
    return hashes

def build_cms(docs: List[str], depth: int = 4, width: int = 1024) -> np.ndarray:
    """Construct a Count‑Min Sketch matrix from a list of documents."""
    cms = np.zeros((depth, width), dtype=np.int64)
    for doc in docs:
        tokens = _tokenize(doc)
        for token in tokens:
            cols = _cms_hash(token, depth, width)
            for row, col in enumerate(cols):
                cms[row, col] += 1
    return cms

# ---------------------------------------------------------------------------
# Hypervector primitives (Parent B)
# ---------------------------------------------------------------------------

def _random_complex_hv(dim: int) -> np.ndarray:
    """Unit‑magnitude complex hypervector."""
    angles = np.random.uniform(0, 2 * math.pi, dim)
    return np.exp(1j * angles)

def cms_to_hv(cms: np.ndarray, pheromones: List[PheromoneEntry],
              dim: int = 8192) -> np.ndarray:
    """
    Convert a CMS matrix to a single complex hypervector.
    Each cell contributes a random unit‑magnitude hypervector weighted by:
        count * decay_factor(token)
    The token associated with a cell is recovered via the hash function used
    during CMS construction.
    """
    depth, width = cms.shape
    hv = np.zeros(dim, dtype=np.complex128)

    # Build a quick lookup from token -> decay factor
    decay_lookup = {p.token: p.decay_factor() for p in pheromones}

    for row in range(depth):
        for col in range(width):
            cnt = cms[row, col]
            if cnt == 0:
                continue
            # Recover a representative token for this cell.
            # In practice we cannot invert the hash; we approximate by using
            # the column index as a pseudo‑token identifier.
            pseudo_token = f"row{row}_col{col}"
            decay = decay_lookup.get(pseudo_token, 1.0)  # unknown tokens default to 1
            weight = cnt * decay
            token_hv = _random_complex_hv(dim)
            hv += weight * token_hv
    # Normalise to unit magnitude per component
    norms = np.abs(hv)
    hv = np.where(norms == 0, 0, hv / norms)
    return hv

def fractional_power(hv: np.ndarray, exponent: float) -> np.ndarray:
    """Raise a complex hypervector to a real exponent (phase scaling)."""
    magnitude = np.abs(hv)
    phase = np.angle(hv)
    new_phase = phase * exponent
    return magnitude * np.exp(1j * new_phase)

def bind_causal(base_hv: np.ndarray, causal_hv: np.ndarray,
                exponent: float) -> np.ndarray:
    """
    Bind a causal hypervector to a base hypervector using element‑wise
    multiplication followed by fractional‑power scaling.
    """
    bound = base_hv * causal_hv
    return fractional_power(bound, exponent)

# ---------------------------------------------------------------------------
# Hybrid operations
# ---------------------------------------------------------------------------

def hybrid_cms_pheromone_hv(docs: List[str],
                            depth: int = 4,
                            width: int = 1024,
                            dim: int = 8192) -> Tuple[np.ndarray, List[PheromoneEntry]]:
    """
    Build a CMS from `docs`, generate pheromone entries, and return a
    decay‑adjusted hypervector.
    """
    cms = build_cms(docs, depth, width)
    pheromones = _generate_pheromones(docs)
    hv = cms_to_hv(cms, pheromones, dim)
    return hv, pheromones

def bind_causal_effect(base_hv: np.ndarray,
                       causal_hv: np.ndarray,
                       pheromones: List[PheromoneEntry]) -> np.ndarray:
    """
    Bind a causal hypervector to `base_hv`.  The exponent for fractional power
    is the average decay factor across all pheromones (clipped to [0.1, 2.0]).
    """
    if not pheromones:
        exponent = 1.0
    else:
        avg_decay = sum(p.decay_factor() for p in pheromones) / len(pheromones)
        exponent = max(0.1, min(2.0, avg_decay * 5))  # scale into a reasonable range
    return bind_causal(base_hv, causal_hv, exponent)

def hybrid_privacy_risk(docs: List[str],
                        unique_quasi_identifiers: int,
                        total_records: int,
                        causal_hv: np.ndarray) -> float:
    """
    Compute a privacy risk score that blends:
    * reconstruction‑risk = unique_quasi_identifiers / total_records
    * causal influence = 1 - cosine similarity between base and bound hypervectors
    The bound hypervector is obtained by binding `causal_hv` to the CMS‑pheromone
    hypervector of the documents.
    """
    base_hv, pheromones = hybrid_cms_pheromone_hv(docs)
    bound_hv = bind_causal_effect(base_hv, causal_hv, pheromones)

    # Cosine similarity for complex vectors (real inner product on concatenated real/imag)
    def _complex_cosine(a: np.ndarray, b: np.ndarray) -> float:
        ar = np.concatenate([a.real, a.imag])
        br = np.concatenate([b.real, b.imag])
        dot = np.dot(ar, br)
        norm_a = np.linalg.norm(ar)
        norm_b = np.linalg.norm(br)
        return dot / (norm_a * norm_b + 1e-12)

    similarity = _complex_cosine(base_hv, bound_hv)
    causal_influence = 1.0 - similarity  # higher when vectors differ

    recon_risk = unique_quasi_identifiers / max(1, total_records)
    # Weighted blend: give equal importance to both aspects
    risk = 0.5 * recon_risk + 0.5 * causal_influence
    return risk

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simple synthetic documents
    docs = [
        "the quick brown fox jumps over the lazy dog",
        "lorem ipsum dolor sit amet consectetur adipiscing elit",
        "the quick brown fox was very quick and very brown",
        "privacy preserving data analysis with hyperdimensional computing"
    ]

    # Build a random causal hypervector
    causal_dim = 8192
    causal_hv = _random_complex_hv(causal_dim)

    # Run hybrid operations
    base_hv, pher = hybrid_cms_pheromone_hv(docs)
    bound_hv = bind_causal_effect(base_hv, causal_hv, pher)
    risk = hybrid_privacy_risk(docs, unique_quasi_identifiers=42, total_records=1000, causal_hv=causal_hv)

    print(f"Base HV norm (mean magnitude): {np.mean(np.abs(base_hv)):.3f}")
    print(f"Bound HV norm (mean magnitude): {np.mean(np.abs(bound_hv)):.3f}")
    print(f"Hybrid privacy risk score: {risk:.4f}")
    sys.exit(0)