# DARWIN HAMMER — match 1818, survivor 4
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hdc_serpentin_m619_s1.py (gen4)
# parent_b: hybrid_hybrid_decision_hygi_ternary_lens_audit_m19_s2.py (gen2)
# born: 2026-05-29T23:38:58Z

"""hybrid_minhash_hdc_decision_entropy.py
Fusion of:
* Parent A – Hyperdimensional MinHash Serpentina Self‑Righting Morphology
* Parent B – Hybrid Ternary Lens Audit & Decision‑Hygiene Module

Mathematical bridge
------------------
Parent A produces a high‑dimensional morphology vector **M** and a MinHash
signature **h** (list of 64‑bit integers).  The signature can be normalised
to a real‑valued vector **H** and bound to **M** by element‑wise multiplication:

  **B** = **M** ⊙ **H**

Parent B extracts a nine‑dimensional feature‑count vector **v** from free‑text,
computes a hygiene score *s* = **w⁺**·**v** − **w⁻**·**v**, and an entropy
*H* from the probability distribution *p* = **v**/Σ**v**.  The hybrid score is

  Sₕ = s·(1 + H/Hₘₐₓ), Hₘₐₓ = log₂ 9.

The fusion treats the bound hyper‑vector **B** as a “state” vector whose
cosine similarity to a goal vector **G** expresses recovery priority.
The final unified metric combines this similarity with the decision‑hygiene
score:

  Score = (cos θ(**B**, **G**))·Sₕ

Thus the MinHash‑HD‑binding supplies a geometric similarity term, while the
regex‑driven feature counts supply a hygiene‑entropy weighting term.  The
module implements this combined pipeline.

"""

import hashlib
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# ----------------------------------------------------------------------
# Parent A – morphology & MinHash utilities
# ----------------------------------------------------------------------
MAX64 = (1 << 64) - 1
Vector = List[float]


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float
    tokens: List[str]


def _hash(seed: int, token: str) -> int:
    """Deterministic 64‑bit hash of a token with a seed."""
    data = seed.to_bytes(4, "big") + token.encode("utf-8", errors="ignore")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


def minhash_signature(tokens: List[str], k: int = 128) -> List[int]:
    """MinHash signature of a token set."""
    toks = {t for t in tokens if t}
    if k <= 0:
        raise ValueError("k must be positive")
    if not toks:
        return [MAX64] * k
    return [min(_hash(i, t) for t in toks) for i in range(k)]


def morphology_vector(m: Morphology, dim: int = 10000) -> Vector:
    """Create a high‑dimensional vector representing the morphology."""
    seed_bytes = hashlib.sha256(
        f"{m.length}{m.width}{m.height}{m.mass}".encode("utf-8")
    ).digest()[:8]
    seed = int.from_bytes(seed_bytes, "big")
    random.seed(seed)  # deterministic per morphology
    vec = [random.random() for _ in range(dim)]
    # Modulate by physical attributes (repeat to match dim)
    attrs = np.array([m.length, m.width, m.height, m.mass])
    repeats = dim // len(attrs) + 1
    mod = np.tile(attrs, repeats)[:dim]
    return (np.array(vec) * mod).tolist()


def bind(a: Vector, b: Vector) -> Vector:
    """Element‑wise binding (Hadamard product)."""
    if len(a) != len(b):
        raise ValueError("vectors must be the same length for binding")
    return (np.array(a) * np.array(b)).tolist()


def cosine_similarity(x: Vector, y: Vector) -> float:
    """Cosine similarity between two real vectors."""
    xv = np.array(x)
    yv = np.array(y)
    norm_x = np.linalg.norm(xv)
    norm_y = np.linalg.norm(yv)
    if norm_x == 0 or norm_y == 0:
        return 0.0
    return float(np.dot(xv, yv) / (norm_x * norm_y))


# ----------------------------------------------------------------------
# Parent B – regex feature extraction, hygiene & entropy
# ----------------------------------------------------------------------
# Nine illustrative regex categories (the original uses 9; we provide 9 here)
FEATURE_REGEXES = [
    # 0 – evidence / verification
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    # 1 – planning / roadmap
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    # 2 – delay / pause
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|delay)\b",
    # 3 – risk
    r"\b(?:risk|danger|threat|vulnerability|exposure|hazard)\b",
    # 4 – compliance
    r"\b(?:compliance|regulation|standard|policy|governance|audit|certification)\b",
    # 5 – performance
    r"\b(?:performance|latency|throughput|speed|efficiency|optimisation|optimize)\b",
    # 6 – security
    r"\b(?:security|authenticat(?:ion|e)|encryption|firewall|intrusion|malware)\b",
    # 7 – cost
    r"\b(?:cost|price|budget|expense|charge|fee|spend)\b",
    # 8 – quality
    r"\b(?:quality|reliability|robust|stable|precision|accuracy)\b",
]

# Fixed positive/negative weight vectors (could be learned; here deterministic)
W_POS = np.array([1.2, 0.9, 0.5, 1.0, 0.8, 1.1, 1.3, 0.7, 0.6])
W_NEG = np.array([0.3, 0.4, 0.6, 0.2, 0.5, 0.3, 0.4, 0.5, 0.2])


def extract_feature_counts(text: str) -> np.ndarray:
    """Return a 9‑dimensional count vector for the regex feature set."""
    counts = []
    for pat in FEATURE_REGEXES:
        cnt = len(re.findall(pat, text, flags=re.IGNORECASE))
        counts.append(cnt)
    return np.array(counts, dtype=float)


def hygiene_score(v: np.ndarray) -> float:
    """Compute s = w⁺·v − w⁻·v."""
    return float(np.dot(W_POS, v) - np.dot(W_NEG, v))


def shannon_entropy(v: np.ndarray) -> float:
    """Entropy H = −∑ pᵢ log₂ pᵢ where p = v/Σv."""
    total = v.sum()
    if total == 0:
        return 0.0
    p = v / total
    # filter zero probabilities to avoid log2(0)
    p_nonzero = p[p > 0]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))


def hybrid_hygiene_entropy_score(v: np.ndarray) -> float:
    """Combine hygiene and entropy as Sₕ = s·(1 + H/Hₘₐₓ)."""
    s = hygiene_score(v)
    H = shannon_entropy(v)
    H_max = math.log2(len(v))  # log₂ 9
    return s * (1.0 + H / H_max)


# ----------------------------------------------------------------------
# Fusion layer – combine hyperdimensional binding with decision‑hygiene
# ----------------------------------------------------------------------
def fused_metric(
    morphology: Morphology,
    goal_vector: Vector,
    k: int = 128,
    dim: int = 10000,
) -> float:
    """
    Compute the unified score for a morphology.

    Steps
    -----
    1. Build morphology vector **M** (dim‑dimensional).
    2. Compute MinHash signature **h** of the token set, normalise to [0,1] → **H**.
    3. Bind → **B** = **M** ⊙ **H**.
    4. Cosine similarity ρ = cos θ(**B**, **G**) with the supplied goal vector.
    5. Concatenate all token strings, run regex extraction → count vector **v**.
    6. Compute hybrid hygiene‑entropy score Sₕ.
    7. Final unified score = ρ · Sₕ.
    """
    # 1. Morphology vector
    M = morphology_vector(morphology, dim=dim)

    # 2. MinHash signature → float vector
    h_ints = minhash_signature(morphology.tokens, k=k)
    # Normalise each 64‑bit integer to [0,1]
    H = [h / MAX64 for h in h_ints]
    # Extend H to the same dimension as M by tiling
    repeats = dim // len(H) + 1
    H_expanded = (np.tile(H, repeats)[:dim]).tolist()

    # 3. Bind
    B = bind(M, H_expanded)

    # 4. Cosine similarity with goal
    rho = cosine_similarity(B, goal_vector)

    # 5. Feature extraction from textual representation of tokens
    combined_text = " ".join(morphology.tokens)
    v = extract_feature_counts(combined_text)

    # 6. Hybrid hygiene‑entropy score
    Sh = hybrid_hygiene_entropy_score(v)

    # 7. Unified metric
    return rho * Sh


# ----------------------------------------------------------------------
# Example usage / smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a dummy morphology
    demo_morph = Morphology(
        length=2.5,
        width=1.8,
        height=0.9,
        mass=3.2,
        tokens=[
            "evidence of compliance",
            "plan the roadmap",
            "delay due to risk",
            "security audit passed",
            "cost estimate 5000",
            "quality assurance check",
        ],
    )

    # Goal vector: a random but deterministic vector (seeded)
    random.seed(42)
    goal = [random.random() for _ in range(10000)]

    score = fused_metric(demo_morph, goal)
    print(f"Unified fused score: {score:.6f}")
    sys.exit(0)