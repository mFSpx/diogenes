# DARWIN HAMMER — match 69, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_hybrid_minimu_korpus_text_m128_s1.py (gen3)
# parent_b: hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s1.py (gen3)
# born: 2026-05-29T23:28:03Z

"""Hybrid Epistemic‑Text‑VRAM‑Geometric Module
Parents:
* hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s3.py (CertaintyFlag, weighted hybrid score)
* hybrid_hybrid_model_vram_sc_hybrid_geometric_pro_m4_s1.py (VRAM budget scheduler, Clifford‑algebra‑style rotor)

Mathematical Bridge
------------------
The epistemic weight *w* (confidence_bps / 10000) is modulated by the
available‑VRAM fraction *g* returned by the scheduler:

    ŵ = w · g ,  g = (total‑reserve‑used) / total   ∈ [0,1]

The text embedding *v* is first rotated by a Clifford‑algebra rotor *R*,
realised as an orthogonal matrix (R·Rᵀ = I).  The rotated vector

    vʹ = R · v

is used for the cosine similarity term.  The hybrid score therefore becomes

    S = ŵ · (α·J + β·cos(R·v, R·r)) · exp(‑γ·E)

where *J* is the MinHash Jaccard estimate, *E* the Shannon entropy of the
observation, and *(α,β,γ)* are tunable scalars.  This fuses:
* epistemic certainty (parent A)
* statistical text features (parent A)
* resource‑aware geometric transformation (parent B)

The module implements the full pipeline and provides utilities to update
the CertaintyFlag based on the computed hybrid score.
"""

import math
import random
import sys
import pathlib
import subprocess
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Tuple, Iterable, List, Dict, Any

import numpy as np

# ----------------------------------------------------------------------
# Parent A – Epistemic certainty helpers
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 … 10000
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())

    @property
    def weight(self) -> float:
        """Epistemic weight w ∈ [0,1] derived from confidence."""
        return self.confidence_bps / 10000.0


# ----------------------------------------------------------------------
# Parent B – VRAM scheduler & geometric rotor helpers
# ----------------------------------------------------------------------
DEFAULT_BUDGET_MB = int(os.getenv("LUCIDOTA_VRAM_BUDGET_MB", "4096"))
DEFAULT_RESERVE_MB = int(os.getenv("LUCIDOTA_VRAM_RESERVE_MB", "768"))


def _run_nvidia_smi() -> Dict[str, Any]:
    """Query nvidia‑smi; fallback to a simulated single‑GPU status."""
    if not shutil.which("nvidia-smi"):
        # Simulated environment: 8 GB total, 2 GB used
        return {"total": 8192, "used": 2048, "free": 6144, "status": "simulated"}
    try:
        cp = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
        )
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.strip())
        total_str, used_str, free_str = cp.stdout.strip().split(",")
        return {
            "total": int(total_str),
            "used": int(used_str),
            "free": int(free_str),
            "status": "nvidia-smi",
        }
    except Exception as exc:
        # Graceful degradation to simulated values
        return {"total": 8192, "used": 2048, "free": 6144, "status": f"error:{exc}"}


def vram_availability_fraction(budget_mb: int = DEFAULT_BUDGET_MB,
                               reserve_mb: int = DEFAULT_RESERVE_MB) -> float:
    """Return g = (budget‑reserve‑used) / budget, clipped to [0,1]."""
    status = _run_nvidia_smi()
    used = status.get("used", 0)
    available = max(budget_mb - reserve_mb - used, 0)
    g = available / budget_mb
    return min(max(g, 0.0), 1.0)


def generate_rotor(dim: int, seed: int | None = None) -> np.ndarray:
    """Generate a random orthogonal matrix R ∈ ℝ^{dim×dim} (a rotor).

    The matrix is obtained via QR decomposition of a Gaussian matrix.
    """
    rng = np.random.default_rng(seed)
    # Gaussian random matrix
    a = rng.normal(size=(dim, dim))
    # QR decomposition; Q is orthogonal
    q, _ = np.linalg.qr(a)
    return q.astype(np.float32)


# ----------------------------------------------------------------------
# Text‑processing utilities (Parent A)
# ----------------------------------------------------------------------
def shingle(text: str, k: int = 5) -> List[str]:
    """Return k‑shingles (character n‑grams) of the text."""
    text = text.replace("\n", " ")
    return [text[i : i + k] for i in range(len(text) - k + 1)]


def minhash_signature(text: str, num_perm: int = 128, seed: int = 0) -> np.ndarray:
    """Compute a MinHash signature (uint64 array) for the given text."""
    rng = random.Random(seed)
    # Generate a list of random hash functions of the form (a*x + b) mod prime
    max_hash = (1 << 61) - 1
    a_vals = [rng.randrange(1, max_hash) for _ in range(num_perm)]
    b_vals = [rng.randrange(0, max_hash) for _ in range(num_perm)]

    signature = np.full(num_perm, max_hash, dtype=np.uint64)
    for sh in shingle(text):
        h = hash(sh) & max_hash
        for i, (a, b) in enumerate(zip(a_vals, b_vals)):
            cand = (a * h + b) % max_hash
            if cand < signature[i]:
                signature[i] = cand
    return signature


def jaccard_estimate(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if sig1.shape != sig2.shape:
        raise ValueError("signatures must have the same length")
    return float(np.mean(sig1 == sig2))


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy (bits) of the character distribution."""
    if not text:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    total = len(text)
    entropy = -sum((c / total) * math.log2(c / total) for c in freq.values())
    return entropy


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two 1‑D arrays."""
    if vec1.shape != vec2.shape:
        raise ValueError("vectors must have the same shape")
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


# ----------------------------------------------------------------------
# Core hybrid operation (fusion of both parents)
# ----------------------------------------------------------------------
def compute_hybrid_score(
    observation_text: str,
    reference_text: str,
    reference_vec: np.ndarray,
    flag: CertaintyFlag,
    alpha: float = 0.6,
    beta: float = 0.4,
    gamma: float = 1.0,
    rotor_seed: int | None = None,
) -> float:
    """
    Compute the fused hybrid score S.

    Steps
    -----
    1. Epistemic weight w = flag.weight.
    2. VRAM availability fraction g → ŵ = w·g.
    3. MinHash Jaccard estimate J between observation and reference texts.
    4. Entropy E of the observation text.
    5. Generate rotor R (dim = len(reference_vec)) and rotate both
       observation and reference embeddings.
    6. Cosine similarity C between the rotated embeddings.
    7. Combine as S = ŵ·(α·J + β·C)·exp(‑γ·E).
    """
    # 1‑2. Epistemic weight scaled by VRAM
    w = flag.weight
    g = vram_availability_fraction()
    w_hat = w * g

    # 3. MinHash Jaccard
    sig_obs = minhash_signature(observation_text)
    sig_ref = minhash_signature(reference_text)
    J = jaccard_estimate(sig_obs, sig_ref)

    # 4. Entropy
    E = shannon_entropy(observation_text)

    # 5. Rotor generation & rotation
    dim = reference_vec.shape[0]
    R = generate_rotor(dim, seed=rotor_seed)

    # For the observation embedding we use a simple bag‑of‑words hash vector
    # (deterministic for reproducibility).
    obs_vec = _text_to_vector(observation_text, dim, seed=rotor_seed)

    obs_rot = R @ obs_vec
    ref_rot = R @ reference_vec

    # 6. Cosine similarity
    C = cosine_similarity(obs_rot, ref_rot)

    # 7. Final hybrid score
    score = w_hat * (alpha * J + beta * C) * math.exp(-gamma * E)
    return score


def _text_to_vector(text: str, dim: int, seed: int | None = None) -> np.ndarray:
    """
    Deterministic pseudo‑embedding: hash each shingle into a bin and count.
    The resulting histogram is L2‑normalized.
    """
    rng = np.random.default_rng(seed)
    # Random projection matrix for hashing into dim bins
    proj = rng.integers(0, dim, size=(dim,))
    vec = np.zeros(dim, dtype=np.float32)
    for sh in shingle(text):
        idx = hash(sh) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def update_certainty_flag(flag: CertaintyFlag, score: float, learning_rate: float = 0.1) -> CertaintyFlag:
    """
    Bayesian‑style update: increase confidence proportionally to the
    hybrid score (clamped to [0,1]).  The update respects the original label.
    """
    # Map score to a delta in basis points
    delta = int(learning_rate * score * 10000)
    new_conf = max(0, min(10000, flag.confidence_bps + delta))
    return CertaintyFlag(
        label=flag.label,
        confidence_bps=new_conf,
        authority_class=flag.authority_class,
        rationale=f"Updated with hybrid score {score:.4f}",
        evidence_refs=flag.evidence_refs,
    )


# ----------------------------------------------------------------------
# Demonstration functions
# ----------------------------------------------------------------------
def demo_hybrid_process() -> None:
    """Run a small demo of the fused pipeline."""
    # Sample data
    obs = "The quick brown fox jumps over the lazy dog."
    ref = "A swift auburn animal leaped above a sleepy canine."
    dim = 128

    # Reference embedding (random but fixed for reproducibility)
    rng = np.random.default_rng(42)
    reference_embedding = rng.normal(size=dim).astype(np.float32)
    reference_embedding /= np.linalg.norm(reference_embedding)

    # Base certainty flag
    base_flag = CertaintyFlag(
        label="PROBABLE",
        confidence_bps=6500,
        authority_class="AI_MODEL",
        rationale="Initial assessment",
    )

    # Compute hybrid score
    score = compute_hybrid_score(
        observation_text=obs,
        reference_text=ref,
        reference_vec=reference_embedding,
        flag=base_flag,
        rotor_seed=123,
    )
    print(f"Hybrid score: {score:.6f}")

    # Update certainty
    updated_flag = update_certainty_flag(base_flag, score)
    print("Updated CertaintyFlag:", asdict(updated_flag))


if __name__ == "__main__":
    demo_hybrid_process()