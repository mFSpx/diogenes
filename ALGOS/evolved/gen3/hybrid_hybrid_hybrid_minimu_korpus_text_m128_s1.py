# DARWIN HAMMER — match 128, survivor 1
# gen: 3
# parent_a: hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s3.py (gen2)
# parent_b: korpus_text.py (gen0)
# born: 2026-05-29T23:25:46Z

"""Hybrid module combining epistemic certainty flags with text‑minhash/entropy/vector math.

Parent A (hybrid_hybrid_minimum_cost__epistemic_certainty_m48_s3) provides a
`CertaintyFlag` data class whose `confidence_bps` (basis points, 0‑10000) encodes
a scalar weight w = confidence_bps / 10000.

Parent B (korpus_text) supplies three pure‑text transformations:
* MinHash signatures (set‑based Jaccard estimate)
* Shannon entropy (distributional uncertainty)
* Quantised embedding vectors (fixed‑length integer literals)

The mathematical bridge is the **weight‑scaled similarity**:
for a given text observation we compute a MinHash Jaccard estimate J and a
cosine similarity C between the observation’s embedding and a reference
embedding. Both are multiplied by the epistemic weight w, and the entropy
E (0‑log₂|Σ|) is used to attenuate w (high entropy → lower trust).  The resulting
hybrid score

    S = w * (α·J + β·C) * exp(‑γ·E)

provides a single scalar that fuses epistemic certainty with the statistical
properties of the text.  The functions below implement this pipeline and also
offer a Bayesian‑style update of a `CertaintyFlag` using the hybrid score.

Only the Python standard library and NumPy are used.
"""

import math
import random
import sys
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Tuple, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Minimal re‑implementation of the epistemic certainty helpers (Parent A)
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS: Tuple[str, ...] = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # 0 .. 10000
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
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


def filesystem_observation(*, sha256: str, path: str, mtime_utc: str | None = None) -> CertaintyFlag:
    refs = [f"sha256:{sha256}", f"path:{path}"]
    if mtime_utc:
        refs.append(f"mtime:{mtime_utc}")
    return certainty(
        "FACT",
        confidence_bps=10000,
        authority_class="filesystem_observation",
        rationale="Local file bytes were hashed and copied into CAS; this proves byte custody, not semantic truth.",
        evidence_refs=refs,
    )


def parser_extraction(*, sha256: str, extract_method: str, injection_detected: bool = False) -> CertaintyFlag:
    label = "BULLSHIT" if injection_detected else "FACT"
    return certainty(
        label,
        confidence_bps=0 if injection_detected else 9000,
        authority_class="parser_extraction",
        rationale=f"Extraction via {extract_method}",
        evidence_refs=[f"sha256:{sha256}", f"method:{extract_method}"],
    )


# ----------------------------------------------------------------------
# Text‑processing helpers (Parent B) – re‑implemented locally
# ----------------------------------------------------------------------
def _shingles(text: str, width: int = 5) -> List[str]:
    """Return a list of overlapping substrings (shingles) of length `width`."""
    if len(text) < width:
        return [text]
    return [text[i : i + width] for i in range(len(text) - width + 1)]


def _minhash_signature(shingles_set: Iterable[str], k: int = 64) -> List[int]:
    """
    Very simple MinHash: for each of k hash functions (simulated by
    seeding Python's built‑in hash with a different salt) keep the minimal
    hash value over the shingle set.
    """
    signature = []
    for seed in range(k):
        min_hash = None
        for sh in shingles_set:
            # combine shingle with seed to emulate a distinct hash function
            h = hash((sh, seed))
            if (min_hash is None) or (h < min_hash):
                min_hash = h
        # Normalise to unsigned 64‑bit integer space
        signature.append(min_hash & ((1 << 64) - 1))
    return signature


def minhash_for_text(text: str, k: int = 64) -> List[int]:
    """Compute a MinHash signature for `text`."""
    cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
    return _minhash_signature(_shingles(cleaned, width=5), k=k)


def shannon_entropy(sequence: List[str]) -> float:
    """Shannon entropy (bits) of a sequence of symbols."""
    if not sequence:
        return 0.0
    counts: Dict[str, int] = {}
    for sym in sequence:
        counts[sym] = counts.get(sym, 0) + 1
    total = len(sequence)
    ent = -sum((c / total) * math.log2(c / total) for c in counts.values())
    return ent


def entropy_for_text(text: str) -> float:
    """Entropy of the first 10 000 characters of `text`."""
    return shannon_entropy(list((text or "")[:10000])) if text else 0.0


INT16_MAX = 32767


def hash_quantized_embedding(text: str, dim: int = 16) -> List[int]:
    """
    Produce a deterministic pseudo‑random int16 vector from `text`.
    The vector is *quantised*: each component lies in [‑INT16_MAX, INT16_MAX].
    """
    rng = random.Random(hash(text))
    return [rng.randint(-INT16_MAX, INT16_MAX) for _ in range(dim)]


def vector_literal(text: str) -> str:
    """Return a string literal of a normalised embedding vector."""
    vec = hash_quantized_embedding(text or "")
    return "[" + ",".join(f"{float(v) / float(INT16_MAX):.8f}" for v in vec) + "]"


# ----------------------------------------------------------------------
# Hybrid operations – mathematical fusion of A and B
# ----------------------------------------------------------------------
def jaccard_estimate(sig1: List[int], sig2: List[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if len(sig1) != len(sig2):
        raise ValueError("Signatures must be of equal length")
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two 1‑D NumPy arrays."""
    if v1.shape != v2.shape:
        raise ValueError("Vectors must have the same shape")
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def hybrid_score(
    flag: CertaintyFlag,
    text_signature: List[int],
    reference_signature: List[int],
    text_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    text_entropy: float,
    *,
    alpha: float = 0.6,
    beta: float = 0.4,
    gamma: float = 0.5,
) -> float:
    """
    Compute a fused certainty score.

    Parameters
    ----------
    flag : CertaintyFlag
        Epistemic weight w = confidence_bps / 10000.
    text_signature / reference_signature : List[int]
        MinHash signatures for the observation and a reference.
    text_embedding / reference_embedding : np.ndarray
        Quantised embedding vectors (float32).
    text_entropy : float
        Shannon entropy of the observation text.
    alpha, beta, gamma : float
        Tunable coefficients (α·J + β·C)·exp(‑γ·E) scaled by w.

    Returns
    -------
    float
        Hybrid score in the range [0, w].
    """
    w = flag.confidence_bps / 10000.0
    J = jaccard_estimate(text_signature, reference_signature)
    C = cosine_similarity(text_embedding, reference_embedding)
    E = text_entropy  # already in bits; higher E → lower trust
    attenuation = math.exp(-gamma * E)
    return w * (alpha * J + beta * C) * attenuation


def update_certainty_with_text(
    prior: CertaintyFlag,
    text: str,
    reference_text: str,
    *,
    alpha: float = 0.6,
    beta: float = 0.4,
    gamma: float = 0.5,
) -> CertaintyFlag:
    """
    Bayesian‑style update: treat the hybrid score as a likelihood and
    combine it with the prior confidence to produce a new CertaintyFlag.
    The new confidence is capped at 10000 bps.
    """
    # Compute all required primitives
    sig_obs = minhash_for_text(text)
    sig_ref = minhash_for_text(reference_text)
    emb_obs = np.array(hash_quantized_embedding(text), dtype=np.float32)
    emb_ref = np.array(hash_quantized_embedding(reference_text), dtype=np.float32)
    ent_obs = entropy_for_text(text)

    # Hybrid likelihood (0..1)
    likelihood = hybrid_score(
        prior,
        sig_obs,
        sig_ref,
        emb_obs,
        emb_ref,
        ent_obs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )

    # Simple linear update: new_confidence = prior + (1‑prior)*likelihood
    prior_w = prior.confidence_bps / 10000.0
    new_w = prior_w + (1.0 - prior_w) * likelihood
    new_confidence = int(round(new_w * 10000))
    new_confidence = min(10000, max(0, new_confidence))

    return CertaintyFlag(
        label=prior.label,
        confidence_bps=new_confidence,
        authority_class=prior.authority_class,
        rationale=f"Updated with text similarity (likelihood={likelihood:.4f})",
        evidence_refs=prior.evidence_refs + ("text_update",),
    )


def hybrid_report(flag: CertaintyFlag, text: str, reference_text: str) -> str:
    """
    Produce a human‑readable report that shows the intermediate values
    and the final hybrid score.
    """
    sig_obs = minhash_for_text(text)
    sig_ref = minhash_for_text(reference_text)
    J = jaccard_estimate(sig_obs, sig_ref)

    emb_obs = np.array(hash_quantized_embedding(text), dtype=np.float32)
    emb_ref = np.array(hash_quantized_embedding(reference_text), dtype=np.float32)
    C = cosine_similarity(emb_obs, emb_ref)

    E = entropy_for_text(text)
    score = hybrid_score(flag, sig_obs, sig_ref, emb_obs, emb_ref, E)

    report = (
        f"CertaintyFlag: {flag.label} (confidence {flag.confidence_bps}/10000)\n"
        f"Jaccard (MinHash) similarity      : {J:.4f}\n"
        f"Cosine similarity (embedding)    : {C:.4f}\n"
        f"Shannon entropy (bits)           : {E:.4f}\n"
        f"Hybrid score (weighted)          : {score:.4f}\n"
    )
    return report


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Create a base certainty flag from a fake filesystem observation
    base_flag = filesystem_observation(
        sha256="deadbeef"*8,
        path="/tmp/example.txt",
        mtime_utc="2026-05-28T12:34:56Z",
    )

    # Two short texts – one is a paraphrase of the other
    obs_text = "The quick brown fox jumps over the lazy dog."
    ref_text = "A fast dark-colored fox leaps above a sleepy canine."

    # Compute a report before update
    print("--- INITIAL REPORT ---")
    print(hybrid_report(base_flag, obs_text, ref_text))

    # Update the certainty flag using the observed text
    updated_flag = update_certainty_with_text(base_flag, obs_text, ref_text)

    # Report after update
    print("\n--- AFTER UPDATE ---")
    print(hybrid_report(updated_flag, obs_text, ref_text))
    print(f"Updated confidence: {updated_flag.confidence_bps}/10000")