# DARWIN HAMMER — match 4753, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_decrea_hybrid_hybrid_hybrid_m1941_s2.py (gen6)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m923_s0.py (gen4)
# born: 2026-05-29T23:57:55Z

"""Hybrid Algorithm Fusion of:
- Parent A: hybrid_hybrid_hybrid_decrea_hybrid_hybrid_hybrid_m1941_s2.py
- Parent B: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_decisi_m923_s0.py

Mathematical Bridge
-------------------
Parent A provides a *Liquid Time‑Constant* (LTC) dynamical system whose weight matrix
`W(t)` is modulated by a pruning probability `p(t) = λ·exp(‑α·t)` and by *pheromone*
signals that encode epistemic certainty.  
Parent B extracts categorical token counts from text, computes Shannon entropy
`H = –∑ p_i log p_i` and a lightweight MinHash signature that approximates the
Jaccard similarity between two token sets.

The fusion treats the **entropy** of a document as a scalar *certainty* that
scales the pheromone update, while the **MinHash similarity** supplies a
pair‑wise coupling term that is injected into the LTC dynamics as an
outer‑product stimulus.  Thus the LTC weight update becomes


W←(1‑p(Δt))·W + η·sim·(h⊗h)   with   η = β·H


where `p(Δt)` is the pruning probability, `sim` the MinHash similarity, `h`
the normalized feature vector, and `β` a tunable entropy‑gain factor.
The resulting system jointly adapts its temporal response (LTC) and its
information‑theoretic confidence (entropy) while respecting similarity‑based
leadership (pheromone) cues.

The module below implements this hybrid mathematically, exposing three core
functions:
1. `extract_features(text)` – regex‑based token categorisation (Parent B).
2. `minhash_similarity(tokens_a, tokens_b)` – MinHash Jaccard estimate.
3. `ltc_step(W, h, dt, sim, entropy, beta=0.5)` – LTC weight update using the
   bridge described above (Parent A + B).
A small smoke test demonstrates end‑to‑end execution.
"""

import math
import random
import sys
import pathlib
import re
from collections import Counter
import numpy as np

# ----------------------------------------------------------------------
# Constants & utilities from Parent A
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

def prune_probability(t: float, lam: float = 1.0, alpha: float = 0.2) -> float:
    """Exponential pruning probability used to decay LTC weights."""
    if t < 0 or lam < 0 or alpha < 0:
        raise ValueError('t, lam, and alpha must be non‑negative')
    return min(1.0, lam * math.exp(-alpha * t))

def length(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def bayes_marginal(prior: float, likelihood: float, false_positive: float) -> float:
    """Bayesian marginal used for epistemic flag handling."""
    if not all(0 <= x <= 1 for x in (prior, likelihood, false_positive)):
        raise ValueError("probabilities must be in [0,1]")
    return likelihood * prior + false_positive * (1.0 - prior)

# ----------------------------------------------------------------------
# Regex feature extraction – derived from Parent B
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
    re.I,
)
RISK_RE = re.compile(
    r"\b(?:risk|danger|threat|vulnerab|expose|exploit|attack|breach|leak|loss|damage|dangerous|unsafe)\b",
    re.I,
)

_REGEX_CATEGORIES = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
    "risk": RISK_RE,
}

def extract_features(text: str) -> Counter:
    """
    Count occurrences of each semantic category in *text*.
    Returns a Counter mapping category → count.
    """
    counts = Counter()
    for cat, pattern in _REGEX_CATEGORIES.items():
        matches = pattern.findall(text)
        if matches:
            counts[cat] = len(matches)
    return counts

# ----------------------------------------------------------------------
# Entropy (Shannon) – also from Parent B
# ----------------------------------------------------------------------
def shannon_entropy(counts: Counter) -> float:
    """Compute Shannon entropy of a categorical count distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()], dtype=float)
    return -np.sum(probs * np.log(probs + 1e-12))

# ----------------------------------------------------------------------
# MinHash similarity – lightweight bridge between the two parents
# ----------------------------------------------------------------------
def _minhash_signature(tokens: list[str], num_perm: int = 64) -> np.ndarray:
    """
    Compute a simple MinHash signature.
    For each permutation we use a random hash seed; the minimum hash value
    over the token set is stored.
    """
    rng = np.random.default_rng(42)  # deterministic seed for reproducibility
    seeds = rng.integers(0, 2**31 - 1, size=num_perm, dtype=np.int64)

    sig = np.full(num_perm, np.iinfo(np.int64).max, dtype=np.int64)
    for token in tokens:
        token_bytes = token.encode('utf-8', errors='ignore')
        token_hash = np.frombuffer(token_bytes, dtype=np.uint8).view(np.int64)
        # Broadcast seeds xor token hash to obtain permuted hashes
        permuted = np.bitwise_xor(seeds[:, None], token_hash[0])
        sig = np.minimum(sig, permuted.min(axis=1))
    return sig

def minhash_similarity(tokens_a: list[str], tokens_b: list[str], num_perm: int = 64) -> float:
    """
    Approximate Jaccard similarity between two token sets via MinHash.
    Returns a value in [0,1].
    """
    if not tokens_a or not tokens_b:
        return 0.0
    sig_a = _minhash_signature(tokens_a, num_perm)
    sig_b = _minhash_signature(tokens_b, num_perm)
    return np.mean(sig_a == sig_b)

# ----------------------------------------------------------------------
# Liquid Time‑Constant (LTC) dynamics fused with entropy & similarity
# ----------------------------------------------------------------------
def ltc_step(
    W: np.ndarray,
    h: np.ndarray,
    dt: float,
    sim: float,
    entropy: float,
    beta: float = 0.5,
    lam: float = 1.0,
    alpha: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one LTC update step.

    Parameters
    ----------
    W : np.ndarray (n×n)
        Current weight matrix.
    h : np.ndarray (n,)
        Normalised feature vector (L2‑norm = 1).
    dt : float
        Time elapsed since the previous step.
    sim : float
        MinHash similarity with a peer document (0‑1).
    entropy : float
        Shannon entropy of the current document (≥0).
    beta : float
        Scaling factor that maps entropy to a pheromone‑like gain.
    lam, alpha : float
        Parameters of the pruning probability.

    Returns
    -------
    W_new : np.ndarray
        Updated weight matrix.
    y : np.ndarray
        Network output after applying a tanh non‑linearity.
    """
    # 1. Decay existing weights
    p = prune_probability(dt, lam, alpha)          # pruning probability
    W_decay = (1.0 - p) * W

    # 2. Compute pheromone gain from entropy
    eta = beta * entropy  # higher entropy ⇒ stronger update

    # 3. Outer‑product stimulus modulated by similarity
    stimulus = eta * sim * np.outer(h, h)

    # 4. Updated weight matrix
    W_new = W_decay + stimulus

    # 5. Network output (tanh activation)
    y = np.tanh(W_new @ h)

    return W_new, y

# ----------------------------------------------------------------------
# High‑level hybrid operation
# ----------------------------------------------------------------------
def hybrid_process(text_a: str, text_b: str, dt: float = 1.0) -> dict:
    """
    Process two documents through the fused system.

    Returns a dictionary containing:
        - feature vectors,
        - entropy values,
        - MinHash similarity,
        - final LTC output vector.
    """
    # Feature extraction
    feats_a = extract_features(text_a)
    feats_b = extract_features(text_b)

    # Build a unified feature order
    all_keys = sorted(set(feats_a) | set(feats_b))
    vec_a = np.array([feats_a.get(k, 0) for k in all_keys], dtype=float)
    vec_b = np.array([feats_b.get(k, 0) for k in all_keys], dtype=float)

    # Normalise to unit length (required by LTC formulation)
    def _norm(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    h_a = _norm(vec_a)
    h_b = _norm(vec_b)

    # Entropy (acts as epistemic certainty)
    ent_a = shannon_entropy(feats_a)
    ent_b = shannon_entropy(feats_b)

    # Tokenisation for MinHash (simple whitespace split)
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()

    # Similarity between the two documents
    sim = minhash_similarity(tokens_a, tokens_b)

    # Initialise LTC weight matrix (size = number of features)
    n = len(all_keys)
    rng = np.random.default_rng(123)
    W = rng.normal(loc=0.0, scale=0.1, size=(n, n))

    # Perform one update step for each document (demonstrates bidirectional influence)
    W, out_a = ltc_step(W, h_a, dt, sim, ent_a)
    W, out_b = ltc_step(W, h_b, dt, sim, ent_b)

    return {
        "features_order": all_keys,
        "vector_a": h_a,
        "vector_b": h_b,
        "entropy_a": ent_a,
        "entropy_b": ent_b,
        "minhash_similarity": sim,
        "ltc_output_a": out_a,
        "ltc_output_b": out_b,
    }

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_a = (
        "The team prepared a detailed plan and checklist. "
        "Evidence was logged in the audit trail. "
        "We will wait until tomorrow before proceeding."
    )
    sample_b = (
        "Urgent: fix the security breach now! "
        "Risk of data loss is high. "
        "Support will be provided after the immediate fix."
    )
    result = hybrid_process(sample_a, sample_b, dt=0.5)

    print("Feature order:", result["features_order"])
    print("Entropy A:", result["entropy_a"])
    print("Entropy B:", result["entropy_b"])
    print("MinHash similarity:", result["minhash_similarity"])
    print("LTC output A:", result["ltc_output_a"])
    print("LTC output B:", result["ltc_output_b"])
    sys.exit(0)