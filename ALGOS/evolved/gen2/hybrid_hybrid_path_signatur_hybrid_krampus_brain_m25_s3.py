# DARWIN HAMMER — match 25, survivor 3
# gen: 2
# parent_a: hybrid_path_signature_kan_m30_s1.py (gen1)
# parent_b: hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s4.py (gen1)
# born: 2026-05-29T23:25:14Z

"""
Hybrid Module: Path Signature + Feature Extraction (Parents A & B)

Parent A (hybrid_path_signature_kan_m30_s1.py) provides:
- lead‑lag transformation of a multivariate path
- level‑1 and level‑2 iterated‑integral signatures

Parent B (hybrid_krampus_brainmap_ollivier_ricci_curva_m13_s4.py) provides:
- a deterministic high‑dimensional feature extractor from free‑form text
- a compact “master vector” derived from those features

Mathematical Bridge
------------------
Given a **sequence of texts** 𝚃 = (t₁,…,tₙ) we first map each
tᵢ → 𝒙ᵢ ∈ ℝᴰ using the master‑vector extractor (Parent B).  
The ordered set X = (𝒙₁,…,𝒙ₙ) is interpreted as a discrete path in ℝᴰ.
Applying the **lead‑lag transform** (Parent A) yields a causally‑aware
augmented path 𝑋̃.  From 𝑋̃ we compute the **level‑1** (total increment) and
**level‑2** (iterated‑integral tensor) signatures.  

The KAN‑style approximation (conceptually from Parent A) is realised here
as a shallow network with a single hidden layer whose activation mimics a
piece‑wise polynomial basis.  This network takes the flattened signature
vector as input and produces a fused embedding that simultaneously
captures the text‑derived geometry (Parent B) and the path‑signature
algebra (Parent A).

The module therefore fuses the two topologies into a single, mathematically
coherent pipeline.
"""

import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------------------------------------------------
# Parent B – Feature extraction utilities
# ----------------------------------------------------------------------
def extract_full_features(text: str) -> Dict[str, float]:
    """Deterministic pseudo‑random feature vector from a string."""
    rnd = random.Random(hash(text))
    keys = [
        "operator_visceral_ratio", "operator_tech_ratio",
        "operator_legal_osint_ratio", "operator_ledger_density",
        "operator_recursion_score", "operator_directive_ratio",
        "operator_target_density", "psyche_forensic_shield_ratio",
        "psyche_poetic_entropy", "psyche_dissociative_index",
        "psyche_wrath_velocity", "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric", "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index", "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax", "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density", "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio", "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline", "telemetry_manic_velocity",
    ]
    return {k: rnd.random() * 10.0 for k in keys}


def extract_master_vector(text: str) -> Dict[str, float]:
    """Reduced master vector – a subset of the full feature map."""
    if not text.strip():
        return {}
    f = extract_full_features(text)
    return {
        "visceral_ratio": f.get("operator_visceral_ratio", 0.0),
        "tech_ratio": f.get("operator_tech_ratio", 0.0),
        "legal_osint_ratio": f.get("operator_legal_osint_ratio", 0.0),
        "ledger_density": f.get("operator_ledger_density", 0.0),
        "recursion_score": f.get("operator_recursion_score", 0.0),
        "directive_ratio": f.get("operator_directive_ratio", 0.0),
        "target_density": f.get("operator_target_density", 0.0),
        "forensic_shield_ratio": f.get("psyche_forensic_shield_ratio", 0.0),
        "poetic_entropy": f.get("psyche_poetic_entropy", 0.0),
        "dissociative_index": f.get("psyche_dissociative_index", 0.0),
        "wrath_velocity": f.get("psyche_wrath_velocity", 0.0),
        "bureaucratic_weaponization_index": f.get(
            "resilience_bureaucratic_weaponization_index", 0.0
        ),
        "resource_exhaustion_metric": f.get(
            "resilience_resource_exhaustion_metric", 0.0
        ),
        "swarm_orchestration_density": f.get(
            "resilience_swarm_orchestration_density", 0.0
        ),
        "logic_crucifixion_index": f.get(
            "resilience_logic_crucifixion_index", 0.0
        ),
        "conspiracy_grounding_ratio": f.get(
            "resilience_conspiracy_grounding_ratio", 0.0
        ),
        "chaotic_good_tax": f.get("resilience_chaotic_good_tax", 0.0),
        "corporate_grit_tension": f.get("rainmaker_corporate_grit_tension", 0.0),
        "countdown_density": f.get("rainmaker_countdown_density", 0.0),
        "asset_structuring_weight": f.get("rainmaker_asset_structuring_weight", 0.0),
        "pitch_formatting_ratio": f.get("rainmaker_pitch_formatting_ratio", 0.0),
        "agent_symmetry_ratio": f.get("telemetry_agent_symmetry_ratio", 0.0),
        "protocol_discipline": f.get("telemetry_protocol_discipline", 0.0),
        "manic_velocity": f.get("telemetry_manic_velocity", 0.0),
    }

# ----------------------------------------------------------------------
# Parent A – Path‑signature utilities
# ----------------------------------------------------------------------
def lead_lag_transform(path: np.ndarray) -> np.ndarray:
    """
    Lead‑lag transform: interleave (lead, lag) channels.

    Parameters
    ----------
    path : np.ndarray, shape (T, d)
        Original discrete path.

    Returns
    -------
    np.ndarray, shape (2T‑1, 2d)
        Lead‑lag augmented path.
    """
    path = np.asarray(path, dtype=float)
    T, d = path.shape
    out = np.empty((2 * T - 1, 2 * d), dtype=float)
    for t in range(T - 1):
        out[2 * t] = np.concatenate([path[t], path[t]])          # (lead, lag) equal
        out[2 * t + 1] = np.concatenate([path[t + 1], path[t]])  # lead moves, lag holds
    out[2 * (T - 1)] = np.concatenate([path[T - 1], path[T - 1]])
    return out


def signature_level1(path: np.ndarray) -> np.ndarray:
    """
    Level‑1 signature (total increment).

    Parameters
    ----------
    path : np.ndarray, shape (T, d)

    Returns
    -------
    np.ndarray, shape (d,)
    """
    path = np.asarray(path, dtype=float)
    return path[-1] - path[0]


def signature_level2(path: np.ndarray) -> np.ndarray:
    """
    Level‑2 iterated‑integral tensor using left‑point Riemann sums.

    Parameters
    ----------
    path : np.ndarray, shape (T, d)

    Returns
    -------
    np.ndarray, shape (d, d)
    """
    path = np.asarray(path, dtype=float)
    increments = np.diff(path, axis=0)               # (T‑1, d)
    running = path[:-1] - path[0]                    # (T‑1, d)
    # Tensor contraction over time axis
    S2 = running.T @ increments                      # (d, d)
    return S2

# ----------------------------------------------------------------------
# Hybrid components
# ----------------------------------------------------------------------
def kan_approximate_signature(sig_vec: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """
    Very lightweight KAN‑style approximation.

    The KAN (Kolmogorov‑Arnold Network) expresses a multivariate function as
    a composition of univariate “basis” functions.  Here we emulate that
    behaviour with a single hidden layer where each hidden unit applies a
    smooth non‑linear (tanh) to a linear projection of the signature.

    Parameters
    ----------
    sig_vec : np.ndarray, shape (p,)
        Flattened signature (level‑1 concatenated with level‑2).
    weight_matrix : np.ndarray, shape (h, p)
        Trainable (or pseudo‑random) weights; h is hidden dimension.

    Returns
    -------
    np.ndarray, shape (h,)
        Hidden representation – the “KAN‑approximated” signature embedding.
    """
    # Linear projection
    proj = weight_matrix @ sig_vec                     # (h,)
    # Smooth univariate basis (tanh acts as a simple spline‑like basis)
    return np.tanh(proj)


def text_sequence_to_path(texts: List[str]) -> np.ndarray:
    """
    Convert a list of texts into a multivariate path.

    Each text is transformed into its master vector (dimension D),
    and the resulting sequence of vectors forms the path X ∈ ℝ^{T×D}.

    Parameters
    ----------
    texts : list of str
        Ordered corpus.

    Returns
    -------
    np.ndarray, shape (T, D)
        Path suitable for signature computation.
    """
    vectors = []
    for txt in texts:
        master = extract_master_vector(txt)
        # Ensure a deterministic ordering of dimensions
        ordered = [master[key] for key in sorted(master.keys())]
        vectors.append(ordered)
    return np.asarray(vectors, dtype=float)


def hybrid_signature_embedding(texts: List[str],
                               hidden_dim: int = 32,
                               random_seed: int = 42) -> np.ndarray:
    """
    End‑to‑end hybrid pipeline:
    1. Extract master vectors → path.
    2. Lead‑lag transform → enriched path.
    3. Compute level‑1 and level‑2 signatures.
    4. Flatten and feed into a KAN‑style approximator.
    5. Return the hidden embedding.

    Parameters
    ----------
    texts : list of str
        Input sequence.
    hidden_dim : int, optional
        Dimensionality of the KAN hidden layer.
    random_seed : int, optional
        Seed for reproducible pseudo‑random weights.

    Returns
    -------
    np.ndarray, shape (hidden_dim,)
        Hybrid embedding that couples textual semantics with path‑signature algebra.
    """
    if len(texts) < 2:
        raise ValueError("At least two texts are required to form a path.")
    # 1. Path from master vectors
    path = text_sequence_to_path(texts)                # (T, D)

    # 2. Lead‑lag augmentation
    aug_path = lead_lag_transform(path)                # (2T‑1, 2D)

    # 3. Signatures
    s1 = signature_level1(aug_path)                    # (2D,)
    s2 = signature_level2(aug_path)                    # (2D, 2D)

    # Flatten signatures into a single vector
    sig_vec = np.concatenate([s1.ravel(), s2.ravel()])  # shape (p,)

    # 4. KAN‑style approximation
    rng = np.random.default_rng(random_seed)
    weight_matrix = rng.standard_normal((hidden_dim, sig_vec.shape[0]))
    embedding = kan_approximate_signature(sig_vec, weight_matrix)

    return embedding

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "In quantum mechanics, particles exhibit wave‑particle duality.",
        "Artificial intelligence blends statistics with symbolic reasoning.",
        "Cryptographic hashes provide integrity guarantees."
    ]

    try:
        emb = hybrid_signature_embedding(sample_texts, hidden_dim=16)
        print("Hybrid embedding shape:", emb.shape)
        print("Embedding values (first 5):", emb[:5])
    except Exception as e:
        print("Error during hybrid computation:", e, file=sys.stderr)
        sys.exit(1)