# DARWIN HAMMER — match 2815, survivor 1
# gen: 5
# parent_a: hybrid_krampus_chrono_hybrid_possum_filter_m34_s1.py (gen3)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_infotaxis_hyb_m2474_s4.py (gen4)
# born: 2026-05-29T23:46:03Z

"""Hybrid algorithm merging temporal‑spatial resource modeling (Parent A) with
stylometry‑morphology information‑theoretic affinity (Parent B).

Mathematical bridge:
- Both parents construct a resource/feature matrix **A** whose rows correspond to
  objects (entities, models, documents) and whose columns encode orthogonal
  resource dimensions.
- Parent A supplies the first three columns: spatial distance, privacy load,
  temporal weight.
- Parent B supplies the last two columns: stylometric mass and morphology load.
- The unified matrix **A** (size (N_objects × 5)) is subjected to a single
  linear feasibility test **Aᵀ·x ≤ b**, where **b** concatenates the spatial,
  privacy, temporal, stylometric and morphology budgets.
The code below builds this matrix, evaluates a simple linear feasibility
criterion, and returns the objects that satisfy the combined budget.
"""

import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import re
import hashlib
from collections import Counter

# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    timestamp: str = ""  # ISO‑8601 string


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_gb: float
    tau: float
    mu: float


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    morphology: Tuple[float, float, float, float]  # (length, width, height, mass)


# ----------------------------------------------------------------------
# Helper functions – Parent A side
# ----------------------------------------------------------------------
def haversine_m(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Great‑circle distance in metres between two (lat,lon) points."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000 * math.sqrt(h)


def extract_temporal_weight(timestamp: str, reference: datetime) -> float:
    """
    Convert an ISO‑8601 timestamp into a temporal weight.
    The weight is the absolute difference in seconds, scaled to [0,1] by a
    simple exponential decay.
    """
    try:
        ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except Exception:
        return 0.0
    delta = abs((ts - reference).total_seconds())
    # decay with half‑life of 30 days (2592000 s)
    half_life = 2_592_000
    weight = math.exp(-math.log(2) * delta / half_life)
    return weight


def compute_entity_vector(
    entity: Entity,
    reference_loc: Tuple[float, float],
    all_signatures: Dict[str, int],
    temporal_ref: datetime,
) -> np.ndarray:
    """
    Return the 3‑dimensional resource vector for an Entity:
    [spatial_distance, privacy_load, temporal_weight]
    """
    # spatial distance (metres)
    d = haversine_m((entity.lat, entity.lon), reference_loc)

    # privacy load: β·σ where σ=1 if signature collides, else 0
    beta = 0.5  # tunable constant
    sigma = 1 if all_signatures.get(entity.address_signature, 0) > 1 else 0
    p = beta * sigma

    # temporal weight
    t = extract_temporal_weight(entity.timestamp, temporal_ref)

    return np.array([d, p, t], dtype=float)


# ----------------------------------------------------------------------
# Helper functions – Parent B side
# ----------------------------------------------------------------------
FUNCTION_CATS = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}


def words(text: str) -> List[str]:
    """Tokenise lower‑case alphabetic words."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())


def lsm_vector(text: str) -> np.ndarray:
    """
    Simple stylometric vector: normalized frequencies of function‑category words.
    Returns a vector of length len(FUNCTION_CATS).
    """
    toks = words(text)
    total = len(toks) or 1
    cat_counts = []
    for cat_words in FUNCTION_CATS.values():
        count = sum(1 for w in toks if w in cat_words)
        cat_counts.append(count / total)
    return np.array(cat_counts, dtype=float)


def stylometric_mass(lsm_vec: np.ndarray) -> float:
    """Aggregate the stylometric vector into a scalar mass (L2 norm)."""
    return float(np.linalg.norm(lsm_vec))


def morphology_load(morph: Tuple[float, float, float, float]) -> float:
    """Return the mass component of a morphology tuple."""
    _, _, _, mass = morph
    return float(mass)


def compute_document_vector(doc: Document) -> np.ndarray:
    """
    Return the 2‑dimensional vector for a Document:
    [stylometric_mass, morphology_load]
    """
    lsm = lsm_vector(doc.text)
    s_mass = stylometric_mass(lsm)
    m_load = morphology_load(doc.morphology)
    return np.array([s_mass, m_load], dtype=float)


# ----------------------------------------------------------------------
# Core hybrid operations
# ----------------------------------------------------------------------
def build_resource_matrix(
    entities: List[Entity],
    models: List[ModelTier],
    documents: List[Document],
    reference_loc: Tuple[float, float],
    temporal_ref: datetime,
) -> np.ndarray:
    """
    Assemble the unified resource matrix A.
    Columns: [spatial, privacy, temporal, stylometric, morphology]
    Rows: entities + models + documents.
    """
    # Prepare signature collision map for privacy load
    sig_counts: Dict[str, int] = Counter(e.address_signature for e in entities)

    rows: List[np.ndarray] = []

    # Entity rows (first three columns filled, last two zero)
    for e in entities:
        vec3 = compute_entity_vector(e, reference_loc, sig_counts, temporal_ref)
        rows.append(np.concatenate([vec3, np.zeros(2)]))

    # Model rows (reuse parent A's model vector, pad with zeros for B)
    for m in models:
        # ModelTier vector from parent A: [RAM, α·τ·μ]
        alpha = 0.3
        model_vec = np.array([m.ram_gb, alpha * m.tau * m.mu], dtype=float)
        rows.append(np.concatenate([model_vec, np.zeros(3)]))

    # Document rows (first three columns zero, last two filled)
    for d in documents:
        vec2 = compute_document_vector(d)
        rows.append(np.concatenate([np.zeros(3), vec2]))

    return np.vstack(rows)


def compute_hybrid_scores(
    A: np.ndarray,
    budgets: np.ndarray,
) -> np.ndarray:
    """
    Compute a feasibility score for each row.
    We solve the linear inequality Aᵀ·x ≤ budgets for a non‑negative weight vector x.
    Because a full LP solver is unavailable, we approximate by projecting the
    budgets onto each row direction:

        score_i = min_j ( budgets_j / (A_ij + ε) )

    Larger scores indicate that the row comfortably fits under all budgets.
    """
    eps = 1e-9
    # Avoid division by zero
    ratios = budgets / (A + eps)
    # The limiting budget for a row is the smallest ratio across columns
    scores = np.min(ratios, axis=1)
    return scores


def select_feasible_objects(
    objects: List[Any],
    scores: np.ndarray,
    threshold: float = 1.0,
) -> List[Any]:
    """
    Return the subset of objects whose hybrid score meets or exceeds the threshold.
    """
    return [obj for obj, sc in zip(objects, scores) if sc >= threshold]


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Reference points
    REF_LOC = (40.7128, -74.0060)  # New York City
    REF_TIME = datetime.utcnow()

    # Dummy entities
    ents = [
        Entity(
            id="e1",
            lat=40.730610,
            lon=-73.935242,
            category="sensor",
            address_signature="sigA",
            timestamp=(datetime.utcnow()).isoformat(),
        ),
        Entity(
            id="e2",
            lat=34.052235,
            lon=-118.243683,
            category="camera",
            address_signature="sigB",
            timestamp=(datetime.utcnow()).isoformat(),
        ),
        Entity(
            id="e3",
            lat=51.507351,
            lon=-0.127758,
            category="drone",
            address_signature="sigA",  # collides with e1
            timestamp=(datetime.utcnow()).isoformat(),
        ),
    ]

    # Dummy models
    mods = [
        ModelTier(name="low", ram_gb=4.0, tau=1.2, mu=0.8),
        ModelTier(name="high", ram_gb=16.0, tau=0.9, mu=1.1),
    ]

    # Dummy documents
    docs = [
        Document(
            id="d1",
            text="The quick brown fox jumps over the lazy dog. It is a common pangram.",
            morphology=(10.0, 5.0, 2.0, 0.3),
        ),
        Document(
            id="d2",
            text="In a galaxy far far away, the rebels fought against the empire.",
            morphology=(12.0, 6.0, 3.0, 0.5),
        ),
    ]

    # Build matrix
    A = build_resource_matrix(ents, mods, docs, REF_LOC, REF_TIME)

    # Define combined budgets (arbitrary but reasonable)
    # spatial (metres), privacy, temporal, stylometric, morphology
    budgets = np.array([1e7, 2.0, 1.0, 5.0, 1.0], dtype=float)

    # Compute scores
    scores = compute_hybrid_scores(A, budgets)

    # Assemble master object list in the same order as rows of A
    master_objects: List[Any] = ents + mods + docs

    # Select feasible ones
    feasible = select_feasible_objects(master_objects, scores, threshold=0.5)

    print("Hybrid scores per object:")
    for obj, sc in zip(master_objects, scores):
        print(f"{getattr(obj, 'id', getattr(obj, 'name', 'unknown'))}: {sc:.3f}")

    print("\nObjects meeting the threshold:")
    for obj in feasible:
        print(f"- {getattr(obj, 'id', getattr(obj, 'name', 'unknown'))}")

    sys.exit(0)