# DARWIN HAMMER — match 1977, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_sparse_wta_hy_m1142_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_krampu_hybrid_possum_filter_m1001_s0.py (gen4)
# born: 2026-05-29T23:40:18Z

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

FUNCTION_CATS: Dict[str, set] = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: int


@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class Entity:
    id: str
    lat: float
    lon: float
    category: str
    score: float = 0.0
    address_signature: str = ""
    morphology: 'Morphology' = None


@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float


def extract_stylometry_features(text: str) -> np.ndarray:
    tokens = [
        t.strip(PUNCT).lower()
        for t in text.split()
        if t.strip(PUNCT)
    ]
    vec = np.zeros(len(FUNCTION_CATS), dtype=float)
    for idx, cat in enumerate(FUNCTION_CATS):
        cat_words = FUNCTION_CATS[cat]
        count = sum(1 for w in tokens if w in cat_words)
        vec[idx] = count
    if vec.sum() > 0:
        vec = vec / vec.sum()
    return vec


def sparse_winner_take_all(matrix: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros_like(matrix)
    if k >= matrix.shape[1]:
        return matrix.copy()
    idx = np.argpartition(-matrix, kth=k - 1, axis=1)[:, :k]
    wta = np.zeros_like(matrix)
    rows = np.arange(matrix.shape[0])[:, None]
    wta[rows, idx] = matrix[rows, idx]
    return wta


def reconstruction_risk(original: np.ndarray, reconstructed: np.ndarray) -> float:
    diff = original - reconstructed
    return float(np.linalg.norm(diff, ord='fro'))


def haversine(coord_a: Tuple[float, float], coord_b: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, coord_a)
    lat2, lon2 = map(math.radians, coord_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6_371_000.0 * math.asin(min(1.0, math.sqrt(h)))


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine((coords[i, 0], coords[i, 1]), (coords[j, 0], coords[j, 1]))
            D[i, j] = D[j, i] = d
    return D


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalized = features / norm
    return normalized @ normalized.T


def combined_edge_weights(similarity: np.ndarray, distance: np.ndarray) -> np.ndarray:
    max_d = distance.max() if distance.max() > 0 else 1.0
    norm_dist = 1.0 - (distance / max_d)  
    alpha = 0.6  
    beta = 0.4   
    return alpha * similarity + beta * norm_dist


def approx_ollivier_ricci_curvature(weights: np.ndarray) -> np.ndarray:
    deg = weights.sum(axis=1, keepdims=True)
    denom = deg + deg.T
    with np.errstate(divide='ignore', invalid='ignore'):
        curvature = 1.0 - np.where(denom > 0, weights / denom, 0.0)
    np.fill_diagonal(curvature, 0.0)
    return curvature


def regret_weighted_actions(
    actions: List[MathAction],
    curvature: np.ndarray,
    entity_ids: List[str],
) -> List[MathAction]:
    id_to_idx = {eid: idx for idx, eid in enumerate(entity_ids)}
    adjusted = []
    for act in actions:
        idx = id_to_idx.get(act.id)
        if idx is not None:
            coeff = curvature[idx].mean()
        else:
            coeff = 1.0
        new_ev = act.expected_value * coeff
        adjusted.append(MathAction(id=act.id, expected_value=new_ev,
                                   cost=act.cost, risk=act.risk))
    return adjusted


def hybrid_pipeline(
    texts: List[str],
    coords: List[Tuple[float, float]],
    actions: List[MathAction],
    wta_k: int = 3,
) -> Tuple[np.ndarray, List[MathAction]]:
    features = np.array([extract_stylometry_features(t) for t in texts])
    wta_features = sparse_winner_take_all(features, wta_k)
    risk = reconstruction_risk(features, wta_features)
    
    coords = np.array(coords)
    D = distance_matrix(coords)
    similarity = cosine_similarity_matrix(features)
    weights = combined_edge_weights(similarity, D)
    curvature = approx_ollivier_ricci_curvature(weights)
    
    entity_ids = [e.id for e in coords.T]
    adjusted_actions = regret_weighted_actions(actions, curvature, entity_ids)
    
    return np.array([risk]), adjusted_actions