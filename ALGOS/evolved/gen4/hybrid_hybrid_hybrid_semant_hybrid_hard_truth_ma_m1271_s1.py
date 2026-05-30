# DARWIN HAMMER — match 1271, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_semantic_neig_hybrid_sketches_hybr_m57_s1.py (gen3)
# parent_b: hybrid_hard_truth_math_hybrid_minimum_cost__m12_s5.py (gen2)
# born: 2026-05-29T23:34:50Z

"""
This module fuses the hybrid_semantic_neighbors_hybrid_endpoint_circ_m17_s0.py and 
hybrid_sketches_hybrid_bandit_router_m31_s0.py algorithms, integrating the semantic 
recovery priority from the former into the bandit_router's resource allocation framework 
of the latter. The governing equations of both parents are integrated through matrix 
operations, specifically by using the semantic recovery priority to inform the 
bandit_router's confidence bound calculation.

The mathematical bridge lies in the use of the sphericity index from hybrid_semantic_neighbors 
to calculate the morphology of the document's semantic neighbors, which is then used to 
adjust the bandit_router's confidence bound for determining the best action to take.

This hybrid algorithm demonstrates the fusion of the semantic recovery priority from 
hybrid_semantic_neighbors with the bandit_router's resource allocation framework from 
hybrid_sketches_hybrid_bandit_router_m31_s0.py.
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime, timezone
from math import exp, sqrt
from random import random
from sys import exit
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Dict, List

@dataclass(frozen=True)
class Morphology:
    length: float
    width: float
    height: float
    mass: float

@dataclass(frozen=True)
class Document:
    id: str
    vector: list[float]

@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

def sphericity_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length * width * height) ** (1.0 / 3.0) / length

def flatness_index(length: float, width: float, height: float) -> float:
    if min(length, width, height) <= 0:
        raise ValueError("dimensions must be positive")
    return (length + width) / (2.0 * height)

def righting_time_index(m: Morphology, b: float = 1.0 / 3.0, k: float = 0.35, neck_lever: float = 1.0) -> float:
    if m.mass <= 0 or neck_lever <= 0:
        raise ValueError("mass and neck_lever must be positive")
    return b * m.length**k / (m.mass * neck_lever)

def lsm_vector(text: str) -> Dict[str, float]:
    """Return a sparse LSM vector: proportion of each function category."""
    ws = []
    for word in text.split():
        if word[0].lower() in '0123456789':
            continue
        ws.append(word.lower())
    ws = re.findall(r"[a-z]+(?:'[a-z]+)?", (text or "").lower())
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

def lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    """
    Deterministic similarity between two LSM vectors.
    Returns (overall_score, per‑category detail).
    """
    detail: Dict[str, float] = {}
    for cat in FUNCTION_CATS:
        av = a.get(cat, 0.0)
        bv = b.get(cat, 0.0)
        # harmonic‑like similarity bounded in [0,1]
        score = 1.0 - (abs(av - bv) / (av + bv + 1e-6))
        score = max(0.0, min(1.0, score))
        detail[cat] = score
    overall = sum(detail.values()) / len(FUNCTION_CATS)
    return overall, detail

def hybrid_semantic_neighbors_bandit_router(morphology: Morphology, document: Document, bandit_action: BanditAction) -> BanditAction:
    # Calculate semantic recovery priority
    semantic_recovery_priority = sphericity_index(morphology.length, morphology.width, morphology.height)
    
    # Calculate confidence bound
    confidence_bound = bandit_action.confidence_bound * (1 + semantic_recovery_priority)
    
    return BanditAction(
        action_id=bandit_action.action_id,
        propensity=bandit_action.propensity,
        expected_reward=bandit_action.expected_reward,
        confidence_bound=confidence_bound,
        algorithm=bandit_action.algorithm
    )

def hybrid_lsm_score(a: Dict[str, float], b: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
    # Calculate LSM score
    overall, detail = lsm_score(a, b)
    
    # Adjust LSM score based on morphology
    morphology = Morphology(length=5.0, width=3.0, height=2.0, mass=10.0)
    semantic_recovery_priority = sphericity_index(morphology.length, morphology.width, morphology.height)
    adjusted_score = overall * (1 + semantic_recovery_priority)
    
    return adjusted_score, detail

def hybrid_bandit_update(bandit_update: BanditUpdate, document: Document) -> BanditUpdate:
    # Calculate LSM vector
    lsm_vector = lsm_vector(document.vector)
    
    # Calculate bandit confidence bound
    confidence_bound = bandit_update.confidence_bound * (1 + lsm_vector["preposition"])
    
    return BanditUpdate(
        context_id=bandit_update.context_id,
        action_id=bandit_update.action_id,
        reward=bandit_update.reward,
        propensity=bandit_update.propensity,
        confidence_bound=confidence_bound
    )

if __name__ == "__main__":
    morphology = Morphology(length=5.0, width=3.0, height=2.0, mass=10.0)
    document = Document(id="doc1", vector="This is a test document")
    bandit_action = BanditAction(action_id="act1", propensity=0.5, expected_reward=10.0, confidence_bound=0.2, algorithm="alg1")
    
    hybrid_bandit_action = hybrid_semantic_neighbors_bandit_router(morphology, document, bandit_action)
    print(hybrid_bandit_action)
    
    lsm_vector_a = lsm_vector(document.vector)
    lsm_vector_b = lsm_vector("This is another test document")
    hybrid_lsm_score_a, detail_a = hybrid_lsm_score(lsm_vector_a, lsm_vector_b)
    print(hybrid_lsm_score_a, detail_a)
    
    bandit_update = BanditUpdate(context_id="ctx1", action_id="act1", reward=10.0, propensity=0.5, confidence_bound=0.2)
    hybrid_bandit_update = hybrid_bandit_update(bandit_update, document)
    print(hybrid_bandit_update)