# DARWIN HAMMER — match 764, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s0.py (gen2)
# parent_b: hybrid_hybrid_xgboost_objec_hybrid_hybrid_label__m144_s1.py (gen4)
# born: 2026-05-29T23:30:45Z

"""
Hybrid of hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s0.py and hybrid_hybrid_xgboost_objec_hybrid_hybrid_label__m144_s1.py:
This module integrates the pheromone-based surface usage tracking and entropy-based action selection
from hybrid_hybrid_pheromone_inf_hybrid_decision_hygi_m37_s0.py with the XGBoost-based labeling
and decision-making from hybrid_hybrid_xgboost_objec_hybrid_hybrid_label__m144_s1.py. The mathematical
bridge between the two lies in using the entropy calculation to analyze the distribution of decision
hygiene scores, which are then used as inputs to the XGBoost labeling function. Moreover, the pheromone
probabilities are used to inform the decision hygiene scoring, ultimately guiding the selection of
actions based on surface usage patterns and decision-making process.
"""

import numpy as np
import math
import random
import sys
from pathlib import Path
import re

def calculate_pheromone_probabilities(surface_key, limit, db_url):
    """Calculates pheromone probabilities from the database."""
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('''SELECT signal_value FROM lucidota_runtime.surface_pheromone 
                            WHERE surface_key=%s ORDER BY created_at DESC LIMIT %s''', (surface_key, limit))
            pheromones = [r['signal_value'] for r in cur.fetchall()]
            total = sum(pheromones)
            return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    """Calculates the entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    """Calculates the expected entropy of an action."""
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

def decision_hygiene_score(text: str) -> dict[str, int]:
    EVIDENCE_RE = re.compile(r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|")
    matches = EVIDENCE_RE.findall(text)
    return {match: len(matches) for match in set(matches)}

def sigmoid(margin: np.ndarray | float) -> np.ndarray | float:
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-margin))

def binary_logistic_grad_hess(y_true: np.ndarray, margin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and second gradients for binary logistic loss in margin space."""
    p = sigmoid(margin)
    g = p - y_true
    h = p * (1.0 - p)
    return g, h

def optimal_leaf_weight(gradient_sum: float, hessian_sum: float, reg_lambda: float = 1.0) -> float:
    """Optimal leaf weight for XGBoost."""
    return -float(gradient_sum) / (float(hessian_sum) + float(reg_lambda))

def split_gain(
    left_gradient: float,
    left_hessian: float,
    right_gradient: float,
    right_hessian: float,
    *,
    reg_lambda: float = 1.0,
    gamma: float = 0.0,
) -> float:
    """Split gain for XGBoost."""
    gl, hl = float(left_gradient), float(left_hessian)
    gr, hr = float(right_gradient), float(right_hessian)
    parent = (gl + gr) ** 2 / (hl + hr + reg_lambda)
    children = gl**2 / (hl + reg_lambda) + gr**2 / (hr + reg_lambda)
    return parent - children

def labeling_function(name: str|None=None):
    def deco(fn: callable): 
        fn.lf_name=name or fn.__name__; 
        return fn
    return deco

def aggregate_labels(batches: list[list[dict]]) -> list[dict]:
    votes = {}
    for batch in batches:
        for result in batch:
            lf_name = result.get('lf_name')
            doc_id = result.get('doc_id')
            label = result.get('label')
            if lf_name not in votes:
                votes[lf_name] = {}
            if doc_id not in votes[lf_name]:
                votes[lf_name][doc_id] = []
            votes[lf_name][doc_id].append(label)
    return votes

def hybrid_pheromone_labeling(surface_key, limit, db_url, text):
    pheromone_probabilities = calculate_pheromone_probabilities(surface_key, limit, db_url)
    decision_hygiene = decision_hygiene_score(text)
    entropy_score = entropy(pheromone_probabilities)
    return {
        'pheromone_probabilities': pheromone_probabilities,
        'decision_hygiene': decision_hygiene,
        'entropy_score': entropy_score
    }

def hybrid_xgboost_labeling(margin, y_true):
    sigmoid_score = sigmoid(margin)
    gradient, hessian = binary_logistic_grad_hess(y_true, margin)
    return {
        'sigmoid_score': sigmoid_score,
        'gradient': gradient,
        'hessian': hessian
    }

def hybrid_fusion(margin, y_true, surface_key, limit, db_url, text):
    pheromone_labeling = hybrid_pheromone_labeling(surface_key, limit, db_url, text)
    xgboost_labeling = hybrid_xgboost_labeling(margin, y_true)
    return {
        'pheromone_probabilities': pheromone_labeling['pheromone_probabilities'],
        'decision_hygiene': pheromone_labeling['decision_hygiene'],
        'entropy_score': pheromone_labeling['entropy_score'],
        'sigmoid_score': xgboost_labeling['sigmoid_score'],
        'gradient': xgboost_labeling['gradient'],
        'hessian': xgboost_labeling['hessian']
    }

if __name__ == "__main__":
    surface_key = 'test_surface'
    limit = 10
    db_url = 'postgresql://user:password@host:port/dbname'
    text = 'This is a test text with evidence and verification.'
    margin = np.array([0.5, 0.5])
    y_true = np.array([1, 0])
    result = hybrid_fusion(margin, y_true, surface_key, limit, db_url, text)
    print(result)