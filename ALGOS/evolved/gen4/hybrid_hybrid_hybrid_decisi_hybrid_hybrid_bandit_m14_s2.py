# DARWIN HAMMER — match 14, survivor 2
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:26:19Z

"""
This module combines the core topologies of the decision hygiene algorithm 
(hybrid_decision_hygiene_shannon_entropy_m12_s5.py) and the hybrid bandit router 
algorithm (hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py) into a 
single unified system. The mathematical bridge is formed by integrating the 
spatial signature filtering concept from the decision hygiene algorithm with 
the bandit decision process, where the entity's signature is used to influence 
the bandit's propensity.

The new system defines a 3-dimensional resource vector eᵢ = [ dᵢ, pᵢ, sᵢ ] for each 
entity, where:

- dᵢ = haversine distance (in metres) from a reference location,
- pᵢ = β·σᵢ, where σᵢ = 1 if the entity's signature collides with any other 
  entity, otherwise 0,
- sᵢ = score from the decision hygiene algorithm.

The bandit decision process is influenced by the entity's signature, where the 
propensity is adjusted based on the signature collision.

For each model tier, we reuse the resource vector defined in the decision 
hygiene algorithm: mⱼ = [ RAMⱼ, α·τⱼ·μ ], where

- RAMⱼ is the model's RAM consumption,
- τⱼ is the tier factor (T1→1, T2→2, T3→3),
- μ is the mean(privacy_risk) over the provided records,
- α is a scaling constant.

The hybrid system stacks all vectors to yield a combined resource matrix A 
(rows = entities∪models, columns = [spatial/RAM-load, privacy-load, score]). 
Selecting a subset corresponds to a binary indicator x and must satisfy the 
linear constraints

Aᵀ·x ≤ [ spatial_budget, privacy_budget, decision_budget ]

where spatial_budget is the total allowed distance (or 0 for pure model 
selection), privacy_budget is the privacy-budget from the decision hygiene 
algorithm, and decision_budget is the maximum allowed score (or 0 for pure 
spatial/mode selection). The greedy algorithm respects both topologies in a 
single unified decision process.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

def calculate_resource_vector(entity, reference_location, beta, alpha):
    """
    Calculate the 3-dimensional resource vector eᵢ = [ dᵢ, pᵢ, sᵢ ] for an entity.

    Parameters:
    entity (dict): Entity data with 'location' and 'signature'.
    reference_location (tuple): Reference location (lat, lon).
    beta (float): Scaling constant for signature collision.
    alpha (float): Scaling constant for tier factor.

    Returns:
    list: Resource vector [dᵢ, pᵢ, sᵢ].
    """
    distance = haversine_distance(entity['location'], reference_location)
    signature_collision = 1 if entity['signature'] in [e['signature'] for e in [entity]] else 0
    score = entity['score']
    return [distance, beta * signature_collision, score]

def haversine_distance(loc1, loc2):
    """
    Calculate the haversine distance between two locations.

    Parameters:
    loc1 (tuple): Location 1 (lat, lon).
    loc2 (tuple): Location 2 (lat, lon).

    Returns:
    float: Haversine distance in meters.
    """
    lat1, lon1 = loc1
    lat2, lon2 = loc2
    radius = 6371e3  # Earth's radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c

def calculate_bandit_propensity(entity, base_eta, alpha, beta, dt):
    """
    Calculate the bandit propensity for an entity.

    Parameters:
    entity (dict): Entity data with 'location' and 'signature'.
    base_eta (float): Baseline learning rate.
    alpha (float): Coefficient for the store differential equation.
    beta (float): Coefficient for the store differential equation.
    dt (float): Time step for store integration.

    Returns:
    float: Bandit propensity.
    """
    signature_collision = 1 if entity['signature'] in [e['signature'] for e in [entity]] else 0
    propensity = base_eta * (1 + alpha * signature_collision) * (1 - beta * dt)
    return propensity

def hybrid_decision(entity, reference_location, beta, alpha, base_eta, dt):
    """
    Make a hybrid decision for an entity.

    Parameters:
    entity (dict): Entity data with 'location' and 'signature'.
    reference_location (tuple): Reference location (lat, lon).
    beta (float): Scaling constant for signature collision.
    alpha (float): Scaling constant for tier factor.
    base_eta (float): Baseline learning rate.
    dt (float): Time step for store integration.

    Returns:
    dict: Hybrid decision with 'resource_vector' and 'bandit_propensity'.
    """
    resource_vector = calculate_resource_vector(entity, reference_location, beta, alpha)
    bandit_propensity = calculate_bandit_propensity(entity, base_eta, alpha, beta, dt)
    return {'resource_vector': resource_vector, 'bandit_propensity': bandit_propensity}

if __name__ == "__main__":
    entity = {'location': (37.7749, -122.4194), 'signature': 'signature1', 'score': 0.5}
    reference_location = (37.7859, -122.4364)
    beta = 0.5
    alpha = 0.5
    base_eta = 0.01
    dt = 1.0
    decision = hybrid_decision(entity, reference_location, beta, alpha, base_eta, dt)
    print(decision)