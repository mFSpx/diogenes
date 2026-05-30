# DARWIN HAMMER — match 14, survivor 1
# gen: 4
# parent_a: hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py (gen3)
# parent_b: hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py (gen2)
# born: 2026-05-29T23:26:19Z

"""
This module fuses the core topologies of the hybrid_hybrid_decision_hygi_hybrid_possum_filter_m22_s3.py 
and the hybrid_hybrid_bandit_router_hybrid_model_vram_sc_m35_s5.py algorithms into a single unified system.
The mathematical bridge is formed by integrating the resource vector formulation from the decision hygiene 
algorithm with the bandit's propensity and expected reward from the bandit router algorithm.
The new system defines a 3-dimensional resource vector eᵢ = [ dᵢ, pᵢ, sᵢ ] for each entity, where:
- dᵢ = haversine distance (in metres) from a reference location,
- pᵢ = β·σᵢ, where σᵢ = 1 if the entity's signature collides with any other entity, otherwise 0,
- sᵢ = score from the decision hygiene algorithm, modulated by the bandit's expected reward.
For each model tier, we reuse the resource vector defined in the decision hygiene algorithm: 
mⱼ = [ RAMⱼ, α·τⱼ·μ ], where
- RAMⱼ is the model's RAM consumption,
- τⱼ is the tier factor (T1→1, T2→2, T3→3),
- μ is the mean(privacy_risk) over the provided records,
- α is a scaling constant.
The bandit's propensity and expected reward are integrated into the resource vector formulation, 
creating a deeper feedback loop between the decision hygiene and bandit router algorithms.
"""

import math
import random
import sys
from pathlib import Path
import numpy as np

def calculate_resource_vector(entity, reference_location, beta, sigma):
    """
    Calculate the 3-dimensional resource vector eᵢ for an entity.
    
    Parameters:
    entity (dict): Entity with 'location' and 'signature' attributes.
    reference_location (tuple): Reference location (latitude, longitude).
    beta (float): Scaling constant for σᵢ.
    sigma (float): 1 if the entity's signature collides with any other entity, otherwise 0.
    
    Returns:
    list: 3-dimensional resource vector eᵢ.
    """
    distance = haversine_distance(entity['location'], reference_location)
    pi = beta * sigma
    score = entity['score']
    return [distance, pi, score]

def haversine_distance(location1, location2):
    """
    Calculate the haversine distance between two locations.
    
    Parameters:
    location1 (tuple): Location 1 (latitude, longitude).
    location2 (tuple): Location 2 (latitude, longitude).
    
    Returns:
    float: Haversine distance between the two locations.
    """
    lat1, lon1 = location1
    lat2, lon2 = location2
    radius = 6371  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = radius * c * 1000  # Convert to meters
    return distance

def calculate_bandit_propensity(entity, base_eta, alpha, beta, dt, store_decay):
    """
    Calculate the bandit's propensity for an entity.
    
    Parameters:
    entity (dict): Entity with 'action_id' and 'expected_reward' attributes.
    base_eta (float): Baseline learning rate.
    alpha (float): Coefficient for the store differential equation.
    beta (float): Coefficient for the store differential equation.
    dt (float): Time step for store integration.
    store_decay (float): Exponential decay applied to the store each step.
    
    Returns:
    float: Bandit's propensity for the entity.
    """
    action_id = entity['action_id']
    expected_reward = entity['expected_reward']
    propensity = base_eta * (1 + alpha * expected_reward) * (1 - beta * store_decay * dt)
    return propensity

def integrate_resource_vector_and_bandit_propensity(entities, reference_location, beta, sigma, base_eta, alpha, beta_bandit, dt, store_decay):
    """
    Integrate the resource vector formulation with the bandit's propensity.
    
    Parameters:
    entities (list): List of entities with 'location', 'signature', 'score', 'action_id', and 'expected_reward' attributes.
    reference_location (tuple): Reference location (latitude, longitude).
    beta (float): Scaling constant for σᵢ.
    sigma (float): 1 if the entity's signature collides with any other entity, otherwise 0.
    base_eta (float): Baseline learning rate.
    alpha (float): Coefficient for the store differential equation.
    beta_bandit (float): Coefficient for the bandit's propensity.
    dt (float): Time step for store integration.
    store_decay (float): Exponential decay applied to the store each step.
    
    Returns:
    list: Integrated resource vectors with bandit's propensity.
    """
    integrated_vectors = []
    for entity in entities:
        resource_vector = calculate_resource_vector(entity, reference_location, beta, sigma)
        propensity = calculate_bandit_propensity(entity, base_eta, alpha, beta_bandit, dt, store_decay)
        integrated_vector = resource_vector + [propensity]
        integrated_vectors.append(integrated_vector)
    return integrated_vectors

if __name__ == "__main__":
    entities = [
        {'location': (40.7128, -74.0060), 'signature': 'entity1', 'score': 0.5, 'action_id': 'action1', 'expected_reward': 0.8},
        {'location': (34.0522, -118.2437), 'signature': 'entity2', 'score': 0.7, 'action_id': 'action2', 'expected_reward': 0.9}
    ]
    reference_location = (37.7749, -122.4194)
    beta = 0.5
    sigma = 0.2
    base_eta = 0.01
    alpha = 0.1
    beta_bandit = 0.2
    dt = 0.1
    store_decay = 0.9
    integrated_vectors = integrate_resource_vector_and_bandit_propensity(entities, reference_location, beta, sigma, base_eta, alpha, beta_bandit, dt, store_decay)
    print(integrated_vectors)