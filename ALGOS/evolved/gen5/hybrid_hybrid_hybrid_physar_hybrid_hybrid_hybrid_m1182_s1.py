# DARWIN HAMMER — match 1182, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_physarum_netw_hybrid_sparse_wta_hy_m336_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_bandit_hybrid_voronoi_parti_m169_s0.py (gen3)
# born: 2026-05-29T23:33:15Z

"""
Module for the hybrid algorithm that combines the Flux-based conductance update primitive from 
physarum_network_hybrid_hybrid_bandit_m11_s3.py and the Voronoi partition and circuit-breaker 
functionality from hybrid_voronoi_partition_hybrid_endpoint_circ_m47_s3.py. The mathematical 
bridge between these two structures lies in the concept of conductance and propensity. In the 
context of Physarum networks, conductance represents the ease with which material can flow 
between points. Similarly, in the context of the Hybrid Bandit TTT model, propensity represents 
the inflow rate of a bandit action. By integrating these concepts, we can create a hybrid system 
that updates the conductance of a network based on the propensity of bandit actions and uses the 
Euclidean distance in the Voronoi partition to inform action selection.
"""

import numpy as np
import random
import math
import sys
import pathlib

def flux(conductance, edge_length, pressure_a, pressure_b, eps=1e-12):
    if edge_length <= 0:
        raise ValueError('edge_length must be positive')
    return conductance / max(edge_length, eps) * (pressure_a - pressure_b)

def update_conductance(conductance, q, dt=1.0, gain=1.0, decay=0.05):
    if dt < 0 or decay < 0:
        raise ValueError('dt and decay must be non-negative')
    return max(0.0, conductance + dt * (gain * abs(q) - decay * conductance))

def hybrid_bandit_update(conductance, propensity, reward, dt=1.0, gain=1.0, decay=0.05):
    q = propensity * reward
    return update_conductance(conductance, q, dt, gain, decay)

def euclidean_distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def hybrid_voronoi_update(conductance, point, dt=1.0, gain=1.0, decay=0.05):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # N, S, E, W
    closest_distance = float('inf')
    closest_point = None
    for dx, dy in neighbors:
        neighbor_point = (point[0] + dx, point[1] + dy)
        distance = euclidean_distance(point, neighbor_point)
        if distance < closest_distance:
            closest_distance = distance
            closest_point = neighbor_point
    q = conductance / closest_distance
    return update_conductance(conductance, q, dt, gain, decay)

def hybrid_select_action(conductance, points, dt=1.0, gain=1.0, decay=0.05):
    actions = []
    for point in points:
        actions.append((point, hybrid_voronoi_update(conductance, point, dt, gain, decay)))
    return max(actions, key=lambda x: x[1][0])[0]

if __name__ == "__main__":
    conductance = 1.0
    points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    dt = 1.0
    gain = 1.0
    decay = 0.05
    action = hybrid_select_action(conductance, points, dt, gain, decay)
    print(action)