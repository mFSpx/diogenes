# DARWIN HAMMER — match 5584, survivor 4
# gen: 6
# parent_a: hybrid_rlct_grokking_dendritic_compartmen_m61_s1.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_ternar_m599_s0.py (gen5)
# born: 2026-05-30T00:03:11Z

import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np

@dataclass
class Endpoint:
    name: str
    health_score: float = 0.0
    failure_rate: float = 0.0
    recovery_priority: float = 0.0
    observations: int = 0

def calculate_membrane_potential(
    V: float,
    C_m: float,
    g_L: float,
    E_L: float,
    g_Na: float,
    E_Na: float,
    m: float,
    h: float,
    g_K: float,
    E_K: float,
    n_gate: float,
    I_syn: float,
) -> float:
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n_gate ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    dV_dt = (-I_L - I_Na - I_K + I_syn) / C_m
    return V + dV_dt

def calculate_free_energy(
    n: int,
    L0: float,
    lambda_rlct: float,
    m_param: int = 1,
) -> float:
    if n <= 1:
        raise ValueError("n must be greater than 1 for logarithms.")
    term1 = n * L0
    term2 = lambda_rlct * math.log(n)
    term3 = (m_param - 1) * math.log(math.log(n)) if m_param > 1 else 0.0
    return term1 + term2 + term3

def hoeffding_bound(r: float, delta: float, n: int) -> float:
    if n <= 0:
        return float("inf")
    return r * math.sqrt(math.log(2.0 / delta) / (2.0 * n))

def health_score_from_physics(V: float, free_energy: float, alpha: float = 0.01, beta: float = 0.001) -> float:
    exponent = -alpha * (V ** 2) - beta * free_energy
    exponent = max(exponent, -700)
    return math.exp(exponent)

def select_endpoint(endpoints: List[Endpoint], delta: float = 0.05, r: float = 1.0) -> Endpoint:
    best_ep = None
    best_ucb = -float("inf")
    for ep in endpoints:
        bound = hoeffding_bound(r, delta, ep.observations)
        ucb = ep.health_score + bound
        if ucb > best_ucb:
            best_ucb = ucb
            best_ep = ep
    return best_ep

def route_packet(packet: Dict[str, Any], endpoints: List[Endpoint], delta: float = 0.05) -> Dict[str, Any]:
    chosen = select_endpoint(endpoints, delta=delta)
    chosen.observations += 1
    routed = packet.copy()
    routed["routed_to"] = chosen.name
    routed["routing_success"] = True
    return routed

def hybrid_step(
    V: float,
    C_m: float,
    g_L: float,
    E_L: float,
    g_Na: float,
    E_Na: float,
    m_gate: float,
    h_gate: float,
    g_K: float,
    E_K: float,
    n_gate: float,
    I_syn: float,
    n_data: int,
    L0: float,
    lambda_rlct: float,
    endpoints: List[Endpoint],
    learning_rate: float = 0.1,
) -> Dict[str, Any]:
    V_new = calculate_membrane_potential(
        V, C_m, g_L, E_L, g_Na, E_Na, m_gate, h_gate, g_K, E_K, n_gate, I_syn
    )
    F = calculate_free_energy(n_data, L0, lambda_rlct)
    for ep in endpoints:
        ep.health_score = health_score_from_physics(V_new, F)
        ep.health_score += learning_rate * (ep.health_score - health_score_from_physics(V, F))
    dummy_packet = {"payload": "example", "timestamp": sys.maxsize}
    routed_packet = route_packet(dummy_packet, endpoints)
    return {
        "V_before": V,
        "V_after": V_new,
        "free_energy": F,
        "selected_endpoint": routed_packet["routed_to"],
        "packet": routed_packet,
    }

if __name__ == "__main__":
    V0 = -65.0
    C_m = 1.0
    g_L = 0.1
    E_L = -65.0
    g_Na = 120.0
    E_Na = 50.0
    m_gate = 0.05
    h_gate = 0.6
    g_K = 36.0
    E_K = -77.0
    n_gate = 0.32
    I_syn = 10.0
    n_data = 10000
    L0 = 0.02
    lambda_rlct = 0.5
    endpoint_names = ["alpha", "beta", "gamma"]
    endpoints = [Endpoint(name=n) for n in endpoint_names]
    for _ in range(100):
        result = hybrid_step(
            V0, C_m, g_L, E_L, g_Na, E_Na, m_gate, h_gate, g_K, E_K, n_gate, I_syn, n_data, L0, lambda_rlct, endpoints
        )
        V0 = result["V_after"]
        print(result["selected_endpoint"])