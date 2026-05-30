# DARWIN HAMMER — match 1549, survivor 2
# gen: 6
# parent_a: hybrid_hybrid_hybrid_infota_hybrid_fold_change_d_m128_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1139_s2.py (gen5)
# born: 2026-05-29T23:37:20Z

"""Hybrid Entropic‑Pheromone‑Morphology Fusion (HEPMF)

Parents:
- **Parent A**: hybrid_hybrid_hybrid_infota_hybrid_fold_change_d_m128_s0.py  
  Provides MinHash‑based entropic signatures and a bandit policy whose
  propensity is modulated by Hamming similarity of those signatures.

- **Parent B**: hybrid_hybrid_hybrid_pherom_hybrid_hybrid_hybrid_m1139_s2.py  
  Provides a morphology‑derived recovery priority *p* ∈ [0,1], a pheromone
  decay function ϕ(t)=v₀·0.5^{t/τ}, and a structural similarity σ(x,y)∈[0,1].

**Mathematical Bridge**  
The bridge treats the Hamming similarity between MinHash signatures as an
alternative to the SSIM similarity σ.  For a pair of agents *i* and *j* we
define  

    σ̂_{ij} = HammingSim(sig_i, sig_j) ∈ [0,1]

and fuse the three quantities exactly as in Parent B:

    H_{ij} = p_i·ϕ_i(t) + (1‑p_i)·σ̂_{ij}

When many agents are present, stacking the morphologies yields a diagonal
matrix **Pdiag** of priorities *p_i*.  The pheromone decay yields a vector
ϕ(t) which is broadcast into a diagonal matrix **Φ**.  The similarity matrix
**Ŝ** is built from σ̂_{ij}.  The fused influence matrix is therefore

    F = Pdiag·Φ + (I‑Pdiag)·Ŝ

The resulting matrix **F** drives the drag‑limited integration of each
agent’s state (position, velocity) while the bandit policy is updated with
the same similarity‑derived reward, completing a mathematically unified
system.  
"""

import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple
import numpy as np

# ----------------------------------------------------------------------
# Global bandit policy store (Parent A)
# ----------------------------------------------------------------------
_POLICY: Dict[str, List[float]] = {}

def reset_policy() -> None:
    """Reset the global bandit policy."""
    _POLICY.clear()

def _reward(action: str) -> float:
    """Average reward observed for *action*."""
    total, n = _POLICY.get(action, [0.0, 0.0])
    return total / n if n else 0.0

def _count(action: str) -> float:
    """Number of times *action* has been taken."""
    _, n = _POLICY.get(action, [0.0, 0.0])
    return n

# ----------------------------------------------------------------------
# Parent A – Entropic MinHash utilities
# ----------------------------------------------------------------------
def entropic_minhash(probs: np.ndarray, num_perm: int = 64) -> np.ndarray:
    """
    Build a MinHash signature from a discrete probability distribution.

    The signature consists of the minimum hash value observed for each of
    ``num_perm`` random hash functions applied to the categorical indices.
    """
    if probs.ndim != 1:
        raise ValueError("Probability vector must be 1‑D")
    probs = probs / probs.sum()
    # Sample categorical indices according to the distribution
    categories = np.arange(len(probs))
    draws = np.random.choice(categories, size=1024, p=probs)
    signature = np.full(num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)
    rng = np.random.default_rng()
    # Random hash coefficients a,b for universal hashing: h(x) = (a*x + b) mod prime
    prime = (1 << 61) - 1
    a = rng.integers(1, prime, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, prime, size=num_perm, dtype=np.uint64)
    for x in draws:
        hx = (a * np.uint64(x) + b) % prime
        signature = np.minimum(signature, hx)
    return signature

def hamming_similarity(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """
    Normalised Hamming similarity between two MinHash signatures.
    Returns a value in [0,1] where 1 means identical signatures.
    """
    if sig1.shape != sig2.shape:
        raise ValueError("Signatures must have the same shape")
    equal = np.sum(sig1 == sig2)
    return equal / sig1.size

def bandit_policy_update(context_id: str,
                         action_id: str,
                         reward: float,
                         propensity: float) -> None:
    """
    Update the global bandit policy using the supplied reward.
    The update follows a simple additive rule:
        total_reward += reward
        count        += 1
    """
    total, n = _POLICY.get(action_id, [0.0, 0.0])
    _POLICY[action_id] = [total + reward, n + 1]

# ----------------------------------------------------------------------
# Parent B – Morphology, pheromone decay and SSIM‑like similarity
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Morphology:
    length: float
    width:  float
    height: float
    mass:   float

def righting_time_index(morph: Morphology) -> float:
    """
    Compute a scalar recovery priority p ∈ [0,1] from morphology.
    A simple proxy: normalised mass‑to‑volume ratio.
    """
    volume = morph.length * morph.width * morph.height + 1e-12
    ratio = morph.mass / volume
    # Normalise using a soft‑plus to keep the result in (0,1)
    return 1.0 / (1.0 + math.exp(-ratio))

def pheromone_decay(t: float, v0: float = 1.0, tau: float = 10.0) -> float:
    """Temporal decay ϕ(t)=v₀·0.5^{t/τ}."""
    return v0 * (0.5 ** (t / tau))

def ssim_placeholder(x: np.ndarray, y: np.ndarray) -> float:
    """
    Placeholder for structural similarity (SSIM).  Returns a pseudo‑random
    similarity in [0,1] based on the Euclidean distance of the inputs.
    """
    dist = np.linalg.norm(x - y)
    return math.exp(-dist)  # decays with distance, bounded in (0,1]

# ----------------------------------------------------------------------
# Hybrid core – fusion of both parents
# ----------------------------------------------------------------------
@dataclass
class HybridAgent:
    """State holder for a single agent."""
    agent_id: str
    morphology: Morphology
    signature: np.ndarray          # MinHash signature (Parent A)
    position: float = 0.0
    velocity: float = 0.0
    pheromone_grid: float = 1.0    # scalar representing local pheromone level

def compute_fused_influence(agents: List[HybridAgent],
                            t: float,
                            v0: float = 1.0,
                            tau: float = 10.0) -> np.ndarray:
    """
    Build the fused influence matrix F = Pdiag·Φ + (I‑Pdiag)·Ŝ.

    - Pdiag : diagonal matrix of recovery priorities p_i.
    - Φ    : diagonal matrix of pheromone decays ϕ_i(t) = pheromone_decay(t)·grid_i.
    - Ŝ    : similarity matrix built from Hamming similarity of MinHash signatures.
    """
    n = len(agents)
    if n == 0:
        return np.empty((0, 0))

    # 1. Recovery priorities
    p_vec = np.array([righting_time_index(a.morphology) for a in agents])
    Pdiag = np.diag(p_vec)

    # 2. Pheromone decay vector (broadcast to diagonal)
    phi_vec = np.array([pheromone_decay(t, v0, tau) * a.pheromone_grid for a in agents])
    Phi = np.diag(phi_vec)

    # 3. Similarity matrix from MinHash signatures
    S = np.empty((n, n), dtype=float)
    for i, ai in enumerate(agents):
        for j, aj in enumerate(agents):
            if i <= j:
                sim = hamming_similarity(ai.signature, aj.signature)
                S[i, j] = S[j, i] = sim
    # 4. Fuse
    identity = np.eye(n)
    F = Pdiag @ Phi + (identity - Pdiag) @ S
    return F

def hybrid_strike(agents: List[HybridAgent],
                  dt: float = 0.1,
                  steps: int = 10,
                  v0: float = 1.0,
                  tau: float = 10.0) -> List[HybridAgent]:
    """
    Run a drag‑limited integration for each agent.
    The force applied to agent *i* at step *k* is taken as the i‑th row sum
    of the fused influence matrix F_k.
    After the force is computed, the bandit policy is updated with a reward
    proportional to the force magnitude.
    """
    for step in range(steps):
        t = step * dt
        F = compute_fused_influence(agents, t, v0, tau)

        for i, agent in enumerate(agents):
            # Force is the row sum – a scalar summarising influence on this agent
            force = F[i].sum()
            # Simple drag model: a = force - c·v (c = 0.1)
            drag_coeff = 0.1
            acceleration = force - drag_coeff * agent.velocity

            # Update kinematics (Euler integration)
            agent.velocity += acceleration * dt
            agent.position += agent.velocity * dt

            # Bandit update: treat larger positive force as higher reward
            reward = max(force, 0.0)
            propensity = 1.0 / (1.0 + math.exp(-force))  # logistic mapping
            bandit_policy_update(context_id=agent.agent_id,
                                 action_id=f"move_{agent.agent_id}",
                                 reward=reward,
                                 propensity=propensity)
    return agents

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    # Create two synthetic agents
    agents: List[HybridAgent] = []
    for idx in range(2):
        # Random morphology
        morph = Morphology(
            length=random.uniform(0.5, 2.0),
            width =random.uniform(0.5, 2.0),
            height=random.uniform(0.5, 2.0),
            mass  =random.uniform(1.0, 10.0)
        )
        # Random probability distribution for MinHash
        probs = np.random.rand(20)
        probs /= probs.sum()
        sig = entropic_minhash(probs)

        agents.append(
            HybridAgent(
                agent_id=f"agent_{idx}",
                morphology=morph,
                signature=sig,
                position=0.0,
                velocity=0.0,
                pheromone_grid=random.uniform(0.5, 1.5)
            )
        )

    # Run the hybrid strike simulation
    final_agents = hybrid_strike(agents, dt=0.05, steps=20)

    # Print final states and a snapshot of the bandit policy
    for ag in final_agents:
        print(f"{ag.agent_id}: pos={ag.position:.3f}, vel={ag.velocity:.3f}")

    print("\nBandit policy snapshot:")
    for act, (tot, cnt) in _POLICY.items():
        print(f"{act}: total_reward={tot:.3f}, count={cnt}")