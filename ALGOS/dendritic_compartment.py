#!/usr/bin/env python3
"""
Multi-Compartment Dendritic ODEs — Hodgkin-Huxley cable model.

ODE for each compartment i:

    C_m * dV_i/dt = -g_L*(V_i - E_L)
                  + sum_{j in N(i)} g_ij * (V_j - V_i)
                  + I_ion(V_i)
                  + I_syn(t)

Where:
  V_i      — membrane potential of compartment i (mV)
  C_m      — membrane capacitance (uF/cm^2)
  g_L      — passive leak conductance (mS/cm^2)
  E_L      — leak reversal potential (mV)
  g_ij     — axial coupling conductance between compartments i and j (mS/cm^2)
  I_ion    — nonlinear ion channel currents: Na+, K+, Ca2+ (here Na+ and K+)
  I_syn    — synaptic input modeled as conductance change

Hodgkin-Huxley ion channels:

    I_Na = g_Na * m^3 * h * (V - E_Na)      (fast inward, transient)
    I_K  = g_K  * n^4    * (V - E_K)        (delayed rectifier, outward)
    I_L  = g_L           * (V - E_L)        (passive leak)

Gate dynamics: dx/dt = alpha_x(V)*(1-x) - beta_x(V)*x
  m — Na+ activation   (fast)
  h — Na+ inactivation (slow)
  n — K+  activation   (medium)

The cute note: a single biological neuron with a branching dendritic tree IS
a deep neural network.  Each dendritic branch performs a thresholded nonlinear
integration before passing signals toward the soma.  The soma is just the last
layer.  The tree is the network.  Evolution found deep learning in one cell.
This is the Poirazi & Mel (2003) result.  A layer-5 cortical neuron has ~200
compartments; each branch acts as a two-layer sigmoid unit.  Dendritic spikes
are the hidden-layer activations.  The tree computes; the soma reads out.

References:
  Hodgkin & Huxley (1952) J Physiol 117:500-544
  Poirazi & Mel (2003) Neuron 37:977-987
  Koch & Segev (2000) Nature Neurosci 3:1171-1177
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "sodium_current",
    "potassium_current",
    "leak_current",
    "alpha_beta_gates",
    "compartment_step",
    "simulate_tree",
    "build_linear_dendrite",
]

# ---------------------------------------------------------------------------
# Ion channel currents
# ---------------------------------------------------------------------------

def sodium_current(V, m, h, g_Na=120.0, E_Na=50.0):
    """Hodgkin-Huxley sodium current.

    I_Na = g_Na * m^3 * h * (V - E_Na)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    m:
        Na+ activation gate variable, in [0, 1].
    h:
        Na+ inactivation gate variable, in [0, 1].
    g_Na:
        Maximum Na+ conductance (mS/cm^2). Default 120.0.
    E_Na:
        Na+ reversal (Nernst) potential (mV). Default 50.0.

    Returns
    -------
    float or np.ndarray
        Sodium current I_Na (uA/cm^2). Inward (depolarizing) when V < E_Na.
    """
    return g_Na * (m ** 3) * h * (V - E_Na)


def potassium_current(V, n, g_K=36.0, E_K=-77.0):
    """Hodgkin-Huxley potassium current.

    I_K = g_K * n^4 * (V - E_K)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    n:
        K+ activation gate variable, in [0, 1].
    g_K:
        Maximum K+ conductance (mS/cm^2). Default 36.0.
    E_K:
        K+ reversal potential (mV). Default -77.0.

    Returns
    -------
    float or np.ndarray
        Potassium current I_K (uA/cm^2). Outward (repolarizing) when V > E_K.
    """
    return g_K * (n ** 4) * (V - E_K)


def leak_current(V, g_L=0.3, E_L=-54.4):
    """Passive leak current.

    I_L = g_L * (V - E_L)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.
    g_L:
        Leak conductance (mS/cm^2). Default 0.3.
    E_L:
        Leak reversal potential (mV). Default -54.4.

    Returns
    -------
    float or np.ndarray
        Leak current I_L (uA/cm^2).
    """
    return g_L * (V - E_L)


# ---------------------------------------------------------------------------
# Gate kinetics
# ---------------------------------------------------------------------------

def alpha_beta_gates(V):
    """Hodgkin-Huxley gating variable rate functions.

    Standard HH formulas for the squid axon at 6.3 degC.  V is measured
    relative to resting potential (convention: rest ~ -65 mV).  The original
    HH convention used depolarization as positive from rest; here V is the
    absolute membrane potential in mV so all expressions are shifted by -65.

    Rate equations (per ms):
        alpha_m = 0.1*(V+40) / (1 - exp(-(V+40)/10))     [care: removable singularity at V=-40]
        beta_m  = 4 * exp(-(V+65)/18)
        alpha_h = 0.07 * exp(-(V+65)/20)
        beta_h  = 1 / (1 + exp(-(V+35)/10))
        alpha_n = 0.01*(V+55) / (1 - exp(-(V+55)/10))    [removable singularity at V=-55]
        beta_n  = 0.125 * exp(-(V+65)/80)

    Parameters
    ----------
    V:
        Membrane potential (mV). Scalar or numpy array.

    Returns
    -------
    dict
        Keys: alpha_m, beta_m, alpha_h, beta_h, alpha_n, beta_n.
        Values are floats or arrays matching V shape.
    """
    V = np.asarray(V, dtype=float)

    # alpha_m: handle removable singularity at V = -40
    dv_m = V + 40.0
    alpha_m = np.where(
        np.abs(dv_m) < 1e-7,
        1.0,                                         # limit value
        0.1 * dv_m / (1.0 - np.exp(-dv_m / 10.0))
    )

    beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)

    alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
    beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))

    # alpha_n: handle removable singularity at V = -55
    dv_n = V + 55.0
    alpha_n = np.where(
        np.abs(dv_n) < 1e-7,
        0.1,                                         # limit value
        0.01 * dv_n / (1.0 - np.exp(-dv_n / 10.0))
    )

    beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)

    # Return scalars when V was scalar
    if V.ndim == 0:
        return {
            "alpha_m": float(alpha_m),
            "beta_m":  float(beta_m),
            "alpha_h": float(alpha_h),
            "beta_h":  float(beta_h),
            "alpha_n": float(alpha_n),
            "beta_n":  float(beta_n),
        }
    return {
        "alpha_m": alpha_m,
        "beta_m":  beta_m,
        "alpha_h": alpha_h,
        "beta_h":  beta_h,
        "alpha_n": alpha_n,
        "beta_n":  beta_n,
    }


# ---------------------------------------------------------------------------
# Single compartment Euler step
# ---------------------------------------------------------------------------

def compartment_step(V, gating, adj_V, g_axial, I_syn, C_m=1.0, g_L=0.3, E_L=-65.0, dt=0.025):
    """One forward-Euler step for a single compartment.

    Solves:
        C_m * dV/dt = -I_L - I_Na - I_K
                    + sum_j g_ij*(V_j - V)
                    + I_syn

    Gate update (forward Euler):
        dx/dt = alpha_x*(1-x) - beta_x*x

    Note: I_ion sign convention — inward current is positive in HH; we
    subtract it from the RHS because inward Na+ is depolarizing only when
    the driving force (V - E_Na) is negative, i.e., I_Na as computed is
    negative when inward.  No sign flip needed: the formula already
    encodes direction through the driving force (V - E_reversal).

    Parameters
    ----------
    V:
        Scalar membrane potential of this compartment (mV).
    gating:
        Dict with keys 'm', 'h', 'n' — current gate states (floats in [0,1]).
    adj_V:
        List of (V_j, g_ij) tuples for each neighboring compartment j.
        V_j: neighbor potential (mV). g_ij: axial conductance (mS/cm^2).
    g_axial:
        Not used directly; axial terms are encoded in adj_V. Kept for API
        symmetry; pass 0.0 if unused.
    I_syn:
        Scalar synaptic current input (uA/cm^2) at this time step.
        Positive = depolarizing.
    C_m:
        Membrane capacitance (uF/cm^2). Default 1.0.
    g_L:
        Leak conductance (mS/cm^2). Default 0.3.
    E_L:
        Leak reversal potential (mV). Default -65.0.
    dt:
        Time step (ms). Default 0.025.

    Returns
    -------
    tuple
        (V_new, gating_new) where V_new is a float and gating_new is a dict
        with updated 'm', 'h', 'n'.
    """
    m = gating["m"]
    h = gating["h"]
    n = gating["n"]

    rates = alpha_beta_gates(V)
    am, bm = rates["alpha_m"], rates["beta_m"]
    ah, bh = rates["alpha_h"], rates["beta_h"]
    an, bn = rates["alpha_n"], rates["beta_n"]

    # Ion currents (sign: positive = outward)
    I_Na = sodium_current(V, m, h)      # inward when V << E_Na  -> negative value
    I_K  = potassium_current(V, n)      # outward when V >> E_K  -> positive value
    I_L  = leak_current(V, g_L, E_L)   # passive

    # Axial coupling: sum g_ij * (V_j - V)
    I_axial = 0.0
    for V_j, g_ij in adj_V:
        I_axial += g_ij * (V_j - V)

    # Membrane equation (subtract outward, add inward-is-negative, add coupling, add syn)
    dV = (-(I_Na + I_K + I_L) + I_axial + I_syn) / C_m

    V_new = V + dt * dV

    # Gate ODEs: forward Euler
    m_new = m + dt * (am * (1.0 - m) - bm * m)
    h_new = h + dt * (ah * (1.0 - h) - bh * h)
    n_new = n + dt * (an * (1.0 - n) - bn * n)

    # Clamp gates to [0, 1]
    m_new = float(np.clip(m_new, 0.0, 1.0))
    h_new = float(np.clip(h_new, 0.0, 1.0))
    n_new = float(np.clip(n_new, 0.0, 1.0))

    gating_new = {"m": m_new, "h": h_new, "n": n_new}
    return float(V_new), gating_new


# ---------------------------------------------------------------------------
# Tree simulation
# ---------------------------------------------------------------------------

def simulate_tree(tree_adj, V0, I_syn_seq, dt=0.025, T=100.0):
    """Simulate a full dendritic tree over time T.

    Uses forward Euler stepping.  All compartments are updated in parallel
    (previous-step voltages used for coupling, so order is irrelevant).

    Parameters
    ----------
    tree_adj:
        Dict mapping node id -> list of (neighbor_id, g_axial) pairs.
        Node ids can be any hashable (int, str, etc.).
    V0:
        Dict node -> float initial membrane potential (mV).
    I_syn_seq:
        Dict node -> 1-D numpy array of synaptic currents over time.
        Array length must equal n_steps = int(T / dt).
        Nodes not present receive zero synaptic input.
    dt:
        Time step (ms). Default 0.025.
    T:
        Total simulation time (ms). Default 100.0.

    Returns
    -------
    dict
        V_traces: node -> numpy array of shape (n_steps+1,) containing
        membrane potential at every time step including t=0.
    """
    n_steps = int(T / dt)
    nodes = list(tree_adj.keys())

    # Initialize state
    V = {node: float(V0.get(node, -65.0)) for node in nodes}

    # Steady-state gate initialization at resting V
    def _steady_gates(v):
        r = alpha_beta_gates(v)
        m_inf = r["alpha_m"] / (r["alpha_m"] + r["beta_m"])
        h_inf = r["alpha_h"] / (r["alpha_h"] + r["beta_h"])
        n_inf = r["alpha_n"] / (r["alpha_n"] + r["beta_n"])
        return {"m": m_inf, "h": h_inf, "n": n_inf}

    gating = {node: _steady_gates(V[node]) for node in nodes}

    # Pre-allocate traces
    V_traces = {node: np.empty(n_steps + 1) for node in nodes}
    for node in nodes:
        V_traces[node][0] = V[node]

    # Zero-pad synaptic sequences for any missing nodes
    I_syn_arrays = {}
    for node in nodes:
        if node in I_syn_seq:
            I_syn_arrays[node] = np.asarray(I_syn_seq[node], dtype=float)
        else:
            I_syn_arrays[node] = np.zeros(n_steps)

    for step in range(n_steps):
        V_prev = dict(V)   # snapshot for coupling
        for node in nodes:
            # Build adjacency list using previous-step voltages
            adj = [(V_prev[nb], g_ax) for nb, g_ax in tree_adj[node]]
            i_syn = float(I_syn_arrays[node][step])
            V[node], gating[node] = compartment_step(
                V_prev[node], gating[node], adj, 0.0, i_syn, dt=dt
            )
            V_traces[node][step + 1] = V[node]

    return V_traces


# ---------------------------------------------------------------------------
# Helper: build linear dendritic cable
# ---------------------------------------------------------------------------

def build_linear_dendrite(n_compartments, g_axial=0.1):
    """Build a linear dendritic cable as a tree_adj dict.

    Compartments are indexed 0 .. n_compartments-1.
    0 is the distal tip; n_compartments-1 is the soma.
    Each adjacent pair is connected with g_axial.

    Parameters
    ----------
    n_compartments:
        Number of compartments in the cable.
    g_axial:
        Axial coupling conductance (mS/cm^2). Default 0.1.

    Returns
    -------
    dict
        tree_adj: int -> list of (neighbor_int, g_axial).
    """
    tree_adj = {}
    for i in range(n_compartments):
        neighbors = []
        if i > 0:
            neighbors.append((i - 1, g_axial))
        if i < n_compartments - 1:
            neighbors.append((i + 1, g_axial))
        tree_adj[i] = neighbors
    return tree_adj


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Setup: 5-compartment linear dendrite
    #   0 (distal tip) -- 1 -- 2 -- 3 -- 4 (soma)
    # Inject current only at distal tip (compartment 0).
    # Watch the signal propagate to soma (compartment 4).
    # ------------------------------------------------------------------
    N = 5
    dt = 0.025          # ms
    T  = 80.0           # ms
    n_steps = int(T / dt)
    t_axis  = np.linspace(0.0, T, n_steps + 1)

    tree_adj = build_linear_dendrite(N, g_axial=0.5)

    V0 = {i: -65.0 for i in range(N)}   # all compartments at rest

    # Synaptic input: square pulse at compartment 0 from t=5ms to t=25ms
    I_syn_seq = {}
    for i in range(N):
        arr = np.zeros(n_steps)
        if i == 0:
            t_start = int(5.0 / dt)
            t_end   = int(25.0 / dt)
            arr[t_start:t_end] = 20.0   # uA/cm^2 drive at the tip
        I_syn_seq[i] = arr

    V_traces = simulate_tree(tree_adj, V0, I_syn_seq, dt=dt, T=T)

    # ------------------------------------------------------------------
    # Linear comparison: what would the soma see if we just summed the
    # tip current attenuated by a simple RC cable (no HH nonlinearity)?
    # A passive cable attenuates by exp(-x/lambda) per compartment.
    # With 4 hops and a rough lambda ~ 2 compartments, linear expects ~exp(-2).
    # ------------------------------------------------------------------
    attenuation = np.exp(-4.0 / 2.0)     # crude passive estimate
    soma_linear_peak = 20.0 * attenuation

    soma_V  = V_traces[4]
    tip_V   = V_traces[0]
    soma_peak = float(np.max(soma_V) - (-65.0))   # depolarization above rest

    # Print summary
    print("=" * 62)
    print("Dendritic Compartment Model — 5-compartment linear cable")
    print(f"  dt={dt} ms  T={T} ms  g_axial=0.5 mS/cm^2")
    print(f"  Synaptic drive: compartment 0 (tip), 20 uA/cm^2, t=5-25 ms")
    print("=" * 62)

    # Voltage traces at key time points
    probe_times_ms = [0.0, 5.0, 10.0, 20.0, 30.0, 50.0, 70.0]
    probe_idx = [min(int(pt / dt), n_steps) for pt in probe_times_ms]
    header = f"  {'t(ms)':>7}  " + "  ".join(f"C{i:d}(mV)" for i in range(N))
    print(header)
    print("-" * 62)
    for idx, pt in zip(probe_idx, probe_times_ms):
        row = f"  {pt:>7.1f}  " + "  ".join(f"{V_traces[i][idx]:>7.2f}" for i in range(N))
        print(row)
    print("-" * 62)

    print()
    print("Peak depolarization per compartment (mV above rest):")
    for i in range(N):
        peak_dep = float(np.max(V_traces[i]) - (-65.0))
        label = " (tip)" if i == 0 else (" (soma)" if i == N - 1 else "")
        print(f"  C{i}{label}: {peak_dep:+.3f} mV")

    print()
    print("Nonlinear vs linear comparison at soma:")
    print(f"  Passive cable linear estimate: ~{soma_linear_peak:.3f} uA/cm^2 equivalent")
    print(f"  HH soma peak depolarization:   {soma_peak:+.3f} mV above rest")
    if soma_peak > 30.0:
        print("  -> Dendritic spike propagated: soma fired. Nonlinear amplification.")
    elif soma_peak > 5.0:
        print("  -> Sub-threshold but nonlinearly amplified at soma (EPSP).")
    else:
        print("  -> Signal attenuated; soma near rest. Increase drive or g_axial.")

    print()
    print("Biological insight (Poirazi & Mel 2003):")
    print("  Each dendritic branch is a thresholded sigmoidal unit.")
    print("  The branching tree IS a multi-layer neural network in one cell.")
    print("  Soma = last layer readout. Nonlinear integration precedes it.")
    print("=" * 62)
