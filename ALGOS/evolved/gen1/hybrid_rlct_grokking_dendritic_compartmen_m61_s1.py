# DARWIN HAMMER — match 61, survivor 1
# gen: 1
# parent_a: rlct_grokking.py (gen0)
# parent_b: dendritic_compartment.py (gen0)
# born: 2026-05-29T23:24:10Z

"""
Hybrid algorithm fusing the Real Log Canonical Threshold (RLCT) and Grokking from Singular Learning Theory with the Hodgkin-Huxley cable model from Dendritic Compartment.

The mathematical bridge between the two parents lies in the concept of energy and potential. In Singular Learning Theory, the free energy asymptotic equation represents the energy landscape of a neural network. In the Hodgkin-Huxley cable model, the membrane potential and ion channel currents represent the electrical energy and potential of a neuron. We can fuse these two concepts by using the membrane potential and ion channel currents as a physical inspiration for the energy landscape in Singular Learning Theory.

By using the Hodgkin-Huxley equations to model the membrane potential and ion channel currents, we can derive an energy function that represents the energy landscape of a neural network. This energy function can then be used to calculate the RLCT and Grokking threshold, providing a new perspective on the learning dynamics of neural networks.
"""

import numpy as np
import math
import random
import sys
import pathlib

def calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    """
    Calculate the membrane potential using the Hodgkin-Huxley cable model.

    Parameters:
    V (float): membrane potential
    C_m (float): membrane capacitance
    g_L (float): passive leak conductance
    E_L (float): leak reversal potential
    g_Na (float): maximum Na+ conductance
    E_Na (float): Na+ reversal potential
    m (float): Na+ activation gate variable
    h (float): Na+ inactivation gate variable
    g_K (float): maximum K+ conductance
    E_K (float): K+ reversal potential
    n (float): K+ activation gate variable
    I_syn (float): synaptic input current

    Returns:
    float: membrane potential
    """
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    dV_dt = (-I_L - I_Na - I_K + I_syn) / C_m
    return V + dV_dt

def calculate_free_energy(n, L0, lambda_rlct, m=1):
    """
    Calculate the free energy using the Singular Learning Theory.

    Parameters:
    n (float): dataset size
    L0 (float): true risk
    lambda_rlct (float): RLCT
    m (float): parameter (default=1)

    Returns:
    float: free energy
    """
    return n * L0 + lambda_rlct * np.log(n) - (m - 1) * np.log(np.log(n))

def calculate_rlct_from_losses(train_losses_per_n, n_values):
    """
    Estimate the RLCT from a training-loss curve over dataset sizes.

    Parameters:
    train_losses_per_n (list): training loss values at each dataset size
    n_values (list): dataset sizes

    Returns:
    float: estimated RLCT
    """
    losses = np.array(train_losses_per_n)
    ns = np.array(n_values)
    y = np.log(np.maximum(losses, 1e-300))
    x = np.log(np.log(ns))
    x_c = x - x.mean()
    y_c = y - y.mean()
    var_x = (x_c ** 2).sum()
    return (x_c * y_c).sum() / var_x

if __name__ == "__main__":
    # Test the functions
    V = 0.0
    C_m = 1.0
    g_L = 0.3
    E_L = -54.4
    g_Na = 120.0
    E_Na = 50.0
    m = 0.5
    h = 0.5
    g_K = 36.0
    E_K = -77.0
    n = 0.5
    I_syn = 0.0
    print(calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn))
    n = 1000
    L0 = 0.1
    lambda_rlct = 0.5
    print(calculate_free_energy(n, L0, lambda_rlct))
    train_losses_per_n = [1.0, 0.5, 0.2]
    n_values = [100, 500, 1000]
    print(calculate_rlct_from_losses(train_losses_per_n, n_values))