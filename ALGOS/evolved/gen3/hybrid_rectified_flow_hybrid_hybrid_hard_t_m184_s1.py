# DARWIN HAMMER — match 184, survivor 1
# gen: 3
# parent_a: rectified_flow.py (gen0)
# parent_b: hybrid_hybrid_hard_truth_ma_kan_m27_s0.py (gen2)
# born: 2026-05-29T23:27:22Z

"""
Rectified Flow and Kolmogorov-Arnold Networks (KAN) Hybrid.

This module fuses the Rectified Flow Matching algorithm (rectified_flow.py) 
with the Kolmogorov-Arnold Networks (KAN) algorithm (hybrid_hybrid_hard_truth_ma_kan_m27_s0.py). 
The mathematical bridge between the two structures is found by integrating 
the straight-line interpolant of Rectified Flow with the B-spline basis 
and deep KAN composition of the KAN algorithm.

The Rectified Flow's straight-line interpolant is used to generate input 
features for the KAN layers, which are then used to predict the output 
vector field of the Rectified Flow.

Imports:
    numpy
    standard library
    math
    random
    sys
    pathlib
"""

import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import math
import random
import sys
import pathlib

# ----------------------------------------------------------------------
# Parent A – Rectified Flow utilities
# ----------------------------------------------------------------------

def interpolant(x0, x1, t):
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0.

    Broadcasts t over a leading batch dimension.  If x0 has shape (B, d) and t
    has shape (B,), t is reshaped to (B, 1) 
    """
    t = np.reshape(t, (-1, 1))
    return t * x1 + (1 - t) * x0

def flow_target(x0, x1):
    """Target vector field: v_theta(Z_t, t) = (X_1 - X_0)."""
    return x1 - x0

# ----------------------------------------------------------------------
# Parent B – KAN utilities
# ----------------------------------------------------------------------

FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}
PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def words(text: str) -> List[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def lsm_vector(text: str) -> dict[str, float]:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    return {
        cat: sum(cnt[w] for w in vocab) / total
        for cat, vocab in FUNCTION_CATS.items()
    }

# ----------------------------------------------------------------------
# Hybrid Rectified Flow and KAN
# ----------------------------------------------------------------------

class HybridRectifiedFlowKAN:
    def __init__(self, input_dim, output_dim, num_kAN_layers):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_kAN_layers = num_kAN_layers
        self.kan_weights = np.random.rand(num_kAN_layers, input_dim, output_dim)

    def kan_layer(self, inputs, weights):
        return np.dot(inputs, weights)

    def hybrid_operation(self, x0, x1, t):
        z_t = interpolant(x0, x1, t)
        lsm_features = lsm_vector(str(z_t))
        kan_inputs = np.array(list(lsm_features.values()))
        kan_outputs = np.zeros((self.output_dim,))
        for i in range(self.num_kAN_layers):
            kan_outputs = self.kan_layer(kan_inputs, self.kan_weights[i])
            kan_inputs = kan_outputs
        v_theta = kan_outputs
        loss = np.mean((v_theta - flow_target(x0, x1)) ** 2)
        return loss

if __name__ == "__main__":
    np.random.seed(0)
    x0 = np.random.rand(10)
    x1 = np.random.rand(10)
    t = np.random.rand()
    hybrid_rf_kAN = HybridRectifiedFlowKAN(8, 10, 2)
    loss = hybrid_rf_kAN.hybrid_operation(x0, x1, t)
    print(loss)