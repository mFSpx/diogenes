# DARWIN HAMMER — match 4972, survivor 4
# gen: 7
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_bandit_m1896_s2.py (gen6)
# parent_b: hybrid_hybrid_privacy_model_hybrid_hybrid_fisher_m33_s0.py (gen4)
# born: 2026-05-29T23:59:14Z

import numpy as np
import math
import random
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable, List, Dict, Set, Tuple

# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------
def now_z() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# ----------------------------------------------------------------------
# Parent A – Model Pool and Stylometry
# ----------------------------------------------------------------------
FUNCTION_CATS: dict[str, set[str]] = {
    "pronoun": set("i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()),
    "article": set("a an the".split()),
    "preposition": set("about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()),
    "auxiliary": set("am are be been being can could did do does had has have is may might must shall should was were will would".split()),
    "conjunction": set("and but or nor so yet because although while if when where whereas unless until".split()),
    "negation": set("no not never none neither cannot can't won't don't didn't isn't aren't wasn't weren't".split()),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set("very really just still already also even only then there here now often always sometimes".split()),
}

class ModelTier:
    def __init__(self, name: str, ram_mb: int, tier: str):
        self.name = name
        self.ram_mb = ram_mb
        self.tier = tier

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def update_loaded(self, name: str, ram_mb: int) -> None:
        self.loaded[name] = ram_mb

# ----------------------------------------------------------------------
# Parent B – Bandit core
# ----------------------------------------------------------------------
def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0:
        raise ValueError('width must be positive')
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError('samples must have equal length')
    if not x:
        raise ValueError('samples must not be empty')
    n = len(x)
    mx = np.mean(x)
    my = np.mean(y)
    vx = np.var(x)
    vy = np.var(y)
    cov = np.mean((x - mx) * (y - my))
    c1 = (k1 * dynamic_range) ** 2
    c2 = (k2 * dynamic_range) ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Return a normalized reconstruction risk in [0,1]."""
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers / total_records))

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_privacy_risk_vector(model_pool: ModelPool, quasi_identifiers: List[int], records: int) -> np.ndarray:
    risk_vector = np.zeros(len(quasi_identifiers))
    for i, quasi_identifier in enumerate(quasi_identifiers):
        risk_vector[i] = reconstruction_risk_score(quasi_identifier, records)
    return risk_vector

def hybrid_model_resource_matrix(model_pool: ModelPool, ram_ceiling_mb: int = 6000) -> np.ndarray:
    model_resources = np.zeros((len(model_pool.loaded), len(model_pool.loaded)))
    for i, (name, ram_mb) in enumerate(model_pool.loaded.items()):
        for j, (other_name, other_ram_mb) in enumerate(model_pool.loaded.items()):
            model_resources[i, j] = gaussian_beam(ram_mb, 0, ram_ceiling_mb)
    # Apply SSIM to adjust the weights
    for i in range(len(model_pool.loaded)):
        for j in range(len(model_pool.loaded)):
            if i != j:
                model_resources[i, j] *= ssim(np.array([ram_mb]), np.array([other_ram_mb]))
    return model_resources

def hybrid_select_models(model_pool: ModelPool, quasi_identifiers: List[int], records: int, ram_ceiling_mb: int = 6000) -> List[str]:
    risk_vector = hybrid_privacy_risk_vector(model_pool, quasi_identifiers, records)
    model_resources = hybrid_model_resource_matrix(model_pool, ram_ceiling_mb)
    # Use Fisher score as a weighting factor
    weights = np.linalg.inv(model_resources) @ risk_vector
    weights *= np.array([fisher_score(ram_mb, 0, ram_ceiling_mb) for ram_mb in model_pool.loaded.values()])
    return [name for name, _ in model_pool.loaded.items() if weights[list(model_pool.loaded.keys()).index(name)] > 0.5]

# ----------------------------------------------------------------------
# Smoke Test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    model_pool = ModelPool(ram_ceiling_mb=6000)
    model_pool.update_loaded("model1", 1000)
    model_pool.update_loaded("model2", 2000)
    quasi_identifiers = [10, 20]
    records = 10000
    selected_models = hybrid_select_models(model_pool, quasi_identifiers, records)
    print(selected_models)