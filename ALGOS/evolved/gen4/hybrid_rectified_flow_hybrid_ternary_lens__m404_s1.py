# DARWIN HAMMER — match 404, survivor 1
# gen: 4
# parent_a: rectified_flow.py (gen0)
# parent_b: hybrid_ternary_lens_audit_hybrid_hybrid_privac_m154_s1.py (gen3)
# born: 2026-05-29T23:28:49Z

from __future__ import annotations
from typing import Any, Iterable
from dataclasses import dataclass
import numpy as np
from math import exp

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

@dataclass
class Candidate:
    candidate_key: str
    family: str
    notes: str
    classification: str
    fast_path_compatible: bool
    benchmark_required: bool
    benchmark_evidence: bool

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded: dict[str, ModelTier] = {}
        self.sensitive_records: list[Any] = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def load(self, model: ModelTier, candidate: Candidate) -> None:
        if candidate.classification == "unsafe_for_fastpath" and model.tier == "T3":
            if self._used_ram() + model.ram_mb > self.ram_ceiling_mb:
                raise RuntimeError("RAM ceiling exceeded")
            if self._used_vram() + model.vram_mb > self.vram_ceiling_mb:
                raise RuntimeError("VRAM ceiling exceeded")
            self.loaded[model.name] = model

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    if total_records == 0:
        return 1.0  # or np.inf, depending on the application
    return exp(-unique_quasi_identifiers / total_records)

def interpolant(x0: np.ndarray, x1: np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Straight-line interpolant: Z_t = t * x1 + (1 - t) * x0.

    Broadcasts t over a leading batch dimension.  If x0 has shape (B, d) and t
    has shape (B,), t is reshaped to (B, 1) 
    """
    return t * x1 + (1 - t) * x0

def flow_target(x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
    """Constant-velocity vector field: v = (X_1 - X_0)"""
    return x1 - x0

def hybrid_risk_score(model_pool: ModelPool, candidate: Candidate, 
                      x0: np.ndarray, x1: np.ndarray, t: float | np.ndarray) -> float:
    z_t = interpolant(x0, x1, t)
    v = flow_target(x0, x1)
    risk_score = reconstruction_risk_score(len(model_pool.sensitive_records), 
                                           len(model_pool.loaded) + 1)  # Adjust total_records
    return risk_score * np.linalg.norm(v)

def evaluate_hybrid_risk(model_pool: ModelPool, candidates: Iterable[Candidate], 
                         x0: np.ndarray, x1: np.ndarray, t: float | np.ndarray) -> dict:
    risk_scores = {}
    for candidate in candidates:
        risk_scores[candidate.candidate_key] = hybrid_risk_score(model_pool, 
                                                                candidate, 
                                                                x0, x1, t)
    return risk_scores

if __name__ == "__main__":
    model_pool = ModelPool()
    candidate1 = Candidate("key1", "family1", "notes1", "T1", True, False, False)
    candidate2 = Candidate("key2", "family2", "notes2", "T2", False, True, True)
    x0 = np.array([1.0, 2.0])
    x1 = np.array([3.0, 4.0])
    t = 0.5
    risk_scores = evaluate_hybrid_risk(model_pool, [candidate1, candidate2], 
                                       x0, x1, t)
    print(risk_scores)