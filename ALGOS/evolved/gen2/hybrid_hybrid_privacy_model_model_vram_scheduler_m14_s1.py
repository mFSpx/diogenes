# DARWIN HAMMER — match 14, survivor 1
# gen: 2
# parent_a: hybrid_privacy_model_pool_m7_s1.py (gen1)
# parent_b: model_vram_scheduler.py (gen0)
# born: 2026-05-29T23:22:41Z

"""
This module integrates the privacy/anonymization scoring helpers from 'hybrid_privacy_model_pool_m7_s1.py' 
and the model vram scheduler from 'model_vram_scheduler.py'. 
The mathematical bridge between these two structures is the application of 
reconstruction risk scores to predict the likelihood of RAM or VRAM exhaustion, 
thereby informing model loading, eviction, and vram scheduling decisions.
"""

from __future__ import annotations
from typing import Any, Iterable
from dataclasses import dataclass
import numpy as np
import random
import sys
import pathlib
from math import exp

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    vram_mb: int

TIER_T1_QWEN_0_5B = ModelTier("qwen-0.5b", 512, "T1", 1024)
TIER_T2_REASONING = ModelTier("reasoning-t2", 3000, "T2", 2048)
TIER_T2_TOOL = ModelTier("tool-t2", 2600, "T2", 2048)
TIER_T3_QWEN_7B = ModelTier("qwen-7b", 7000, "T3", 4096)

def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    return 0.0 if total_records <= 0 else max(0.0, min(1.0, unique_quasi_identifiers/total_records))

def anonymize_for_indexing(record: dict[str, Any], redact_keys: set[str]|None=None) -> dict[str, Any]:
    redact = redact_keys or {'email', 'phone', 'ssn', 'secret', 'token', 'password'}
    return {k: ('<redacted>' if k.lower() in redact else v) for k, v in record.items()}

def dp_aggregate(values: Iterable[float], epsilon: float=1.0, sensitivity: float=1.0) -> float:
    return sum(values)  # deterministic core; add noise only at runtime boundary

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000, vram_ceiling_mb: int = 4096):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.vram_ceiling_mb = vram_ceiling_mb
        self.loaded = {}
        self.sensitive_records = []

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used_ram(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _used_vram(self) -> int:
        return sum(m.vram_mb for m in self.loaded.values())

    def load(self, model: ModelTier) -> None:
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")
        if model.vram_mb + self._used_vram() > self.vram_ceiling_mb:
            raise RuntimeError("VRAM ceiling exceeded")
        self.loaded[model.name] = model

    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used_ram() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        while self.loaded and model.vram_mb + self._used_vram() > self.vram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)

    def anonymize_model_metadata(self, model: ModelTier) -> dict[str, Any]:
        metadata = {'name': model.name, 'ram_mb': model.ram_mb, 'tier': model.tier, 'vram_mb': model.vram_mb}
        return anonymize_for_indexing(metadata)

    def predict_ram_exhaustion(self, model: ModelTier) -> float:
        unique_quasi_identifiers = len(self.loaded) + 1
        total_records = self.ram_ceiling_mb // model.ram_mb
        return reconstruction_risk_score(unique_quasi_identifiers, total_records)

    def predict_vram_exhaustion(self, model: ModelTier) -> float:
        unique_quasi_identifiers = len(self.loaded) + 1
        total_records = self.vram_ceiling_mb // model.vram_mb
        return reconstruction_risk_score(unique_quasi_identifiers, total_records)

def demonstrate_hybrid_operation():
    pool = ModelPool()
    model = TIER_T1_QWEN_0_5B
    print("Loading model:", model.name)
    pool.load(model)
    print("Anonymized model metadata:", pool.anonymize_model_metadata(model))
    print("Predicted RAM exhaustion risk:", pool.predict_ram_exhaustion(model))
    print("Predicted VRAM exhaustion risk:", pool.predict_vram_exhaustion(model))

def run_hybrid_model_pooling_simulation():
    pool = ModelPool()
    models = [TIER_T1_QWEN_0_5B, TIER_T2_REASONING, TIER_T2_TOOL, TIER_T3_QWEN_7B]
    random.shuffle(models)
    for model in models:
        try:
            pool.load_with_eviction(model)
            print("Loaded model:", model.name)
        except RuntimeError as e:
            print("Failed to load model:", model.name, "Error:", e)

def compare_ram_vram_exhaustion_risks():
    pool = ModelPool()
    model = TIER_T1_QWEN_0_5B
    print("RAM Exhaustion Risk:", pool.predict_ram_exhaustion(model))
    print("VRAM Exhaustion Risk:", pool.predict_vram_exhaustion(model))

if __name__ == "__main__":
    demonstrate_hybrid_operation()
    run_hybrid_model_pooling_simulation()
    compare_ram_vram_exhaustion_risks()