from __future__ import annotations
from dataclasses import dataclass

class ModelLoadError(RuntimeError): pass

@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str

TIER_T1_QWEN_0_5B=ModelTier("qwen-0.5b", 512, "T1")
TIER_T2_REASONING=ModelTier("reasoning-t2", 3000, "T2")
TIER_T2_TOOL=ModelTier("tool-t2", 2600, "T2")
TIER_T3_QWEN_7B=ModelTier("qwen-7b", 7000, "T3")

class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb=ram_ceiling_mb; self.loaded={}
    def is_loaded(self, name: str) -> bool: return name in self.loaded
    def _used(self) -> int: return sum(m.ram_mb for m in self.loaded.values())
    def load(self, model: ModelTier) -> None:
        if model.tier=="T3" and any(m.tier=="T2" for m in self.loaded.values()): raise ModelLoadError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb: raise ModelLoadError("RAM ceiling exceeded")
        self.loaded[model.name]=model
    def load_with_eviction(self, model: ModelTier) -> None:
        while self.loaded and model.ram_mb + self._used() > self.ram_ceiling_mb:
            self.loaded.pop(next(iter(self.loaded)))
        self.load(model)
