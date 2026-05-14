#!/usr/bin/env python3
"""Endpoint circuit-breaker and pool selection primitives."""
from __future__ import annotations
class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int=3): self.failure_threshold=failure_threshold; self.failures=0; self.open=False
    def record_success(self): self.failures=0; self.open=False
    def record_failure(self): self.failures+=1; self.open=self.failures>=self.failure_threshold
    def allow(self) -> bool: return not self.open
class EndpointPool:
    def __init__(self, endpoints: list[str]): self.endpoints=endpoints; self.breakers={e:EndpointCircuitBreaker() for e in endpoints}
    def available(self) -> list[str]: return [e for e in self.endpoints if self.breakers[e].allow()]
