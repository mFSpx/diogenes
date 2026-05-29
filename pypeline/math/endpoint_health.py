from __future__ import annotations
import time

class EndpointCircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold=failure_threshold; self.cooldown_seconds=cooldown_seconds; self.failures={}; self.opened={}
    def record_failure(self, endpoint: str, reason: str = "") -> None:
        self.failures[endpoint]=self.failures.get(endpoint,0)+1
        if self.failures[endpoint]>=self.failure_threshold: self.opened[endpoint]=time.time()
    def is_open(self, endpoint: str) -> bool:
        t=self.opened.get(endpoint)
        if t is None: return False
        if time.time()-t>self.cooldown_seconds:
            self.opened.pop(endpoint,None); self.failures[endpoint]=0; return False
        return True

class EndpointPool:
    def __init__(self, failure_threshold: int = 3, fallback: str = "ollama-hoebrain"):
        self.breaker=EndpointCircuitBreaker(failure_threshold=failure_threshold); self.fallback=fallback
    def record_failure(self, endpoint: str, reason: str = "") -> None: self.breaker.record_failure(endpoint, reason)
    def pick_endpoint(self, preferred: str) -> str: return self.fallback if self.breaker.is_open(preferred) else preferred
