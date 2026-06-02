"""Unified LUCIDOTA model routing client.

Singleton pattern. All components should use this for model routing.
Uses bandit algorithm from pypeline.math.bandit_router.
"""
from __future__ import annotations
import os, time, threading
import psycopg2
from psycopg2 import sql
from typing import Optional

from pypeline.math.bandit_router import select_action, update_policy, BanditUpdate, reset_policy, _POLICY as BANDIT_POLICY

_DSN = os.environ["LUCIDOTA_GO_STATE_DSN"]


class LucidotaModelRouter:
    _instance: Optional["LucidotaModelRouter"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "LucidotaModelRouter":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._cache: dict = {}
        self._cache_time: float = 0.0
        self._ttl = 60.0
        self._initialized = True
        self._refresh_cache()

    def _refresh_cache(self) -> None:
        """Load full model_routing_policy table from Postgres."""
        try:
            with psycopg2.connect(_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT workload_type, provider, bandit_weight, success_rate, health FROM lucidota_runtime.model_routing_policy")
                    rows = cur.fetchall()
                    new_cache = {}
                    for row in rows:
                        wt, prov, bw, sr, health = row
                        if wt not in new_cache:
                            new_cache[wt] = []
                        new_cache[wt].append({
                            "provider": prov,
                            "bandit_weight": bw,
                            "success_rate": sr,
                            "health": health
                        })
                    self._cache = new_cache
                    self._cache_time = time.monotonic()
        except Exception:
            if not self._cache:
                raise RuntimeError("Failed to load initial model_routing_policy and no cache")

    def _ensure_fresh(self) -> None:
        """Refresh cache if TTL expired."""
        if time.monotonic() - self._cache_time > self._ttl:
            self._refresh_cache()

    def _get_actions(self, workload_type: str) -> list[str]:
        """Get list of provider names for a workload type."""
        self._ensure_fresh()
        entries = self._cache.get(workload_type, [])
        return [e["provider"] for e in entries]

    def _get_entry(self, workload_type: str, provider: str) -> dict:
        """Get specific entry for workload_type + provider."""
        self._ensure_fresh()
        for e in self._cache.get(workload_type, []):
            if e["provider"] == provider:
                return e
        raise KeyError(f"No entry for {workload_type} + {provider}")

    def route(self, workload_type: str, exclude_offline: bool = True) -> str:
        """Return best provider: filter health='ok', sort by bandit_weight * success_rate DESC."""
        self._ensure_fresh()
        entries = self._cache.get(workload_type, [])
        if exclude_offline:
            entries = [e for e in entries if e["health"] == "ok"]
        if not entries:
            raise ValueError(f"No healthy providers for {workload_type}")
        best = max(entries, key=lambda e: e["bandit_weight"] * e["success_rate"])
        return best["provider"]

    def route_with_fallback(self, workload_type: str, fallback: str = "groq") -> str:
        """Try primary route, fall back to fallback on any error."""
        try:
            return self.route(workload_type, exclude_offline=True)
        except Exception:
            return fallback

    def update_outcome(self, workload_type: str, provider: str, success: bool) -> None:
        """Update bandit policy via epsilon-greedy and write new weight back to DB."""
        reward = 1.0 if success else 0.0
        
        # Update bandit policy in memory
        update_policy([BanditUpdate(
            context_id=workload_type,
            action_id=provider,
            reward=reward,
            propensity=1.0
        )])
        
        # Get updated mean from bandit policy
        stats = BANDIT_POLICY.get(provider, [0.0, 0.0])
        new_weight = stats[0] / stats[1] if stats[1] > 0 else 0.0
        
        # Write back to Postgres
        try:
            with psycopg2.connect(_DSN) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("UPDATE lucidota_runtime.model_routing_policy SET bandit_weight = %s WHERE workload_type = %s AND provider = %s"),
                        (new_weight, workload_type, provider)
                    )
                    conn.commit()
        except Exception:
            pass
        
        # Refresh cache to pick up the change
        self._refresh_cache()


# Module-level singleton accessor
def get_router() -> LucidotaModelRouter:
    return LucidotaModelRouter()


# Convenience module-level functions that delegate to singleton
_route = LucidotaModelRouter()

def route(workload_type: str, exclude_offline: bool = True) -> str:
    return _route.route(workload_type, exclude_offline)

def route_with_fallback(workload_type: str, fallback: str = "groq") -> str:
    return _route.route_with_fallback(workload_type, fallback)

def update_outcome(workload_type: str, provider: str, success: bool) -> None:
    return _route.update_outcome(workload_type, provider, success)
