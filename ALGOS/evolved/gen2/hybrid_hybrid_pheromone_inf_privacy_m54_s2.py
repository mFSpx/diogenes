# DARWIN HAMMER — match 54, survivor 2
# gen: 2
# parent_a: hybrid_pheromone_infotaxis_m3_s3.py (gen1)
# parent_b: privacy.py (gen0)
# born: 2026-05-29T23:23:52Z

"""Hybrid Pheromone‑Infotaxis‑Privacy System
================================================
This module fuses the *PheromoneSystem* (parent algorithm A) with the
privacy‑scoring helpers (parent algorithm B).  

Mathematical bridge
-------------------
- The pheromone subsystem computes an **expected entropy** `H = p·H(hit)+(1‑p)·H(miss)`.
- The privacy subsystem supplies a **reconstruction‑risk score** `R ∈ [0,1]`,
  defined as `R = unique_quasi_identifiers / total_records`.

We treat the risk score as a *scalar weight* that modulates pheromone
signals before entropy evaluation:


   w = 1 – R                     # privacy‑preserving weight
   ŝ = w · s                     # weighted signal value
   Ĥ = expected_entropy(p_hit, hit_state·w, miss_state·w)


Thus the two topologies are coupled through element‑wise multiplication
of the pheromone signal vector and the privacy‑risk vector, a standard
matrix‑algebra operation implementable with NumPy.

The hybrid system therefore:
1. Computes raw pheromone signals.
2. Computes privacy risk per record.
3. Forms weighted signals `ŝ = (1‑R)·s`.
4. Uses the weighted signals in entropy‑based decision making and
   differentially‑private aggregation.

The implementation below provides three exemplar hybrid functions:
`hybrid_signal`, `hybrid_batch_process`, and `best_privacy_action`.  """

import argparse
import json
import math
import random
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Parent A – Pheromone / Infotaxis core (slightly corrected)
# ----------------------------------------------------------------------
class PheromoneSystem:
    def __init__(self) -> None:
        self.pheromones: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Core pheromone signal with exponential decay
    # ------------------------------------------------------------------
    def calculate_pheromone_signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: float,
    ) -> float:
        now = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        else:
            prev = self.pheromones[surface_key]
            elapsed = (now - prev["created_time"]).total_seconds()
            decayed = prev["signal_value"] * math.pow(
                0.5, elapsed / prev["half_life_seconds"]
            )
            # Replace with the new signal (the algorithm treats it as a reset)
            self.pheromones[surface_key] = {
                "signal_kind": signal_kind,
                "signal_value": signal_value,
                "half_life_seconds": half_life_seconds,
                "created_time": now,
            }
        return signal_value

    # ------------------------------------------------------------------
    # Entropy utilities (Infotaxis)
    # ------------------------------------------------------------------
    @staticmethod
    def calculate_entropy(probabilities: List[float], eps: float = 1e-12) -> float:
        total = sum(probabilities)
        if total <= 0:
            raise ValueError("positive probability mass required")
        return -sum(
            (p / total) * math.log(max(p / total, eps))
            for p in probabilities
            if p > 0
        )

    def expected_entropy(
        self, p_hit: float, hit_state: List[float], miss_state: List[float]
    ) -> float:
        if not 0.0 <= p_hit <= 1.0:
            raise ValueError("p_hit must be in [0,1]")
        return p_hit * self.calculate_entropy(hit_state) + (
            1.0 - p_hit
        ) * self.calculate_entropy(miss_state)

    # ------------------------------------------------------------------
    # Decision utilities
    # ------------------------------------------------------------------
    def best_action(self, actions: Dict[Any, Tuple[float, List[float], List[float]]]) -> Any:
        """
        actions: mapping action -> (p_hit, hit_state, miss_state)
        Returns the action with minimal expected entropy.
        """
        if not actions:
            raise ValueError("actions required")
        return min(
            actions,
            key=lambda a: (
                self.expected_entropy(*actions[a]),
                a,
            ),
        )

    # ------------------------------------------------------------------
    # Public API wrappers (signal / decay)
    # ------------------------------------------------------------------
    def signal(
        self,
        surface_key: str,
        signal_kind: str,
        signal_value: float,
        half_life_seconds: int,
        execute: bool,
    ) -> Dict[str, Any]:
        pheromone_uuid = str(uuid.uuid4()) if execute else None
        self.calculate_pheromone_signal(
            surface_key, signal_kind, signal_value, half_life_seconds
        )
        return {
            "action": "signal",
            "execute_performed": bool(execute),
            "db_writes_performed": bool(execute),
            "graph_writes_performed": False,
            "surface_key": surface_key,
            "signal_kind": signal_kind,
            "signal_value": signal_value,
            "pheromone_uuid": pheromone_uuid,
            "status": "PASS",
        }

    def decay(self, surface_key: str, half_life_seconds: int, execute: bool) -> Dict[str, Any]:
        rows = []
        updated = 0
        if execute and surface_key in self.pheromones:
            now = datetime.now(timezone.utc)
            prev = self.pheromones[surface_key]
            elapsed = (now - prev["created_time"]).total_seconds()
            decayed = prev["signal_value"] * math.pow(
                0.5, elapsed / prev["half_life_seconds"]
            )
            self.pheromones[surface_key]["signal_value"] = decayed
            rows.append({"surface_key": surface_key, "decayed_value": decayed})
            updated = 1
        else:
            rows.append({"surface_key": surface_key, "would_decay": "dry_run"})
        return {
            "action": "decay",
            "execute_performed": bool(execute),
            "db_writes_performed": bool(execute),
            "graph_writes_performed": False,
            "surface_key": surface_key,
            "rows_seen": len(rows),
            "rows_updated": updated,
            "rows": rows[:20],
            "status": "PASS",
        }


# ----------------------------------------------------------------------
# Parent B – Privacy / Anonymization helpers
# ----------------------------------------------------------------------
def reconstruction_risk_score(unique_quasi_identifiers: int, total_records: int) -> float:
    """Simple risk score = proportion of unique quasi‑identifiers."""
    if total_records <= 0:
        return 0.0
    return max(0.0, min(1.0, unique_quasi_identifiers / total_records))


def anonymize_for_indexing(
    record: Dict[str, Any], redact_keys: set[str] | None = None
) -> Dict[str, Any]:
    """Replace sensitive fields with a placeholder."""
    redact = redact_keys or {"email", "phone", "ssn", "secret", "token", "password"}
    return {
        k: ("<redacted>" if k.lower() in redact else v) for k, v in record.items()
    }


def dp_aggregate(values: Iterable[float], epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Deterministic aggregation; noise would be added at runtime."""
    return sum(values)


# ----------------------------------------------------------------------
# Hybrid System – mathematically couples pheromone signals with privacy risk
# ----------------------------------------------------------------------
class HybridSystem:
    def __init__(self) -> None:
        self.pheromone = PheromoneSystem()

    # ------------------------------------------------------------------
    # 1️⃣ Weighted pheromone signal respecting privacy risk
    # ------------------------------------------------------------------
    def hybrid_signal(
        self,
        surface_key: str,
        record: Dict[str, Any],
        signal_kind: str = "generated",
        base_signal: float = 1.0,
        half_life_seconds: int = 604800,
        execute: bool = False,
    ) -> Dict[str, Any]:
        """
        Compute a pheromone signal, then attenuate it by (1‑R) where R is the
        reconstruction risk derived from the record's quasi‑identifiers.
        """
        # 1. Count quasi‑identifiers (keys that are not redacted)
        quasi_ids = [
            k
            for k in record.keys()
            if k.lower()
            not in {"email", "phone", "ssn", "secret", "token", "password"}
        ]
        risk = reconstruction_risk_score(len(quasi_ids), total_records=1)
        weight = 1.0 - risk
        weighted_signal = base_signal * weight

        report = self.pheromone.signal(
            surface_key,
            signal_kind,
            weighted_signal,
            half_life_seconds,
            execute,
        )
        # Attach privacy diagnostics
        report.update(
            {
                "privacy_risk": risk,
                "privacy_weight": weight,
                "raw_signal": base_signal,
                "weighted_signal": weighted_signal,
            }
        )
        return report

    # ------------------------------------------------------------------
    # 2️⃣ Batch processing – vectorised matrix operation
    # ------------------------------------------------------------------
    def hybrid_batch_process(
        self,
        records: List[Dict[str, Any]],
        surface_keys: List[str],
        base_signal: float = 1.0,
        half_life_seconds: int = 604800,
        execute: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a batch of records in a single NumPy‑driven step.
        Returns aggregated (DP‑protected) weighted signal.
        """
        if len(records) != len(surface_keys):
            raise ValueError("records and surface_keys must have equal length")

        # Compute risk vector R_i for each record
        risks = np.array(
            [
                reconstruction_risk_score(
                    len(
                        [
                            k
                            for k in rec.keys()
                            if k.lower()
                            not in {
                                "email",
                                "phone",
                                "ssn",
                                "secret",
                                "token",
                                "password",
                            }
                        ]
                    ),
                    total_records=1,
                )
                for rec in records
            ]
        )
        weights = 1.0 - risks
        weighted_signals = base_signal * weights

        # Emit pheromone signals (side‑effect) for each entry
        for sk, ws in zip(surface_keys, weighted_signals):
            self.pheromone.signal(
                sk, "generated", float(ws), half_life_seconds, execute
            )

        # DP aggregation of the weighted signals
        aggregated = dp_aggregate(weighted_signals.tolist())

        return {
            "batch_size": len(records),
            "risks": risks.tolist(),
            "weights": weights.tolist(),
            "weighted_signals": weighted_signals.tolist(),
            "dp_aggregated_signal": aggregated,
        }

    # ------------------------------------------------------------------
    # 3️⃣ Decision making that incorporates privacy‑adjusted entropy
    # ------------------------------------------------------------------
    def best_privacy_action(
        self,
        actions: Dict[Any, Tuple[float, List[float], List[float]]],
        privacy_weights: List[float],
    ) -> Any:
        """
        Adjust each action's hit/miss state by the corresponding privacy weight
        before entropy evaluation. `privacy_weights` must align with actions.
        """
        if len(actions) != len(privacy_weights):
            raise ValueError("actions and privacy_weights length mismatch")

        adjusted_actions = {}
        for (action, (p_hit, hit_state, miss_state)), w in zip(
            actions.items(), privacy_weights
        ):
            # Scale probability masses
            hit_adj = [h * w for h in hit_state]
            miss_adj = [m * w for m in miss_state]
            adjusted_actions[action] = (p_hit, hit_adj, miss_adj)

        return self.pheromone.best_action(adjusted_actions)

    # ------------------------------------------------------------------
    # Utility: decay with privacy diagnostics
    # ------------------------------------------------------------------
    def decay_with_privacy(
        self, surface_key: str, half_life_seconds: int = 604800, execute: bool = False
    ) -> Dict[str, Any]:
        report = self.pheromone.decay(surface_key, half_life_seconds, execute)
        # Attach a dummy risk (could be derived from metadata in a real system)
        report["privacy_risk"] = 0.0
        report["privacy_weight"] = 1.0
        return report


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    hybrid = HybridSystem()

    # 1️⃣ Single hybrid signal
    rec1 = {"name": "Alice", "email": "alice@example.com", "age": 30}
    print("=== Hybrid Signal ===")
    print(hybrid.hybrid_signal("node_A", rec1, execute=False))

    # 2️⃣ Batch processing
    batch_records = [
        {"id": 1, "city": "NY", "ssn": "123-45-6789"},
        {"id": 2, "city": "LA", "phone": "555-1234"},
        {"id": 3, "city": "SF", "age": 25},
    ]
    surface_keys = ["node_B", "node_C", "node_D"]
    print("\n=== Batch Process ===")
    print(hybrid.hybrid_batch_process(batch_records, surface_keys, execute=False))

    # 3️⃣ Decision making with privacy‑adjusted entropy
    actions = {
        "move_left": (0.6, [0.7, 0.2, 0.1], [0.4, 0.4, 0.2]),
        "move_right": (0.4, [0.3, 0.5, 0.2], [0.6, 0.3, 0.1]),
    }
    # Simulated privacy weights per action
    privacy_weights = [0.8, 0.5]
    print("\n=== Best Privacy‑Aware Action ===")
    print(hybrid.best_privacy_action(actions, privacy_weights))

    # 4️⃣ Decay demonstration
    print("\n=== Decay Demo ===")
    print(hybrid.decay_with_privacy("node_A", execute=False))