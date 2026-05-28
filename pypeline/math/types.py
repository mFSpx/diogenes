from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MathClaim:
    id: str
    text: str
    subject: str
    predicate: str
    object: str
    source: str
    timestamp: str

@dataclass(frozen=True)
class MathEvidence:
    id: str
    source: str
    hash: str
    admissibility: float = 1.0
    reliability: float = 1.0

@dataclass(frozen=True)
class MathHypothesis:
    id: str
    prior: float
    posterior: float
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class MathAction:
    id: str
    type: str
    expected_value: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class MathCounterfactual:
    id: str
    action: str
    predicted_actor_response: str = ""
    probability: float = 0.0
    loss_bound: float = 0.0
