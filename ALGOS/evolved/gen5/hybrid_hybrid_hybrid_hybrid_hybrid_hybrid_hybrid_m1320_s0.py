# DARWIN HAMMER — match 1320, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s4.py (gen4)
# parent_b: hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s2.py (gen4)
# born: 2026-05-29T23:35:14Z

"""hybrid_fusion_regret_certainty.py
Fusion of:
- Parent A (hybrid_hybrid_hybrid_regret_hybrid_sparse_wta_hy_m173_s4.py) – a ModelPool with
  RAM constraints, mutual‑exclusivity tiers and a regret‑style eviction score.
- Parent B (hybrid_hybrid_hybrid_minimu_hybrid_hybrid_hybrid_m108_s2.py) – epistemic
  certainty flags, labeling functions and probabilistic label aggregation.

Mathematical Bridge
------------------
Both parents manipulate *scores* that drive selection:
* A uses a scalar “regret” score for eviction.
* B uses a confidence (basis‑points) attached to epistemic flags and propagates it
  as a weight in label aggregation.

The fusion defines a unified score matrix **S** ∈ ℝ^{M×A} where M is the number of
loaded models and A the number of candidate actions.  Each entry

    S_{m,a} = (EV_a) * (1 + c_m/10_000) – λ·R_m

combines:
- **EV_a** – the expected value of action *a* (from `MathAction.expected_value`);
- **c_m** – the certainty confidence of model *m* (from `CertaintyFlag.confidence_bps`);
- **R_m** – a regret term derived from the model’s recent load‑timestamp (newer
  models incur lower regret);
- **λ** – a tunable regularisation constant (default 0.01).

The matrix is used for:
1. Selecting the best (model, action) pair (max S_{m,a});
2. Computing a weighted‑average probabilistic label for a document, where the
   weights are the row‑wise maxima of **S** (i.e. the most confident model for each
   action).

The following implementation provides three core functions that expose this
hybrid behaviour:
- `compute_score_matrix(models, actions, certainties, lambda_regret)`
- `select_best_model_action(models, actions, certainties, lambda_regret)`
- `aggregate_labels(label_results, certainties)`
"""

import sys
import random
import math
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Dict, Tuple, Callable, Any

import numpy as np

# ----------------------------------------------------------------------
# Data structures from Parent A
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class MathAction:
    id: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0


@dataclass(frozen=True)
class MathCounterfactual:
    action_id: str
    outcome_value: float
    probability: float = 1.0


@dataclass(frozen=True)
class ModelTier:
    name: str
    ram_mb: int
    tier: str
    reference_tokens: Tuple[str, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------
# Model pool with sophisticated eviction (Parent A)
# ----------------------------------------------------------------------
class ModelPool:
    def __init__(self, ram_ceiling_mb: int = 6000):
        self.ram_ceiling_mb = ram_ceiling_mb
        self.loaded: Dict[str, ModelTier] = {}
        self.load_timestamp: Dict[str, float] = {}

    def is_loaded(self, name: str) -> bool:
        return name in self.loaded

    def _used(self) -> int:
        return sum(m.ram_mb for m in self.loaded.values())

    def _check_constraints(self, model: ModelTier) -> None:
        # Mutual exclusivity: T3 cannot coexist with any T2
        if model.tier == "T3" and any(m.tier == "T2" for m in self.loaded.values()):
            raise RuntimeError("T3 mutually exclusive with T2 resident")
        if model.ram_mb + self._used() > self.ram_ceiling_mb:
            raise RuntimeError("RAM ceiling exceeded")

    def load(self, model: ModelTier) -> None:
        self._check_constraints(model)
        self.loaded[model.name] = model
        self.load_timestamp[model.name] = datetime.now(tz=timezone.utc).timestamp()

    def _evict_one(self, score_fn: Callable[[ModelTier, float], float]) -> None:
        """Evict the model with the lowest regret‑adjusted score."""
        if not self.loaded:
            return
        # Compute score for each loaded model using its timestamp
        scores = {
            name: score_fn(self.loaded[name], self.load_timestamp[name])
            for name in self.loaded
        }
        victim_name = min(scores, key=scores.get)
        del self.loaded[victim_name]
        del self.load_timestamp[victim_name]


# ----------------------------------------------------------------------
# Data structures from Parent B
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")

@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int
    authority_class: str
    rationale: str
    evidence_refs: Tuple[str, ...] = ()
    generated_at: str = ""

    def __post_init__(self) -> None:
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10_000:
            raise ValueError("confidence_bps must be in 0..10000")
        if not self.generated_at:
            object.__setattr__(
                self,
                "generated_at",
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence_bps": self.confidence_bps,
            "authority_class": self.authority_class,
            "rationale": self.rationale,
            "evidence_refs": self.evidence_refs,
            "generated_at": self.generated_at,
        }


def certainty(
    label: str,
    *,
    confidence_bps: int,
    authority_class: str,
    rationale: str,
    evidence_refs: Iterable[str] = (),
) -> CertaintyFlag:
    return CertaintyFlag(
        label=label,
        confidence_bps=int(confidence_bps),
        authority_class=authority_class,
        rationale=rationale,
        evidence_refs=tuple(str(x) for x in evidence_refs if x is not None),
    )


@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int


@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float


@dataclass(frozen=True)
class LabelError:
    doc_id: str
    given_label: int
    suggested_label: int
    error_probability: float


def labeling_function(name: str | None = None):
    def deco(fn: Callable[[dict], int]):
        fn.lf_name = name or fn.__name__
        return fn

    return deco


# ----------------------------------------------------------------------
# Hybrid Core – Mathematical Fusion
# ----------------------------------------------------------------------
def compute_score_matrix(
    models: List[ModelTier],
    actions: List[MathAction],
    certainties: Dict[str, CertaintyFlag],
    lambda_regret: float = 0.01,
) -> np.ndarray:
    """
    Build the S matrix where each entry combines action expected value,
    model certainty and a simple regret term based on load age.

    Parameters
    ----------
    models : List[ModelTier]
        Loaded models (order defines rows).
    actions : List[MathAction]
        Candidate actions (order defines columns).
    certainties : Dict[str, CertaintyFlag]
        Mapping model name → CertaintyFlag.
    lambda_regret : float
        Weight of the regret penalty.

    Returns
    -------
    np.ndarray
        Shape (M, A) score matrix.
    """
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    M = len(models)
    A = len(actions)

    # Vectorised expected values
    ev = np.array([a.expected_value for a in actions], dtype=float)  # shape (A,)

    # Prepare model‑wise terms
    certainty_vec = np.empty(M, dtype=float)
    regret_vec = np.empty(M, dtype=float)

    for i, model in enumerate(models):
        cf = certainties.get(model.name)
        confidence = cf.confidence_bps if cf else 0
        certainty_vec[i] = 1.0 + confidence / 10_000.0

        # Regret term: newer models have lower regret (age in seconds)
        load_ts = model_pool.load_timestamp.get(model.name, now_ts)
        age = now_ts - load_ts
        regret_vec[i] = math.log1p(age)  # smooth increasing penalty

    # Broadcast to matrix
    score_mat = ev * certainty_vec[:, None] - lambda_regret * regret_vec[:, None]
    return score_mat


def select_best_model_action(
    models: List[ModelTier],
    actions: List[MathAction],
    certainties: Dict[str, CertaintyFlag],
    lambda_regret: float = 0.01,
) -> Tuple[ModelTier, MathAction, float]:
    """
    Choose the (model, action) pair that maximises the fused score.

    Returns
    -------
    (model, action, score) tuple.
    """
    score_mat = compute_score_matrix(models, actions, certainties, lambda_regret)
    idx_flat = int(np.argmax(score_mat))
    row, col = divmod(idx_flat, score_mat.shape[1])
    best_model = models[row]
    best_action = actions[col]
    best_score = float(score_mat[row, col])
    return best_model, best_action, best_score


def aggregate_labels(
    label_results: List[LabelingFunctionResult],
    certainties: Dict[str, CertaintyFlag],
) -> List[ProbabilisticLabel]:
    """
    Produce a probabilistic label per document by weighting each LF's output
    with the certainty of the model that generated it (if any).  The weight for
    a labeling function result is:

        w = 1 + confidence_bps / 10_000

    The final confidence for a label is the normalised sum of weights for that
    label.
    """
    # Group by doc_id
    doc_groups: Dict[str, List[LabelingFunctionResult]] = {}
    for lf_res in label_results:
        doc_groups.setdefault(lf_res.doc_id, []).append(lf_res)

    aggregated: List[ProbabilisticLabel] = []
    for doc_id, results in doc_groups.items():
        # Accumulate weighted counts per label value
        weight_per_label: Dict[int, float] = {}
        total_weight = 0.0
        for res in results:
            cf = certainties.get(res.lf_name)
            confidence = cf.confidence_bps if cf else 0
            w = 1.0 + confidence / 10_000.0
            weight_per_label[res.label] = weight_per_label.get(res.label, 0.0) + w
            total_weight += w
        # Normalise to probabilities and pick the most probable label
        if total_weight == 0:
            # fallback to uniform probability
            chosen_label = results[0].label
            prob = 1.0 / len(weight_per_label) if weight_per_label else 1.0
        else:
            chosen_label = max(weight_per_label, key=weight_per_label.get)
            prob = weight_per_label[chosen_label] / total_weight
        aggregated.append(ProbabilisticLabel(doc_id=doc_id, label=chosen_label, confidence=prob))
    return aggregated


# ----------------------------------------------------------------------
# Simple smoke‑test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Instantiate a global ModelPool (used by score computation for timestamps)
    model_pool = ModelPool(ram_ceiling_mb=8000)

    # Define some models
    models = [
        ModelTier(name="alpha", ram_mb=1500, tier="T1"),
        ModelTier(name="beta", ram_mb=2000, tier="T2"),
        ModelTier(name="gamma", ram_mb=1200, tier="T3"),
    ]

    # Load them (respecting constraints)
    for m in models:
        try:
            model_pool.load(m)
        except RuntimeError as e:
            print(f"Could not load {m.name}: {e}", file=sys.stderr)

    # Certainty flags per model (using model name as key)
    certainties = {
        "alpha": certainty(
            label="FACT",
            confidence_bps=8500,
            authority_class="A",
            rationale="benchmark performance",
            evidence_refs=["ref1", "ref2"],
        ),
        "beta": certainty(
            label="PROBABLE",
            confidence_bps=6200,
            authority_class="B",
            rationale="recent paper",
            evidence_refs=["ref3"],
        ),
        "gamma": certainty(
            label="POSSIBLE",
            confidence_bps=3000,
            authority_class="C",
            rationale="prototype stage",
        ),
    }

    # Define candidate actions
    actions = [
        MathAction(id="act1", expected_value=1.2),
        MathAction(id="act2", expected_value=0.8),
        MathAction(id="act3", expected_value=1.5),
    ]

    # Hybrid selection
    best_model, best_action, best_score = select_best_model_action(
        models=list(model_pool.loaded.values()),
        actions=actions,
        certainties=certainties,
        lambda_regret=0.02,
    )
    print(
        f"Best pair -> Model: {best_model.name}, Action: {best_action.id}, Score: {best_score:.4f}"
    )

    # Simulate labeling function outputs
    @labeling_function(name="alpha")
    def lf_alpha(doc):
        return 1 if doc["value"] > 0 else 0

    @labeling_function(name="beta")
    def lf_beta(doc):
        return 0

    docs = [{"doc_id": "D1", "value": 0.3}, {"doc_id": "D2", "value": -0.1}]
    label_results = []
    for doc in docs:
        label_results.append(
            LabelingFunctionResult(lf_name=lf_alpha.lf_name, doc_id=doc["doc_id"], label=lf_alpha(doc))
        )
        label_results.append(
            LabelingFunctionResult(lf_name=lf_beta.lf_name, doc_id=doc["doc_id"], label=lf_beta(doc))
        )

    # Aggregate probabilistic labels
    prob_labels = aggregate_labels(label_results, certainties)
    for pl in prob_labels:
        print(f"Doc {pl.doc_id}: label={pl.label}, confidence={pl.confidence:.2%}")

    # Demonstrate eviction based on regret‑adjusted score
    def regret_score(model: ModelTier, load_ts: float) -> float:
        age = datetime.now(tz=timezone.utc).timestamp() - load_ts
        # Higher age → higher regret → larger score (we evict the lowest)
        return math.log1p(age) * 0.01

    # Force eviction of the least promising model
    model_pool._evict_one(regret_score)
    print("Remaining models after eviction:", list(model_pool.loaded.keys()))