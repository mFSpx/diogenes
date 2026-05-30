# DARWIN HAMMER — match 2372, survivor 0
# gen: 5
# parent_a: hybrid_hybrid_decision_hygi_hybrid_rete_bandit_g_m544_s0.py (gen3)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m569_s1.py (gen4)
# born: 2026-05-29T23:42:09Z

"""Hybrid Decision‑Regret‑Certainty Engine
Parents:
- hybrid_hybrid_decision_hygi_hybrid_rete_bandit_g_m544_s0.py (Regret‑minimizing bandit with information‑gain decision hygiene)
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m569_s1.py (Sheaf‑certainty cohomology with liquid‑time‑constant workshare)

Mathematical Bridge:
The bandit’s regret‑minimization selects an allocation strategy (an “arm”) using a reward signal derived from
Shannon information‑gain computed on textual evidence.  Each arm carries a multivector‑like weight matrix **W**
whose entries are scaled by epistemic certainty flags (confidence in basis vectors).  The liquid‑time‑constant
**λ(t)**, modulated by the day‑of‑week, multiplies the geometric‑product update W←W+λ·(v⊗v), thereby coupling the
certainty‑cohomology dynamics to the bandit’s regret‑driven learning loop.  This creates a single unified system
where information‑theoretic gain drives regret‑based selection while epistemic certainty shapes the weight
evolution via Clifford‑algebra‑inspired updates.
"""

import re
import math
import random
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

# ----------------------------------------------------------------------
# Parent A – regexes and information‑gain helpers
# ----------------------------------------------------------------------
EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
    re.I,
)
PLANNING_RE = re.compile(
    r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
    re.I,
)
DELAY_RE = re.compile(
    r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
    re.I,
)
SUPPORT_RE = re.compile(
    r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
    re.I,
)
BOUNDARY_RE = re.compile(
    r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
    re.I,
)
OUTCOME_RE = re.compile(
    r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
    re.I,
)
IMPULSIVE_RE = re.compile(
    r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name)\b",
    re.I,
)

REGEX_GROUPS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
    "outcome": OUTCOME_RE,
    "impulsive": IMPULSIVE_RE,
    "scarcity": SCARCITY_RE,
}


def _category_counts(text: str) -> dict:
    """Return a dict mapping each category to the number of matches in *text*."""
    counts = {}
    for name, regex in REGEX_GROUPS.items():
        counts[name] = len(regex.findall(text))
    return counts


def shannon_entropy(probs: np.ndarray) -> float:
    """Standard Shannon entropy (base‑2). Zero‑probability entries are ignored."""
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def compute_information_gain(text_before: str, text_after: str) -> float:
    """
    Compute the reduction in Shannon entropy when moving from *text_before* to *text_after*.
    The probability distribution is built from the normalized category counts.
    """
    before = np.array(list(_category_counts(text_before).values()), dtype=float)
    after = np.array(list(_category_counts(text_after).values()), dtype=float)

    # avoid division by zero – add a tiny epsilon
    eps = 1e-12
    before_prob = before + eps
    after_prob = after + eps

    before_prob /= before_prob.sum()
    after_prob /= after_prob.sum()

    h_before = shannon_entropy(before_prob)
    h_after = shannon_entropy(after_prob)

    return h_before - h_after  # positive => information gain


# ----------------------------------------------------------------------
# Parent B – certainty flags and liquid‑time‑constant workshare
# ----------------------------------------------------------------------
EPISTEMIC_FLAGS = ("FACT", "PROBABLE", "POSSIBLE", "BULLSHIT", "SURE_MAYBE")


@dataclass(frozen=True)
class CertaintyFlag:
    label: str
    confidence_bps: int  # basis points, 0..10000
    authority_class: str
    rationale: str
    evidence_refs: tuple = ()
    generated_at: str = ""

    def __post_init__(self):
        if self.label not in EPISTEMIC_FLAGS:
            raise ValueError(f"unknown epistemic flag: {self.label!r}")
        if not 0 <= int(self.confidence_bps) <= 10000:
            raise ValueError("confidence_bps must be 0..10000")
        if not self.generated_at:
            object.__setattr__(self, "generated_at", datetime.now(timezone.utc).isoformat())


class HybridSheafCertainty:
    """Container for a sheaf (dictionary of nodes) together with certainty flags."""

    def __init__(self, sheaf_data: dict):
        self.sheaf_data = sheaf_data
        self.flags = []  # type: list[CertaintyFlag]

    def add_flag(self, flag: CertaintyFlag):
        self.flags.append(flag)


def liquid_time_constant(day_of_week: int, base: float = 1.0, info_gain: float = 0.0) -> float:
    """
    Day‑of‑week (0=Mon … 6=Sun) modulates a sinusoidal factor.
    The resulting λ is scaled by *info_gain* to couple information theory to the dynamics.
    """
    sinusoid = 1.0 + math.sin(2 * math.pi * day_of_week / 7.0)  # in [0,2]
    return base * sinusoid * (1.0 + info_gain)


def geometric_product_update(W: np.ndarray, v: np.ndarray, lam: float) -> np.ndarray:
    """
    Approximate a Clifford‑algebra geometric product update:
        W ← W + λ * (v ⊗ v)
    where ⊗ denotes the outer (tensor) product.
    """
    outer = np.outer(v, v)
    return W + lam * outer


# ----------------------------------------------------------------------
# Hybrid core: regret‑minimizing bandit driven by information gain
# ----------------------------------------------------------------------
class RegretBandit:
    """
    Simple Exp3‑style bandit where each arm i has a weight vector w_i.
    Rewards are scaled by epistemic confidence.
    Regret is accumulated to drive exploration vs exploitation.
    """

    def __init__(self, num_arms: int, dim: int, certainty_flags: list[CertaintyFlag]):
        self.num_arms = num_arms
        self.dim = dim
        self.certainty_flags = certainty_flags

        # Initialise weight matrices (multivector analogue) randomly
        self.W = np.random.rand(num_arms, dim, dim)
        # Cumulative estimated reward for each arm
        self.cum_reward = np.zeros(num_arms)
        # Exploration parameter
        self.gamma = 0.1

    def _confidence_vector(self) -> np.ndarray:
        """Map certainty flags to a confidence vector of length *dim* (normalized)."""
        # Simple mapping: distribute confidence evenly across dimensions
        total = sum(f.confidence_bps for f in self.certainty_flags) + 1e-9
        vec = np.array([f.confidence_bps for f in self.certainty_flags], dtype=float)
        if len(vec) < self.dim:
            # pad with small values
            vec = np.pad(vec, (0, self.dim - len(vec)), constant_values=1.0)
        elif len(vec) > self.dim:
            vec = vec[:self.dim]
        return vec / total

    def select_arm(self, day_idx: int, info_gain: float) -> int:
        """
        Compute a UCB‑style score for each arm.
        The liquid time constant λ(t) injects day‑of‑week and information‑gain dynamics.
        """
        day_of_week = (datetime.now(timezone.utc).weekday())  # 0=Mon
        lam = liquid_time_constant(day_of_week, base=0.5, info_gain=info_gain)

        # Expected reward estimate = trace(W_i) * confidence
        conf_vec = self._confidence_vector()
        est = np.array([
            np.trace(self.W[i]) * np.dot(conf_vec, conf_vec) for i in range(self.num_arms)
        ])

        # Upper‑confidence bound
        total_counts = max(1.0, np.sum(self.cum_reward))
        ucb = est + lam * np.sqrt(math.log(total_counts + 1) / (self.cum_reward + 1e-9))

        chosen = int(np.argmax(ucb))
        return chosen

    def update(self, arm: int, reward: float, info_gain: float, day_idx: int):
        """
        Update the selected arm’s weight matrix using the geometric product
        and adjust cumulative reward for regret tracking.
        """
        day_of_week = (datetime.now(timezone.utc).weekday())
        lam = liquid_time_constant(day_of_week, base=0.5, info_gain=info_gain)

        # Build a direction vector from reward and confidence
        conf_vec = self._confidence_vector()
        v = conf_vec * reward  # shape (dim,)

        # Update the arm’s matrix
        self.W[arm] = geometric_product_update(self.W[arm], v, lam)

        # Update regret (cumulative reward)
        self.cum_reward[arm] += reward


# ----------------------------------------------------------------------
# Public API – three demonstrative functions
# ----------------------------------------------------------------------
def simulate_day(text_before: str, text_after: str, bandit: RegretBandit, day_idx: int) -> dict:
    """
    Run one day of the hybrid system:
      1. Compute information gain from textual evolution.
      2. Choose an arm via regret‑minimizing bandit.
      3. Derive a synthetic reward (here: normalized info_gain).
      4. Update the bandit.
    Returns a dict summarising the step.
    """
    info_gain = compute_information_gain(text_before, text_after)
    arm = bandit.select_arm(day_idx, info_gain)

    # Synthetic reward: map info_gain ∈ ℝ to [0,1] via sigmoid
    reward = 1.0 / (1.0 + math.exp(-info_gain))

    bandit.update(arm, reward, info_gain, day_idx)

    return {
        "day_index": day_idx,
        "info_gain": info_gain,
        "selected_arm": arm,
        "reward": reward,
        "arm_trace": float(np.trace(bandit.W[arm])),
    }


def aggregate_certainty(sheaf: HybridSheafCertainty) -> np.ndarray:
    """
    Produce a single confidence vector from all certainty flags attached to *sheaf*.
    The vector length equals the number of distinct authority classes (capped at 8 for simplicity).
    """
    classes = list({f.authority_class for f in sheaf.flags})
    dim = min(8, len(classes))
    vec = np.zeros(dim)

    class_to_idx = {cls: i for i, cls in enumerate(classes[:dim])}
    for flag in sheaf.flags:
        idx = class_to_idx.get(flag.authority_class)
        if idx is not None:
            vec[idx] += flag.confidence_bps

    total = vec.sum() + 1e-9
    return vec / total


def run_hybrid_simulation(days: int = 7) -> None:
    """
    Smoke‑test driver:
    - creates dummy certainty flags,
    - initialises the bandit,
    - runs *days* iterations with synthetic text snippets,
    - prints a concise log.
    """
    # Dummy certainty flags
    flags = [
        CertaintyFlag(label="FACT", confidence_bps=8000, authority_class="Science", rationale="empirical"),
        CertaintyFlag(label="PROBABLE", confidence_bps=1500, authority_class="Expert", rationale="consensus"),
        CertaintyFlag(label="POSSIBLE", confidence_bps=500, authority_class="Anecdote", rationale="observation"),
    ]

    dim = 4  # dimensionality of the multivector space
    bandit = RegretBandit(num_arms=5, dim=dim, certainty_flags=flags)

    # Very simple synthetic texts that evolve over days
    base = "plan schedule"
    for d in range(days):
        before = base + " " + " ".join(random.choices(list(REGEX_GROUPS.keys()), k=2))
        after = base + " " + " ".join(random.choices(list(REGEX_GROUPS.keys()), k=4))
        result = simulate_day(before, after, bandit, d)
        print(
            f"Day {d+1:02d} | Gain={result['info_gain']:.4f} | Arm={result['selected_arm']} "
            f"| Reward={result['reward']:.4f} | Trace={result['arm_trace']:.4f}"
        )


if __name__ == "__main__":
    # Run a quick 7‑day demo; any exception will surface as a test failure.
    run_hybrid_simulation(days=7)