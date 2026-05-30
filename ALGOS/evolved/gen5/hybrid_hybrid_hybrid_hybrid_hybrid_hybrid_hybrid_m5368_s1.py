# DARWIN HAMMER — match 5368, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m2597_s0.py (gen4)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m39_s2.py (gen4)
# born: 2026-05-30T00:01:35Z

import math
import random
import re
import numpy as np

# Regex feature definitions
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
    r"\b(?:boundary|boundaries|walk away|no contact|do not|block|ignore|distance|safe|safe distance|no talk|no communication|stop|stop talking|stop interaction|no interaction|leave)\b",
    re.I,
)

REGEX_PATTERNS = {
    "evidence": EVIDENCE_RE,
    "planning": PLANNING_RE,
    "delay": DELAY_RE,
    "support": SUPPORT_RE,
    "boundary": BOUNDARY_RE,
}

# Bandit core
@dataclass
class ThompsonBandit:
    """Simple Thompson‑sampling bandit with Beta priors."""
    successes: float = 1.0
    failures: float = 1.0
    # dynamic confidence bound (upper‑confidence index)
    confidence_scale: float = 1.0

    def sample(self) -> float:
        """Draw a propensity sample from the posterior Beta distribution."""
        return random.betavariate(self.successes, self.failures)

    @property
    def propensity(self) -> float:
        """Mean of the Beta posterior – used as deterministic propensity."""
        return self.successes / (self.successes + self.failures)

    @property
    def confidence_bound(self) -> float:
        """A simple confidence interval width."""
        var = (self.successes * self.failures) / (
            (self.successes + self.failures) ** 2 * (self.successes + self.failures + 1)
        )
        return self.confidence_scale * math.sqrt(var)

    def update(self, reward: float) -> None:
        """Update posterior with binary reward (1 = success, 0 = failure)."""
        if reward >= 0.5:
            self.successes += 1
        else:
            self.failures += 1


# Fractional‑memory utilities
def _caputo_kernel(alpha: float, delta: int, propensity: float, conf: float) -> float:
    """
    Fractional‑memory kernel for a lag `delta` (t‑k) with bandit modulation.
    K = propensity * (delta)^{-alpha} * exp(-conf * delta)
    """
    if delta <= 0:
        return 0.0
    return propensity * (delta ** (-alpha)) * math.exp(-conf * delta)


def fractional_memory(
    past_alloc: np.ndarray,
    alpha: float,
    propensity: float,
    confidence: float,
) -> float:
    """
    Compute the Caputo‑type memory contribution for the current step.
    `past_alloc` is a 1‑D array of previous allocations for a single group.
    """
    if past_alloc.size == 0:
        return 0.0
    deltas = np.arange(past_alloc.size, 0, -1)  # t‑k = 1,2,…,n
    kernels = np.vectorize(_caputo_kernel)(alpha, deltas, propensity, confidence)
    return float(np.dot(kernels, past_alloc))


# Hybrid system data container
@dataclass
class HybridState:
    """Holds all mutable state required for the hybrid algorithm."""
    groups: Tuple[str, ...] = ("codex", "groq", "cohere", "local_models")
    # deterministic/LLM split λ_g for each group (sum to 1 across groups)
    lambda_split: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.3, 0.2, 0.1]))
    # pheromone priority matrix M_{g,t} (initialized uniformly)
    pheromone: np.ndarray = field(init=False)  # shape (n_groups, )
    # allocation history per group (list of lists)
    history: List[List[float]] = field(default_factory=lambda: [[] for _ in range(4)])
    # bandit instance
    bandit: ThompsonBandit = field(default_factory=ThompsonBandit)
    # fractional order α (0<α<1)
    alpha: float = 0.7
    # scaling factor β for effective time constant τ_t = sigmoid(β·s_t)
    beta: float = 0.05
    # kernel regularization parameter
    kernel_reg: float = 0.01

    def __post_init__(self):
        self.pheromone = np.ones(len(self.groups)) / len(self.groups)


# Core hybrid functions
def init_hybrid_system() -> HybridState:
    """Create a fresh HybridState with sensible defaults."""
    return HybridState()


def score_features(text: str) -> Dict[str, int]:
    """
    Apply the regex catalogue to `text` and return a dict of hit counts.
    Mirrors Parent B's feature extraction.
    """
    scores = {}
    for name, pattern in REGEX_PATTERNS.items():
        scores[name] = len(pattern.findall(text))
    return scores


def effective_time_constant(total_score: float, beta: float) -> float:
    """
    Map the aggregated feature score `total_score` to an effective time constant τ∈(0,1)
    using a logistic sigmoid.
    """
    return 1.0 / (1.0 + math.exp(-beta * total_score))


def hybrid_allocate(state: HybridState, text: str, baseline_llm_share: float = 0.6) -> Dict[str, float]:
    """
    Compute per‑group allocation for 
    """
    scores = score_features(text)
    total_score = sum(scores.values())
    tau_t = effective_time_constant(total_score, state.beta)
    propensity = state.bandit.propensity
    confidence = state.bandit.confidence_bound
    allocations = {}
    for i, group in enumerate(state.groups):
        past_alloc = np.array(state.history[i])
        memory_contribution = fractional_memory(past_alloc, state.alpha, propensity, confidence)
        memory_contribution = max(0.0, memory_contribution)  # ensure non-negative
        lambda_g = state.lambda_split[i]
        L_t = baseline_llm_share
        M_g_t = state.pheromone[i] * (1 + memory_contribution * state.kernel_reg)
        a_g_t = lambda_g * (1 - tau_t) * L_t + (1 - lambda_g) * tau_t * M_g_t
        allocations[group] = a_g_t
        state.history[i].append(a_g_t)
    return allocations


def update_bandit_memory(state: HybridState, reward: float) -> None:
    """
    Update the bandit with a reward signal.
    """
    state.bandit.update(reward)


# Smoke test
if __name__ == "__main__":
    state = init_hybrid_system()
    text = "This is a test sentence with some evidence and planning."
    allocations = hybrid_allocate(state, text)
    print(allocations)
    update_bandit_memory(state, 1.0)  # reward signal