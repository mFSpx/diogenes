# DARWIN HAMMER — match 407, survivor 5
# gen: 5
# parent_a: hybrid_hybrid_fold_change_d_hybrid_hybrid_pherom_m306_s2.py (gen4)
# parent_b: state_space_duality.py (gen0)
# born: 2026-05-29T23:28:49Z

"""Hybrid Bandit-SSM Algorithm
Combines:
- Parent A: hybrid bandit router with fold‑change detection, pheromone infotaxis and decision hygiene.
- Parent B: State‑Space Model (SSM) duality (sequential scan ↔ semiseparable parallel matrix).

Mathematical Bridge
------------------
Each action *a* is treated as a dimension of a hidden state vector **h**∈ℝⁿ.
When a BanditUpdate arrives, the reward vector **x** has a non‑zero entry only at the
chosen action index.  A single SSM step

    h_{t+1} = A h_t + B x_t
    y_t   = C h_{t+1}

updates the internal belief (state) about all actions simultaneously.
The matrices are chosen as

    A = decay·I               (decay of past evidence)
    B = I                     (direct injection of reward)
    C = I                     (exposes the updated state)

Thus the SSM furnishes a *global*, temporally smoothed estimate of each action’s
expected reward, which is then fused with the classic bandit store‑factor,
fold‑change detection and pheromone‑infotaxis when selecting the next action.

The parallel form of the SSM yields a lower‑triangular semiseparable matrix **M**
that encodes the entire history of updates; this matrix can be used for
hardware‑friendly batch evaluation or analysis of the bandit dynamics.

The module provides three core hybrid operations:
1. `hybrid_initialize` – creates policy tables and the hidden state.
2. `hybrid_ssm_update` – incorporates a BanditUpdate via an SSM step.
3. `hybrid_select_action` – chooses an action using the fused score.
Additional utilities expose the semiseparable matrix and verify duality.
"""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import numpy as np

# ----------------------------------------------------------------------
# Parent A – Bandit / Pheromone components
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class BanditAction:
    action_id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str

@dataclass(frozen=True)
class BanditUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

_POLICY: dict = defaultdict(lambda: [0.0, 0.0])          # action_id → [cum_reward, count]
_STATE: dict = {}                                      # will hold hidden state vector (np.ndarray)
_ACTION_INDEX: dict = {}                               # action_id → index in state vector

def reset_policy() -> None:
    """Reset the bandit policy and hidden state."""
    _POLICY.clear()
    _STATE.clear()
    _ACTION_INDEX.clear()

def _reward(action: str) -> float:
    total, n = _POLICY[action]
    return total / n if n else 0.0

def _count(action: str) -> float:
    return _POLICY[action][1]

def _hybrid_store_factor(action_id: str, count: float, log_count_ratio: float) -> float:
    return log_count_ratio * count

def _fold_change_detection(x: float, eps: float = 1e-12) -> float:
    return math.log(x / max(abs(x), eps)) if x != 0 else 0.0

def _pheromone_infotaxis(pheromone: float, log_count_ratio: float) -> float:
    return pheromone * log_count_ratio if pheromone != 0 and log_count_ratio != 0 else 0.0

def _decision_hygiene_shannon_entropy(pheromone: float, log_count_ratio: float) -> float:
    infotax = _pheromone_infotaxis(pheromone, log_count_ratio)
    return -infotax * math.log(pheromone) if infotax != 0 and pheromone != 0 else 0.0

# ----------------------------------------------------------------------
# Parent B – State‑Space Model (SSM) duality
# ----------------------------------------------------------------------
def ssm_step(
    h: np.ndarray,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Single sequential SSM step."""
    h_new = A @ h + B @ x
    y = C @ h_new
    return h_new, y

def ssm_sequential(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
) -> np.ndarray:
    """Run SSM sequentially over a sequence."""
    T, _ = x_seq.shape
    state_dim = A.shape[0]
    if h0 is None:
        h = np.zeros(state_dim)
    else:
        h = h0.copy()
    outputs = []
    for t in range(T):
        h, y = ssm_step(h, x_seq[t], A, B, C)
        outputs.append(y)
    return np.stack(outputs, axis=0)

def semiseparable_matrix(
    A_seq: list[np.ndarray],
    B_seq: list[np.ndarray],
    C_seq: list[np.ndarray],
) -> np.ndarray:
    """
    Build the lower‑triangular semiseparable matrix M such that Y = M X,
    where X stacks the input vectors x_t and Y stacks the outputs y_t.
    """
    T = len(A_seq)
    # Dimensions (assume all A are (D,D), all B are (D,U), all C are (O,D))
    D = A_seq[0].shape[0]
    U = B_seq[0].shape[1]
    O = C_seq[0].shape[0]

    M = np.zeros((T * O, T * U))
    # Pre‑compute cumulative products of A for efficiency
    prod_A = [np.eye(D, dtype=A_seq[0].dtype)]
    for i in range(1, T):
        prod_A.append(A_seq[i] @ prod_A[-1])

    for i in range(T):          # output row block
        for j in range(i + 1):  # input column block (causal)
            # product A_{j+1} .. A_i  (empty product = I)
            if i == j:
                prod = np.eye(D, dtype=A_seq[0].dtype)
            else:
                prod = prod_A[i] @ np.linalg.inv(prod_A[j])
            block = C_seq[i] @ prod @ B_seq[j]
            M[i*O:(i+1)*O, j*U:(j+1)*U] = block
    return M

def ssm_parallel(
    x_seq: np.ndarray,
    A_seq: list[np.ndarray],
    B_seq: list[np.ndarray],
    C_seq: list[np.ndarray],
) -> np.ndarray:
    """Parallel evaluation via the semiseparable matrix."""
    M = semiseparable_matrix(A_seq, B_seq, C_seq)
    X = x_seq.reshape(-1, 1) if x_seq.ndim == 1 else x_seq.reshape(x_seq.shape[0], -1)
    Y = M @ X
    # Reshape to (T, output_dim)
    O = C_seq[0].shape[0]
    return Y.reshape(-1, O)

def verify_duality(
    x_seq: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    h0: np.ndarray | None = None,
    atol: float = 1e-6,
) -> bool:
    """Check that sequential and parallel forms produce the same outputs."""
    seq = ssm_sequential(x_seq, A, B, C, h0)
    A_seq = [A] * len(x_seq)
    B_seq = [B] * len(x_seq)
    C_seq = [C] * len(x_seq)
    par = ssm_parallel(x_seq, A_seq, B_seq, C_seq)
    return np.allclose(seq, par, atol=atol)

# ----------------------------------------------------------------------
# Hybrid Operations
# ----------------------------------------------------------------------
def hybrid_initialize(actions: list[BanditAction]) -> None:
    """
    Initialise policy tables and the hidden state vector.
    Each action obtains an index; the hidden state starts at zero.
    """
    reset_policy()
    n = len(actions)
    h = np.zeros(n, dtype=float)
    for idx, act in enumerate(actions):
        _POLICY[act.action_id] = [0.0, 0.0]
        _ACTION_INDEX[act.action_id] = idx
    _STATE["h"] = h

def _state_vector() -> np.ndarray:
    """Retrieve the current hidden state vector."""
    return _STATE["h"]

def _set_state_vector(new_h: np.ndarray) -> None:
    _STATE["h"] = new_h

def hybrid_ssm_update(
    update: BanditUpdate,
    log_count_ratio: float,
    decay: float = 0.95,
) -> None:
    """
    Incorporate a BanditUpdate using a single SSM step.
    The reward is injected only into the chosen action dimension.
    """
    # Update policy counters first (used by other components)
    total, cnt = _POLICY[update.action_id]
    _POLICY[update.action_id] = [total + update.reward, cnt + 1]

    # Build SSM matrices for this *global* step
    n = len(_ACTION_INDEX)
    A = decay * np.eye(n, dtype=float)          # decay of past belief
    B = np.eye(n, dtype=float)                  # direct reward injection
    C = np.eye(n, dtype=float)                  # expose updated belief

    # Input vector x has reward at the selected index, zero elsewhere
    x = np.zeros(n, dtype=float)
    idx = _ACTION_INDEX[update.action_id]
    x[idx] = update.reward * update.propensity   # weight by propensity

    h = _state_vector()
    h_new, _ = ssm_step(h, x, A, B, C)
    _set_state_vector(h_new)

def hybrid_select_action(
    actions: list[BanditAction],
    log_count_ratio: float,
) -> str:
    """
    Select an action by fusing:
      * hybrid store factor,
      * SSM‑derived belief (current hidden state),
      * fold‑change detection on empirical reward,
      * pheromone‑infotaxis and decision‑hygiene entropy.
    The action with maximal combined score is returned.
    """
    h = _state_vector()
    best_action = None
    best_score = -math.inf

    for act in actions:
        idx = _ACTION_INDEX[act.action_id]

        store = _hybrid_store_factor(act.action_id, _count(act.action_id), log_count_ratio)
        belief = h[idx]                                 # SSM belief
        fc = _fold_change_detection(_reward(act.action_id))
        inf = _pheromone_infotaxis(act.propensity, log_count_ratio)
        entropy = _decision_hygiene_shannon_entropy(act.propensity, log_count_ratio)

        score = store + belief + fc + inf - entropy   # entropy penalises uncertainty
        if score > best_score:
            best_score = score
            best_action = act.action_id
    return best_action if best_action is not None else ""

def hybrid_semiseparable_matrix(
    actions: list[BanditAction],
    log_count_ratio: float,
    decay: float = 0.95,
) -> np.ndarray:
    """
    Construct the semiseparable matrix that represents the whole
    history of hybrid updates up to the current time.
    For demonstration we treat each time‑step as using the same A, B, C
    (the decay‑based A and identity B, C).
    """
    n = len(actions)
    A = decay * np.eye(n, dtype=float)
    B = np.eye(n, dtype=float)
    C = np.eye(n, dtype=float)

    # In a real scenario we would keep a list of past x_t vectors.
    # Here we simply return the matrix for a single step (useful for unit‑tests).
    return semiseparable_matrix([A], [B], [C])

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Define a tiny action set
    actions = [
        BanditAction(action_id="a", propensity=0.8, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
        BanditAction(action_id="b", propensity=0.5, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
        BanditAction(action_id="c", propensity=0.3, expected_reward=0.0,
                     confidence_bound=0.0, algorithm="hybrid"),
    ]

    hybrid_initialize(actions)

    # Simulate a few updates
    updates = [
        BanditUpdate(context_id="c0", action_id="a", reward=1.0, propensity=0.8),
        BanditUpdate(context_id="c1", action_id="b", reward=0.2, propensity=0.5),
        BanditUpdate(context_id="c2", action_id="a", reward=0.7, propensity=0.8),
        BanditUpdate(context_id="c3", action_id="c", reward=0.5, propensity=0.3),
    ]

    log_count_ratio = 0.6  # arbitrary example value

    for upd in updates:
        hybrid_ssm_update(upd, log_count_ratio)

    # Select next action
    chosen = hybrid_select_action(actions, log_count_ratio)
    print(f"Chosen action: {chosen}")

    # Verify SSM duality on a synthetic sequence
    seq_len = 5
    x_seq = np.random.randn(seq_len, len(actions))
    A = 0.9 * np.eye(len(actions))
    B = np.eye(len(actions))