# DARWIN HAMMER — match 3093, survivor 2
# gen: 7
# parent_a: hybrid_hybrid_hybrid_ternar_hybrid_hybrid_hybrid_m744_s0.py (gen5)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m1625_s1.py (gen6)
# born: 2026-05-29T23:47:47Z

import numpy as np

@dataclass(frozen=True)
class HybridAction:
    id: str
    propensity: float
    expected_reward: float
    confidence_bound: float
    algorithm: str
    expected_value: float
    cost: float = 0.0
    risk: float = 0.0

@dataclass(frozen=True)
class HybridUpdate:
    context_id: str
    action_id: str
    reward: float
    propensity: float

@dataclass
class StoreState:
    level: float = 0.0
    alpha: float = 1.0
    beta: float = 1.0
    dt: float = 1.0
    base: float = 1.0
    gain: float = 1.0
    limit: float = 10.0
    _last_delta: float = 0.0

    def update(self, inflow: list[float], outflow: list[float]) -> tuple[float, float]:
        delta = self.alpha * sum(inflow) - self.beta * sum(outflow)
        self.level = max(0.0, self.level + self.dt * delta)
        self._last_delta = delta
        return self.level, delta

    @property
    def dance(self) -> float:
        return max(0.0, min(self.limit, self.base + self.gain * self._last_delta))

_POLICY: dict[str, list[tuple[str, float, float]]] = {}

def reset_policy() -> None:
    global _POLICY
    _POLICY = {}

def calculate_variational_free_energy(action: HybridAction, store_state: StoreState) -> float:
    return action.propensity * store_state.level

def update_hybrid_policy(update: HybridUpdate, store_state: StoreState) -> None:
    global _POLICY
    context_id = update.context_id
    action_id = update.action_id
    reward = update.reward
    propensity = update.propensity

    if context_id not in _POLICY:
        _POLICY[context_id] = []

    _POLICY[context_id].append((action_id, reward, propensity))

    store_state.update([propensity], [reward])

def get_hybrid_action(context_id: str, store_state: StoreState) -> HybridAction:
    if context_id not in _POLICY:
        return HybridAction(id="default", propensity=0.0, expected_reward=0.0, confidence_bound=0.0, algorithm="default", expected_value=0.0)

    actions = _POLICY[context_id]
    action_ids = [action[0] for action in actions]
    propensities = [action[2] for action in actions]

    free_energies = [calculate_variational_free_energy(HybridAction(id=action_id, propensity=propensity, expected_reward=0.0, confidence_bound=0.0, algorithm="default", expected_value=0.0), store_state) for action_id, propensity in zip(action_ids, propensities)]

    selected_action_id = action_ids[np.argmin(free_energies)]
    selected_propensity = propensities[np.argmin(free_energies)]

    return HybridAction(id=selected_action_id, propensity=selected_propensity, expected_reward=0.0, confidence_bound=0.0, algorithm="default", expected_value=0.0)

def main() -> None:
    store_state = StoreState()
    update = HybridUpdate(context_id="context1", action_id="action1", reward=1.0, propensity=0.5)
    update_hybrid_policy(update, store_state)
    action = get_hybrid_action("context1", store_state)
    print(action)

if __name__ == "__main__":
    main()