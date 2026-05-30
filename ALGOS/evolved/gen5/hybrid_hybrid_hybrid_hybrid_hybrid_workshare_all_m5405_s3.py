# DARWIN HAMMER — match 5405, survivor 3
# gen: 5
# parent_a: hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s2.py (gen4)
# parent_b: hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (gen1)
# born: 2026-05-30T00:01:41Z

"""Hybrid Allocation Algorithm

Parents:
- hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_hybrid_m84_s2.py (Multivector with pheromone‑modulated grades)
- hybrid_workshare_allocator_doomsday_calendar_m14_s0.py (Workshare allocator combined with Doomsday calendar)

Mathematical Bridge:
Both parents manipulate a scalar quantity (total work units) and distribute it across a set of
basis elements.  In the Multivector formulation each basis blade represents a “group” and the
grade‑1 part encodes the LLM‑share.  The workshare allocator splits the same total into a
deterministic part and a LLM part.  The bridge therefore consists of:
1. Building a grade‑0 (deterministic) and grade‑1 (LLM) multivector from the allocator’s split.
2. Modulating the grade‑1 blades by a factor `store_state * (1 + pheromone_signal)` (from the
   pheromone‑time‑constant idea) and additionally by the day‑of‑week factor `day_of_week/7`
   (from the Doomsday calendar).
3. Extracting the resulting grade‑1 components back into a dictionary of per‑group allocations.

The resulting hybrid algorithm yields a day‑aware, pheromone‑aware workshare distribution
expressed through the geometric‑algebraic structure of a multivector."""
import math
import random
import sys
from pathlib import Path
from datetime import date
import numpy as np

# ----------------------------------------------------------------------
# Constants & Helpers
# ----------------------------------------------------------------------
GROUPS = ("codex", "groq", "cohere", "local_models")
GROUP_INDEX = {name: i for i, name in enumerate(GROUPS)}

def _pct(value: float) -> float:
    """Round a float to six decimal places."""
    return round(float(value), 6)

def _blade_sign(indices):
    """Return (sorted_blade, sign) after bubble‑sorting index list.
    Duplicate indices cancel each other (Grassmann algebra rule)."""
    lst = list(indices)
    sign = 1
    n = len(lst)
    i = 0
    while i < n - 1:
        j = 0
        while j < n - 1 - i:
            if lst[j] > lst[j + 1]:
                lst[j], lst[j + 1] = lst[j + 1], lst[j]
                sign *= -1
            elif lst[j] == lst[j + 1]:
                # cancel the pair
                lst.pop(j)
                lst.pop(j)  # second element now occupies position j
                n -= 2
                i = -1  # restart outer loop because length changed
                break
            j += 1
        i += 1
    return lst, sign

def _multiply_blades(blade_a, blade_b):
    """Multiply two basis blades (each a frozenset of indices)."""
    combined = list(blade_a) + list(blade_b)
    result, sign = _blade_sign(combined)
    return frozenset(result), sign

# ----------------------------------------------------------------------
# Multivector core (from Parent A)
# ----------------------------------------------------------------------
class Multivector:
    """Element of Cl(n,0) represented as a sum of basis blades."""
    def __init__(self, components: dict[frozenset[int], float], n: int):
        # store only non‑zero components
        self.components = {k: float(v) for k, v in components.items() if abs(v) > 1e-12}
        self.n = int(n)

    def grade(self, k: int, store_state: float, pheromone_signal: float):
        """Return a new Multivector keeping only grade‑k blades,
        each multiplied by the modulation factor
        `store_state * (1 + pheromone_signal)`."""
        factor = store_state * (1.0 + pheromone_signal)
        new_comps = {}
        for blade, val in self.components.items():
            if len(blade) == k:
                new_comps[blade] = _pct(val * factor)
        return Multivector(new_comps, self.n)

    def __repr__(self):
        return f"Multivector({self.components})"

# ----------------------------------------------------------------------
# Allocation utilities (from Parent B)
# ----------------------------------------------------------------------
def doomsday(year: int, month: int, day: int) -> int:
    """Return day of week as 0=Sunday … 6=Saturday (same convention as Python's weekday)."""
    return (date(year, month, day).weekday() + 1) % 7

def _split_total_units(total_units: float, deterministic_target_pct: float):
    """Perform the deterministic / LLM split used by the original allocator."""
    deterministic_units = total_units * deterministic_target_pct / 100.0
    llm_units = total_units - deterministic_units
    return deterministic_units, llm_units

# ----------------------------------------------------------------------
# Hybrid Functions
# ----------------------------------------------------------------------
def build_allocation_multivector(total_units: float,
                                 deterministic_target_pct: float = 90.0) -> Multivector:
    """Create a multivector whose grade‑0 blade holds the deterministic share
    and whose grade‑1 blades hold the per‑group LLM shares."""
    deterministic_units, llm_units = _split_total_units(total_units, deterministic_target_pct)
    per_group = llm_units / len(GROUPS)

    components = {
        frozenset(): _pct(deterministic_units)  # scalar (grade‑0)
    }
    for name, idx in GROUP_INDEX.items():
        components[frozenset({idx})] = _pct(per_group)

    return Multivector(components, n=len(GROUPS))

def allocate_hybrid(total_units: float,
                    store_state: float,
                    pheromone_signal: float,
                    year: int,
                    month: int,
                    day: int,
                    deterministic_target_pct: float = 90.0) -> dict:
    """
    Hybrid allocation:
    1. Build a multivector from the deterministic / LLM split.
    2. Extract the grade‑1 (LLM) part and modulate it by pheromone & store state.
    3. Further modulate by the day‑of‑week factor from the Doomsday calendar.
    4. Return a dictionary mirroring the original workshare allocator output.
    """
    if total_units <= 0:
        raise ValueError("total_units must be positive")
    if not 0 <= deterministic_target_pct <= 100:
        raise ValueError("deterministic_target_pct must be between 0 and 100")

    # 1. base multivector
    base_mv = build_allocation_multivector(total_units, deterministic_target_pct)

    # 2. pheromone‑store modulation on grade‑1 blades
    mod_mv = base_mv.grade(k=1, store_state=store_state, pheromone_signal=pheromone_signal)

    # 3. day‑of‑week factor
    dow = doomsday(year, month, day)          # 0 … 6
    day_factor = dow / 7.0                    # [0, 6/7]

    # Extract per‑group values
    per_group_alloc = []
    for name in GROUPS:
        idx = GROUP_INDEX[name]
        blade = frozenset({idx})
        raw_val = mod_mv.components.get(blade, 0.0)
        day_adj = _pct(raw_val * day_factor)
        per_group_alloc.append({
            "group": name,
            "llm_units": day_adj,
            "llm_share_pct": _pct(100.0 / len(GROUPS) * day_factor),
            "proof_required": True,
        })

    # Assemble final dict (matches allocate_workshare output shape)
    deterministic_units = base_mv.components.get(frozenset(), 0.0)
    llm_units_total = sum(item["llm_units"] for item in per_group_alloc)

    allocation = {
        "total_units": _pct(total_units),
        "deterministic_target_pct": _pct(deterministic_target_pct),
        "deterministic_units": _pct(deterministic_units),
        "llm_units": _pct(llm_units_total),
        "lanes": per_group_alloc,
        "day_of_week": dow,
        "day_factor": _pct(day_factor),
    }
    return allocation

def summarize_allocation(allocation: dict) -> str:
    """Return a human‑readable summary of the hybrid allocation."""
    lines = [
        f"Total Units: {allocation['total_units']}",
        f"Deterministic ({allocation['deterministic_target_pct']}%): {allocation['deterministic_units']}",
        f"LLM Units (after modulation): {allocation['llm_units']}",
        f"Day of Week (0=Sun): {allocation['day_of_week']}  Day Factor: {allocation['day_factor']}",
        "Per‑Group LLM Allocation:"
    ]
    for lane in allocation["lanes"]:
        lines.append(
            f"  - {lane['group']}: {lane['llm_units']} units "
            f"({lane['llm_share_pct']}% of total LLM, proof_required={lane['proof_required']})"
        )
    return "\n".join(lines)

def multivector_to_numpy(mv: Multivector) -> np.ndarray:
    """Convert the grade‑1 part of a multivector into a NumPy vector ordered by GROUPS."""
    vec = np.zeros(len(GROUPS))
    for name, idx in GROUP_INDEX.items():
        blade = frozenset({idx})
        vec[idx] = mv.components.get(blade, 0.0)
    return vec

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Example parameters
    TOTAL_UNITS = 1000.0
    STORE_STATE = 0.8               # e.g., 80 % of capacity used
    PHEROMONE_SIGNAL = 0.15         # modest positive signal
    YEAR, MONTH, DAY = 2026, 5, 30  # today

    alloc = allocate_hybrid(
        total_units=TOTAL_UNITS,
        store_state=STORE_STATE,
        pheromone_signal=PHEROMONE_SIGNAL,
        year=YEAR,
        month=MONTH,
        day=DAY,
        deterministic_target_pct=90.0,
    )
    print(summarize_allocation(alloc))

    # Demonstrate conversion to NumPy vector
    mv = build_allocation_multivector(TOTAL_UNITS)
    mod_mv = mv.grade(k=1, store_state=STORE_STATE, pheromone_signal=PHEROMONE_SIGNAL)
    np_vec = multivector_to_numpy(mod_mv)
    print("\nNumPy LLM vector after pheromone‑store modulation:", np_vec)