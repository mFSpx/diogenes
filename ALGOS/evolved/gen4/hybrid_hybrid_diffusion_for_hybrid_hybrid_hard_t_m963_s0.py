# DARWIN HAMMER — match 963, survivor 0
# gen: 4
# parent_a: hybrid_diffusion_forcing_hybrid_bandit_router_m175_s0.py (gen2)
# parent_b: hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py (gen3)
# born: 2026-05-29T23:31:50Z

"""
This module integrates the Diffusion Forcing algorithm from hybrid_diffusion_forcing_hybrid_bandit_router_m175_s0.py 
and the Hybrid Hard Truth Math Model from hybrid_hybrid_hard_truth_ma_hybrid_hybrid_endpoi_m127_s3.py.
The mathematical bridge between the two structures is found in the concept of 
noise schedules and stylometry features. The Diffusion Forcing algorithm uses a 
noise schedule to corrupt input tokens, while the Hybrid Hard Truth Math Model uses 
stylometry features to compute the proportion of words belonging to each function category.
By combining these concepts, we can create a hybrid algorithm that uses a noise schedule 
to corrupt input tokens and stylometry features to select words based on their function categories.
"""

import numpy as np
import math
import random
import sys
import pathlib

# Define the function categories
FUNCTION_CATS = {
    "pronoun": set(
        "i me my mine myself you your yours yourself he him his she her hers they them their theirs we us our ours".split()
    ),
    "article": set("a an the".split()),
    "preposition": set(
        "about above after against around as at before behind below between by during for from in into of off on onto over through to under with without".split()
    ),
    "auxiliary": set(
        "am are be been being can could did do does had has have is may might must shall should was were will would".split()
    ),
    "conjunction": set(
        "and but or nor so yet because although while if when where whereas unless until".split()
    ),
    "negation": set(
        "no not never none neither cannot can't won't don't didn't isn't aren't was wasn't weren't".split()
    ),
    "quantifier": set("all any both each few many more most much none several some such".split()),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

PUNCT = "!?;:,.—-()[]{}\"'`/\\|@#$%^&*+=~"

def noise_schedule(T: int, schedule: str = "cosine") -> np.ndarray:
    """Return the cumulative noise schedule alpha_bar, shape (T+1,).

    alpha_bar[0] = 1.0  (clean)
    alpha_bar[T] ~ 0.0  (pure noise)

    Parameters
    ----------
    T:
        Total number of diffusion timesteps.
    schedule:
        "cosine" (Nichol & Dhariwal 2021) or "linear" (Ho et al. 2020).

    Returns
    -------
    np.ndarray shape (T+1,) with values in (0, 1].
    """
    if schedule == "cosine":
        s = 0.008
        steps = np.arange(T + 1, dtype=np.float64)
        f = np.cos(((steps / T) + s) / (1.0 + s) * np.pi / 2.0) ** 2
        alpha_bars = f / f[0]
        alpha_bars = np.clip(alpha_bars, a_min=1e-5, a_max=1.0)
        return alpha_bars
    else:
        raise ValueError("Invalid schedule")

def words(text: str) -> list[str]:
    """Lower‑case alphabetic tokens (apostrophe‑aware)."""
    return [w for w in text.lower().split() if w.isalpha()]

def stylometry_features(text: str) -> np.ndarray:
    """
    Produce a deterministic 96‑dimensional numeric representation of *text*.
    The implementation uses a SHA‑256 hash to seed a pseudo‑random generator,
    guaranteeing reproducibility without external corpora.
    """
    import hashlib
    h = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.random(96)

def lsm_vector(text: str) -> np.ndarray:
    """
    Compute the proportion of words belonging to each FUNCTION_CAT.
    Returns a 7-dimensional vector.
    """
    word_count = len(words(text))
    if word_count == 0:
        return np.zeros(7)
    proportions = np.zeros(7)
    function_cat_list = list(FUNCTION_CATS.keys())
    for i, function_cat in enumerate(function_cat_list):
        count = sum(1 for word in words(text) if word in FUNCTION_CATS[function_cat])
        proportions[i] = count / word_count
    return proportions

def hybrid_diffusion_stylometry(
    context: dict[str, float],
    actions: list[str],
    store: float,
    T: int,
    schedule: str = "cosine",
    epsilon: float = 0.1,
    eta: float = 0.1,
    seed: int | str | None = 7,
) -> str:
    alpha_bars = noise_schedule(T, schedule)
    rng = random.Random(seed)
    store_factor = 1.0 + store / (store + 1.0)

    # Corrupt input tokens using noise schedule
    corrupted_context = {
        key: value * alpha_bars[rng.randint(0, T)] for key, value in context.items()
    }

    # Calculate stylometry features for actions
    stylometry_features_list = [stylometry_features(action) for action in actions]

    # Select action based on stylometry features and corrupted context
    scores = [
        np.dot(list(corrupted_context.values()), stylometry_features_list[i])
        for i in range(len(actions))
    ]
    selected_action = actions[np.argmax(scores)]

    return selected_action

def update_store(
    store: float,
    inflow: list[float],
    outflow: list[float],
    alpha: float = 1.0,
    beta: float = 1.0,
    dt: float = 1.0,
) -> tuple[float, float]:
    delta = alpha * sum(inflow) - beta * sum(outflow)
    new_store = max(0.0, store + dt * delta)
    return new_store, delta

if __name__ == "__main__":
    context = {"token1": 0.5, "token2": 0.3}
    actions = ["action1", "action2", "action3"]
    store = 10.0
    T = 100
    schedule = "cosine"
    epsilon = 0.1
    eta = 0.1
    seed = 7

    selected_action = hybrid_diffusion_stylometry(
        context, actions, store, T, schedule, epsilon, eta, seed
    )
    print(f"Selected action: {selected_action}")

    new_store, delta = update_store([1.0, 2.0], [0.5, 1.0], 1.0, 1.0, 1.0)
    print(f"New store: {new_store}, delta: {delta}")

    print("Noise schedule:")
    print(noise_schedule(T, schedule))

    print("Stylometry features:")
    print(stylometry_features("This is a test sentence."))

    print("LSM vector:")
    print(lsm_vector("This is a test sentence."))