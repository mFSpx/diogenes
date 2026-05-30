# DARWIN HAMMER — match 608, survivor 1
# gen: 5
# parent_a: hybrid_cockpit_metrics_rectified_flow_m10_s0.py (gen1)
# parent_b: hybrid_hybrid_hybrid_hard_t_hybrid_hybrid_hybrid_m385_s0.py (gen4)
# born: 2026-05-29T23:30:04Z

import numpy as np
from collections import Counter

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
    "quantifier": set(
        "all any both each few many more most much none several some such".split()
    ),
    "adverb_common": set(
        "very really just still already also even only then there here now often always sometimes".split()
    ),
}

def words(text: str) -> list[str]:
    return [word for word in (text or "").lower().split() if word.isalpha()]

def anti_slop_ratio(claims_with_evidence: int, total_claims_emitted: int) -> float:
    return 1.0 if total_claims_emitted <= 0 else max(0.0, min(1.0, claims_with_evidence / total_claims_emitted))

def cockpit_honesty(displayed_ok: int, unknown_displayed_as_ok: int) -> float:
    total = displayed_ok + unknown_displayed_as_ok
    return 1.0 if total <= 0 else max(0.0, min(1.0, displayed_ok / total))

def stylometry_features(text: str, dim: int) -> np.ndarray:
    ws = words(text)
    total = max(1, len(ws))
    cnt = Counter(ws)
    vocab = [cat for cat in FUNCTION_CATS.keys()]
    return np.array([
        sum(cnt[w] for w in FUNCTION_CATS[vocab[i]]) / total
        for i in range(dim)
    ])

def lsm_vector(text: str) -> np.ndarray:
    return stylometry_features(text, 6)

def flow_target(x0, x1):
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    return x1 - x0

def flow_loss(v_pred, x0, x1):
    v_pred = np.asarray(v_pred, dtype=np.float64)
    target = flow_target(x0, x1)
    diff = v_pred - target
    return float(np.mean(diff ** 2))

def euler_solve(v_fn, x0, steps=10):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, lsm_vector("This is a sample text"), float(t), anti_slop_ratio(10, 100), cockpit_honesty(10, 0))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj

def hybrid_solve(v_fn, x0, claims_with_evidence, total_claims_emitted, displayed_ok, unknown_displayed_as_ok, text: str, steps=10):
    x0 = np.asarray(x0, dtype=np.float64)
    dt = 1.0 / steps
    ts = np.linspace(0.0, 1.0 - dt, steps)
    traj = np.empty((steps + 1,) + x0.shape, dtype=np.float64)
    traj[0] = x0
    z = x0.copy()
    for k, t in enumerate(ts):
        v = v_fn(z, lsm_vector(text), float(t), anti_slop_ratio(claims_with_evidence, total_claims_emitted), cockpit_honesty(displayed_ok, unknown_displayed_as_ok))
        z = z + dt * np.asarray(v, dtype=np.float64)
        traj[k + 1] = z
    return traj

def improved_v_fn(z, lsm_vector, t, anti_slop, cockpit_honesty):
    return np.array([1.0 * anti_slop * cockpit_honesty, 2.0 * anti_slop * cockpit_honesty])

if __name__ == "__main__":
    text = "This is a sample text"
    x0 = np.array([0.0, 0.0])
    v = lambda z, _lsm_vector, t, anti_slop, cockpit_honesty: improved_v_fn(z, _lsm_vector, t, anti_slop, cockpit_honesty)
    traj = hybrid_solve(v, x0, 10, 100, 10, 0, text, steps=10)
    print(traj)