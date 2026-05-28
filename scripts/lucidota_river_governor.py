#!/usr/bin/env python3
"""River ML governor — system learns its own limits.

No daemon. No polling. Call learn() after each batch, predict() before each batch.
ADWIN detects hw drift. Hoeffding Tree learns optimal batch_size from telemetry.
Decisions written to DB as RULE facts — queryable by any LLM or flow.

Query current learned recommendation:
  SELECT fact_value FROM lucidota_control.runtime_status_fact
  WHERE subsystem='river_governor' AND fact_key='batch_recommendation';
"""
import json, os, pathlib, pickle, sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODEL_PATH = ROOT / "04_RUNTIME" / "river_governor_model.pkl"
_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT))


def _load_model():
    from river import tree, drift, compose, preprocessing
    if _MODEL_PATH.exists():
        try:
            return pickle.loads(_MODEL_PATH.read_bytes())
        except Exception:
            pass
    return {
        "tree": tree.HoeffdingTreeRegressor(),
        "adwin": drift.ADWIN(),
        "n_samples": 0,
        "last_drift_at": None,
    }


def _save_model(model: dict) -> None:
    _MODEL_PATH.write_bytes(pickle.dumps(model))


def _features(hw: dict) -> dict:
    t = hw.get("telemetry", hw)
    return {
        "ram_avail_mb": float(t.get("ram_avail_mb", 4000)),
        "vram_used_pct": float(t.get("vram_used_pct", 0)),
        "cpu_load_per_core": float(t.get("cpu_load_per_core", 0.3)),
        "disk_used_pct": float(t.get("disk_used_pct", 70)),
    }


def predict(hw: dict) -> int:
    """Return recommended batch_size for current hw_state."""
    model = _load_model()
    x = _features(hw)
    if model["n_samples"] < 5:
        return max(4, int(os.cpu_count() or 4) * 2)
    pred = model["tree"].predict_one(x)
    return max(1, min(64, int(round(pred or 4))))


def learn(hw: dict, batch_size: int, throughput_per_sec: float, oom: bool = False) -> dict:
    """Feed one sample into River. Called after each batch completes."""
    model = _load_model()
    x = _features(hw)
    target = 0.0 if oom else throughput_per_sec

    drift_detected = model["adwin"].update(throughput_per_sec)
    if drift_detected:
        from river import tree
        model["tree"] = tree.HoeffdingTreeRegressor()
        model["last_drift_at"] = datetime.now(timezone.utc).isoformat()

    model["tree"].learn_one(x, target)
    model["n_samples"] += 1
    _save_model(model)

    recommendation = predict(hw)
    result = {
        "n_samples": model["n_samples"],
        "drift_detected": bool(drift_detected),
        "last_drift_at": model["last_drift_at"],
        "recommended_batch_size": recommendation,
        "throughput_input": throughput_per_sec,
        "oom": oom,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_recommendation_to_db(result)
    return result


def _write_recommendation_to_db(result: dict) -> None:
    try:
        from core.runtime_dsns import resolve_state_dsn
        import psycopg2, psycopg2.extras
        dsn = resolve_state_dsn("postgresql://mfspx@/lucidota_state")
        conn = psycopg2.connect(dsn)
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.runtime_status_fact
                    (subsystem, fact_key, fact_value, evidence_refs, derived_at)
                VALUES ('river_governor', 'batch_recommendation', %s, '[]'::jsonb, now())
                ON CONFLICT (subsystem, fact_key)
                DO UPDATE SET fact_value = EXCLUDED.fact_value, derived_at = now()
            """, (psycopg2.extras.Json(result),))
        conn.close()
    except Exception:
        pass


if __name__ == "__main__":
    from lucidota_hw_gate import read_hw_state
    hw = read_hw_state()
    print(f"Current hw: {hw['mode']}, predicted batch: {predict(hw)}")
    print("Seeding 1 sample (16 files/s, no OOM)...")
    r = learn(hw, batch_size=16, throughput_per_sec=16.0)
    print(json.dumps(r, indent=2))
