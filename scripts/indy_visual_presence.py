#!/usr/bin/env python3
"""Visual presence detection — PHASE_08 WO-08.1.

OpenCV state-only: operator_in_seat / guest_in_seat / empty_chair.
NO recording. Single frame, state detection, immediate release.
Writes to lucidota_control.runtime_status_fact (subsystem='visual_channel').
Stages presence EVENT (@08) candidate to 05_OUTPUTS/.

Policy: cadence ≥60s. Local only. No frames stored.
"""
import cv2, json, os, pathlib, sys, time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
_CADENCE_S = int(os.getenv("LUCIDOTA_PRESENCE_CADENCE_S", "60"))
_CURSOR = ROOT / "04_RUNTIME" / "visual_presence_last.json"
_CURSOR.parent.mkdir(parents=True, exist_ok=True)


def detect_once() -> dict:
    """Capture one frame, detect presence, release immediately. No storage."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"state": "camera_unavailable", "confidence": 0.0}
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return {"state": "camera_unavailable", "confidence": 0.0}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    del frame, gray  # no retention

    n = len(faces)
    if n == 0:
        return {"state": "empty_chair", "confidence": 0.85, "faces_detected": 0}
    elif n == 1:
        return {"state": "operator_in_seat", "confidence": 0.80, "faces_detected": 1}
    else:
        return {"state": "guest_in_seat", "confidence": 0.70, "faces_detected": int(n)}


def sample(write_db: bool = True) -> dict:
    detection = detect_once()
    result = {
        "go25_term": "EVENT",
        "go25_id": "@08",
        "subsystem": "visual_channel",
        "policy": "no_recording_state_only",
        **detection,
        "sampled_at": datetime.now(timezone.utc).isoformat(),
    }
    _CURSOR.write_text(json.dumps(result))

    if write_db:
        _write_to_db(result)
        _stage_candidate(result)

    return result


def _write_to_db(result: dict) -> None:
    try:
        from core.runtime_dsns import resolve_state_dsn
        import psycopg2, psycopg2.extras
        dsn = resolve_state_dsn("postgresql://mfspx@/lucidota_state")
        conn = psycopg2.connect(dsn)
        with conn, conn.cursor() as cur:
            cur.execute("""
                INSERT INTO lucidota_control.runtime_status_fact
                    (subsystem, fact_key, fact_value, evidence_refs, derived_at)
                VALUES ('visual_channel', 'presence_state', %s, '[]'::jsonb, now())
                ON CONFLICT (subsystem, fact_key)
                DO UPDATE SET fact_value = EXCLUDED.fact_value, derived_at = now()
            """, (psycopg2.extras.Json(result),))
        conn.close()
    except Exception:
        pass


def _stage_candidate(result: dict) -> None:
    out = ROOT / "05_OUTPUTS" / "visual_presence"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (out / f"presence_{ts}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    continuous = "--watch" in sys.argv
    if continuous:
        print(f"Presence watch every {_CADENCE_S}s. Ctrl-C to stop.")
        while True:
            r = sample(write_db=True)
            print(f"[{r['sampled_at']}] {r['state']} (faces={r.get('faces_detected',0)})")
            time.sleep(_CADENCE_S)
    else:
        r = sample(write_db=True)
        print(json.dumps(r, indent=2))
