#!/usr/bin/env python3
"""
eros_presence_yolo.py — Indy_READs Operator Presence & Telemetry
=================================================================
IronClaw daemon component. Watches the webcam, detects who is in the pilot
seat, tracks Northern's pupils for operator telemetry, logs intruders.

Nancy Drew mode: ultralight inference, receipts for everything.

Stack (all ultralight, GPU + CPU pipeline):
  1. YOLO11n        — person detection      (GPU, ~10ms, 5.4MB)
  2. Haar cascade   — face detection         (CPU, instant, built into OpenCV)
  3. Haar cascade   — eye region extraction  (CPU, instant, built into OpenCV)
  4. Threshold blob — pupil center + radius  (CPU, trivial)
  5. Grayscale embed— Northern vs Guest      (CPU, L2 distance on 64x64 face)

States (written to presence_state.json):
  "pilot-northern"  — Northern in the pilot seat, pupils tracked
  "pilot-guest"     — someone else in the seat, snapshot saved
  "empty"           — nobody in the seat

Operator telemetry (only when "pilot-northern"):
  pupil_dilation    — pupil_radius / eye_width  (0.0–1.0)
  gaze_direction    — "left" | "center" | "right"
  attention_score   — rolling window mean dilation

Intruder log (only when "pilot-guest"):
  Snapshot saved to intruder_log/ with timestamp + face crop.
  JSONL entry with timestamp, confidence, face_embedding hash.

Calibration:
  --calibrate captures 20 face embeddings, stores mean vector in
  BOOKS/.indy_reads/northern_face_embedding.npy

Usage:
  source scripts/lucidota_safe_ops_env.sh
  python3 scripts/eros_presence_yolo.py --calibrate       # one-time Northern calibration
  python3 scripts/eros_presence_yolo.py --daemon           # IronClaw background loop
  python3 scripts/eros_presence_yolo.py --once             # single capture + analysis + print
  python3 scripts/eros_presence_yolo.py --once --verbose   # with debug overlay values

Output files:
  04_RUNTIME/presence_state.json     — current state + telemetry (daemon reads this)
  04_RUNTIME/intruder_log/           — guest snapshots + JSONL log
  BOOKS/.indy_reads/northern_face_embedding.npy — calibration reference

Requires (all already in .venv):
  opencv-python, ultralytics, numpy, torch
"""

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RUNTIME_DIR = PROJECT_ROOT / "04_RUNTIME"
STATE_FILE = RUNTIME_DIR / "presence_state.json"
INTRUDER_DIR = RUNTIME_DIR / "intruder_log"
CALIBRATION_FILE = PROJECT_ROOT / "BOOKS" / ".indy_reads" / "northern_face_embedding.npy"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EMBED_SIZE = 64          # face resize dimensions (64x64 = 4096-d vector)
CALIBRATION_FRAMES = 20  # number of frames to capture for Northern embedding
DAEMON_INTERVAL = 2.0    # seconds between captures in daemon mode
NORTHERN_THRESHOLD = 12.0  # L2 distance threshold for Northern match (tune after calibration)
PILOT_SEAT_ZONE = (0.15, 0.1, 0.85, 0.85)  # (x1%, y1%, x2%, y2%) of frame — pilot seat region

# COCO class index
PERSON_CLASS = 0

# ---------------------------------------------------------------------------
# Haar cascades (built into OpenCV, no extra files needed)
# ---------------------------------------------------------------------------
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def ts() -> str:
    return datetime.datetime.now().isoformat()


def ts_filename() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# YOLO model (lazy load)
# ---------------------------------------------------------------------------
_yolo_model = None

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO("yolo11n.pt")
    return _yolo_model


# ---------------------------------------------------------------------------
# Face embedding — ultralight 64x64 grayscale L2
# ---------------------------------------------------------------------------

def compute_face_embedding(face_roi: np.ndarray) -> np.ndarray:
    """Convert face region to normalized 64x64 grayscale embedding vector."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (EMBED_SIZE, EMBED_SIZE))
    flat = resized.flatten().astype(np.float32)
    norm = np.linalg.norm(flat)
    if norm > 0:
        flat /= norm
    return flat


def load_calibration() -> np.ndarray | None:
    if CALIBRATION_FILE.exists():
        return np.load(CALIBRATION_FILE)
    return None


def save_calibration(embedding: np.ndarray):
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CALIBRATION_FILE, embedding)


def is_northern(embedding: np.ndarray, reference: np.ndarray, threshold: float = NORTHERN_THRESHOLD) -> tuple[bool, float]:
    """Returns (is_match, distance)."""
    dist = np.linalg.norm(embedding - reference)
    return dist < threshold, float(dist)


# ---------------------------------------------------------------------------
# Pupil analysis from eye region
# ---------------------------------------------------------------------------

def analyze_pupil(eye_roi_gray: np.ndarray) -> dict | None:
    """
    Extract pupil center and dilation from an eye region image.
    Returns {cx, cy, radius, eye_w, eye_h, dilation, gaze_x_ratio} or None.
    dilation = pupil_radius / eye_width
    gaze_x_ratio = pupil_cx / eye_width  (<0.4=left, 0.4–0.6=center, >0.6=right)
    """
    if eye_roi_gray.size == 0:
        return None

    h, w = eye_roi_gray.shape
    if w < 15 or h < 8:
        return None

    # Blur to reduce noise
    blurred = cv2.GaussianBlur(eye_roi_gray, (5, 5), 0)

    # Adaptive threshold to isolate dark pupil region
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 4
    )

    # Find contours — pupil should be the largest dark blob in the center region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Score contours: prefer round, centered, reasonable size
    best = None
    best_score = -1
    for c in contours:
        area = cv2.contourArea(c)
        if area < 4 or area > (w * h * 0.4):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        # Prefer contours near horizontal center of eye
        center_x = x + cw / 2
        center_y = y + ch / 2
        horiz_score = 1.0 - abs(center_x / w - 0.5) * 2  # 1.0 at center, 0 at edges
        # Prefer roughly circular
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        score = area * horiz_score * circularity
        if score > best_score:
            best_score = score
            best = (x, y, cw, ch, area)

    if best is None:
        return None

    x, y, cw, ch, area = best
    cx = x + cw / 2
    cy = y + ch / 2
    # Estimate pupil radius from area
    radius = float(np.sqrt(area / np.pi))
    dilation = radius / float(w)  # pupil radius relative to eye width
    gaze_x_ratio = cx / float(w)

    return {
        "cx": float(cx),
        "cy": float(cy),
        "radius": radius,
        "eye_w": w,
        "eye_h": h,
        "dilation": round(dilation, 4),
        "gaze_x_ratio": round(gaze_x_ratio, 4),
    }


def gaze_direction(gaze_x_ratio: float) -> str:
    if gaze_x_ratio < 0.38:
        return "left"
    elif gaze_x_ratio > 0.62:
        return "right"
    return "center"


# ---------------------------------------------------------------------------
# Core analysis: frame → state + telemetry
# ---------------------------------------------------------------------------

def analyze_frame(frame: np.ndarray, northern_ref: np.ndarray | None = None) -> dict:
    """
    Run full detection pipeline on a single frame.
    Returns a result dict with state, telemetry, and debug info.
    """
    h, w = frame.shape[:2]
    result = {
        "timestamp": ts(),
        "state": "empty",
        "frame_w": w,
        "frame_h": h,
        "person_detected": False,
        "person_conf": 0.0,
        "person_box": None,
        "face_detected": False,
        "face_box": None,
        "is_northern": False,
        "northern_distance": None,
        "pupil_left": None,
        "pupil_right": None,
        "gaze_direction": "unknown",
        "avg_dilation": None,
        "intruder_snapped": False,
    }

    # Step 1: YOLO person detection
    model = get_yolo()
    detections = model(frame, verbose=False, classes=[PERSON_CLASS], device="cuda:0")
    persons = []
    for det in detections:
        boxes = det.boxes
        if boxes is not None:
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == PERSON_CLASS and conf > 0.4:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    persons.append((x1, y1, x2, y2, conf))

    if not persons:
        return result

    # Use highest-confidence person
    best_person = max(persons, key=lambda p: p[4])
    px1, py1, px2, py2, pconf = best_person
    result["person_detected"] = True
    result["person_conf"] = round(pconf, 3)
    result["person_box"] = [round(v) for v in (px1, py1, px2, py2)]

    # Check if person is in pilot seat zone
    zone_x1 = int(w * PILOT_SEAT_ZONE[0])
    zone_y1 = int(h * PILOT_SEAT_ZONE[1])
    zone_x2 = int(w * PILOT_SEAT_ZONE[2])
    zone_y2 = int(h * PILOT_SEAT_ZONE[3])

    person_cx = (px1 + px2) / 2
    person_cy = (py1 + py2) / 2

    in_pilot_seat = (zone_x1 <= person_cx <= zone_x2 and zone_y1 <= person_cy <= zone_y2)
    if not in_pilot_seat:
        # Person detected but not in pilot seat → guest elsewhere in frame
        result["state"] = "pilot-guest"
        return result

    # Step 2: Face detection (Haar cascade on full frame — fast at 640x480)
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray_full, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))

    if len(faces) == 0:
        # Try alt cascade as fallback
        alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        faces = alt_cascade.detectMultiScale(gray_full, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40))

    if len(faces) == 0:
        result["state"] = "pilot-guest"
        return result

    # Match face to this person: face center must be within person box
    matched_faces = []
    for fx, fy, fw_face, fh_face in faces:
        face_cx = fx + fw_face / 2
        face_cy = fy + fh_face / 2
        if px1 <= face_cx <= px2 and py1 <= face_cy <= py2:
            matched_faces.append((fx, fy, fw_face, fh_face))

    if not matched_faces:
        result["state"] = "pilot-guest"
        return result

    # Use largest matched face
    fx, fy, fw_face, fh_face = max(matched_faces, key=lambda f: f[2] * f[3])
    abs_fx = int(fx)
    abs_fy = int(fy)
    fw_face = int(fw_face)
    fh_face = int(fh_face)
    result["face_detected"] = True
    result["face_box"] = [abs_fx, abs_fy, fw_face, fh_face]

    # Extract face region for embedding
    face_img = frame[abs_fy:abs_fy + fh_face, abs_fx:abs_fx + fw_face]
    face_embedding = compute_face_embedding(face_img)

    # Step 3: Northern vs Guest
    if northern_ref is not None:
        match, dist = is_northern(face_embedding, northern_ref)
        result["is_northern"] = match
        result["northern_distance"] = round(dist, 4)
    else:
        # No calibration → assume Northern (first-run behavior)
        result["is_northern"] = True
        result["northern_distance"] = 0.0

    if not result["is_northern"]:
        result["state"] = "pilot-guest"
        return result

    result["state"] = "pilot-northern"

    # Step 4: Eye detection for pupil analysis
    eye_region_y1 = max(0, abs_fy + int(fh_face * 0.2))
    eye_region_y2 = max(0, abs_fy + int(fh_face * 0.55))
    eye_region_x1 = max(0, abs_fx)
    eye_region_x2 = min(w, abs_fx + fw_face)

    eye_roi = frame[eye_region_y1:eye_region_y2, eye_region_x1:eye_region_x2]
    if eye_roi.size == 0:
        return result

    gray_eye_roi = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    eyes = EYE_CASCADE.detectMultiScale(gray_eye_roi, scaleFactor=1.1, minNeighbors=4, minSize=(18, 10))

    pupil_results = []
    for ex, ey, ew, eh in eyes:
        eye_crop_gray = gray_eye_roi[ey:ey + eh, ex:ex + ew]
        pupil = analyze_pupil(eye_crop_gray)
        if pupil:
            # Map to absolute coordinates
            pupil["abs_x"] = eye_region_x1 + ex + int(pupil["cx"])
            pupil["abs_y"] = eye_region_y1 + ey + int(pupil["cy"])
            pupil_results.append(pupil)

    if len(pupil_results) >= 2:
        # Sort by horizontal position → left eye first, right eye second
        pupil_results.sort(key=lambda p: p["abs_x"])
        result["pupil_left"] = pupil_results[0]
        result["pupil_right"] = pupil_results[1]
        avg_gaze = (pupil_results[0]["gaze_x_ratio"] + pupil_results[1]["gaze_x_ratio"]) / 2
        result["gaze_direction"] = gaze_direction(avg_gaze)
        result["avg_dilation"] = round(
            (pupil_results[0]["dilation"] + pupil_results[1]["dilation"]) / 2, 4
        )
    elif len(pupil_results) == 1:
        result["pupil_left"] = pupil_results[0]
        result["gaze_direction"] = gaze_direction(pupil_results[0]["gaze_x_ratio"])
        result["avg_dilation"] = pupil_results[0]["dilation"]

    return result


# ---------------------------------------------------------------------------
# Intruder snapshot
# ---------------------------------------------------------------------------

def save_intruder(frame: np.ndarray, face_box: list | None, result: dict):
    INTRUDER_DIR.mkdir(parents=True, exist_ok=True)
    stamp = ts_filename()

    # Full frame
    full_path = INTRUDER_DIR / f"intruder_{stamp}.jpg"
    cv2.imwrite(str(full_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Face crop
    face_path = None
    if face_box:
        fx, fy, fw, fh = face_box
        face_crop = frame[max(0, fy):fy + fh, max(0, fx):fx + fw]
        if face_crop.size > 0:
            face_path = INTRUDER_DIR / f"intruder_face_{stamp}.jpg"
            cv2.imwrite(str(face_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # JSONL log
    log_entry = {
        "timestamp": ts(),
        "full_frame": str(full_path),
        "face_crop": str(face_path) if face_path else None,
        "person_conf": result["person_conf"],
        "person_box": result["person_box"],
    }
    log_file = INTRUDER_DIR / "intruder_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    return full_path


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(device: int):
    """Capture CALIBRATION_FRAMES face embeddings and store the mean."""
    import os
    os.environ["YOLO_VERBOSE"] = "False"

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[CALIBRATE] Cannot open camera {device}")
        sys.exit(1)
    for _ in range(5):
        cap.read()

    embeddings = []
    print(f"[CALIBRATE] Capturing {CALIBRATION_FRAMES} face embeddings for Northern.")
    print("[CALIBRATE] Sit still, face the camera. SPACE to capture each frame, Q to skip.")

    while len(embeddings) < CALIBRATION_FRAMES:
        ret, frame = cap.read()
        if not ret:
            continue

        display = frame.copy()
        cv2.putText(display, f"Northern Calibration: {len(embeddings)}/{CALIBRATION_FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 200), 2)
        cv2.putText(display, "SPACE=Capture | Q=Quit",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Run face detection on frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cv2.rectangle(display, (fx, fy), (fx + fw, fy + fh), (0, 255, 200), 2)

        cv2.imshow("Northern Calibration", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" ") and len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_img = frame[fy:fy + fh, fx:fx + fw]
            emb = compute_face_embedding(face_img)
            embeddings.append(emb)
            print(f"  [{len(embeddings)}/{CALIBRATION_FRAMES}] Captured.")
        elif key in (ord("q"), ord("Q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 3:
        print(f"[CALIBRATE] Not enough captures ({len(embeddings)}). Need at least 3. Aborting.")
        sys.exit(1)

    mean_embedding = np.mean(embeddings, axis=0)
    save_calibration(mean_embedding)
    print(f"[CALIBRATE] Northern embedding saved ({len(embeddings)} frames, {EMBED_SIZE}x{EMBED_SIZE}={EMBED_SIZE*EMBED_SIZE}d)")
    print(f"[CALIBRATE] Stored at: {CALIBRATION_FILE}")

    # Compute in-group distances to suggest threshold
    distances = []
    for emb in embeddings:
        distances.append(np.linalg.norm(emb - mean_embedding))
    mean_dist = np.mean(distances)
    max_dist = np.max(distances)
    print(f"[CALIBRATE] In-group L2 distances: mean={mean_dist:.4f}, max={max_dist:.4f}")
    print(f"[CALIBRATE] Suggested NORTHERN_THRESHOLD: {max_dist * 2.0:.1f} (2x max in-group)")


# ---------------------------------------------------------------------------
# State persistence (operator telemetry output)
# ---------------------------------------------------------------------------

def write_state(result: dict):
    """Write presence state + telemetry to 04_RUNTIME/presence_state.json"""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    telemetry = None
    if result["state"] == "pilot-northern" and result["avg_dilation"] is not None:
        telemetry = {
            "pupil_dilation": result["avg_dilation"],
            "gaze_direction": result["gaze_direction"],
            "pupil_left_dilation": result["pupil_left"]["dilation"] if result["pupil_left"] else None,
            "pupil_right_dilation": result["pupil_right"]["dilation"] if result["pupil_right"] else None,
        }

    state = {
        "timestamp": result["timestamp"],
        "presence": result["state"],
        "person_confidence": result["person_conf"],
        "is_northern": result["is_northern"],
        "telemetry": telemetry,
    }

    # Write atomically
    tmp = STATE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def once(device: int, verbose: bool = False):
    """Single capture, analyze, print JSON to stdout."""
    import os
    os.environ["YOLO_VERBOSE"] = "False"

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(json.dumps({"error": f"Cannot open camera {device}"}))
        sys.exit(1)
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(json.dumps({"error": "Failed to capture frame"}))
        sys.exit(1)

    northern_ref = load_calibration()
    result = analyze_frame(frame, northern_ref)

    write_state(result)

    if result["state"] == "pilot-guest" and result["person_detected"]:
        saved = save_intruder(frame, result.get("face_box"), result)
        result["intruder_snapped"] = True
        result["intruder_path"] = str(saved)

    if verbose:
        # Add debug info
        result["_calibration_loaded"] = northern_ref is not None
        result["_calibration_file"] = str(CALIBRATION_FILE)

    print(json.dumps(result, indent=2, default=str))


def daemon_mode(device: int):
    """Run continuously as IronClaw background thread."""
    import os
    os.environ["YOLO_VERBOSE"] = "False"

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f"[PRESENCE-YOLO] Cannot open camera {device}", file=sys.stderr)
        sys.exit(1)
    for _ in range(5):
        cap.read()

    northern_ref = load_calibration()
    if northern_ref is None:
        print("[PRESENCE-YOLO] WARNING: No calibration file. Run --calibrate first. Assuming all faces are Northern.", file=sys.stderr)
    else:
        print(f"[PRESENCE-YOLO] Northern calibration loaded ({EMBED_SIZE}x{EMBED_SIZE} embedding)", file=sys.stderr)

    print(f"[PRESENCE-YOLO] Daemon mode: capturing every {DAEMON_INTERVAL}s. Ctrl+C to stop.", file=sys.stderr)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[PRESENCE-YOLO] Frame capture failed", file=sys.stderr)
                time.sleep(DAEMON_INTERVAL)
                continue

            result = analyze_frame(frame, northern_ref)
            write_state(result)

            if result["state"] == "pilot-guest" and result["person_detected"]:
                save_intruder(frame, result.get("face_box"), result)
                print(f"[PRESENCE-YOLO] INTRUDER: snapshot saved at {ts()}", file=sys.stderr)

            if result["state"] == "pilot-northern" and result["avg_dilation"]:
                print(
                    f"[PRESENCE-YOLO] {ts()} | Northern | "
                    f"gaze={result['gaze_direction']} | "
                    f"dilation={result['avg_dilation']:.4f}",
                    file=sys.stderr,
                )

            time.sleep(DAEMON_INTERVAL)

    except KeyboardInterrupt:
        print("[PRESENCE-YOLO] Daemon stopped.", file=sys.stderr)
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Indy_READs Operator Presence & Telemetry — YOLO + Haar + pupil tracking"
    )
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default 0)")
    parser.add_argument("--calibrate", action="store_true", help="Capture Northern's face embedding")
    parser.add_argument("--daemon", action="store_true", help="Run continuously for IronClaw integration")
    parser.add_argument("--once", action="store_true", help="Single capture + analysis + JSON to stdout")
    parser.add_argument("--verbose", action="store_true", help="Include debug info in --once output")
    args = parser.parse_args()

    if args.calibrate:
        calibrate(args.device)
    elif args.daemon:
        daemon_mode(args.device)
    elif args.once:
        once(args.device, verbose=args.verbose)
    else:
        # Default: single capture
        once(args.device, verbose=True)


if __name__ == "__main__":
    main()
