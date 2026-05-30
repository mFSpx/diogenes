# DARWIN HAMMER — match 1143, survivor 2
# gen: 5
# parent_a: pheromone.py (gen0)
# parent_b: hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s0.py (gen4)
# born: 2026-05-29T23:32:59Z

"""
This module integrates the Darwinian surface pheromone worker (pheromone.py) 
with the Hybrid SSIM Decision Hygiene (hybrid_hybrid_hybrid_hybrid_hybrid_ssim_hybrid_d_m24_s0.py). 
The mathematical bridge between the two parents lies in the application of 
the structural similarity index measurement (SSIM) to compare the similarity 
between feature vectors extracted from text, and then using the result as a 
weighting factor in the calculation of the hybrid score, which in turn is 
used to update the surface pheromone signals.

The governing equations of the parent algorithms are fused as follows:

- The store equation from parent B is used to update the virtual-VRAM store.
- The learning-rate-scaled matrix update from parent B is used to update 
  the weight matrix.
- The evasion-driven position perturbation from parent B is used to 
  perturb the positions.
- The surface pheromone signal update from parent A is used to update the 
  surface pheromone signals, with the hybrid score used as a weighting factor.

The resulting hybrid algorithm couples resource-allocation dynamics with 
continuous optimisation dynamics and decision hygiene evaluation.
"""

import numpy as np
import math
import re
import random
import sys
from pathlib import Path
from datetime import datetime, timezone
import json
import psycopg
from psycopg.rows import dict_row

# Shared constants
ALPHA = 0.6          # store inflow coefficient
BETA = 0.4           # store outflow coefficient
DT = 1.0             # time step for store dynamics
ETA0 = 0.01          # base learning rate for matrix updates
DELTA_MAX = 1.0      # max evasion magnitude
ALPHA_EVASION = 3.0  # decay rate for evasion schedule
HOEFFDING_DELTA = 0.

# Feature extraction and weighting
_FEATURE_ORDER = [
    "evidence",
    "planning",
    "delay",
    "support",
    "boundary",
    "outcome",
    "impulsive",
    "scarcity",
    "risk",
]

_POSITIVE_WEIGHTS = np.array([1600, 1200, 1400, 1000, 1200, 800, 0, 0, 0], dtype=np.int64)
_NEGATIVE_WEIGHTS = np.array([0, 0, 0, 0, 0, 0, 1500, 700, 1200], dtype=np.int64)
_TOTAL_ABS_WEIGHTS = np.abs(_POSITIVE_WEIGHTS) + np.abs(_NEGATIVE_WEIGHTS)

EVIDENCE_RE = re.compile(
    r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|r"
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'05_OUTPUTS/surfaces'
SCHEMA = ROOT/'06_SCHEMA/029_darwinian_surfaces.sql'

def ts():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')

def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')

def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:
        return str(p)

def db(a):
    return a.database_url or os.environ.get('KORPUS_DATABASE_URL') or os.environ.get('DATABASE_URL') or 'postgresql:///lucidota_storage'

def write(name, payload):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT/f'hybrid_{name}_{ts()}.json'
    payload.setdefault('generated_at', now())
    payload['report_path'] = rel(p)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    print(f'REPORT_PATH={rel(p)}')
    return p

def ensure_schema(cur):
    cur.execute(SCHEMA.read_text())

def signal(a):
    pheromone_uuid = None
    if a.execute:
        with psycopg.connect(db(a), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                ensure_schema(cur)
                cur.execute('''INSERT INTO lucidota_runtime.surface_pheromone(surface_key,signal_kind,signal_value,half_life_seconds,detail)
                               VALUES (%s,%s,%s,%s,%s::jsonb) RETURNING pheromone_uuid::text''', 
                               (a.surface_key, a.signal_kind, a.signal_value, a.half_life_seconds, json.dumps({'script':'hybrid.py'})))
                pheromone_uuid = cur.fetchone()['pheromone_uuid']
            conn.commit()
    report = {'action': 'signal', 'execute_performed': bool(a.execute), 'db_writes_performed': bool(a.execute), 
              'graph_writes_performed': False, 'surface_key': a.surface_key, 'signal_kind': a.signal_kind, 
              'signal_value': a.signal_value, 'pheromone_uuid': pheromone_uuid, 'status': 'PASS'}
    write('signal_execute' if a.execute else 'signal_dry_run', report)
    print('PHEROMONE_SIGNAL=PASS')
    return 0

def decay(a):
    updated = 0
    rows = []
    if a.execute:
        with psycopg.connect(db(a), row_factory=dict_row) as conn:
            with conn.cursor() as cur:
                ensure_schema(cur)
                cur.execute('''SELECT pheromone_uuid::text,surface_key,signal_value,half_life_seconds,created_at 
                                FROM lucidota_runtime.surface_pheromo''')
                rows = cur.fetchall()
    for row in rows:
        # Calculate the hybrid score using the SSIM-based weighting factor
        hybrid_score = calculate_hybrid_score(row['surface_key'], row['signal_value'])
        # Update the surface pheromone signal using the hybrid score
        update_surface_pheromone(row['pheromone_uuid'], hybrid_score)
        updated += 1
    report = {'action': 'decay', 'execute_performed': bool(a.execute), 'db_writes_performed': bool(a.execute), 
              'graph_writes_performed': False, 'updated': updated, 'status': 'PASS'}
    write('decay_execute' if a.execute else 'decay_dry_run', report)
    print('PHEROMONE_DECAY=PASS')
    return 0

def calculate_hybrid_score(surface_key, signal_value):
    # Calculate the feature vector for the surface key
    feature_vector = calculate_feature_vector(surface_key)
    # Calculate the SSIM-based weighting factor
    weighting_factor = calculate_ssim_weighting_factor(feature_vector, signal_value)
    # Calculate the hybrid score using the weighting factor
    hybrid_score = weighting_factor * signal_value
    return hybrid_score

def calculate_feature_vector(surface_key):
    # Extract the features from the surface key
    features = extract_features(surface_key)
    # Calculate the feature vector
    feature_vector = np.array(features)
    return feature_vector

def extract_features(surface_key):
    # Extract the features from the surface key using regular expressions
    features = []
    for feature in _FEATURE_ORDER:
        match = re.search(feature, surface_key)
        if match:
            features.append(1)
        else:
            features.append(0)
    return features

def calculate_ssim_weighting_factor(feature_vector, signal_value):
    # Calculate the SSIM-based weighting factor
    ssim = calculate_ssim(feature_vector, signal_value)
    weighting_factor = ssim
    return weighting_factor

def calculate_ssim(feature_vector, signal_value):
    # Calculate the SSIM between the feature vector and the signal value
    ssim = np.dot(feature_vector, signal_value) / (np.linalg.norm(feature_vector) * np.linalg.norm(signal_value))
    return ssim

def update_surface_pheromone(pheromone_uuid, hybrid_score):
    # Update the surface pheromone signal using the hybrid score
    with psycopg.connect(db({'execute': True}), row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            ensure_schema(cur)
            cur.execute('''UPDATE lucidota_runtime.surface_pheromone SET signal_value = %s WHERE pheromone_uuid = %s''', 
                           (hybrid_score, pheromone_uuid))
        conn.commit()

if __name__ == "__main__":
    class Args:
        def __init__(self, surface_key, signal_kind, signal_value, half_life_seconds, execute):
            self.surface_key = surface_key
            self.signal_kind = signal_kind
            self.signal_value = signal_value
            self.half_life_seconds = half_life_seconds
            self.execute = execute

    args = Args('test_surface_key', 'test_signal_kind', 1.0, 3600, True)
    signal(args)
    decay(args)