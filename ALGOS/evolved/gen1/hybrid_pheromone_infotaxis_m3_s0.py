# DARWIN HAMMER — match 3, survivor 0
# gen: 1
# parent_a: pheromone.py (gen0)
# parent_b: infotaxis.py (gen0)
# born: 2026-05-29T23:14:18Z

"""Hybrid of pheromone.py and infotaxis.py: This module integrates the pheromone-based surface usage tracking 
from pheromone.py with the entropy-based action selection from infotaxis.py. The mathematical bridge between 
the two lies in the idea of using pheromone signals as probabilities to inform the entropy calculation, 
ultimately guiding the selection of actions based on surface usage patterns."""

import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

def calculate_pheromone_probabilities(surface_key, limit, db_url):
    """Calculates pheromone probabilities from the database."""
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('''SELECT signal_value FROM lucidota_runtime.surface_pheromone 
                            WHERE surface_key=%s ORDER BY created_at DESC LIMIT %s''', (surface_key, limit))
            pheromones = [r['signal_value'] for r in cur.fetchall()]
            total = sum(pheromones)
            return [p / total for p in pheromones]

def entropy(probabilities, eps=1e-12):
    """Calculates the entropy of a probability distribution."""
    total = sum(probabilities)
    if total <= 0:
        raise ValueError('positive probability mass required')
    return -sum((p / total) * math.log(max(p / total, eps)) for p in probabilities if p > 0)

def expected_entropy(p_hit, hit_state, miss_state):
    """Calculates the expected entropy of an action."""
    if not 0 <= p_hit <= 1:
        raise ValueError('p_hit must be in [0,1]')
    return p_hit * entropy(hit_state) + (1.0 - p_hit) * entropy(miss_state)

def best_action(actions, surface_key, limit, db_url):
    """Selects the best action based on pheromone probabilities and entropy."""
    pheromone_probabilities = calculate_pheromone_probabilities(surface_key, limit, db_url)
    best_action = min(actions, key=lambda a: (expected_entropy(*actions[a]), a))
    return best_action

def signal(surface_key, signal_kind, signal_value, half_life_seconds, execute, db_url):
    """Signals a surface usage event."""
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(db_url, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute('''INSERT INTO lucidota_runtime.surface_pheromone(surface_key, signal_kind, signal_value, half_life_seconds)
                            VALUES (%s, %s, %s, %s)''', (surface_key, signal_kind, signal_value, half_life_seconds))
            if execute:
                conn.commit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--surface-key', default='marrow_loop_status')
    parser.add_argument('--signal-kind', choices=['generated', 'used', 'promoted', 'forked', 'decayed', 'archived', 'operator_selected'], default='used')
    parser.add_argument('--signal-value', type=float, default=1.0)
    parser.add_argument('--half-life-seconds', type=int, default=604800)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--execute', action='store_true')

    args = parser.parse_args()

    actions = {
        'action1': (0.5, [0.2, 0.3, 0.5], [0.1, 0.4, 0.5]),
        'action2': (0.7, [0.6, 0.2, 0.2], [0.3, 0.6, 0.1]),
    }

    signal(args.surface_key, args.signal_kind, args.signal_value, args.half_life_seconds, args.execute, args.database_url)
    best = best_action(actions, args.surface_key, args.limit, args.database_url)
    print(f"Best action: {best}")

if __name__ == "__main__":
    main()