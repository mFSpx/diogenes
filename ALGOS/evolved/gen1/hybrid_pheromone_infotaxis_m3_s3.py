# DARWIN HAMMER — match 3, survivor 3
# gen: 1
# parent_a: pheromone.py (gen0)
# parent_b: infotaxis.py (gen0)
# born: 2026-05-29T23:14:18Z

import argparse
import json
import math
import numpy as np
import os
import pathlib
import random
import sys
from datetime import datetime, timezone

class PheromoneSystem:
    def __init__(self):
        self.pheromones = {}

    def calculate_pheromone_signal(self, surface_key, signal_kind, signal_value, half_life_seconds):
        current_time = datetime.now(timezone.utc)
        if surface_key not in self.pheromones:
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        else:
            previous_signal_value = self.pheromones[surface_key]['signal_value']
            previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
            previous_created_time = self.pheromones[surface_key]['created_time']
            elapsed_time = (current_time - previous_created_time).total_seconds()
            decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
            self.pheromones[surface_key] = {'signal_kind': signal_kind, 'signal_value': signal_value, 'half_life_seconds': half_life_seconds, 'created_time': current_time}
        return signal_value

    def calculate_entropy(self, probabilities, eps=1e-12):
        total = sum(probabilities)
        if total <= 0:
            raise ValueError('positive probability mass required')
        return -sum((p/total) * math.log(max(p/total, eps)) for p in probabilities if p > 0)

    def expected_entropy(self, p_hit, hit_state, miss_state):
        if not 0 <= p_hit <= 1:
            raise ValueError('p_hit must be in [0,1]')
        return p_hit * self.calculate_entropy(hit_state) + (1.0 - p_hit) * self.calculate_entropy(miss_state)

    def best_action(self, actions):
        if not actions:
            raise ValueError('actions required')
        return min(actions, key=lambda a: (self.expected_entropy(*actions[a]), a))

    def signal(self, surface_key, signal_kind, signal_value, half_life_seconds, execute):
        pheromone_uuid = None
        if execute:
            pheromone_uuid = str(random.uuid4())
        report = {
            'action': 'signal',
            'execute_performed': bool(execute),
            'db_writes_performed': bool(execute),
            'graph_writes_performed': False,
            'surface_key': surface_key,
            'signal_kind': signal_kind,
            'signal_value': signal_value,
            'pheromone_uuid': pheromone_uuid,
            'status': 'PASS'
        }
        self.calculate_pheromone_signal(surface_key, signal_kind, signal_value, half_life_seconds)
        return report

    def decay(self, surface_key, half_life_seconds, execute):
        updated = 0
        rows = []
        if execute:
            current_time = datetime.now(timezone.utc)
            if surface_key in self.pheromones:
                previous_signal_value = self.pheromones[surface_key]['signal_value']
                previous_half_life_seconds = self.pheromones[surface_key]['half_life_seconds']
                previous_created_time = self.pheromones[surface_key]['created_time']
                elapsed_time = (current_time - previous_created_time).total_seconds()
                decayed_signal_value = previous_signal_value * math.pow(0.5, elapsed_time / previous_half_life_seconds)
                self.pheromones[surface_key] = {'signal_kind': self.pheromones[surface_key]['signal_kind'], 'signal_value': decayed_signal_value, 'half_life_seconds': previous_half_life_seconds, 'created_time': previous_created_time}
                rows.append({'surface_key': surface_key, 'decayed_value': decayed_signal_value})
                updated += 1
        else:
            rows = [{'surface_key': surface_key, 'would_decay': 'dry_run'}]
        report = {
            'action': 'decay',
            'execute_performed': bool(execute),
            'db_writes_performed': bool(execute),
            'graph_writes_performed': False,
            'surface_key': surface_key,
            'rows_seen': len(rows),
            'rows_updated': updated,
            'rows': rows[:20],
            'status': 'PASS'
        }
        return report

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('signal')
    s.add_argument('--surface-key', default='marrow_loop_status')
    s.add_argument('--signal-kind', choices=['generated', 'used', 'promoted', 'forked', 'decayed', 'archived', 'operator_selected'], default='used')
    s.add_argument('--signal-value', type=float, default=1.0)
    s.add_argument('--half-life-seconds', type=int, default=604800)
    s.add_argument('--execute', action='store_true')
    d = sub.add_parser('decay')
    d.add_argument('--surface-key', default='marrow_loop_status')
    d.add_argument('--limit', type=int, default=20)
    d.add_argument('--execute', action='store_true')
    a = p.parse_args()
    pheromone_system = PheromoneSystem()
    if a.cmd == 'signal':
        print(pheromone_system.signal(a.surface_key, a.signal_kind, a.signal_value, a.half_life_seconds, a.execute))
    elif a.cmd == 'decay':
        print(pheromone_system.decay(a.surface_key, 604800, a.execute))

if __name__ == "__main__":
    main()