from __future__ import annotations
from pathlib import Path
import hashlib

def _hash(path: Path) -> str: return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
def compute_phash(path: Path) -> str: return _hash(Path(path))
def compute_dhash(path: Path) -> str: return _hash(Path(path))
def hamming_distance(a: str, b: str) -> int:
    return sum(x!=y for x,y in zip(a,b)) + abs(len(a)-len(b))
def cluster_by_phash(paths, hamming_threshold: int = 4):
    clusters=[]
    for p in paths:
        p=Path(p); hp=compute_phash(p); placed=False
        for c in clusters:
            if hamming_distance(hp, compute_phash(c[0])) <= hamming_threshold:
                c.append(p); placed=True; break
        if not placed: clusters.append([p])
    return clusters
