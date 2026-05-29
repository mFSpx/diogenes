from __future__ import annotations
from dataclasses import dataclass
import uuid, re

@dataclass(frozen=True)
class NeighborResult:
    neighbor_id: str
    distance: float
    retrieval_receipt_id: str
    truth_band: str = "OBSERVED"

_ENCLAVES: dict[str, dict[str, dict]] = {}

def clear_enclave(enclave_id: str = "default") -> None:
    _ENCLAVES.pop(enclave_id, None)

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def register_document(enclave_id: str, doc_id: str, text: str, truth_band: str = "OBSERVED") -> None:
    _ENCLAVES.setdefault(enclave_id, {})[doc_id] = {"text": text, "tokens": _tokens(text), "truth_band": truth_band}

def semantic_neighbors(query: str, enclave_id: str = "default", k: int = 5, backend: str = "memory", truth_band_filter: list[str] | None = None) -> list[NeighborResult]:
    if backend == "memory":
        docs=_ENCLAVES.get(enclave_id, {})
        if not docs: return []
        q=_tokens(query); receipt="receipt_"+uuid.uuid4().hex[:12]
        filt=set(truth_band_filter or [])
        rows=[]
        for doc_id,d in docs.items():
            if filt and d["truth_band"] not in filt: continue
            union=q|d["tokens"]
            sim=(len(q & d["tokens"])/len(union)) if union else 0.0
            rows.append((sim, NeighborResult(doc_id, 1.0-sim, receipt, d["truth_band"])))
        rows.sort(key=lambda x:x[0], reverse=True)
        return [r for _,r in rows[:k]]
    if backend in {"chroma","faiss"}: raise NotImplementedError(f"{backend} backend not installed")
    if backend in {"qdrant","pgvector"}: raise ImportError(f"{backend} backend not installed")
    raise ValueError(f"Unknown backend {backend}")
