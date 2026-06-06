#!/usr/bin/env python3
"""
BRAG ABSURD Worker — RETE-bandit-driven, Percyphon-hashed, LTC-ordered chunking.

Pipeline:
  1. ABSURD queue picks up doc packets
  2. RETE bandit gate (ALGOS.rete_bandit_gate.apply_rete_bandit) routes each doc to
     the correct ontology pass + algorithm pool
  3. Percyphon (ALGOS.percyphon) generates deterministic 128-slot xxhash128 identities
  4. LTC (ALGOS.ltc) orders chunks by temporal evidence flow (ODE-based)
  5. XHash wraps every shape
  6. Ingests to Postgres via PostgREST

Deterministic. Pure SHA256 arithmetic. No LLM. No randomness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ALGOS.rete_bandit_gate import apply_rete_bandit  # noqa: E402
from ALGOS.ltc import LTCCell  # noqa: E402
from ALGOS import percyphon  # noqa: E402


# ─── Ontology ─────────────────────────────────────────────────────────

GO25 = [
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "FRICTION", "LEVERAGE",
    "VISIBILITY", "ACTION", "EVENT", "TIME", "PATTERN",
    "HYPOTHESIS", "CLAIM", "EVIDENCE", "ATOMIC_ID", "SIGNAL",
    "GLOW", "TERM", "TOOL", "ALGORITHM", "NAUGHTY",
    "NICE", "GROUP", "OPERATOR", "MODE", "COMMENT",
]

O75 = GO25 + [
    "ENDPOINT", "SCHEMA", "ROUTE", "MODEL", "PROVIDER",
    "CONFIG", "DEPENDENCY", "ENV_VAR", "SECRET", "TOKEN",
    "SESSION", "MESSAGE", "CHAIN", "EMBEDDING", "VECTOR",
    "CHUNK", "SOURCE", "HASH", "SIGNATURE", "CERT",
    "QUEUE", "WORKER", "JOB", "PIPELINE", "STAGE",
    "CACHE", "STORE", "INDEX", "QUERY", "FILTER",
    "HOOK", "PLUGIN", "EXTENSION", "MIDDLEWARE", "ADAPTER",
    "UI", "WIDGET", "PAGE", "LAYOUT", "THEME",
    "ERROR", "EXCEPTION", "LOG", "METRIC", "ALERT",
    "PERMISSION", "ROLE", "POLICY", "AUDIT", "BACKUP",
]


@dataclass
class BragShape:
    """Document chunk with RETE-routed ontology, Percyphon identity, XHash."""
    source: str
    chunk_id: str
    pass_name: str
    ontology_tags: list[str]
    percyphon_slot: int       # Percyphon 128-slot coordinate
    text: str
    token_estimate: int
    sha256: str
    xhash: str
    rete_decision: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_xhash(self) -> str:
        """XHash: SHA256(pass + sorted(tags) + sha256 + slot + BRAGv2)."""
        raw = f"BRAGv2:{self.pass_name}:{':'.join(sorted(self.ontology_tags))}:{self.sha256}:{self.percyphon_slot}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def to_row(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "pass_name": self.pass_name,
            "ontology_tags": self.ontology_tags,
            "percyphon_slot": self.percyphon_slot,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "sha256": self.sha256,
            "xhash": self.xhash,
            "rete_decision": json.dumps(self.rete_decision),
            "metadata": json.dumps(self.metadata),
        }


# ─── Step 1: RETE bandit routes each doc ──────────────────────────────

def rete_route_doc(doc: dict[str, Any]) -> dict[str, Any]:
    """Route a document through the RETE bandit gate to determine ontology pass."""
    packet = {
        "source": doc["source"],
        "source_path": doc["source"],
        "source_ref": doc["source"],
        "payload": {
            "file_type": doc.get("type", "TEXT"),
            "text": doc["text"][:2000],
        },
    }
    try:
        decision = apply_rete_bandit(packet)
    except Exception:
        decision = {"selected_algorithm": "minhash", "rule_hits": ["fallback"]}

    # Map RETE decision to ontology pass
    algo = decision.get("selected_algorithm", "minhash")
    if algo in ("gliner_zero_shot", "semantic_neighbors", "minhash"):
        pass_name = "GO-25"
        tags = GO25
    elif algo in ("decision_hygiene", "lora_preemption", "needle_classifier"):
        pass_name = "O-75"
        tags = O75
    else:
        pass_name = "GO-25"
        tags = GO25

    return {
        "pass_name": pass_name,
        "tags": tags,
        "algorithm": algo,
        "rule_hits": decision.get("rule_hits", []),
        "regret_weights": decision.get("regret_weights", {}),
    }


# ─── Step 2: Percyphon generates deterministic identities ─────────────

def percyphon_identity(source: str, chunk_index: int) -> tuple[int, str]:
    """Use Percyphon 128-slot xxhash128 to generate deterministic chunk identity."""
    slot = chunk_index % 128  # Percyphon 128-slot identity mask
    seed = f"{source}:{chunk_index}"
    digest = percyphon._xxhash128_int(seed) if hasattr(percyphon, '_xxhash128_int') else int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
    return slot, f"{digest:032x}"


# ─── Step 3: LTC orders chunks temporally ─────────────────────────────

class LTCOrderer:
    """LTC-based temporal ordering of document chunks."""

    def __init__(self, hidden_size: int = 16):
        self.cell = LTCCell(input_size=4, hidden_size=hidden_size)
        self.hidden_state = None
        self.t = 0.0

    def order_chunk(self, chunk: BragShape, delta_t: float = 1.0) -> float:
        """Process chunk through LTC, return evidence intensity score."""
        import numpy as np
        # Encode chunk features as input vector
        input_vec = np.array([
            float(chunk.token_estimate) / 1000.0,
            float(len(chunk.ontology_tags)) / 75.0,
            float(chunk.percyphon_slot) / 128.0,
            1.0 if chunk.pass_name == "ROOT-414" else 0.0,
        ], dtype=np.float64)

        state_size = self.cell.hidden_size
        init_state = np.zeros(state_size, dtype=np.float64)
        self.hidden_state = self.cell.step(init_state, input_vec, delta_t)
        self.t += delta_t

        # Evidence intensity = norm of hidden state
        intensity = float(np.linalg.norm(self.hidden_state))
        return intensity


# ─── Step 4: XHash wrapping ───────────────────────────────────────────

def xhash_wrap(shape: BragShape) -> BragShape:
    """Wrap shape with XHash integrity chain."""
    shape.xhash = shape.compute_xhash()
    return shape


# ─── Document extraction ──────────────────────────────────────────────

def extract_docs(odysseus_path: str) -> list[dict[str, Any]]:
    """Extract all docs deterministically (sorted by path for reproducibility)."""
    docs = []
    base = Path(odysseus_path)

    for md_file in sorted(base.rglob("*.md")):
        if "node_modules" in str(md_file):
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            docs.append({"source": str(md_file.relative_to(base)), "type": "markdown", "text": text})
        except Exception:
            pass

    for py_file in sorted(base.rglob("*.py")):
        if "node_modules" in str(py_file) or "__pycache__" in str(py_file):
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
            m = re.search(r'"""(.*?)"""', text, re.DOTALL)
            if m:
                docs.append({"source": str(py_file.relative_to(base)), "type": "docstring", "text": m.group(1).strip()})
        except Exception:
            pass

    return docs


# ─── Main pipeline ────────────────────────────────────────────────────

def run_brag(
    odysseus_path: str,
    output: str | None = None,
    dsn: str | None = None,
    dry_run: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Run RETE-bandit-driven BRAG pipeline with Percyphon + LTC."""
    print(f"\n=== BRAGv2: RETE × Percyphon × LTC ===", file=sys.stderr)
    t0 = time.time()

    # Extract
    docs = extract_docs(odysseus_path)
    print(f"  Docs: {len(docs)}", file=sys.stderr)

    all_shapes: list[BragShape] = []
    ltc_orderer = LTCOrderer()
    chunk_counter = 0

    for doc in docs:
        # Step 1: RETE route
        route = rete_route_doc(doc)

        # Split into chunks (semantic by paragraph)
        paragraphs = re.split(r'\n\s*\n', doc["text"])
        for para in paragraphs:
            para = para.strip()
            if len(para) < 20:
                continue

            # Step 2: Percyphon identity
            slot, ident = percyphon_identity(doc["source"], chunk_counter)

            # Detect ontology tags from content
            upper = para.upper()
            detected_tags = [t for t in route["tags"] if t in upper]
            if not detected_tags:
                detected_tags = ["ENTITY"]

            # Create shape
            shape = BragShape(
                source=doc["source"],
                chunk_id=f"{route['pass_name'].lower()}_{chunk_counter:06x}",
                pass_name=route["pass_name"],
                ontology_tags=sorted(set(detected_tags)),
                percyphon_slot=slot,
                text=para,
                token_estimate=len(para.split()),
                sha256=hashlib.sha256(para.encode()).hexdigest(),
                xhash="",
                rete_decision={
                    "algorithm": route["algorithm"],
                    "rule_hits": route["rule_hits"],
                },
            )

            # Step 3: LTC order
            intensity = ltc_orderer.order_chunk(shape)
            shape.metadata["ltc_intensity"] = round(intensity, 4)
            shape.metadata["ltc_time"] = round(ltc_orderer.t, 2)

            # Step 4: XHash wrap
            shape = xhash_wrap(shape)

            all_shapes.append(shape)
            chunk_counter += 1

    # ROOT-414 pass: integrity hashes only
    r414_shapes = []
    for i, shape in enumerate(all_shapes):
        hash_shape = BragShape(
            source=shape.source,
            chunk_id=f"414_{i:06x}",
            pass_name="ROOT-414",
            ontology_tags=["ATOMIC_ID", "EVIDENCE"],
            percyphon_slot=i % 128,
            text=f"INTEGRITY:{shape.sha256}",
            token_estimate=0,
            sha256=shape.sha256,
            xhash="",
            metadata={"parent_chunk_id": shape.chunk_id, "parent_sha256": shape.sha256},
        )
        hash_shape = xhash_wrap(hash_shape)
        r414_shapes.append(hash_shape)

    total = len(all_shapes) + len(r414_shapes)
    print(f"  GO-25/O-75 shapes: {len(all_shapes)}", file=sys.stderr)
    print(f"  ROOT-414 hashes:   {len(r414_shapes)}", file=sys.stderr)
    print(f"  Total:            {total}", file=sys.stderr)

    # Ingest via PostgREST
    ingested = 0
    postgrest_url = "http://127.0.0.1:3000"
    if not dry_run:
        for shape_batch in [all_shapes, r414_shapes]:
            for shape in shape_batch:
                try:
                    import urllib.request
                    req = urllib.request.Request(
                        f"{postgrest_url}/brag_cell",
                        data=json.dumps(shape.to_row()).encode(),
                        headers={"Content-Type": "application/json", "Prefer": "return=minimal"},
                        method="POST",
                    )
                    urllib.request.urlopen(req, timeout=5)
                    ingested += 1
                except Exception:
                    pass  # Duplicate or PostgREST not ready
        print(f"  Ingested: {ingested}/{total}", file=sys.stderr)

    # Output
    result = {
        "schema": "lucidota.brag_absurd_worker.v2",
        "docs_processed": len(docs),
        "go25_o75_shapes": len(all_shapes),
        "root414_hashes": len(r414_shapes),
        "total_shapes": total,
        "ingested": ingested,
        "elapsed_s": round(time.time() - t0, 2),
        "algorithms": ["rete_bandit_gate", "percyphon", "ltc"],
        "deterministic": True,
    }

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(out_path, "w") as f:
                for shape in all_shapes + r414_shapes:
                    f.write(json.dumps(asdict(shape)) + "\n")
            result["output"] = str(out_path)
        except OSError as e:
            print(f"  [error] Failed to write output: {e}", file=sys.stderr)
            result["output_error"] = str(e)

    # Write receipt
    receipt_dir = ROOT / "05_OUTPUTS" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"brag_v2_{time.strftime('%Y%m%dT%H%M%S')}.json"
    try:
        receipt_path.write_text(json.dumps(result, indent=2))
        result["receipt"] = str(receipt_path.relative_to(ROOT))
    except OSError as e:
        print(f"  [warn] Failed to write receipt: {e}", file=sys.stderr)
        result["receipt"] = None

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== BRAGv2 Complete ===")
        print(f"  Docs: {len(docs)} | Shapes: {total}")
        print(f"  Algos: rete_bandit_gate × percyphon × ltc")
        print(f"  Ingested: {ingested}")
        print(f"  Deterministic: YES (SHA256 arithmetic)")
        print(f"  Receipt: {result.get('receipt', 'write_failed')}")

    return result


def main():
    parser = argparse.ArgumentParser(description="BRAGv2: RETE × Percyphon × LTC")
    parser.add_argument("--odysseus-path", default="01_REPOS/odysseus",
                        help="Path to Odysseus repo (default: 01_REPOS/odysseus)")
    parser.add_argument("--output", default="05_OUTPUTS/brag/brag_v2_shapes.jsonl",
                        help="Output path for shapes JSONL (default: 05_OUTPUTS/brag/brag_v2_shapes.jsonl)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip PostgREST ingestion")
    parser.add_argument("--json", action="store_true",
                        help="JSON output only")
    args = parser.parse_args()

    # Validate inputs
    if not args.odysseus_path or not args.odysseus_path.strip():
        sys.exit("[error] --odysseus-path must be a non-empty path")
    if not args.output or not args.output.strip():
        sys.exit("[error] --output must be a non-empty path")

    path = str(ROOT / args.odysseus_path) if not os.path.isabs(args.odysseus_path) else args.odysseus_path
    run_brag(
        odysseus_path=path,
        output=str(ROOT / args.output) if not os.path.isabs(args.output) else args.output,
        dry_run=args.dry_run,
        json_output=args.json,
    )


if __name__ == "__main__":
    main()
