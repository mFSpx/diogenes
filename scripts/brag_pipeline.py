#!/usr/bin/env python3
"""
BRAG: Built Really Absurdly Gwarishly — Multi-pass doc chunking + embedding + ingestion.

Pipeline:
  Pass 1 (GO-25):  25-category ontology chunking
  Pass 2 (O-75):   75-category extended ontology chunking
  Pass 3 (ROOT-414): SHA256 integrity hashes on all chunks

Every shape wrapped in XHash. Cells stored in Postgres with PostgREST API.
Designed for the Odysseus Manual as the golden example of manual generation.
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

# ─── Ontology ─────────────────────────────────────────────────────────

GO25_ONTOLOGY = [
    "ENTITY", "ATTRIBUTE", "RELATIONSHIP", "FRICTION", "LEVERAGE",
    "VISIBILITY", "ACTION", "EVENT", "TIME", "PATTERN",
    "HYPOTHESIS", "CLAIM", "EVIDENCE", "ATOMIC_ID", "SIGNAL",
    "GLOW", "TERM", "TOOL", "ALGORITHM", "NAUGHTY",
    "NICE", "GROUP", "OPERATOR", "MODE", "COMMENT",
]

O75_EXTENDED = GO25_ONTOLOGY + [
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
class DocShape:
    """A single document chunk with ontology tags and XHash."""
    source: str
    chunk_id: str
    pass_name: str          # GO-25, O-75, ROOT-414
    ontology_tags: list[str]
    text: str
    token_estimate: int
    sha256: str
    xhash: str              # Wrapped hash
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_xhash(self) -> str:
        """XHash: SHA256 of (pass + ontology_tags + sha256 + salt)."""
        raw = f"{self.pass_name}:{':'.join(sorted(self.ontology_tags))}:{self.sha256}:BRAGv1"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def to_row(self) -> dict[str, Any]:
        """Format as a Postgres/PostgREST row."""
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "pass_name": self.pass_name,
            "ontology_tags": self.ontology_tags,
            "text": self.text,
            "token_estimate": self.token_estimate,
            "sha256": self.sha256,
            "xhash": self.xhash,
            "metadata": json.dumps(self.metadata),
            "embedding": None,  # populated by embedding step
        }


# ─── Doc extraction ───────────────────────────────────────────────────

def extract_all_docs(odysseus_path: Path) -> list[dict[str, Any]]:
    """Extract all documentation from Odysseus repo."""
    docs = []

    # Markdown files
    for md_file in sorted(odysseus_path.rglob("*.md")):
        if "node_modules" in str(md_file):
            continue
        try:
            text = md_file.read_text(encoding="utf-8", errors="replace")
            rel_path = str(md_file.relative_to(odysseus_path))
            docs.append({
                "source": rel_path,
                "type": "markdown",
                "text": text,
            })
        except Exception as e:
            print(f"  [warn] Skipping {md_file}: {e}", file=sys.stderr)

    # Python docstrings
    for py_file in sorted(odysseus_path.rglob("*.py")):
        if "node_modules" in str(py_file) or "__pycache__" in str(py_file):
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
            rel_path = str(py_file.relative_to(odysseus_path))
            # Extract module-level docstring
            docstring_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
            docstring = docstring_match.group(1).strip() if docstring_match else ""
            if docstring:
                docs.append({
                    "source": rel_path,
                    "type": "docstring",
                    "text": docstring,
                })
        except Exception:
            pass

    # Config files
    for cfg_file in sorted(odysseus_path.rglob("*.json")):
        if "node_modules" in str(cfg_file):
            continue
        try:
            text = cfg_file.read_text(encoding="utf-8", errors="replace")
            rel_path = str(cfg_file.relative_to(odysseus_path))
            docs.append({
                "source": rel_path,
                "type": "config",
                "text": text,
            })
        except Exception:
            pass

    return docs


# ─── Chunking strategies ──────────────────────────────────────────────

def chunk_go25(text: str, source: str, chunk_size: int = 512) -> list[DocShape]:
    """Pass 1: GO-25 ontology chunking — semantic boundary detection."""
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if len(para) < 20:
            continue

        # Detect ontology tags from content
        tags = []
        upper = para.upper()
        for term in GO25_ONTOLOGY:
            if term in upper:
                tags.append(term)
        if not tags:
            tags = ["ENTITY"]  # default tag

        # Sub-chunk long paragraphs
        words = para.split()
        if len(words) > chunk_size:
            for j in range(0, len(words), chunk_size):
                sub = " ".join(words[j:j + chunk_size])
                chunks.append(_make_shape(source, f"go25_{i}_{j}", "GO-25", tags, sub))
        else:
            chunks.append(_make_shape(source, f"go25_{i}", "GO-25", tags, para))

    return chunks


def chunk_o75(text: str, source: str, chunk_size: int = 256) -> list[DocShape]:
    """Pass 2: O-75 extended ontology — finer granularity."""
    chunks = []
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current = []
    current_len = 0
    sent_idx = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        words = len(sent.split())

        # Detect O-75 tags
        tags = []
        upper = sent.upper()
        for term in O75_EXTENDED:
            if term in upper:
                tags.append(term)
        if not tags:
            tags = ["ENTITY"]

        if current_len + words > chunk_size and current:
            chunk_text = " ".join(c[0] for c in current)
            chunks.append(_make_shape(
                source, f"o75_{sent_idx}", "O-75",
                _merge_tags([c[1] for c in current]), chunk_text
            ))
            current = []
            current_len = 0

        current.append((sent, tags))
        current_len += words
        sent_idx += 1

    if current:
        chunk_text = " ".join(c[0] for c in current)
        chunks.append(_make_shape(
            source, f"o75_{sent_idx}", "O-75",
            _merge_tags([c[1] for c in current]), chunk_text
        ))

    return chunks


def chunk_root414(chunks: list[DocShape]) -> list[DocShape]:
    """Pass 3: ROOT-414 — integrity hashes only. No text, just SHA256 + XHash."""
    hash_chunks = []
    for i, chunk in enumerate(chunks):
        # Create pure hash shape
        hash_shape = DocShape(
            source=chunk.source,
            chunk_id=f"414_{i:06x}",
            pass_name="ROOT-414",
            ontology_tags=["ATOMIC_ID", "EVIDENCE"],
            text=f"INTEGRITY_CHECK:{chunk.sha256}",
            token_estimate=0,
            sha256=chunk.sha256,
            xhash="",
            metadata={
                "parent_chunk_id": chunk.chunk_id,
                "parent_pass": chunk.pass_name,
                "parent_sha256": chunk.sha256,
            },
        )
        hash_shape.xhash = hash_shape.compute_xhash()
        hash_chunks.append(hash_shape)
    return hash_chunks


def _make_shape(source: str, chunk_id: str, pass_name: str,
                tags: list[str], text: str) -> DocShape:
    """Create a DocShape with computed hashes."""
    shape = DocShape(
        source=source,
        chunk_id=f"{pass_name.lower()}_{chunk_id}",
        pass_name=pass_name,
        ontology_tags=sorted(set(tags)),
        text=text,
        token_estimate=len(text.split()),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        xhash="",
    )
    shape.xhash = shape.compute_xhash()
    return shape


def _merge_tags(tag_lists: list[list[str]]) -> list[str]:
    """Merge multiple tag lists into a sorted unique set."""
    merged = set()
    for tags in tag_lists:
        merged.update(tags)
    return sorted(merged)


# ─── Postgres ingestion ───────────────────────────────────────────────

def ingest_to_postgres(shapes: list[DocShape], dsn: str, schema: str = "lucidota_korpus") -> int:
    """Ingest all shapes into Postgres via the document_parse_span table."""
    import psycopg2

    conn = None
    try:
        conn = psycopg2.connect(dsn)
    except Exception as e:
        print(f"  [error] Postgres connection failed: {e}", file=sys.stderr)
        return 0

    cur = None
    try:
        cur = conn.cursor()

        # Ensure table exists
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {schema}.brag_cell (
                cell_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                chunk_id text NOT NULL,
                source text NOT NULL,
                pass_name text NOT NULL,
                ontology_tags text[] NOT NULL DEFAULT '{{}}',
                text text NOT NULL,
                token_estimate integer NOT NULL DEFAULT 0,
                sha256 text NOT NULL,
                xhash text NOT NULL,
                metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector(384),
                created_at timestamptz NOT NULL DEFAULT now(),
                UNIQUE(chunk_id)
            );
        """)
        conn.commit()

        # Insert rows
        count = 0
        for shape in shapes:
            try:
                cur.execute(f"""
                    INSERT INTO {schema}.brag_cell
                        (chunk_id, source, pass_name, ontology_tags, text,
                         token_estimate, sha256, xhash, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (chunk_id) DO NOTHING
                """, (
                    shape.chunk_id,
                    shape.source,
                    shape.pass_name,
                    shape.ontology_tags,
                    shape.text,
                    shape.token_estimate,
                    shape.sha256,
                    shape.xhash,
                    json.dumps(shape.metadata),
                ))
                count += 1
            except Exception as e:
                print(f"  [warn] Insert failed for {shape.chunk_id}: {e}", file=sys.stderr)

        conn.commit()
        return count
    except Exception as e:
        print(f"  [error] Ingest failed: {e}", file=sys.stderr)
        if conn:
            conn.rollback()
        return 0
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


def check_postgrest_api(base_url: str = "http://127.0.0.1:3000") -> bool:
    """Check if PostgREST is running and has the brag_cell table."""
    import urllib.request
    try:
        resp = urllib.request.urlopen(f"{base_url}/brag_cell?limit=1", timeout=5)
        return resp.status == 200
    except Exception:
        return False


# ─── Main pipeline ────────────────────────────────────────────────────

def run_pipeline(
    odysseus_path: str,
    output: str | None = None,
    dsn: str | None = None,
    dry_run: bool = False,
    json_output: bool = False,
) -> dict[str, Any]:
    """Run the full BRAG pipeline."""
    print(f"\n=== BRAG Pipeline ===", file=sys.stderr)
    t0 = time.time()

    # Phase 1: Extract
    print("  Phase 1: Extracting docs...", file=sys.stderr)
    docs = extract_all_docs(Path(odysseus_path))
    total_chars = sum(len(d["text"]) for d in docs)
    print(f"  Extracted {len(docs)} documents ({total_chars:,} chars)", file=sys.stderr)

    # Phase 2: Chunk — Pass 1 (GO-25)
    print("  Phase 2: Chunking...", file=sys.stderr)
    go25_shapes: list[DocShape] = []
    for doc in docs:
        go25_shapes.extend(chunk_go25(doc["text"], doc["source"]))
    print(f"  Pass 1 (GO-25): {len(go25_shapes)} chunks", file=sys.stderr)

    # Phase 2: Chunk — Pass 2 (O-75)
    o75_shapes: list[DocShape] = []
    for doc in docs:
        o75_shapes.extend(chunk_o75(doc["text"], doc["source"]))
    print(f"  Pass 2 (O-75): {len(o75_shapes)} chunks", file=sys.stderr)

    # Phase 2: Chunk — Pass 3 (ROOT-414)
    all_shapes = go25_shapes + o75_shapes
    r414_shapes = chunk_root414(all_shapes)
    print(f"  Pass 3 (ROOT-414): {len(r414_shapes)} integrity hashes", file=sys.stderr)

    all_with_hashes = all_shapes + r414_shapes
    print(f"  Total shapes: {len(all_with_hashes)}", file=sys.stderr)

    # Phase 3: Embed (placeholder for embedding service)
    # Embeddings generated separately via embedding worker

    # Phase 4: Ingest
    ingested = 0
    if dsn and not dry_run:
        print(f"  Phase 4: Ingesting to Postgres...", file=sys.stderr)
        ingested = ingest_to_postgres(all_with_hashes, dsn)
        print(f"  Ingested: {ingested} rows", file=sys.stderr)

        # Check PostgREST
        if check_postgrest_api():
            print(f"  PostgREST: available — cells accessible via /brag_cell", file=sys.stderr)
        else:
            print(f"  PostgREST: not detected (start with: postgrest GOALS/root_rotor_postgrest.conf)", file=sys.stderr)

    # Write output
    result = {
        "schema": "lucidota.brag_pipeline.v1",
        "documents_extracted": len(docs),
        "total_chars": total_chars,
        "go25_chunks": len(go25_shapes),
        "o75_chunks": len(o75_shapes),
        "root414_hashes": len(r414_shapes),
        "total_shapes": len(all_with_hashes),
        "ingested": ingested,
        "elapsed_s": round(time.time() - t0, 2),
        "ontology_go25": GO25_ONTOLOGY,
        "ontology_o75_count": len(O75_EXTENDED),
    }

    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write shapes as JSONL
        shapes_path = output_path.with_suffix(".jsonl")
        try:
            with open(shapes_path, "w") as f:
                for shape in all_with_hashes:
                    f.write(json.dumps(asdict(shape)) + "\n")
            result["shapes_file"] = str(shapes_path)
        except OSError as e:
            print(f"  [error] Failed to write shapes: {e}", file=sys.stderr)
            result["shapes_file_error"] = str(e)

        # Write result
        result_path = output_path.parent / f"brag_result_{time.strftime('%Y%m%dT%H%M%S')}.json"
        try:
            result_path.write_text(json.dumps(result, indent=2))
            print(f"\n  Shapes: {shapes_path}", file=sys.stderr)
            print(f"  Result: {result_path}", file=sys.stderr)
        except OSError as e:
            print(f"  [error] Failed to write result: {e}", file=sys.stderr)

    if json_output:
        print(json.dumps(result, indent=2))
    else:
        print(f"\n=== BRAG Complete ===")
        print(f"  Docs: {len(docs)} | Chars: {total_chars:,}")
        print(f"  GO-25 chunks: {len(go25_shapes)}")
        print(f"  O-75 chunks:  {len(o75_shapes)}")
        print(f"  ROOT-414:     {len(r414_shapes)} hashes")
        print(f"  Total shapes: {len(all_with_hashes)}")
        print(f"  Ingested:     {ingested} rows")
        print(f"  Elapsed:      {result['elapsed_s']}s")

    return result


def main():
    parser = argparse.ArgumentParser(description="BRAG: Multi-pass doc chunking + embedding pipeline")
    parser.add_argument("--odysseus-path", default="01_REPOS/odysseus",
                        help="Path to Odysseus repo")
    parser.add_argument("--output", default="05_OUTPUTS/brag/odysseus_manual.jsonl",
                        help="Output path for shapes JSONL")
    parser.add_argument("--dsn", help="Postgres DSN for ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Skip Postgres ingestion")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    # Validate inputs
    if not args.odysseus_path or not args.odysseus_path.strip():
        sys.exit("[error] --odysseus-path must be a non-empty path")
    if not args.output or not args.output.strip():
        sys.exit("[error] --output must be a non-empty path")

    path = str(ROOT / args.odysseus_path) if not os.path.isabs(args.odysseus_path) else args.odysseus_path
    # Ensure source path exists
    if not Path(path).exists():
        sys.exit(f"[error] odysseus-path does not exist: {path}")

    dsn = args.dsn or os.environ.get("LUCIDOTA_GO_STATE_DSN", "postgresql:///lucidota_state")

    run_pipeline(
        odysseus_path=path,
        output=str(ROOT / args.output) if not os.path.isabs(args.output) else args.output,
        dsn=dsn,
        dry_run=args.dry_run,
        json_output=args.json,
    )


if __name__ == "__main__":
    main()
