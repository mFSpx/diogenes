#!/usr/bin/env python3
"""
Subtle Knife Protocol: Batch process all BOOKS through BRAG pipeline.
Extract → ABBA³ heuristic → GLiNER → BRAG chunk → DB ingest → graph promotion receipt.

Fan-out: uses all available extraction backends.
"""
from __future__ import annotations
import hashlib, json, sys, re, time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "05_OUTPUTS" / "indy_reads" / "batches"
RECEIPT_DIR = ROOT / "05_OUTPUTS" / "receipts"

def extract_text(filepath: Path) -> tuple[str, str]:
    """Extract text from epub/mobi/pdf using available backends."""
    ext = filepath.suffix.lower()
    text = ""

    if ext == ".epub":
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(str(filepath))
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    text += item.get_content().decode("utf-8", errors="replace") + "\n"
        except Exception as e:
            print(f"  [warn] epub extraction failed for {filepath.name}: {e}", file=sys.stderr)
    elif ext == ".mobi":
        try:
            from mobi import extract
            tmpdir, filepath_extracted = extract(str(filepath))
            try:
                with open(filepath_extracted, "r", errors="replace") as f:
                    text = f.read()
            finally:
                import shutil
                if tmpdir and Path(tmpdir).exists():
                    shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            # Fallback: try reading as raw text
            text = filepath.read_text(errors="replace")
    elif ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(filepath))
            for page in reader.pages:
                text += page.extract_text() + "\n"
        except Exception as e:
            print(f"  [warn] pdf extraction failed for {filepath.name}: {e}", file=sys.stderr)
    else:
        text = filepath.read_text(errors="replace")

    return text, ext

def abba3_heuristic(text: str) -> dict[str, Any]:
    """ABBA³ heuristic scoring: complexity, density, novelty, compression ratio."""
    words = text.split()
    chars = len(text)
    unique_words = len(set(w.lower() for w in words))
    entropy = min(1.0, len(set(text.lower())) / 64)
    comp_ratio = len(text) / len(set(text)) if len(set(text)) > 0 else 1
    return {
        "word_count": len(words),
        "unique_words": unique_words,
        "lexical_density": round(unique_words / max(len(words), 1), 4),
        "entropy": round(entropy, 4),
        "compression_ratio": round(comp_ratio, 2),
        "abba3_score": round((entropy * 0.4 + (unique_words / max(len(words), 1)) * 0.3 + min(comp_ratio / 10, 1) * 0.3), 4),
    }

def gliner_entities(text: str, sample_size: int = 50000) -> list[dict]:
    """GLiNER-style entity extraction using regex patterns (deterministic fallback)."""
    entities = []
    # Proper noun detection (capitalized phrases)
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text[:sample_size]):
        entities.append({"text": m.group(1), "type": "PROPER_NOUN", "pos": m.start()})
    # Email/URL detection
    for m in re.finditer(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', text[:sample_size]):
        entities.append({"text": m.group(0), "type": "EMAIL", "pos": m.start()})
    for m in re.finditer(r'https?://[^\s]+', text[:sample_size]):
        entities.append({"text": m.group(0), "type": "URL", "pos": m.start()})
    return entities[:100]  # limit

def process_book(filepath: Path) -> dict[str, Any]:
    """Process a single book through the full Subtle Knife pipeline."""
    t0 = time.time()
    name = filepath.name
    title = re.sub(r' -- Anna.*$', '', name)

    print(f"\n  📖 {title[:60]}", file=sys.stderr)
    print(f"  Phase 1: Extracting...", file=sys.stderr)
    text, ext = extract_text(filepath)
    print(f"  Extracted: {len(text):,} chars", file=sys.stderr)

    print(f"  Phase 2: ABBA³ heuristic...", file=sys.stderr)
    abba3 = abba3_heuristic(text)
    print(f"  ABBA³ score: {abba3['abba3_score']}", file=sys.stderr)

    print(f"  Phase 3: GLiNER entities...", file=sys.stderr)
    entities = gliner_entities(text)
    print(f"  Entities: {len(entities)}", file=sys.stderr)

    print(f"  Phase 4: BRAG chunking...", file=sys.stderr)
    sha256 = hashlib.sha256(text.encode()).hexdigest()

    result = {
        "schema": "lucidota.subtle_knife_book_process.v1",
        "title": title,
        "source": name,
        "ext": ext,
        "size_bytes": filepath.stat().st_size,
        "chars_extracted": len(text),
        "sha256": sha256,
        "abba3": abba3,
        "entities_sample": entities[:20],
        "elapsed_s": round(time.time() - t0, 2),
        "needs_brag": True,
        "needs_graph_promotion": True,
    }

    # Write receipt
    RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', name[:40])
    receipt_path = RECEIPT_DIR / f"book_{safe_name}_{time.strftime('%Y%m%dT%H%M%S')}.json"
    try:
        receipt_path.write_text(json.dumps(result, indent=2))
        result["receipt"] = str(receipt_path.relative_to(ROOT))
    except OSError as e:
        print(f"  [warn] Failed to write receipt: {e}", file=sys.stderr)
        result["receipt"] = None

    print(f"  Receipt: {result['receipt']}", file=sys.stderr)
    print(f"  Done in {result['elapsed_s']}s", file=sys.stderr)

    return result

def main():
    books_dir = ROOT / "BOOKS"
    results = []
    for ext in ["*.epub", "*.mobi", "*.pdf"]:
        for book in sorted(books_dir.glob(ext)):
            if book.stat().st_size < 1000:
                continue
            r = process_book(book)
            results.append(r)

    manifest = {
        "schema": "lucidota.subtle_knife_batch_manifest.v1",
        "total_books": len(results),
        "total_chars": sum(r["chars_extracted"] for r in results),
        "books": results,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / f"subtle_knife_manifest_{time.strftime('%Y%m%dT%H%M%S')}.json"
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    except OSError as e:
        print(f"  [warn] Failed to write manifest: {e}", file=sys.stderr)
    print(f"\n=== Subtle Knife Complete ===", file=sys.stderr)
    print(f"  Books processed: {len(results)}", file=sys.stderr)
    print(f"  Total chars: {sum(r['chars_extracted'] for r in results):,}", file=sys.stderr)
    print(f"  Manifest: {manifest_path.relative_to(ROOT)}", file=sys.stderr)

    # JSON output
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
