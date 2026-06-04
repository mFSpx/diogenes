#!/usr/bin/env python3
"""Stage INDY_READs BOOK_READER_LORA datasets and Talkie source custody.

LoRA teaches reading behavior; Postgres/RAG/CAS keeps exact text. This script
creates clean 500-token-ish book chunks, bounded conversational reading-card
JSONL, optional local embeddings, target adapter manifests, charts, and receipts.
It does not train or send anything.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import embedding_provider, indy_reads

CARD_TYPES = ["go25_packet", "chapter_map", "entity_claim_extract", "motif_questions", "style_voice"]
ADAPTER_TARGETS: dict[str, dict[str, Any]] = {
    "talkie": {
        "target_model_id": "talkie-lm/talkie-1930-13b-it",
        "runtime_lane": "RUNPOD_TALKIE_FORGE",
        "adapter_output": "PEFT safetensors first; optional runtime conversion after eval",
        "notes": "Talkie adapter is separate from Bonsai adapters; do not assume cross-family LoRA compatibility.",
    },
    "bonsai8b_q1": {
        "target_model_id": "prism-ml/Bonsai-8B Q1_0 runtime lane",
        "runtime_lane": "LOCAL_VRAM_FAST_PATH",
        "local_gguf": "03_VAULT/models/prism-ml/Bonsai-8B-gguf/Bonsai-8B-Q1_0.gguf",
        "adapter_output": "Train on compatible HF base if available; convert/load only after eval receipt.",
        "notes": "Q1 adapter target uses same book cards but needs its own adapter/export; not a Talkie LoRA clone.",
    },
    "bonsai8b_q2": {
        "target_model_id": "prism-ml/Ternary-Bonsai-8B Q2_0 CPU lane",
        "runtime_lane": "LOCAL_CPU_TERNARY_BACKSTOP",
        "local_gguf": "03_VAULT/models/prism-ml/Ternary-Bonsai-8B-gguf/Ternary-Bonsai-8B-Q2_0.gguf",
        "adapter_output": "Train on compatible HF base if available; convert/load only after eval receipt.",
        "notes": "Q2 adapter target is separate from Talkie and Q1; same chunks, separate manifest and eval gate.",
    },
}


def now_z() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def jdump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def sha_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_text(text: str) -> str:
    return sha256(text.encode("utf-8", errors="replace")).hexdigest()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def short_text(text: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def clean_book_text(text: str) -> str:
    """Normalize extraction/OCR/page cruft before AI-facing chunking."""
    text = text.replace("\ufeff", " ").replace("\x00", " ").replace("\f", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)  # de-hyphenate linebreaks
    text = re.sub(r"[ \t\r\v]+", " ", text)
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        # Drop common naked page markers that pollute chunk starts/ends.
        if re.fullmatch(r"(?:page|p\.?)[\s_-]*\d{1,5}", line, flags=re.I):
            continue
        if re.fullmatch(r"\d{1,5}", line):
            continue
        # Drop obvious PDF running headers made entirely of caps/symbols when tiny.
        if len(line) < 80 and sum(c.isupper() for c in line) > max(8, len(line) * 0.65) and not re.search(r"[.!?]", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def sentence_units(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paras:
        bits = re.split(r"(?<=[.!?])\s+", para)
        cur = ""
        for bit in bits:
            bit = bit.strip()
            if not bit:
                continue
            if cur and token_count(cur + " " + bit) > 140:
                units.append(cur.strip())
                cur = bit
            else:
                cur = (cur + " " + bit).strip() if cur else bit
        if cur:
            units.append(cur.strip())
    if not units and text:
        units = [text]
    return units


def split_long_unit(unit: str, chunk_tokens: int) -> list[str]:
    words = re.findall(r"\S+", unit)
    if len(words) <= chunk_tokens:
        return [unit.strip()]
    return [" ".join(words[i : i + chunk_tokens]).strip() for i in range(0, len(words), chunk_tokens)]


def chunk_clean_text(text: str, *, chunk_tokens: int = 500) -> list[str]:
    chunk_tokens = max(100, int(chunk_tokens))
    hard_max = int(chunk_tokens * 1.12)
    chunks: list[str] = []
    cur_parts: list[str] = []
    cur_tokens = 0
    for unit in sentence_units(text):
        for piece in split_long_unit(unit, chunk_tokens):
            n = token_count(piece)
            if cur_parts and cur_tokens + n > hard_max:
                chunks.append(" ".join(cur_parts).strip())
                cur_parts, cur_tokens = [], 0
            cur_parts.append(piece)
            cur_tokens += n
            if cur_tokens >= chunk_tokens:
                chunks.append(" ".join(cur_parts).strip())
                cur_parts, cur_tokens = [], 0
    if cur_parts:
        chunks.append(" ".join(cur_parts).strip())
    return [c for c in chunks if token_count(c) > 0]


def whole_text_for_chunking(book: indy_reads.Book) -> tuple[str, str]:
    path = Path(book.path)
    if book.ext == ".pdf":
        if not shutil.which("pdftotext"):
            raise RuntimeError("pdftotext missing")
        cp = subprocess.run(["pdftotext", str(path), "-"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if cp.returncode != 0:
            raise RuntimeError(cp.stderr.strip() or "pdftotext failed")
        return cp.stdout, "pdftotext-full"
    return indy_reads.whole_text_for_book(path)


def system_prompt() -> str:
    return (
        "You are INDY_READs reading a clean 500-token book chunk through LUCIDOTA. "
        "Create bounded reading cards: claims, motifs, entities, contradictions, questions, and chunk refs. "
        "Do not invent page facts. Exact passages live in RAG/Postgres/receipts; use chunk_ref."
    )


def assistant_payload(card_type: str, book: indy_reads.Book, chunk_ref: str, parsed: dict[str, Any]) -> str:
    terms = parsed.get("terms", [])[:8]
    notes = parsed.get("notes", [])[:4]
    if card_type == "go25_packet":
        obj = {"chunk_ref": chunk_ref, "go_terms": terms, "claim_lifecycle": parsed.get("claim_lifecycle"), "confidence_bps": parsed.get("confidence_bps"), "questions": ["What does the next chunk confirm or contradict?"]}
    elif card_type == "chapter_map":
        obj = {"chunk_ref": chunk_ref, "map": notes, "next_reads": [f"{book.id}.cNEXT"], "uncertainty": "chunk-limited; needs retrieval for exact continuity"}
    elif card_type == "entity_claim_extract":
        obj = {"chunk_ref": chunk_ref, "entities": [], "claims": notes[:2], "evidence_refs": [chunk_ref], "status": "NEEDS_RETRIEVAL_FOR_QUOTES"}
    elif card_type == "motif_questions":
        obj = {"chunk_ref": chunk_ref, "motifs": terms[:5], "unresolved_questions": ["Which motif recurs later?", "What evidence should be retrieved next?"]}
    else:
        obj = {"chunk_ref": chunk_ref, "style_profile": "modern_60_oldtimey_40", "voice_note": "clear modern reading discipline with a light old-timey edge", "do_not_quote_from_memory": True}
    return jdump(obj)


def make_card_from_chunk(book: indy_reads.Book, chunk: dict[str, Any], parsed: dict[str, Any], card_type: str) -> dict[str, Any]:
    excerpt = short_text(chunk.get("text", ""), 1400)
    user = (
        f"BOOK={book.name}\nCHUNK_REF={chunk['chunk_ref']}\n"
        f"TEXT_EXCERPT={excerpt}\nTASK={card_type}: make a reading card."
    )
    return {
        "schema": "lucidota.book_reader_lora.card.v2",
        "adapter_kind": "BOOK_READER_LORA",
        "book_id": book.id,
        "book_name": book.name,
        "chunk_ref": chunk["chunk_ref"],
        "card_type": card_type,
        "source_sha256": chunk.get("source_sha256"),
        "text_sha256": chunk.get("text_sha256"),
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_payload(card_type, book, chunk["chunk_ref"], parsed)},
        ],
    }


def book_chart(books: list[indy_reads.Book]) -> dict[str, Any]:
    by_ext: dict[str, int] = {}
    total = 0
    rows = []
    for b in books:
        by_ext[b.ext] = by_ext.get(b.ext, 0) + 1
        total += b.size_bytes
        rows.append({"id": b.id, "name": b.name, "ext": b.ext, "size_bytes": b.size_bytes, "path": b.path, "sha256": sha_file(Path(b.path))})
    return {
        "schema": "lucidota.indy_reads.book_chart.v1",
        "generated_at": now_z(),
        "book_count": len(books),
        "actual_book_file_count": sum(1 for b in books if b.ext != ".md"),
        "context_pack_count": sum(1 for b in books if b.ext == ".md"),
        "total_size_bytes": total,
        "by_extension": by_ext,
        "books": rows,
    }


def build_book_chunks(books: list[indy_reads.Book], *, chunk_tokens: int, max_chunks_per_book: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    extraction_errors: list[dict[str, Any]] = []
    for b in books:
        try:
            raw, method = whole_text_for_chunking(b)
            cleaned = clean_book_text(raw)
            chunks = chunk_clean_text(cleaned, chunk_tokens=chunk_tokens)
        except Exception as exc:
            extraction_errors.append({"book_id": b.id, "error": str(exc)[:300]})
            continue
        if max_chunks_per_book > 0:
            chunks = chunks[:max_chunks_per_book]
        source_sha = sha_file(Path(b.path))
        for idx, text in enumerate(chunks, 1):
            text_hash = sha_text(text)
            rows.append(
                {
                    "schema": "lucidota.book_reader_lora.chunk.v1",
                    "book_id": b.id,
                    "book_name": b.name,
                    "book_path": b.path,
                    "chunk_index": idx,
                    "chunk_ref": f"{b.id}.c{idx:04d}",
                    "text": text,
                    "token_count": token_count(text),
                    "char_count": len(text),
                    "text_sha256": text_hash,
                    "source_sha256": source_sha,
                    "extract_method": method,
                    "cleaning_policy": "clean_book_text:v1 page_marker_drop dehyphenate whitespace_normalize",
                    "created_at": now_z(),
                    "do_not_infer_beyond_chunk": True,
                }
            )
    return rows, extraction_errors


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(jdump(r) + "\n" for r in rows), encoding="utf-8")


def build_adapter_manifest(target: str, output_root: Path, train_path: Path, val_path: Path, chunks_path: Path, embedding_manifest_path: Path, chunk_tokens: int) -> dict[str, Any]:
    cfg = ADAPTER_TARGETS[target]
    return {
        "schema": "lucidota.book_reader_lora.target_adapter_manifest.v1",
        "adapter_kind": "BOOK_READER_LORA",
        "adapter_target": target,
        "target_model_id": cfg["target_model_id"],
        "runtime_lane": cfg["runtime_lane"],
        "status": "STAGED_DATASET_NOT_TRAINED",
        "requires_rag": True,
        "lora_teaches": "reading_behavior_not_exact_passage_memory",
        "rag_keeps": ["exact_text", "quotes", "chunk_refs", "evidence", "citations", "embeddings"],
        "chunk_tokens": chunk_tokens,
        "chunk_source": rel(chunks_path),
        "embedding_manifest": rel(embedding_manifest_path),
        "train_path": rel(train_path),
        "val_path": rel(val_path),
        "recommended_lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "task_type": "CAUSAL_LM"},
        "adapter_output": cfg["adapter_output"],
        "local_gguf": cfg.get("local_gguf", ""),
        "compatibility_truth": "Same book cards can feed all targets, but each target needs a compatible base/export/eval. Do not reuse Talkie LoRA weights on Bonsai.",
        "notes": cfg["notes"],
    }


def build_ontology_adapter_feasibility(output_root: Path, chunks_path: Path, train_path: Path, val_path: Path) -> dict[str, Any]:
    return {
        "schema": "lucidota.book_reader_lora.ontology_adapter_feasibility.v1",
        "generated_at": now_z(),
        "ontology_curricula": {
            "GO25": {
                "status": "STAGED_FROM_READING_CARDS",
                "role": "fast GO-25 packetization / claims / evidence / uncertainty",
            },
            "GCI_O_75": {
                "status": "PLANNED_CURRICULUM_LAYER",
                "role": "75-term richer routing and contradiction lattice; source ontology needs canonical file before training.",
            },
            "O414": {
                "status": "PLANNED_CURRICULUM_LAYER",
                "role": "414-depth novel-like ontology reading lens; Indy_READs should learn it as staged curriculum, not raw memorization.",
            },
        },
        "model_lanes": {
            "talkie": {
                "lora_status": "STAGED_TARGET",
                "target_manifest": rel(output_root / "adapter_targets" / "talkie" / "adapter_manifest.json"),
            },
            "bonsai8b_q1": {
                "lora_status": "STAGED_TARGET",
                "target_manifest": rel(output_root / "adapter_targets" / "bonsai8b_q1" / "adapter_manifest.json"),
            },
            "bonsai8b_q2": {
                "lora_status": "STAGED_TARGET",
                "target_manifest": rel(output_root / "adapter_targets" / "bonsai8b_q2" / "adapter_manifest.json"),
            },
            "needle": {
                "lora_status": "POSSIBLE_IF_TRAINABLE_TRANSFORMER_BASE",
                "truth": "If Needle is a trainable transformer/HF-compatible model, LoRA can be trained; current local Needle worker is a small routed service, not proven LoRA-loadable runtime.",
                "recommended_first_move": "Train ontology behavior cards for the parent/compatible Needle base, then prove load/eval before claiming hot Needle LoRA.",
            },
            "mamba": {
                "lora_status": "ARCHITECTURE_SPECIFIC_UNVERIFIED_RUNTIME",
                "truth": "Mamba-family adapters are possible in some training stacks, but target modules/runtime loading differ from transformer q_proj/v_proj LoRA and GGUF hot-load support is unverified here.",
                "recommended_first_move": "Stage ontology cards and run a separate Mamba PEFT/adapter feasibility probe on RunPod before promising runtime LoRA.",
            },
        },
        "shared_sources": {
            "chunks": rel(chunks_path),
            "train_cards": rel(train_path),
            "val_cards": rel(val_path),
        },
        "training_law": "Ontology adapters teach routing/reading behavior. Exact ontology definitions and book text stay in RAG/Postgres/receipts.",
    }


def stage_books(
    output_root: Path,
    max_pages_per_book: int,
    cards_per_page: int,
    *,
    chunk_tokens: int = 500,
    max_books: int = 0,
    max_chunks_per_book: int = 0,
    cards_per_chunk: int | None = None,
    embed: bool = False,
) -> dict[str, Any]:
    books = indy_reads.library()
    if max_books > 0:
        books = books[:max_books]
    output_root.mkdir(parents=True, exist_ok=True)
    for sub in ["raw", "custody", "chunks", "cards", "adapter", "adapter_targets", "embeddings", "receipts"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    chart = book_chart(books)
    (output_root / "book_chart.json").write_text(json.dumps(chart, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    # Back-compat: old CLI said pages/cards, new pipeline is chunks/cards.
    if max_chunks_per_book == 0:
        max_chunks_per_book = max(1, max_pages_per_book)
    if cards_per_chunk is None:
        cards_per_chunk = cards_per_page

    chunks, extraction_errors = build_book_chunks(books, chunk_tokens=chunk_tokens, max_chunks_per_book=max_chunks_per_book)
    chunks_path = output_root / "chunks" / f"chunks_{chunk_tokens}tok.jsonl"
    canonical_chunks_path = output_root / "chunks" / "chunks_500tok.jsonl"
    write_jsonl(chunks_path, chunks)
    if chunks_path != canonical_chunks_path:
        write_jsonl(canonical_chunks_path, chunks)

    # Build cards from chunks. Reuse INDY's fast parser by creating page-like packets.
    book_by_id = {b.id: b for b in books}
    cards: list[dict[str, Any]] = []
    for chunk in chunks:
        b = book_by_id[chunk["book_id"]]
        page_like = {
            "book_id": chunk["book_id"],
            "book_name": chunk["book_name"],
            "book_path": chunk["book_path"],
            "page": chunk["chunk_index"],
            "text": chunk["text"],
            "page_hash": chunk["text_sha256"],
            "source_sha256": chunk["source_sha256"],
            "extract_method": chunk["extract_method"],
            "chars": chunk["char_count"],
        }
        parsed = indy_reads.fast_parse(page_like)
        for card_type in CARD_TYPES[: max(1, min(cards_per_chunk, len(CARD_TYPES)) )]:
            cards.append(make_card_from_chunk(b, chunk, parsed, card_type))

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    for i, card in enumerate(cards):
        (val if i % 5 == 0 else train).append(card)
    if cards and not train:
        train, val = cards, []

    train_path = output_root / "cards" / "reading_cards.train.jsonl"
    val_path = output_root / "cards" / "reading_cards.val.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    embeddings_path = output_root / "embeddings" / "chunk_embeddings.jsonl"
    if embed and chunks:
        embed_receipt = embedding_provider.embed_file(chunks_path, embeddings_path, prefer_groq=False)
        embedding_status = "EMBEDDED" if embed_receipt["stats"].get("embedded_local", 0) + embed_receipt["stats"].get("embedded_groq", 0) == len(chunks) else "PARTIAL"
    else:
        embeddings_path.write_text("", encoding="utf-8")
        embed_receipt = {
            "schema": "lucidota.embedding_provider.embed_file.v1",
            "generated_at_utc": now_z(),
            "input_path": str(chunks_path),
            "output_path": str(embeddings_path),
            "rows_written": 0,
            "stats": {"seen": len(chunks), "embedded_groq": 0, "embedded_local": 0, "blocked": 0, "failed": 0, "skipped": len(chunks)},
            "groq_receipts": [],
        }
        embedding_status = "QUEUED_NOT_EMBEDDED"
    embedding_manifest_path = output_root / "embeddings" / "embedding_manifest.json"
    embedding_manifest = {
        "schema": "lucidota.book_reader_lora.embedding_manifest.v1",
        "tool": "scripts.embedding_provider",
        "policy": "scheduled_aux_tool_not_hot_resident; local deterministic fallback used when --embed and no external key",
        "status": embedding_status,
        "input_chunks": rel(chunks_path),
        "output_path": rel(embeddings_path),
        "rows_written": embed_receipt.get("rows_written", 0),
        "stats": embed_receipt.get("stats", {}),
        "max_batch_policy": "bounded; do not import torch/embedder into main LUCI daemon",
    }
    embedding_manifest_path.write_text(json.dumps(embedding_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    target_manifests = {}
    for target in ADAPTER_TARGETS:
        m = build_adapter_manifest(target, output_root, train_path, val_path, chunks_path, embedding_manifest_path, chunk_tokens)
        target_manifests[target] = m
        target_dir = output_root / "adapter_targets" / target
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "adapter_manifest.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    ontology_manifest = build_ontology_adapter_feasibility(output_root, chunks_path, train_path, val_path)
    ontology_manifest_path = output_root / "ontology_adapter_feasibility_manifest.json"
    ontology_manifest_path.write_text(json.dumps(ontology_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    adapter_manifest = {
        "schema": "lucidota.book_reader_lora.adapter_manifest.v2",
        "adapter_kind": "BOOK_READER_LORA",
        "status": "STAGED_DATASET_NOT_TRAINED",
        "requires_rag": True,
        "lora_teaches": "reading_behavior_not_exact_passage_memory",
        "rag_keeps": ["exact_text", "quotes", "chunk_refs", "evidence", "citations", "embeddings"],
        "target_tasks": CARD_TYPES,
        "chunk_tokens": chunk_tokens,
        "chunk_source": rel(chunks_path),
        "embedding_manifest": rel(embedding_manifest_path),
        "adapter_targets": sorted(ADAPTER_TARGETS),
        "target_manifests": {k: rel(output_root / "adapter_targets" / k / "adapter_manifest.json") for k in ADAPTER_TARGETS},
        "ontology_adapter_feasibility_manifest": rel(ontology_manifest_path),
        "recommended_lora": {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "task_type": "CAUSAL_LM"},
        "train_path": rel(train_path),
        "val_path": rel(val_path),
        "base_model_policy": "Train separate adapters against compatible bases; convert adapter to GGUF only after eval receipt.",
    }
    (output_root / "adapter" / "adapter_manifest.json").write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    receipt = {
        "schema": "lucidota.book_reader_lora.stage_receipt.v1",
        "status": "PASS" if cards and chunks else "FAIL",
        "generated_at": now_z(),
        "output_root": str(output_root),
        "book_count": chart["book_count"],
        "actual_book_file_count": chart["actual_book_file_count"],
        "context_pack_count": chart["context_pack_count"],
        "chunk_tokens": chunk_tokens,
        "chunks_written": len(chunks),
        "chunks_path": str(chunks_path),
        "cards_written": len(cards),
        "train_cards": len(train),
        "val_cards": len(val),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "chart_path": str(output_root / "book_chart.json"),
        "adapter_manifest_path": str(output_root / "adapter" / "adapter_manifest.json"),
        "adapter_targets": sorted(ADAPTER_TARGETS),
        "embedding_status": embedding_status,
        "embedding_manifest_path": str(embedding_manifest_path),
        "ontology_adapter_feasibility_manifest_path": str(ontology_manifest_path),
        "extraction_errors": extraction_errors,
    }
    (output_root / "receipts" / "stage_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def talkie_custody(output: Path) -> dict[str, Any]:
    repo = ROOT / "01_REPOS" / "talkie"
    head = git_output(["rev-parse", "HEAD"], repo)
    model_py = repo / "src" / "talkie" / "model.py"
    data = {
        "schema": "lucidota.talkie.source_custody.v1",
        "recorded_at": now_z(),
        "github_url": "https://github.com/talkie-lm/talkie",
        "local_repo": "01_REPOS/talkie",
        "git_head": head,
        "license": (repo / "LICENSE").read_text(encoding="utf-8", errors="ignore").splitlines()[0] if (repo / "LICENSE").exists() else "Apache-2.0",
        "architecture": {
            "class": "decoder_only_gpt",
            "n_layer": 40,
            "n_head": 40,
            "n_embd": 5120,
            "head_dim": 128,
            "activation": "SwiGLU/Silu gate per source model.py",
            "source_file": "01_REPOS/talkie/src/talkie/model.py",
            "source_sha256": sha_file(model_py),
        },
        "models": {
            "instruction_tuned": "talkie-lm/talkie-1930-13b-it",
            "checkpoint_filename": "rl-refined.pt",
            "local_checkpoint": "03_VAULT/models/talkie-lm/talkie-1930-13b-it/rl-refined.pt",
        },
        "runtime_truth": {
            "bf16_vram_requirement_gb": 28,
            "local_8gb_hot_runtime": False,
            "runpod_or_forge_lane": True,
            "note": "Official README requires CUDA GPU with >=28GB VRAM for bf16 inference; local path is source custody/staging, not hot runtime."
        }
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="book-reader-lora-stage")
    ap.add_argument("--output-root", default="04_RUNTIME/BOOK_READER_LORA")
    ap.add_argument("--max-pages-per-book", type=int, default=1, help="Back-compat alias for max chunks per book when --max-chunks-per-book is omitted.")
    ap.add_argument("--cards-per-page", type=int, default=5, help="Back-compat alias for cards per chunk when --cards-per-chunk is omitted.")
    ap.add_argument("--chunk-tokens", type=int, default=500)
    ap.add_argument("--max-books", type=int, default=0)
    ap.add_argument("--max-chunks-per-book", type=int, default=0)
    ap.add_argument("--cards-per-chunk", type=int, default=None)
    ap.add_argument("--embed", action="store_true", help="Embed chunk rows through governed embedding_provider local fallback; no hot resident model import.")
    ap.add_argument("--json", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("talkie-custody")
    c.add_argument("--output", default="05_OUTPUTS/model_runtime/talkie_source_custody.json")
    c.add_argument("--json", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.cmd == "talkie-custody":
        data = talkie_custody(Path(args.output))
        print(json.dumps(data, sort_keys=True) if args.json else json.dumps(data, indent=2, sort_keys=True))
        return 0
    receipt = stage_books(
        Path(args.output_root),
        args.max_pages_per_book,
        args.cards_per_page,
        chunk_tokens=args.chunk_tokens,
        max_books=args.max_books,
        max_chunks_per_book=args.max_chunks_per_book,
        cards_per_chunk=args.cards_per_chunk,
        embed=args.embed,
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True) if args.json else json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if receipt.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
