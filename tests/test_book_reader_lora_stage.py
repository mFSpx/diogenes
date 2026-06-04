import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path("scripts/book_reader_lora_stage.py")


def test_book_reader_lora_stage_counts_books_and_builds_cards(tmp_path):
    out = tmp_path / "BOOK_LORA"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(out),
            "--max-pages-per-book",
            "1",
            "--cards-per-page",
            "4",
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["schema"] == "lucidota.book_reader_lora.stage_receipt.v1"
    assert receipt["book_count"] == 7
    assert receipt["actual_book_file_count"] == 6
    assert receipt["context_pack_count"] == 1
    assert receipt["cards_written"] >= 7 * 4
    assert receipt["train_path"].endswith("reading_cards.train.jsonl")
    assert receipt["val_path"].endswith("reading_cards.val.jsonl")

    chart = json.loads((out / "book_chart.json").read_text(encoding="utf-8"))
    assert chart["book_count"] == 7
    assert {".epub", ".mobi", ".pdf", ".md"}.issubset(chart["by_extension"])

    adapter_manifest = json.loads((out / "adapter" / "adapter_manifest.json").read_text(encoding="utf-8"))
    assert adapter_manifest["adapter_kind"] == "BOOK_READER_LORA"
    assert adapter_manifest["requires_rag"] is True
    assert adapter_manifest["lora_teaches"] == "reading_behavior_not_exact_passage_memory"

    train_rows = [json.loads(line) for line in (out / "cards" / "reading_cards.train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert train_rows
    row = train_rows[0]
    assert "messages" in row
    assert row["card_type"] in {"go25_packet", "chapter_map", "entity_claim_extract", "motif_questions", "style_voice"}
    assert row["chunk_ref"]
    assert len(json.dumps(row)) < 12000


def test_book_reader_lora_stage_builds_clean_500_token_chunks_embeddings_and_three_targets(tmp_path):
    out = tmp_path / "BOOK_LORA"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output-root",
            str(out),
            "--max-books",
            "2",
            "--max-chunks-per-book",
            "2",
            "--chunk-tokens",
            "500",
            "--cards-per-chunk",
            "3",
            "--embed",
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout)
    assert receipt["chunk_tokens"] == 500
    assert receipt["chunks_written"] >= 2
    assert receipt["embedding_status"] == "EMBEDDED"
    assert set(receipt["adapter_targets"]) == {"talkie", "bonsai8b_q1", "bonsai8b_q2"}

    chunks_path = out / "chunks" / "chunks_500tok.jsonl"
    assert chunks_path.exists()
    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(chunks) == receipt["chunks_written"]
    for chunk in chunks:
        assert chunk["schema"] == "lucidota.book_reader_lora.chunk.v1"
        assert ".c" in chunk["chunk_ref"]
        assert 1 <= chunk["token_count"] <= 560
        assert "\f" not in chunk["text"]
        assert not chunk["text"].startswith(("Page ", "PAGE "))
        assert chunk["text_sha256"]

    embeddings_path = out / "embeddings" / "chunk_embeddings.jsonl"
    embeddings = [json.loads(line) for line in embeddings_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(embeddings) == len(chunks)
    assert all(row["status"] == "EMBEDDED" for row in embeddings)
    assert all(row["dimensions"] > 0 for row in embeddings)

    embedding_manifest = json.loads((out / "embeddings" / "embedding_manifest.json").read_text(encoding="utf-8"))
    assert embedding_manifest["tool"] == "scripts.embedding_provider"
    assert embedding_manifest["input_chunks"].endswith("chunks_500tok.jsonl")
    assert embedding_manifest["rows_written"] == len(chunks)

    for target in ["talkie", "bonsai8b_q1", "bonsai8b_q2"]:
        manifest = json.loads((out / "adapter_targets" / target / "adapter_manifest.json").read_text(encoding="utf-8"))
        assert manifest["adapter_target"] == target
        assert manifest["adapter_kind"] == "BOOK_READER_LORA"
        assert manifest["chunk_source"].endswith("chunks_500tok.jsonl")
        assert manifest["requires_rag"] is True

    ontology = json.loads((out / "ontology_adapter_feasibility_manifest.json").read_text(encoding="utf-8"))
    assert {"GO25", "GCI_O_75", "O414"}.issubset(set(ontology["ontology_curricula"]))
    assert ontology["model_lanes"]["talkie"]["lora_status"] == "STAGED_TARGET"
    assert ontology["model_lanes"]["needle"]["lora_status"] == "POSSIBLE_IF_TRAINABLE_TRANSFORMER_BASE"
    assert ontology["model_lanes"]["mamba"]["lora_status"] == "ARCHITECTURE_SPECIFIC_UNVERIFIED_RUNTIME"


def test_talkie_source_custody_receipt_records_architecture_and_runpod_truth(tmp_path):
    out = tmp_path / "talkie_source_custody.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "talkie-custody",
            "--output",
            str(out),
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["schema"] == "lucidota.talkie.source_custody.v1"
    assert data["github_url"] == "https://github.com/talkie-lm/talkie"
    assert data["local_repo"] == "01_REPOS/talkie"
    assert data["architecture"]["n_layer"] == 40
    assert data["architecture"]["n_head"] == 40
    assert data["architecture"]["n_embd"] == 5120
    assert data["runtime_truth"]["bf16_vram_requirement_gb"] >= 28
    assert data["runtime_truth"]["local_8gb_hot_runtime"] is False
    assert out.exists()
