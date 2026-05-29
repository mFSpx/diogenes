from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


def test_safe_probe_uses_pytest_collect_only_for_test_files(monkeypatch, tmp_path):
    import scripts.code_korpus_ingest as ingest

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(ingest.subprocess, "run", fake_run)

    result = ingest.safe_probe(tmp_path / "test_sample.py", "python", {"cli_help_available": False})

    assert result["runtime_probe"] == "imports_ok"
    assert calls[0][:4] == [sys.executable, "-m", "pytest", "--collect-only"]


def test_main_counts_groq_embeddings_and_writes_receipts(monkeypatch, tmp_path):
    import scripts.code_korpus_ingest as ingest

    root = tmp_path
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    sample = root / "scripts" / "sample.py"
    sample.write_text(
        "import json\n\n"
        "def score_case(value):\n"
        "    return value + 1\n\n"
        "if __name__ == '__main__':\n"
        "    print(score_case(1))\n",
        encoding="utf-8",
    )
    (root / "tests").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "test_sample.py").write_text("def test_example():\n    assert True\n", encoding="utf-8")

    fake_provider = SimpleNamespace(
        probe=lambda: {
            "schema": "lucidota.embedding_provider.probe.v1",
            "generated_at_utc": "2026-05-28T00:00:00Z",
            "verdict": "PASS",
            "provider": "groq",
            "groq": {"configured": True, "api_key_present": True, "model": "groq/test-embed", "base_url": "https://api.groq.com/openai/v1", "reason": None},
            "local": {"configured": True, "model": "lucidota.deterministic-hash-384", "dimensions": 384, "provider": "local"},
            "blocked_gap": None,
        },
        embed_text=lambda text, *, source_path="", chunk_id="", prefer_groq=True: SimpleNamespace(
            row={
                "schema": "lucidota.embedding.row.v1",
                "source_path": source_path,
                "chunk_id": chunk_id,
                "provider": "groq",
                "model": "groq/test-embed",
                "dimensions": 2,
                "vector": [0.1, 0.2],
                "text_sha256": "abc123",
                "created_at_utc": "2026-05-28T00:00:00Z",
                "receipt_ref": "05_OUTPUTS/model_invocation_audits/groq_embedding_fake.json",
                "status": "EMBEDDED",
                "error": None,
            },
            groq_receipt={"provider": "groq", "dimensions": 2},
        ),
    )

    monkeypatch.setitem(sys.modules, "embedding_provider", fake_provider)
    monkeypatch.setattr(ingest, "ROOT", root)
    monkeypatch.setattr(ingest, "OUT", root / "05_OUTPUTS" / "code_korpus")
    monkeypatch.setattr(ingest, "RUNTIME", root / "04_RUNTIME" / "code_korpus")
    monkeypatch.setattr(ingest, "DEFAULT_INVENTORY", root / "missing_inventory.jsonl")
    monkeypatch.setattr(sys, "argv", ["code_korpus_ingest.py", "--root", ".", "--batch-size", "10", "--execute", "--json"])

    rc = ingest.main()
    assert rc == 0

    receipt_files = sorted((root / "05_OUTPUTS" / "code_korpus").glob("code_ingest_receipt_*.json"))
    assert receipt_files, "expected a code ingest receipt"
    receipt = json.loads(receipt_files[-1].read_text(encoding="utf-8"))
    assert receipt["embedded_groq"] >= 1
    assert receipt["embedding_provider_probe"]["verdict"] == "PASS"
    assert receipt["graph_candidates"] >= 1
    assert receipt["canonical_graph_writes"] is False
