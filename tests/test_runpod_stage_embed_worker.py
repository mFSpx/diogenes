from __future__ import annotations

import json
from pathlib import Path

from importlib import import_module

mod = import_module("scripts.runpod_stage_embed_worker")


class _FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class _FakeSentenceTransformer:
    def __init__(self, model: str, device: str = "cpu"):
        self.model = model
        self.device = device

    def encode(self, texts, **kwargs):
        # Return deterministic pseudo-embeddings bounded to 4 dims for test visibility.
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), float(len(text) + 1), float(len(text) + 2), float(len(text) + 3)])
        return vectors


def _fake_st_ctor(model, device="cpu"):
    if model == "raise-me":
        raise RuntimeError("constructor failed")
    return _FakeSentenceTransformer(model=model, device=device)


def test_build_jupyter_contents_url_encodes_workspace_path():
    assert (
        mod.build_jupyter_contents_url("https://proxy.example", "/workspace/lucidota_ingest_accel/runpod_embed_worker.py")
        == "https://proxy.example/api/contents/workspace/lucidota_ingest_accel/runpod_embed_worker.py"
    )


def test_remote_worker_source_targets_ingest_accel_workspace():
    source = mod.build_remote_worker_script()
    assert "/workspace/lucidota_ingest_accel" in source
    assert "def run_worker(" in source
    assert "output_sha256" in source
    assert "db writes" in source.lower()
    assert "RUNPOD_EMBED_LOCAL_MODEL" in source
    assert "RUNPOD_EMBED_DEVICE" in source
    assert "sentence_transformers" in source
    assert "_encode_sentence_transformer_batch" in source


def test_stage_worker_uploads_script_via_jupyter_contents_api():
    calls: dict[str, object] = {}

    def fake_put(url, **kwargs):
        calls["url"] = url
        calls["payload"] = kwargs.get("json")
        return _FakeResponse({}, 201)

    result = mod.stage_worker(
        jupyter_url="https://proxy.local",
        jupyter_token="abc",
        remote_path="/workspace/lucidota_ingest_accel/runpod_embed_worker.py",
        request_put=fake_put,
    )

    assert result["status"] == "PASS"
    assert result["request_status"] == 201
    assert calls["url"] == "https://proxy.local/api/contents/workspace/lucidota_ingest_accel/runpod_embed_worker.py"
    payload = calls["payload"]
    assert payload["type"] == "file"
    assert payload["format"] == "text"
    assert payload["name"] == "runpod_embed_worker.py"
    assert payload["content"].startswith("#!/usr/bin/env python3")


def _write_chunks(path: Path) -> None:
    rows = [
        {"chunk_id": "c1", "text": "alpha beta", "source_path": "s1"},
        {"chunk_id": "c2", "text": "gamma delta", "source_path": "s2"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_run_local_worker_uses_deterministic_embeddings_when_model_not_configured(tmp_path):
    inp = tmp_path / "chunks.jsonl"
    out = tmp_path / "embeddings.jsonl"
    rcpt = tmp_path / "receipt.json"
    _write_chunks(inp)

    rcpt_payload = mod.run_local_worker(
        input_jsonl=inp,
        output_jsonl=out,
        receipt_path=rcpt,
        dimensions=8,
    )

    assert rcpt_payload["status"] == "PASS"
    assert rcpt_payload["model_requested"] is False
    assert rcpt_payload["model_used"] is False
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["provider"] == "deterministic" for row in rows)
    assert all(len(row["embedding"]) == 8 for row in rows)

    second = mod.run_local_worker(
        input_jsonl=inp,
        output_jsonl=tmp_path / "embeddings2.jsonl",
        receipt_path=tmp_path / "receipt2.json",
        dimensions=8,
    )
    rows2 = [json.loads(line) for line in (tmp_path / "embeddings2.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["embedding"] == rows2[0]["embedding"]


def test_run_local_worker_uses_chunk_ref_when_chunk_id_is_absent(tmp_path):
    inp = tmp_path / "chunks.jsonl"
    out = tmp_path / "embeddings.jsonl"
    rcpt = tmp_path / "receipt.json"
    inp.write_text(
        json.dumps({"chunk_ref": "book.c0001", "text": "chunk body", "book_path": "book.epub"}) + "\n",
        encoding="utf-8",
    )

    mod.run_local_worker(input_jsonl=inp, output_jsonl=out, receipt_path=rcpt, dimensions=4)

    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["chunk_id"] == "book.c0001"


def test_run_local_worker_uses_model_only_when_configured_and_falls_back_on_failure(tmp_path, monkeypatch):
    inp = tmp_path / "chunks.jsonl"
    out = tmp_path / "embeddings.jsonl"
    rcpt = tmp_path / "receipt.json"
    _write_chunks(inp)

    calls: dict[str, int] = {"count": 0}

    def fake_post(url, *args, **kwargs):
        calls["count"] += 1
        return _FakeResponse({"data": [{"embedding": [0.1, 0.2, 0.3, 0.4]}]}, status_code=200)

    rcpt_payload = mod.run_local_worker(
        input_jsonl=inp,
        output_jsonl=out,
        receipt_path=rcpt,
        dimensions=4,
        model_url="https://model.local",
        model_id="test-model",
        request_post=fake_post,
    )

    assert calls["count"] == 2
    assert rcpt_payload["model_requested"] is True
    assert rcpt_payload["model_used"] is True

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["provider"] == "model" for row in rows)
    assert all(row["model"] == "test-model" for row in rows)


def test_run_local_worker_model_failure_falls_back_to_placeholder(tmp_path):
    inp = tmp_path / "chunks.jsonl"
    out = tmp_path / "embeddings.jsonl"
    rcpt = tmp_path / "receipt.json"
    _write_chunks(inp)

    def fake_post(url, *args, **kwargs):
        return _FakeResponse({}, status_code=500)

    rcpt_payload = mod.run_local_worker(
        input_jsonl=inp,
        output_jsonl=out,
        receipt_path=rcpt,
        dimensions=4,
        model_url="https://model.local",
        model_id="bad-model",
        request_post=fake_post,
    )

    assert rcpt_payload["model_requested"] is True
    assert rcpt_payload["model_used"] is False
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["provider"] == "deterministic" for row in rows)
    assert all(row["model"] == "sha256-placeholder" for row in rows)
    assert rows[0]["error"] == "MODEL_MISSING_OR_FAILED"


def test_build_plan_includes_local_model_env_names():
    plan = mod.build_plan("https://proxy.local", "/workspace/lucidota_ingest_accel/runpod_embed_worker.py")
    assert "RUNPOD_EMBED_LOCAL_MODEL" in plan["model_env"]
    assert "RUNPOD_EMBED_DEVICE" in plan["model_env"]


def test_run_local_worker_uses_sentence_transformers_when_configured(tmp_path):
    inp = tmp_path / "chunks.jsonl"
    out = tmp_path / "embeddings.jsonl"
    rcpt = tmp_path / "receipt.json"
    _write_chunks(inp)

    rcpt_payload = mod.run_local_worker(
        input_jsonl=inp,
        output_jsonl=out,
        receipt_path=rcpt,
        dimensions=4,
        local_model="test-local-model",
        local_device="cpu",
        sentence_transformer_ctor=_fake_st_ctor,
    )

    assert rcpt_payload["model_requested"] is False
    assert rcpt_payload["model_used"] is True
    assert rcpt_payload["output_lines"] == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["provider"] == "sentence_transformers" for row in rows)
    assert all(row["model"] == "test-local-model" for row in rows)
    assert rows[0]["error"] is None


def test_run_local_worker_sentence_transformers_failure_falls_back_to_placeholder(tmp_path):
    inp = tmp_path / "chunks.jsonl"
    out = tmp_path / "embeddings.jsonl"
    rcpt = tmp_path / "receipt.json"
    _write_chunks(inp)

    rcpt_payload = mod.run_local_worker(
        input_jsonl=inp,
        output_jsonl=out,
        receipt_path=rcpt,
        dimensions=4,
        local_model="raise-me",
        local_device="cpu",
        sentence_transformer_ctor=_fake_st_ctor,
    )

    assert rcpt_payload["model_requested"] is False
    assert rcpt_payload["model_used"] is False
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert all(row["provider"] == "deterministic" for row in rows)
    assert rows[0]["error"] == "LOCAL_MODEL_MISSING_OR_FAILED"
