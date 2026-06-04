from __future__ import annotations

import json
import base64
from pathlib import Path

from importlib import import_module

mod = import_module("scripts.runpod_artifact_pull_import")


class _FakeResponse:
    def __init__(self, payload: bytes, status_code: int = 200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = payload.decode("utf-8", errors="replace")
        self.headers = headers or {}

    def iter_content(self, chunk_size: int):
        yield self._payload

    def json(self):
        return json.loads(self.text)


def test_build_contents_url_encodes_nested_artifact_path():
    assert mod.build_contents_url("https://example.proxy", "/workspace/talkie/rl-refined.pt") == (
        "https://example.proxy/api/contents/workspace/talkie/rl-refined.pt?download=1"
    )


def test_pull_artifact_writes_file_and_verifies_size_and_sha256(tmp_path):
    out = tmp_path / "artifact.bin"
    payload = b"runpod artifact body"
    res = mod.pull_artifact(
        jupyter_url="https://example",
        token="abc",
        remote_path="a/b.bin",
        output=out,
        expected_size=len(payload),
        expected_sha256=mod.sha256_bytes(payload),
        request_get=lambda *_args, **_kwargs: _FakeResponse(payload),
    )

    assert res["ok"] is True
    assert res["observed_size"] == len(payload)
    assert res["observed_sha256"] == mod.sha256_bytes(payload)
    assert out.read_bytes() == payload


def test_pull_artifact_reports_size_mismatch(tmp_path):
    out = tmp_path / "artifact.bin"
    payload = b"1234"
    res = mod.pull_artifact(
        jupyter_url="https://example",
        token=None,
        remote_path="a/b.bin",
        output=out,
        expected_size=99,
        request_get=lambda *_args, **_kwargs: _FakeResponse(payload),
    )

    assert res["ok"] is False
    assert res["status"] == "SIZE_MISMATCH"


def test_pull_artifact_decodes_jupyter_contents_json_envelope(tmp_path):
    out = tmp_path / "artifact.jsonl"
    body = b'{"id":"a","embedding":[1,2,3]}\n'
    envelope = json.dumps(
        {
            "name": "artifact.jsonl",
            "path": "workspace/artifact.jsonl",
            "type": "file",
            "format": "base64",
            "content": base64.b64encode(body).decode("ascii"),
        }
    ).encode("utf-8")

    res = mod.pull_artifact(
        jupyter_url="https://example",
        token="abc",
        remote_path="workspace/artifact.jsonl",
        output=out,
        expected_size=len(body),
        expected_sha256=mod.sha256_bytes(body),
        request_get=lambda *_args, **_kwargs: _FakeResponse(
            envelope, headers={"content-type": "application/json"}
        ),
    )

    assert res["ok"] is True
    assert res["observed_size"] == len(body)
    assert out.read_bytes() == body


def test_build_import_plan_reads_csv_header_and_generates_sql(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("id,payload\n1,a\n2,b\n", encoding="utf-8")
    plan = mod.build_import_plan(output=csv_path, table="public.runpod_rows")
    assert plan["ok"] is True
    assert plan["columns"] == ["id", "payload"]
    assert "\\copy" in plan["copy_stmt"]
    assert "ON CONFLICT (id) DO" in plan["upsert_stmt"]


def test_dry_run_mode_generates_import_plan_without_db_credentials(monkeypatch, tmp_path):
    output = tmp_path / "artifact.csv"
    output.write_text("id,payload\n1,hello\n", encoding="utf-8")

    def _fake_pull(**kwargs):
        return {
            "ok": True,
            "status": "PULLED",
            "remote_path": kwargs["remote_path"],
            "output": str(output),
            "observed_size": 0,
            "expected_size": None,
            "observed_sha256": "dummy",
            "expected_sha256": None,
            "sha256_ok": True,
            "size_ok": True,
        }

    monkeypatch.setattr(mod, "pull_artifact", _fake_pull)
    receipt_path = tmp_path / "runpod_receipt.json"
    argv = [
        "--remote-path",
        "workspace/talkie.bin",
        "--output",
        str(output),
        "--jupyter-url",
        "https://proxy",
        "--import-table",
        "public.artifact_rows",
        "--import-columns",
        "id",
        "payload",
        "--dry-run",
        "--receipt",
        str(receipt_path),
        "--json",
    ]
    code = mod.main(argv)
    assert code == 0
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["import_plan"]["ok"] is True
    assert payload["execution_status"] is None
