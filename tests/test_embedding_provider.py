from __future__ import annotations

from types import SimpleNamespace


def test_probe_without_groq_is_blocked(monkeypatch):
    import scripts.embedding_provider as ep

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("LUCIDOTA_GROQ_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GROQ_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)

    report = ep.probe()

    assert report["schema"] == "lucidota.embedding_provider.probe.v1"
    assert report["verdict"] == "BLOCKED"
    assert report["blocked_gap"] == "GROQ_EMBEDDING_NOT_CONFIGURED"
    assert report["provider"] == "local"


def test_empty_text_is_blocked_and_not_embedded(monkeypatch):
    import scripts.embedding_provider as ep

    result = ep.embed_text("", source_path="example.py", chunk_id="c1")

    assert result.row["status"] == "BLOCKED"
    assert result.row["provider"] == "blocked"
    assert result.row["vector"] == []
    assert result.row["dimensions"] == 0


def test_groq_configured_mock_returns_embedding_receipt(monkeypatch, tmp_path):
    import scripts.embedding_provider as ep

    monkeypatch.setattr(ep, "ROOT", tmp_path)
    monkeypatch.setattr(ep, "MODEL_AUDIT_DIR", tmp_path / "05_OUTPUTS" / "model_invocation_audits")
    monkeypatch.setattr(
        ep,
        "groq_embedding_config",
        lambda: {
            "configured": True,
            "api_key_present": True,
            "model": "groq/test-embed",
            "base_url": "https://api.groq.com/openai/v1",
            "reason": None,
        },
    )
    monkeypatch.setattr(ep, "_groq_embed", lambda text, *, model, api_key, base_url: [0.25, 0.75])
    monkeypatch.setattr(ep, "_env", lambda *names: "fake-key")

    result = ep.embed_text("hello world", source_path="example.py", chunk_id="c2")

    assert result.row["status"] == "EMBEDDED"
    assert result.row["provider"] == "groq"
    assert result.row["model"] == "groq/test-embed"
    assert result.row["dimensions"] == 2
    assert result.row["vector"] == [0.25, 0.75]
    assert result.row["receipt_ref"].startswith("05_OUTPUTS/model_invocation_audits/")
    assert result.groq_receipt["provider"] == "groq"
    assert result.groq_receipt["dimensions"] == 2
