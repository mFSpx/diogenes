from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import lucidota_anthropic_llama_proxy as proxy  # noqa: E402


def test_anthropic_proxy_targets_local_llama_server():
    assert proxy.UPSTREAM.startswith("http://127.0.0.1:8080/")
    assert "DeepSeek" in proxy.MODEL
    routed = proxy.anthropic_to_openai({"messages": []})
    assert routed["model"] == proxy.MODEL
    assert routed["messages"][0]["role"] == "user"
    assert routed["stream"] is False


def test_anthropic_proxy_start_script_points_at_local_upstream_only():
    script = Path("scripts/lucidota_start_anthropic_proxy.sh").read_text(encoding="utf-8")
    assert "LUCIDOTA_DEEPSEEK_API_BASE" in script
    assert "LUCIDOTA_ANTHROPIC_API_KEY" not in script
    assert "curl -fsS \"http://127.0.0.1:${PORT}/health\"" in script
    assert "scripts/lucidota_anthropic_llama_proxy.py" in script
