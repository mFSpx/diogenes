from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_line(stdout: str) -> dict:
    for line in reversed([line for line in stdout.splitlines() if line.strip()]):
        if line.lstrip().startswith("{"):
            return json.loads(line)
    raise AssertionError(stdout)


def test_delegate_normalizer_accepts_contract_outputs_and_aliases() -> None:
    from scripts import delegate_output_normalizer as normalizer

    payload = normalizer.normalize_output(
        '{"safe_patch_plan":["*** Begin Patch\\n*** Add File: x.txt\\n+ok\\n*** End Patch"],'
        '"commands_to_run":["python -m pytest -q tests/test_delegate_wrappers.py"],'
        '"next_small_step":"queue the focused test"}'
    )

    assert payload["status"] == "PASS"
    assert payload["patches"][0].startswith("*** Begin Patch")
    assert payload["commands"] == ["python -m pytest -q tests/test_delegate_wrappers.py"]
    assert payload["next_prompt"] == "queue the focused test"
    assert payload["rejection_reason"] is None


def test_delegate_normalizer_rejects_commentary_only_output() -> None:
    from scripts import delegate_output_normalizer as normalizer

    payload = normalizer.normalize_output('{"summary":"looks fine","findings":["ship it"]}')

    assert payload["status"] == "REJECTED"
    assert payload["rejection_reason"] == "commentary_only"
    assert payload["commentary_keys_present"] == ["findings", "summary"]


def test_vibe_delegate_fails_loudly_with_exact_missing_setup_fields(monkeypatch, capsys, tmp_path: Path) -> None:
    from scripts import vibe_delegate as delegate

    monkeypatch.setattr(delegate, "OUT", tmp_path / "out")
    monkeypatch.setattr(delegate, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(
        delegate,
        "discover_vibe_setup",
        lambda: {
            "configured": False,
            "cli_path": ".venv/bin/vibe",
            "config_path": str(Path.home() / ".vibe/config.toml"),
            "required_fields": ["active_model", "default_agent", "providers[].api_key_env_var", "MISTRAL_API_KEY"],
            "missing_setup_fields": ["MISTRAL_API_KEY"],
        },
    )

    assert delegate.main(["--task", "edit scripts/vibe_delegate.py", "--json"]) == 4
    out = capsys.readouterr().out
    assert "VIBE_DELEGATE_NOT_CONFIGURED" in out
    payload = _json_line(out)
    assert payload["status"] == "BLOCKED"
    assert payload["missing_setup_fields"] == ["MISTRAL_API_KEY"]


def test_vibe_delegate_executes_and_normalizes_output(monkeypatch, capsys, tmp_path: Path) -> None:
    from scripts import vibe_delegate as delegate

    monkeypatch.setattr(delegate, "OUT", tmp_path / "out")
    monkeypatch.setattr(delegate, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(
        delegate,
        "discover_vibe_setup",
        lambda: {
            "configured": True,
            "cli_path": ".venv/bin/vibe",
            "config_path": str(Path.home() / ".vibe/config.toml"),
            "required_fields": ["active_model", "default_agent", "providers[].api_key_env_var", "MISTRAL_API_KEY"],
            "missing_setup_fields": [],
        },
    )

    class P:
        returncode = 0
        stderr = ""
        stdout = '{"patches":["*** Begin Patch\\n*** Add File: hi.txt\\n+hello\\n*** End Patch"],"tests":["python -m pytest -q tests/test_delegate_wrappers.py"],"next_prompt":"integrate the smallest patch"}'

    monkeypatch.setattr(delegate.subprocess, "run", lambda *args, **kwargs: P())

    assert delegate.main(["--task", "add a tiny patch", "--json"]) == 0
    payload = _json_line(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert payload["normalized_output"]["tests"] == ["python -m pytest -q tests/test_delegate_wrappers.py"]
    assert payload["normalized_output"]["next_prompt"] == "integrate the smallest patch"


def test_groq_delegate_fails_loudly_with_exact_missing_setup_fields(monkeypatch, capsys, tmp_path: Path) -> None:
    from scripts import groq_delegate as delegate

    monkeypatch.setattr(delegate, "OUT", tmp_path / "out")
    monkeypatch.setattr(delegate, "RUNTIME", tmp_path / "runtime")
    monkeypatch.setattr(
        delegate,
        "discover_groq_setup",
        lambda: {
            "configured": False,
            "adapter_registry_path": "GOALS/plugin_build_mode_bootstrap.json",
            "required_fields": [
                "adapter_registry.groq.env_key_names",
                "adapter_registry.groq.orchestrated_delegate_cmd",
                "adapter_registry.groq.recommended_models.default_cheap_worker",
                "GROQ_API_KEY",
            ],
            "missing_setup_fields": ["GROQ_API_KEY"],
        },
    )

    assert delegate.main(["--task", "review scripts/groq_delegate.py", "--json"]) == 4
    out = capsys.readouterr().out
    assert "GROQ_DELEGATE_NOT_CONFIGURED" in out
    payload = _json_line(out)
    assert payload["status"] == "BLOCKED"
    assert payload["missing_setup_fields"] == ["GROQ_API_KEY"]
