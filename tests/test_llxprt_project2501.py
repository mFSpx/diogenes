from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def test_build_config_writes_groq_alias_profile_project_context(tmp_path) -> None:
    from llxprt_project2501 import configure_llxprt

    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    receipt = configure_llxprt(root=project, home=home, model="openai/gpt-oss-120b")

    provider = json.loads((home / ".llxprt" / "providers" / "lucidota-groq.config").read_text(encoding="utf-8"))
    assert provider["baseProvider"] == "openai"
    assert provider["base-url"] == "https://api.groq.com/openai/v1/"
    assert provider["defaultModel"] == "openai/gpt-oss-120b"
    assert provider["apiKeyEnv"] == "GROQ_API_KEY"

    profile = json.loads((home / ".llxprt" / "profiles" / "lucidota-groq-orchestrator.json").read_text(encoding="utf-8"))
    assert profile["provider"] == "lucidota-groq"
    assert profile["model"] == "openai/gpt-oss-120b"
    assert profile["modelParams"]["max_tokens"] == 8192
    assert profile["ephemeralSettings"]["context-limit"] == 131072
    assert profile["ephemeralSettings"]["streaming"] == "enabled"

    context = (project / "LLXPRT.md").read_text(encoding="utf-8")
    assert "PROJECT 2501" in context
    assert "You are the LUCIDOTA LLXPRT Groq orchestrator" in context
    assert "Bare Steel Rule 4" in context
    assert "Read selectively. Persist globally." in context
    assert "canonical graph materialization remains gated" in context

    settings = json.loads((project / ".llxprt" / "settings.json").read_text(encoding="utf-8"))
    assert settings["ui"]["showMemoryUsage"] is True
    assert settings["ui"]["hideModelInfo"] is False
    assert "LLXPRT.md" in settings["ui"]["contextFileName"]

    assert receipt["status"] == "PASS"
    assert receipt["provider_alias_path"].endswith("lucidota-groq.config")
    assert receipt["profile_path"].endswith("lucidota-groq-orchestrator.json")


def test_launch_dry_run_emits_exact_session_command(tmp_path) -> None:
    from llxprt_project2501 import configure_llxprt, launch_command

    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    configure_llxprt(root=project, home=home)
    cmd, env = launch_command(root=project, home=home, prompt="open the orchestrator lane", dry_run=True)

    assert cmd[0].endswith("llxprt") or cmd[:3] == ["npx", "-y", "@vybestack/llxprt-code"]
    assert "--profile-load" in cmd
    assert "lucidota-groq-orchestrator" in cmd
    assert "--provider" in cmd and "lucidota-groq" in cmd
    assert "--model" in cmd and "openai/gpt-oss-120b" in cmd
    assert "--prompt-interactive" in cmd
    assert "PROJECT 2501" in cmd[-1]
    assert env["HOME"] == str(home)
    assert env.get("GROQ_API_KEY") is None


def test_cli_configure_and_doctor_write_receipts(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    home = tmp_path / "home"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/llxprt_project2501.py",
            "configure",
            "--root",
            str(project),
            "--home",
            str(home),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    payload = next(json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{"))
    assert payload["status"] == "PASS"
    assert "LLXPRT_PROJECT2501=PASS" in proc.stdout

    doctor = subprocess.run(
        [
            sys.executable,
            "scripts/llxprt_project2501.py",
            "doctor",
            "--root",
            str(project),
            "--home",
            str(home),
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert doctor.returncode == 0, doctor.stderr
    report_path = next(line.split("=", 1)[1] for line in doctor.stdout.splitlines() if line.startswith("REPORT_PATH="))
    report = json.loads((ROOT / report_path).read_text(encoding="utf-8"))
    assert report["schema"] == "lucidota.llxprt_project2501.receipt.v1"
    assert report["llxprt"]["binary"]
    assert report["groq"]["model"] == "openai/gpt-oss-120b"
    assert report["profile"]["exists"] is True
