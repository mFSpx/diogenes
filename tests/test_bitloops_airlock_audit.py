#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bitloops_airlock_audit import build_report


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_airlock_blocks_latest_install_and_curl_pipe(tmp_path):
    install = write(tmp_path / "install.sh", "curl -fsSL https://bitloops.com/install.sh | bash\n")

    report = build_report(
        source_tag="latest",
        source_commit="",
        install_script=install,
        daemon_config=None,
        repo_policy=None,
        telemetry_optout_env=False,
    )

    assert report["status"] == "FAIL"
    assert "source_not_pinned" in report["blockers"]
    assert "curl_pipe_shell_install_detected" in report["blockers"]
    assert report["canonical_graph_writes_performed"] is False
    assert report["model_calls_performed"] is False


def test_airlock_blocks_release_latest_installer_even_with_pinned_source(tmp_path):
    install = write(tmp_path / "install.sh", 'API_URL="https://api.github.com/repos/bitloops/bitloops/releases/latest"\n')

    report = build_report(
        source_tag="v0.0.30",
        source_commit="23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48",
        install_script=install,
        daemon_config=None,
        repo_policy=None,
        telemetry_optout_env=True,
    )

    assert report["status"] == "FAIL"
    assert "release_latest_installer_detected" in report["blockers"]


def test_airlock_blocks_cloud_remote_and_telemetry_risks(tmp_path):
    config = write(
        tmp_path / "config.toml",
        """
[telemetry]
enabled = true

[stores.relational]
postgres_dsn = "${BITLOOPS_POSTGRES_DSN}"

[stores.events]
clickhouse_url = "http://localhost:8123"

[stores.blob]
s3_bucket = "bitloops-artifacts"

[inference.runtimes.bitloops_platform_embeddings]
command = "/tmp/bitloops-platform-embeddings"
args = ["--gateway-url", "https://gateway.example/v1/embeddings"]

[inference.profiles.guidance_llm]
driver = "bitloops_platform_chat"
api_key = "${BITLOOPS_PLATFORM_GATEWAY_TOKEN}"
""".strip(),
    )
    policy = write(tmp_path / ".bitloops.local.toml", "[semantic_clones]\nembedding_mode = \"semantic_aware_once\"\n")

    report = build_report(
        source_tag="v0.0.30",
        source_commit="23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48",
        install_script=None,
        daemon_config=config,
        repo_policy=policy,
        telemetry_optout_env=False,
    )

    assert report["status"] == "FAIL"
    for blocker in [
        "telemetry_not_opted_out",
        "remote_relational_store_detected",
        "remote_event_store_detected",
        "remote_blob_store_detected",
        "platform_inference_detected",
        "repo_embeddings_not_off_or_local",
    ]:
        assert blocker in report["blockers"]


def test_airlock_passes_local_only_pinned_config_and_cli_writes_receipt(tmp_path):
    config = write(
        tmp_path / "config.toml",
        """
[telemetry]
enabled = false

[stores.relational]
sqlite_path = "/tmp/bitloops/relational.db"

[stores.events]
duckdb_path = "/tmp/bitloops/events.duckdb"

[stores.blob]
local_path = "/tmp/bitloops/blob"

[inference.runtimes.bitloops_local_embeddings]
command = "/tmp/bitloops-local-embeddings"
""".strip(),
    )
    policy = write(tmp_path / ".bitloops.local.toml", "[semantic_clones]\nembedding_mode = \"off\"\n")

    report = build_report(
        source_tag="v0.0.30",
        source_commit="23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48",
        install_script=None,
        daemon_config=config,
        repo_policy=policy,
        telemetry_optout_env=True,
    )
    assert report["status"] == "PASS"
    assert report["blockers"] == []
    assert report["sovereign_mode"] is True

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/bitloops_airlock_audit.py",
            "--source-tag",
            "v0.0.30",
            "--source-commit",
            "23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48",
            "--daemon-config",
            str(config),
            "--repo-policy",
            str(policy),
            "--telemetry-optout-env",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    line = next(line for line in proc.stdout.splitlines() if line.startswith("REPORT_PATH="))
    receipt = ROOT / line.split("=", 1)[1]
    assert receipt.exists()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["source"]["tag"] == "v0.0.30"


def test_airlock_can_fail_closed_when_live_binary_required(monkeypatch):
    import bitloops_airlock_audit as mod

    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    report = build_report(
        source_tag="v0.0.30",
        source_commit="23e3b4da0404c75cc8ec1fdfb0b40bf3091b9a48",
        telemetry_optout_env=True,
        require_binary=True,
    )

    assert report["status"] == "FAIL"
    assert "bitloops_binary_missing" in report["blockers"]
    assert report["checks"]["bitloops_binary_present"] is False


def test_airlock_receipts_do_not_overwrite_same_timestamp(tmp_path, monkeypatch):
    import bitloops_airlock_audit as mod

    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(mod, "stamp", lambda: "SAMESTAMP")
    base = {
        "schema": "lucidota.bitloops.airlock_audit.v1",
        "generated_at": "now",
        "status": "PASS",
        "blockers": [],
    }

    first = mod.write_report(dict(base))
    second = mod.write_report(dict(base))

    assert first != second
    assert first.exists()
    assert second.exists()
