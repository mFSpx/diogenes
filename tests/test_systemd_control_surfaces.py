from __future__ import annotations

from pathlib import Path


def test_systemd_units_cover_postgrest_absurd_indy_and_audits():
    services = {
        "services/lucidota-postgrest.service": ["postgrest", "root_rotor_postgrest.conf", "Restart=on-failure"],
        "services/lucidota-absurd-worker.service": ["absurd_queue_spine.py", "--action wake-plane --execute", "Restart=on-failure"],
        "services/ironclaw-indy-reads.service": ["Indy_READs 24/7 DB daemon", "lucidota_start_indy_reads_watcher.sh --foreground", "Restart=on-failure"],
        "services/lucidota-root-rotor-audit.service": ["root_rotor_red_team_audit.py", "Type=oneshot"],
        "services/lucidota-root-rotor-audit.timer": ["OnUnitActiveSec=1h", "WantedBy=timers.target"],
        "services/lucidota-krampus-triage.service": ["krampuschewing_quarantine_triage.py", "--dry-run", "Type=oneshot"],
        "services/lucidota-krampus-triage.timer": ["OnUnitActiveSec=6h", "WantedBy=timers.target"],
    }
    for rel_path, needles in services.items():
        text = Path(rel_path).read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text, (rel_path, needle)


def test_recovery_matrix_includes_root_rotor_and_krampus_triage():
    text = Path("scripts/recovery_matrix.py").read_text(encoding="utf-8")
    assert "root_rotor_red_team_audit" in text
    assert "krampuschewing_quarantine_triage.py --dry-run --json" in text
