from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts import absurd_flows as af


def test_file_kind_and_batch_helpers() -> None:
    assert af.file_kind(Path("x.pdf"), "application/pdf") == "document"
    assert af.file_kind(Path("x.png"), "") == "image"
    assert af.batch([Path("a"), Path("b"), Path("c")], 2) == [[Path("a"), Path("b")], [Path("c")]]


def test_process_files_dry_run_uses_receipts_without_db(tmp_path, monkeypatch) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_text("alpha", encoding="utf-8")
    (root / "two.txt").write_text("beta", encoding="utf-8")
    monkeypatch.setattr(af, "ROOT", root)
    monkeypatch.setattr(af, "CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(af, "OUT_DIR", tmp_path / "out")
    payload = af.process_files(root, max_files=None, chunk_size=2, execute=False)
    assert payload["ok"] is True
    assert payload["file_count"] == 2
    assert payload["processed_count"] == 0
    assert len(payload["records"]) == 2


def test_phase1_edge_dedupe_script_dedupes_hashes(tmp_path) -> None:
    target = tmp_path / "krampuschewing_unpacked"
    target.mkdir()
    (target / "a.txt").write_text("same", encoding="utf-8")
    (target / "b.txt").write_text("same", encoding="utf-8")
    (target / "c.txt").write_text("different", encoding="utf-8")
    script = Path("scripts/phase1_edge_dedupe.sh").resolve()
    proc = subprocess.run([str(script), str(target)], cwd=Path.cwd(), text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    receipt_path = Path(proc.stdout.strip())
    assert receipt_path.exists()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["ok"] is True
    mapping = Path(receipt["mapping_path"])
    assert mapping.exists()
    lines = [line for line in mapping.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
