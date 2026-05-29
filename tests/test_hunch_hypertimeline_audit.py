from __future__ import annotations

import json
from pathlib import Path

from scripts.hunch_hypertimeline_audit import (
    build_audit,
    parse_hunch_file,
    write_observation_center,
)

ROOT = Path(__file__).resolve().parents[1]
FIRST_AUDIT = ROOT / "09_STORAGE/krampuschewing_unpacked/docs_Luci-010.zip_10711561291/Luci/RandomReading/Lucidota/LOG/PONYBOY_HUNCH_ACCURACY_AUDIT.md"
DEEP_SCAN = ROOT / "09_STORAGE/krampuschewing_unpacked/docs_Luci-010.zip_10711561291/Luci/RandomReading/Lucidota/LOG/PONYBOY_HUNCH_AUDIT_DEEP_SCAN.md"


def test_parse_hunch_file_extracts_ids_ratings_and_lineage() -> None:
    records = parse_hunch_file(FIRST_AUDIT, root=ROOT)

    ids = {r["hunch_id"] for r in records}
    assert "RR-01" in ids
    assert "SYS-02" in ids
    assert len(records) >= 28

    rr01 = next(r for r in records if r["hunch_id"] == "RR-01")
    assert rr01["rating"] == "FUCKIN RIGHT"
    assert rr01["source_sha256"]
    assert rr01["line_start"] < rr01["line_end"]


def test_build_audit_preserves_count_discrepancy_instead_of_faking_closure() -> None:
    audit = build_audit(
        roots=[FIRST_AUDIT, DEEP_SCAN],
        root=ROOT,
        max_files=50,
        include_repo_scan=False,
    )

    assert audit["schema"] == "lucidota.hunch_hypertimeline_audit.v1"
    assert audit["canonical_known_tracked_total"] == 91
    assert audit["parsed_hunch_headings_total"] >= 91
    assert audit["count_status"] == "DISCREPANCY_REQUIRES_REVIEW"
    assert audit["rating_counts"]["FUCKIN RIGHT"] >= 39
    assert "hunches_require_evidence_paths_before_truth_promotion" in audit["binding_rules"]
    assert audit["graph_writes_performed"] is False


def test_write_observation_center_updates_runtime_and_big_board(tmp_path: Path) -> None:
    audit = build_audit(
        roots=[FIRST_AUDIT, DEEP_SCAN],
        root=ROOT,
        max_files=50,
        include_repo_scan=False,
    )
    fake_root = tmp_path
    (fake_root / "05_OUTPUTS").mkdir(parents=True)
    (fake_root / "05_OUTPUTS" / "big_board.json").write_text(json.dumps({"counters": {}}), encoding="utf-8")

    result = write_observation_center(audit, root=fake_root)

    runtime = fake_root / "04_RUNTIME" / "observation_center" / "hunch_hypertimeline_latest.json"
    big_board = json.loads((fake_root / "05_OUTPUTS" / "big_board.json").read_text(encoding="utf-8"))
    assert runtime.exists()
    assert result["observation_center_path"] == "04_RUNTIME/observation_center/hunch_hypertimeline_latest.json"
    assert big_board["observation_center"]["hunch_hypertimeline"]["canonical_known_tracked_total"] == 91
    assert big_board["counters"]["hunch_hypertimeline_known_total"] == 91
