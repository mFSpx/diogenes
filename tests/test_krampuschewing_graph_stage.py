import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from krampuschewing_graph_stage import build_candidates


def _write_index(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_graph_stage_passes_when_case_index_has_no_active_db(tmp_path):
    index = tmp_path / "index.jsonl"
    out = tmp_path / "candidates.jsonl"
    summary = tmp_path / "summary.json"
    _write_index(
        index,
        [
            {
                "repo_relative_path": "RICKSHAW_ROBBERY/MASTER_CASE_FILE/01_CASE_SUMMARY/RICKSHAW_ROBBERY.md",
                "sha256": "a" * 64,
                "sha256_status": "computed",
                "lane_guess": "CASE_WORK",
                "kind_guess": "markdown",
                "case_guess": "RICKSHAW_ROBBERY",
                "contains_case_terms": True,
            }
        ],
    )

    _, _, result = build_candidates(index, out, summary)

    assert result["verdict"] == "PASS"
    assert result["blockers"] == []
    assert result["candidates_emitted"] == 1
    assert result["active_runtime_db_skipped_count"] == 0


def test_graph_stage_still_excludes_active_runtime_db(tmp_path):
    index = tmp_path / "index.jsonl"
    out = tmp_path / "candidates.jsonl"
    summary = tmp_path / "summary.json"
    _write_index(
        index,
        [
            {
                "repo_relative_path": "CHROMADB/chroma.sqlite3",
                "sha256": "b" * 64,
                "sha256_status": "computed",
                "lane_guess": "UNKNOWN",
                "kind_guess": "unknown",
            },
            {
                "repo_relative_path": "RICKSHAW_ROBBERY/MASTER_CASE_FILE/04_TIMELINE/MASTER_TIMELINE.md",
                "sha256": "c" * 64,
                "sha256_status": "computed",
                "lane_guess": "CASE_WORK",
                "kind_guess": "markdown",
                "case_guess": "RICKSHAW_ROBBERY",
                "contains_case_terms": True,
            },
        ],
    )

    _, _, result = build_candidates(index, out, summary)

    assert result["verdict"] == "PASS"
    assert result["blockers"] == []
    assert result["active_runtime_db_excluded"] is True
    assert result["active_runtime_db_skipped_count"] == 1
    assert result["candidates_emitted"] == 1
