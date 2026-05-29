from pathlib import Path

from scripts.project2501_bytewax_board_stream_service import render_unit, write_unit_file


def test_render_unit_contains_live_cursor_loop():
    unit = render_unit(interval=7, limit=55)
    assert "project2501_bytewax_board_stream.py once --execute --live-cursor --limit 55 --json" in unit
    assert "Restart=always" in unit
    assert "PROJECT2501_BOARD_STREAM_INTERVAL=7" in unit


def test_write_unit_file_is_dry_run_safe(tmp_path):
    payload = write_unit_file(unit_dir=tmp_path, execute=False, interval=5, limit=25)
    assert payload["execute_performed"] is False
    assert payload["unit_written"] is False
    assert payload["canonical_graph_writes_performed"] is False
    assert not Path(payload["unit_path"]).exists()
    payload = write_unit_file(unit_dir=tmp_path, execute=True, interval=5, limit=25)
    assert payload["unit_written"] is True
    assert Path(payload["unit_path"]).exists()
