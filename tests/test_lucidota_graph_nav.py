from scripts.lucidota_graph_nav import build_search_sql, day_evidence_report, render_report


def test_search_sql_parameterizes_query_and_limit():
    sql, params = build_search_sql("Rickshaw", limit=3, term="CASE")
    assert "ILIKE" in sql
    assert "term=%s" in sql
    assert params == ["CASE", "%Rickshaw%", "%Rickshaw%", 3]


def test_render_roots_shows_metric_edge_counts_and_labels():
    report = {
        "mode": "roots",
        "rows": [
            {
                "uuid": "root-1",
                "term": "CASE",
                "label": "RICKSHAW_ROBBERY second-pass case-corpus root",
                "truth_scope": "case_corpus_root_meta_truth_only",
                "metric_edges": 1,
                "created_at": "2026-05-27T18:00:00Z",
            }
        ],
    }
    text = render_report(report)
    assert "RICKSHAW_ROBBERY" in text
    assert "metrics=1" in text
    assert "case_corpus_root_meta_truth_only" in text


def test_chrono_sql_filters_query_and_date_bounds():
    from scripts.lucidota_graph_nav import build_chrono_sql
    sql, params = build_chrono_sql("rickshaw", since="2026-02-01", until="2026-05-27", limit=5)
    assert "current_chrono_timeline_projection" in sql
    assert "resolved_timestamp >= %s::timestamptz" in sql
    assert "resolved_timestamp <= %s::timestamptz" in sql
    assert params == ["2026-02-01", "2026-05-27", "%rickshaw%", "%rickshaw%", "%rickshaw%", 5]


def test_day_chrono_sql_uses_exact_exclusive_date_window():
    from scripts.lucidota_graph_nav import build_day_chrono_sql
    sql, params = build_day_chrono_sql("2026-03-07", limit=7)
    assert "resolved_timestamp >= %s::date" in sql
    assert "resolved_timestamp < (%s::date + interval '1 day')" in sql
    assert params == ["2026-03-07", "2026-03-07", 7]


def test_render_day_shows_bucket_counts_edges_and_chrono_sample():
    report = {
        "mode": "day",
        "day": "2026-03-07",
        "rows": [
            {
                "uuid": "day-1",
                "term": "DATE_BUCKET",
                "label": "Chrono projection day bucket 2026-03-07",
                "truth_scope": "chrono_day_bucket_meta_truth_only",
                "evidence_counts": {"projected_event_count": 2435},
            }
        ],
        "edges": [
            {
                "edge_type": "HAS_CHRONO_DAY_BUCKET",
                "edge_uuid": "edge-1",
                "source_label": "Chrono projection month bucket 2026-03",
                "target_label": "Chrono projection day bucket 2026-03-07",
            }
        ],
        "chrono_rows": [
            {
                "resolved_timestamp": "2026-03-07 12:00:00-08",
                "trust_weight": "0.7",
                "evidence_source": "mtime_snapshot_v1",
                "file_uuid": "file-1",
                "source_sha256": "abc",
            }
        ],
    }
    text = render_report(report)
    assert "projected_event_count=2435" in text
    assert "HAS_CHRONO_DAY_BUCKET" in text
    assert "chrono 2026-03-07 12:00:00-08" in text


def test_render_evidence_shows_refs_and_sources():
    report = {
        "mode": "evidence",
        "target": "root-1",
        "rows": [
            {"ref": "05_OUTPUTS/example.json", "source": "materialization", "count": 1},
            {"ref": "lucidota_go.graph_item:root-1", "source": "edge", "count": 2},
        ],
    }
    text = render_report(report)
    assert "05_OUTPUTS/example.json" in text
    assert "materialization" in text
    assert "count=2" in text


def test_day_evidence_resolves_day_bucket_before_evidence(monkeypatch):
    calls = {}
    def fake_query_rows(database_url, sql, params):
        assert database_url == "postgresql://unit"
        assert "chrono_day_bucket_meta_truth_only" in sql
        assert params == ["2026-03-07"]
        return [{"uuid": "day-uuid"}]
    def fake_evidence_report(database_url, target, limit):
        calls["args"] = (database_url, target, limit)
        return {"mode": "evidence", "target": target, "rows": [{"source": "payload.source_receipts", "ref": "r1", "count": 1}], "status": "PASS"}
    monkeypatch.setattr("scripts.lucidota_graph_nav.query_rows", fake_query_rows)
    monkeypatch.setattr("scripts.lucidota_graph_nav.evidence_report", fake_evidence_report)
    report = day_evidence_report("postgresql://unit", "2026-03-07", 5)
    assert calls["args"] == ("postgresql://unit", "day-uuid", 5)
    assert report["mode"] == "day-evidence"
    assert report["day"] == "2026-03-07"
    assert report["resolved_day_uuid"] == "day-uuid"
