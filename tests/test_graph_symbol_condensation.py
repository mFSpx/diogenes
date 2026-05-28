from scripts import graph_symbol_condensation


class _DummyConn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_graph_symbol_condensation_builds_confidence_scored_claims(monkeypatch):
    monkeypatch.setattr(graph_symbol_condensation.psycopg, "connect", lambda *args, **kwargs: _DummyConn())
    monkeypatch.setattr(
        graph_symbol_condensation,
        "fetch_db_evidence",
        lambda storage_conn, state_conn, term_limit: {
            "active_terms": ["OBJECT", "EVENT", "EDGE", "CLAIM", "EVIDENCE"],
            "runtime_facts": [{"fact_key": "llxprt_groq_login_wired", "fact_value": {"status": "wired"}}],
            "graph_counts": {"graph_item_count": 123, "graph_edge_count": 456, "graph_journal_count": 7},
        },
    )

    report = graph_symbol_condensation.build_report(
        symbols=["OBJECT", "EVENT", "EDGE", "OBJECT"],
        storage_database_url=None,
        state_database_url=None,
        term_limit=10,
        baseline_report=None,
        objective="condense graph and postgres evidence into language",
    )

    assert report["schema"] == "lucidota.graph_symbol_condensation.v1"
    assert report["ontology_mode"] == "GO25_STRICT"
    assert report["model_calls_performed"] is False
    assert report["canonical_graph_writes_performed"] is False
    assert report["claims"]
    assert report["comparison_summary"]["backed_claims"] >= 0
    assert report["evidence_refs"]
    assert all(0 <= claim["confidence_bps"] <= 10000 for claim in report["claims"])
    assert any(claim["status"] in {"backed", "candidate"} for claim in report["claims"])


def test_graph_symbol_condensation_uses_default_symbols_when_empty(monkeypatch):
    monkeypatch.setattr(graph_symbol_condensation.psycopg, "connect", lambda *args, **kwargs: _DummyConn())
    monkeypatch.setattr(
        graph_symbol_condensation,
        "fetch_db_evidence",
        lambda storage_conn, state_conn, term_limit: {
            "active_terms": ["OBJECT", "EVENT", "EDGE"],
            "runtime_facts": [],
            "graph_counts": {},
        },
    )

    report = graph_symbol_condensation.build_report(
        symbols=[],
        storage_database_url=None,
        state_database_url=None,
        term_limit=3,
        baseline_report=None,
        objective="",
    )

    assert report["symbol_lexicon"]
    assert report["language_seed"]
