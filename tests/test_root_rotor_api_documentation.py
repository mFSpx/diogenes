from __future__ import annotations

import json
import re
from pathlib import Path



def test_root_rotor_api_docs_can_render_html_from_db_payload(tmp_path) -> None:
    import scripts.root_rotor_api_documentation as doc

    payload = {
        "schema": "lucidota.root_law_api_payload.v1",
        "generated_at": "2026-06-02T00:00:00Z",
        "manuals": [
            {
                "manual_id": "FLIGHT_MAN",
                "nodes": [
                    {
                        "node_id": "4.100.0",
                        "title": "Route index",
                        "payload": "Root route index",
                        "status": "verified",
                        "version": 2,
                        "node_kind": "REFERENCE",
                        "source_refs": ["api://route/root_law_docs"],
                        "node_sort_key": [4, 100, 0],
                    }
                ],
            }
        ],
        "api_routes": [
            {
                "route_id": "route_a",
                "method": "GET",
                "path_pattern": "/api/a",
                "description": "A route",
                "target": "target",
                "sample_request": "{\"ok\":true}",
                "sample_response": "{\"v\":1}",
                "content_hash": "abc123",
            }
        ],
        "contradictions": {
            "blockers": ["none"],
            "warnings": ["gap_x"],
            "coverage_ratio": 0.88,
            "gap_atlas": [
                {
                    "name": "sidecar",
                    "blockers": 1,
                    "warnings": 1,
                    "coverage_ratio": 0.25,
                    "missing_sidecars": 3,
                    "valid_sidecars": 2,
                    "manifest_files": 5,
                    "symbolic_edge_values": 0,
                },
                {
                    "name": "red_team",
                    "blockers": 1,
                    "warnings": 1,
                    "coverage_ratio": 0.5,
                    "draft_nodes": 3,
                    "verified_nodes": 7,
                    "model_payload_count": 5,
                    "total_nodes": 10,
                },
            ],
            "surfaces": {
                "sidecar": {
                    "blockers": ["sidecar_coverage_incomplete"],
                    "warnings": ["sidecar_edge_symbol_anomaly"],
                    "metrics": {
                        "coverage_ratio": 0.25,
                        "verified_nodes": 2,
                        "draft_nodes": 1,
                        "model_payload_count": 1,
                    },
                },
                "red_team": {
                    "blockers": ["manual_incomplete_draft_nodes"],
                    "warnings": ["not_all_nodes_have_model_payloads"],
                    "metrics": {
                        "coverage_ratio": 0.5,
                        "verified_nodes": 7,
                        "draft_nodes": 3,
                        "model_payload_count": 5,
                    },
                },
            },
        },
        "payload_hash": "deadbeef",
    }
    template_text = """
    <h1>{{ generated_at }}</h1>
    {% for manual in manuals %}
      <h2>{{ manual.manual_id }}</h2>
      {% for node in manual.nodes %}
        <p>{{ node.node_id }} {{ node.title }}</p>
      {% endfor %}
    {% endfor %}
    {% for route in api_routes %}
      <div>{{ route.path_pattern }} {{ route.content_hash }}</div>
    {% endfor %}
    {% for warning in warnings %}
      <span>{{ warning }}</span>
    {% endfor %}
    {% for name, surface in audit_surfaces.items() %}
      <section>{{ name }} {{ surface.metrics.coverage_ratio }}</section>
    {% endfor %}
    {% for item in gap_atlas %}
      <section>{{ item.name }} {{ item.blockers }}</section>
    {% endfor %}
    """
    html = doc.render_html(template_text, payload)
    assert "2026-06-02T00:00:00Z" in html
    assert "FLIGHT_MAN" in html
    assert "4.100.0 Route index" in html
    assert "/api/a" in html
    assert "abc123" in html
    assert "gap_x" in html
    assert "sidecar 0.25" in html
    assert "red_team 1" in html


def test_root_rotor_api_docs_runs_and_writes_artifacts(tmp_path, monkeypatch) -> None:
    import scripts.root_rotor_api_documentation as doc

    def fake_fetch_manual_nodes(dsn: str, manual_ids: list[str]):
        return {"FLIGHT_MAN": [{"node_id": "4.1.0", "title": "x", "payload": "x", "status": "verified", "version": 1, "node_kind": "OBJECT", "source_refs": ["scripts/x.py"], "node_sort_key": [4, 1, 0]}]}

    def fake_fetch_routes(dsn: str):
        return [
            {
                "route_id": "nodes",
                "method": "GET",
                "path_pattern": "/api_bible_nodes",
                "description": "nodes",
                "target": "lucidota_canon.api_bible_nodes",
                "sample_request": "{}",
                "sample_response": "[]",
                "content_hash": "routehash",
            }
        ]

    class Dummy:
        @staticmethod
        def read_default_contradictions(receipt_dir: Path):
            return {
                "blockers": [],
                "warnings": [],
                "coverage_ratio": 1.0,
                "gap_atlas": [
                    {
                        "name": "sidecar",
                        "blockers": 1,
                        "warnings": 1,
                        "coverage_ratio": 0.25,
                        "missing_sidecars": 3,
                        "valid_sidecars": 2,
                        "manifest_files": 5,
                        "symbolic_edge_values": 0,
                    },
                    {
                        "name": "red_team",
                        "blockers": 1,
                        "warnings": 1,
                        "coverage_ratio": 0.5,
                        "draft_nodes": 3,
                        "verified_nodes": 7,
                        "model_payload_count": 5,
                        "total_nodes": 10,
                    },
                ],
                "surfaces": {},
            }

    monkeypatch.setattr(doc, "fetch_manual_nodes", fake_fetch_manual_nodes)
    monkeypatch.setattr(doc, "fetch_route_catalog", fake_fetch_routes)
    monkeypatch.setattr(doc, "read_default_contradictions", Dummy.read_default_contradictions)
    monkeypatch.setattr(doc, "sync_routes_to_bible_nodes", lambda *_, **__: {"upserted": 0, "updated": 0, "errors": []})

    template = tmp_path / "tpl.html"
    template.write_text("<html><body>{% for m in manuals %}{{ m.content_hash }}{% for node in m.nodes %}{{ node.node_id }}{% endfor %}{% endfor %}{% for r in api_routes %}{{ r.content_hash }}{% endfor %}{% for item in gap_atlas %}{{ item.name }}{% endfor %}</body></html>", encoding="utf-8")
    result = doc.run(
        dsn="postgresql:///ignore",
        manual_ids=["FLIGHT_MAN"],
        template=template,
        output_dir=tmp_path / "out",
        receipt_dir=tmp_path,
        sync_routes=False,
        include_all=True,
    )

    assert result["status"] == "PASS"
    assert result["route_count"] == 1
    html_text = Path(result["html_path"]).read_text(encoding="utf-8")
    assert "4.1.0" in html_text
    assert re.search(r"[0-9a-f]{16}", html_text)
    assert Path(result["markdown_path"]).exists()
    md_text = Path(result["markdown_path"]).read_text(encoding="utf-8")
    assert "## Gap Atlas" in md_text
    assert "sidecar: blockers=1 warnings=1 coverage=0.25" in md_text


class FakeCursor:
    def __init__(self, fetchsets):
        self.fetchsets = fetchsets
        self.last_rows = []

    def execute(self, sql, params=()):
        self.sql = sql
        self.params = params
        if "SELECT node_id FROM lucidota_canon.bible_nodes" in sql and "payload_format='json'" in sql:
            self.last_rows = []
        elif "SELECT node_id FROM lucidota_canon.bible_nodes WHERE manual_id" in sql and "LIKE '4.%'" in sql:
            self.last_rows = [("4.1.0",), ("4.2.0",)]
        else:
            self.last_rows = []

    def fetchone(self):
        return self.last_rows[0] if self.last_rows else None

    def fetchall(self):
        rows, self.last_rows = self.last_rows, []
        return rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    def __init__(self):
        self.cur = FakeCursor([])
        self.commits = 0

    def cursor(self):
        return self.cur

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_root_rotor_api_docs_sync_assigns_route_node_ids(monkeypatch) -> None:
    import scripts.root_rotor_api_documentation as doc

    fake = FakeConn()
    monkeypatch.setattr(doc.psycopg, "connect", lambda *_a, **_k: fake)

    routes = [
        {
            "route_id": "root_law_docs",
            "method": "GET",
            "path_pattern": "/root_law_docs",
            "description": "docs",
            "target": "ok",
            "sample_request": "{}",
            "sample_response": "{}",
        }
    ]
    result = doc.sync_routes_to_bible_nodes("postgresql:///ignore", routes, include_all=False)

    assert result["updated"] == 1
    assert not result["errors"]
    # generated id is based on endpoint namespace, then node id should avoid used route ids
    assert result["upserted"] == 1
