from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts import mamba_db_watch


def test_fetch_rows_hits_indy_queue_with_limit_query() -> None:
    seen: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen["path"] = self.path
            body = json.dumps(
                [
                    {
                        "id": "row-1",
                        "event_id": "event-1",
                        "sender_id": "sender",
                        "room_id": "room",
                        "raw_text": "raw body",
                        "clean_text": "clean body",
                        "processed_status": "queued",
                        "receipt_id": "receipt-1",
                    }
                ]
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rows = mamba_db_watch.fetch_rows("indy_queue", base_url=f"http://127.0.0.1:{server.server_port}", limit=9)
    finally:
        server.shutdown()
        thread.join(timeout=2)

    assert seen["path"].startswith("/indy_queue?")
    assert "limit=9" in seen["path"]
    assert rows[0]["processed_status"] == "queued"


def test_poll_once_hits_live_indy_queue_route() -> None:
    try:
        payload = mamba_db_watch.poll_once(base_url="http://127.0.0.1:3000", limit=1, max_items=1)
    except Exception as exc:  # pragma: no cover - live service unavailable
        raise AssertionError(f"live PostgREST unavailable: {exc}") from exc

    assert payload["visible_route"] == "/indy_queue"
    assert payload["schema"] == "lucidota.mamba_db_watch.compact_queue.v1"
    assert payload["row_count"] >= 0
    assert "raw_text" not in payload
