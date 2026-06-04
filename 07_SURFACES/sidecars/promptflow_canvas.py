#!/usr/bin/env python3
"""LUCIDOTA /flow visual canvas.

Small local Promptflow-style canvas for operator composition.  It keeps the
workflow visible: dragging cards attaches typed refs, not full file contents;
explicit Save/Stage/Run/Validate/Promote/Rollback writes receipts under
05_OUTPUTS/flow/<flow_id>/.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import mimetypes
import os
import socket
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "05_OUTPUTS" / "flow"
DB_URL = os.environ.get("ABSURD_SYSTEM_DATABASE_URL") or os.environ.get("DATABASE_URL") or "postgresql:///lucidota_state"
CARD_LIMIT_PER_ROOT = 80
CARD_ROOTS = [
    ("GOALS", ROOT / "GOALS", {".md", ".json"}),
    ("scripts", ROOT / "scripts", {".py", ".sh", ".sql"}),
    ("schemas", ROOT / "06_SCHEMA", {".sql", ".json"}),
    ("runtime", ROOT / "04_RUNTIME", {".json", ".yaml", ".yml", ".md"}),
]
CORE_CARD_PATHS = [
    "luci",
    "scripts/indy_reads.py",
    "scripts/indy_conduit_driver.py",
    "scripts/promptflow_eval_runner.py",
    "GOALS/CURRENT_HANDOFF.md",
    "04_RUNTIME/promptflow_smoke_flow/flow.dag.yaml",
]
ACTIONS = {"save", "stage", "validate", "run", "promote", "rollback"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


def file_card(path: Path, card_type: str = "FILE", *, summary: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if card_type == "FILE":
        if suffix in {".py", ".sh", ".sql"}:
            card_type = "SCRIPT" if suffix != ".sql" else "WORKFLOW"
        elif suffix in {".md", ".txt"}:
            card_type = "PROMPT"
        elif suffix in {".json", ".jsonl", ".yaml", ".yml"}:
            card_type = "DATA"
    size = path.stat().st_size if path.exists() else 0
    return {
        "id": f"file:{rel(path)}",
        "type": card_type,
        "label": path.name,
        "ref": {"kind": "repo_path", "path": rel(path), "sha256": sha256_file(path), "size_bytes": size},
        "summary": summary or f"{card_type} reference at {rel(path)}",
        "custody_state": "repo_ref_only",
    }


def db_cards(limit: int = 40) -> list[dict[str, Any]]:
    try:
        import psycopg
    except Exception:
        return []
    cards: list[dict[str, Any]] = []
    try:
        with psycopg.connect(DB_URL, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT manual_id, title, node_count, max_updated_at::text
                FROM lucidota_canon.api_bible_manuals
                ORDER BY manual_id
                LIMIT %s
                """,
                (min(limit, 20),),
            )
            for manual_id, title, node_count, updated in cur.fetchall():
                cards.append(
                    {
                        "id": f"manual:{manual_id}",
                        "type": "ONTOLOGY",
                        "label": f"{manual_id}: {title}",
                        "ref": {"kind": "postgrest_route", "route": f"/api_bible_nodes?manual_id=eq.{manual_id}", "manual_id": manual_id},
                        "summary": f"DB-backed manual slice, nodes={node_count}, updated={updated}",
                        "custody_state": "db_ref_only",
                    }
                )
            cur.execute(
                """
                SELECT id::text, sender_id, event_id, left(coalesce(clean_text, raw_text, ''), 120)
                FROM ironclaw.waking_dialogue_stream
                WHERE comms_channel='matrix' AND processed_status='queued'
                ORDER BY received_at DESC NULLS LAST, created_at DESC NULLS LAST
                LIMIT %s
                """,
                (min(limit, 10),),
            )
            for row_id, sender_id, event_id, text in cur.fetchall():
                cards.append(
                    {
                        "id": f"dialogue:{row_id}",
                        "type": "DATA",
                        "label": f"queued dialogue {row_id}",
                        "ref": {"kind": "postgres_row", "table": "ironclaw.waking_dialogue_stream", "id": row_id, "event_id": event_id},
                        "summary": f"{sender_id or 'unknown'}: {text}",
                        "custody_state": "db_ref_only",
                    }
                )
    except Exception as exc:
        cards.append(
            {
                "id": "db:unavailable",
                "type": "RECEIPT",
                "label": "DB card index unavailable",
                "ref": {"kind": "error", "database_url": DB_URL},
                "summary": f"DB card scan skipped: {type(exc).__name__}: {str(exc)[:160]}",
                "custody_state": "diagnostic_only",
            }
        )
    return cards


def build_card_index(limit_per_root: int = CARD_LIMIT_PER_ROOT) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rel_path in CORE_CARD_PATHS:
        path = ROOT / rel_path
        if path.exists() and path.is_file():
            card = file_card(path, summary="Core LUCIDOTA operator/runtime surface")
            cards.append(card)
            seen.add(card["id"])
    for label, root, suffixes in CARD_ROOTS:
        if not root.exists():
            continue
        count = 0
        for path in sorted(root.rglob("*")):
            if count >= limit_per_root:
                break
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if any(part in {"__pycache__", ".pytest_cache", "node_modules", "target"} for part in path.parts):
                continue
            card = file_card(path, summary=f"Indexed {label} card; content loads only on expand/stage")
            if card["id"] in seen:
                continue
            seen.add(card["id"])
            cards.append(card)
            count += 1
    cards.extend(db_cards())
    cards.extend(
        [
            {
                "id": "capability:stage-flow",
                "type": "CAPABILITY",
                "label": "Stage Flow",
                "ref": {"kind": "capability", "capability_key": "flow.stage"},
                "summary": "Write a staged flow spec and receipt; no hidden execution.",
                "custody_state": "local_capability",
            },
            {
                "id": "mutation:kronenberg",
                "type": "MUTATION",
                "label": "Kronenberg Mutation Node",
                "ref": {"kind": "mutation_template", "mode": "visible_proposal_only"},
                "summary": "Propose a new workflow/artifact/capability from selected nodes; must be tested before run.",
                "custody_state": "proposal_only",
            },
        ]
    )
    return cards


def normalize_flow_spec(payload: dict[str, Any] | None = None, *, cards: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    payload = dict(payload or {})
    flow_id = str(payload.get("flow_id") or f"flow_{uuid.uuid4().hex[:16]}")
    selected = cards or build_card_index(limit_per_root=8)
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        nodes = [
            {"id": "node-operator", "type": "PROMPT", "label": "Operator Intent", "x": 180, "y": 120, "ref": {"kind": "literal", "text": "/flow"}},
            {"id": "node-indy", "type": "SCRIPT", "label": "Indy_READs Queue", "x": 480, "y": 220, "ref": selected[1].get("ref") if len(selected) > 1 else {}},
        ]
    edges = payload.get("edges")
    if not isinstance(edges, list):
        edges = [{"id": "edge-operator-indy", "from": "node-operator", "to": "node-indy", "label": "visible staged handoff"}]
    return {
        "flow_id": flow_id,
        "name": str(payload.get("name") or "LUCIDOTA Flow Draft"),
        "status": str(payload.get("status") or "draft"),
        "nodes": nodes,
        "edges": edges,
        "selected_object_refs": payload.get("selected_object_refs") or [card.get("ref") for card in selected[:8]],
        "paths_hashes": payload.get("paths_hashes") or [card.get("ref") for card in selected[:8] if card.get("ref", {}).get("path")],
        "command_previews": payload.get("command_previews") or ["./luci flow batch --dag <flow> --eval <data> --run-id <id>"],
        "allowed_read_surfaces": payload.get("allowed_read_surfaces") or ["repo refs", "PostgREST safe GET routes", "ironclaw queued dialogue rows"],
        "allowed_write_surfaces": payload.get("allowed_write_surfaces") or ["05_OUTPUTS/flow/<flow_id>/", "luci_flow.flow_spec when migration exists"],
        "validation_tests": payload.get("validation_tests") or [".venv/bin/python -m pytest -q tests/test_luci_flow_app.py tests/test_luci_flow_wrapper.py"],
        "receipt_output_plan": payload.get("receipt_output_plan") or "05_OUTPUTS/flow/<flow_id>/receipt_<action>_<hash>.json",
        "rollback_stop_conditions": payload.get("rollback_stop_conditions") or ["validation fails", "operator presses Rollback", "DB write unavailable"],
        "created_at": payload.get("created_at") or utc_now(),
        "updated_at": utc_now(),
    }


def flow_dir(flow_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    safe = "".join(ch for ch in flow_id if ch.isalnum() or ch in {"-", "_"})[:96] or f"flow_{uuid.uuid4().hex[:12]}"
    return output_dir / safe


def maybe_write_flow_db(flow: dict[str, Any], receipt: dict[str, Any]) -> str:
    if os.environ.get("LUCI_FLOW_DISABLE_DB_WRITE"):
        return "db_skipped:disabled_by_env"
    try:
        import psycopg
    except Exception as exc:
        return f"db_skipped:{type(exc).__name__}"
    try:
        with psycopg.connect(DB_URL, connect_timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('luci_flow.flow_spec')::text, to_regclass('luci_flow.flow_receipt')::text")
            spec_table, receipt_table = cur.fetchone()
            if not spec_table or not receipt_table:
                return "db_skipped:luci_flow_tables_missing"
            cur.execute(
                """
                INSERT INTO luci_flow.flow_spec (flow_id, name, status, flow_json, nodes, edges, created_by, receipt_id)
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s)
                ON CONFLICT (flow_id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    flow_json=excluded.flow_json,
                    nodes=excluded.nodes,
                    edges=excluded.edges,
                    receipt_id=excluded.receipt_id,
                    updated_at=now()
                """,
                (
                    flow["flow_id"],
                    flow["name"],
                    flow["status"],
                    json.dumps(flow, sort_keys=True),
                    json.dumps(flow["nodes"], sort_keys=True),
                    json.dumps(flow["edges"], sort_keys=True),
                    os.environ.get("USER", "operator"),
                    receipt["receipt_id"],
                ),
            )
            cur.execute(
                """
                INSERT INTO luci_flow.flow_receipt (receipt_id, flow_id, action, status, output_path, output_hash, metrics)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (receipt_id) DO NOTHING
                """,
                (
                    receipt["receipt_id"],
                    flow["flow_id"],
                    receipt["action"],
                    receipt["status"],
                    receipt.get("output_path"),
                    receipt.get("output_hash"),
                    json.dumps(receipt.get("metrics", {}), sort_keys=True),
                ),
            )
            conn.commit()
            return "db_write_ok"
    except Exception as exc:
        return f"db_skipped:{type(exc).__name__}:{str(exc)[:160]}"


def write_flow_action(flow: dict[str, Any], action: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    if action not in ACTIONS:
        raise ValueError(f"unknown flow action: {action}")
    flow = normalize_flow_spec(flow)
    if action != "save":
        flow["status"] = "run" if action == "run" else f"{action}d" if action in {"stage", "validate", "promote"} else action
        flow["updated_at"] = utc_now()
    outdir = flow_dir(flow["flow_id"], output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    spec_path = outdir / f"{flow['flow_id']}.flow.json"
    spec_text = json.dumps(flow, indent=2, sort_keys=True, default=str) + "\n"
    spec_path.write_text(spec_text, encoding="utf-8")
    spec_hash = sha256_bytes(spec_text.encode("utf-8"))
    receipt_id = f"flow_{action}_{uuid.uuid4().hex[:16]}"
    receipt = {
        "schema": "lucidota.flow_receipt.v1",
        "receipt_id": receipt_id,
        "flow_id": flow["flow_id"],
        "action": action,
        "status": "ok",
        "created_at": utc_now(),
        "output_path": rel(spec_path),
        "output_hash": spec_hash,
        "metrics": {"nodes": len(flow.get("nodes", [])), "edges": len(flow.get("edges", []))},
        "command_preview": flow.get("command_previews", []),
        "explicit_operator_action_required": action in {"run", "promote", "rollback"},
    }
    receipt_path = outdir / f"receipt_{action}_{receipt_id}_{spec_hash[:12]}.json"
    receipt["receipt_path"] = rel(receipt_path)
    receipt["db_status"] = maybe_write_flow_db(flow, receipt)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return {"flow": flow, "spec_path": rel(spec_path), "receipt": receipt, "receipt_path": rel(receipt_path)}


def render_html() -> str:
    return """<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\" />
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
<title>LUCI /flow</title>
<style>
:root{color-scheme:dark;--bg:#081014;--panel:#111b22;--ink:#e8f4ff;--muted:#86a1aa;--accent:#7cf7c4;--line:#2a4652;--warn:#ffc857}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;height:100vh;overflow:hidden}.top{height:48px;background:#0c171d;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:10px;padding:8px 12px}.top b{color:var(--accent);font-size:16px}.top button{background:#132630;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:7px 10px;cursor:pointer}.top button:hover{border-color:var(--accent)}.grid{height:calc(100vh - 48px);display:grid;grid-template-columns:290px 1fr 340px;grid-template-rows:1fr 170px;grid-template-areas:'palette canvas inspector' 'console console console'}.palette{grid-area:palette;background:var(--panel);border-right:1px solid var(--line);padding:10px;overflow:auto}.canvas{grid-area:canvas;position:relative;overflow:hidden;background:radial-gradient(circle at 1px 1px,#16313b 1px,transparent 0);background-size:24px 24px}.inspector{grid-area:inspector;background:var(--panel);border-left:1px solid var(--line);padding:12px;overflow:auto}.console{grid-area:console;background:#060a0d;border-top:1px solid var(--line);padding:10px;overflow:auto;white-space:pre-wrap}.search{width:100%;padding:8px;border:1px solid var(--line);border-radius:8px;background:#071116;color:var(--ink);margin-bottom:10px}.card,.node{border:1px solid var(--line);background:#0d1b22;border-radius:10px;padding:8px;margin:8px 0;cursor:grab}.card .type,.node .type{color:var(--accent);font-size:11px}.card .summary{color:var(--muted);font-size:12px;max-height:42px;overflow:hidden}.node{position:absolute;width:210px;box-shadow:0 10px 26px #0008}.node.selected{outline:2px solid var(--accent)}svg.edges{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.hint{color:var(--muted)}code{color:var(--warn)}
</style>
</head>
<body>
<div class=\"top\"><b>LUCI /flow</b><span class=\"hint\">visual Promptflow canvas; refs only until expanded/staged</span><button onclick=\"saveFlow('save')\">Save Flow</button><button onclick=\"saveFlow('stage')\">Stage</button><button onclick=\"saveFlow('validate')\">Validate</button><button onclick=\"saveFlow('run')\">Run</button><button onclick=\"saveFlow('promote')\">Promote</button><button onclick=\"saveFlow('rollback')\">Rollback</button></div>
<div class=\"grid\"><aside class=\"palette\"><input id=\"search\" class=\"search\" placeholder=\"search cards\" oninput=\"paintCards()\"><div id=\"cards\"></div></aside><main id=\"canvas\" class=\"canvas\" ondragover=\"event.preventDefault()\" ondrop=\"dropCard(event)\"><svg class=\"edges\" id=\"edges\"></svg></main><aside class=\"inspector\"><h3>Inspector</h3><div id=\"inspect\" class=\"hint\">Select a node.</div><h4>Command preview</h4><code>./luci flow batch --dag &lt;flow&gt; --eval &lt;data&gt; --run-id &lt;id&gt;</code></aside><section id=\"console\" class=\"console\">/flow ready. Drag cards from left. Shift-click two nodes to wire. Stage/Run require explicit button press.\n</section></div>
<script>
let cards=[], nodes=[], edges=[], selected=null, wireFrom=null;
const log=x=>{document.getElementById('console').textContent += '\\n'+x; document.getElementById('console').scrollTop=999999};
async function init(){cards=await (await fetch('/api/cards')).json(); const t=await (await fetch('/api/flow-template')).json(); nodes=t.nodes; edges=t.edges; paintCards(); paintCanvas();}
function paintCards(){const q=document.getElementById('search').value.toLowerCase(); const el=document.getElementById('cards'); el.innerHTML=''; cards.filter(c=>(c.label+' '+c.type+' '+c.summary).toLowerCase().includes(q)).slice(0,160).forEach(c=>{let d=document.createElement('div'); d.className='card'; d.draggable=true; d.ondragstart=e=>e.dataTransfer.setData('text/plain',JSON.stringify(c)); d.innerHTML=`<div class=type>${c.type}</div><b>${escapeHtml(c.label)}</b><div class=summary>${escapeHtml(c.summary||'')}</div>`; el.appendChild(d);});}
function dropCard(e){e.preventDefault(); let c=JSON.parse(e.dataTransfer.getData('text/plain')); let r=e.currentTarget.getBoundingClientRect(); nodes.push({id:'node-'+crypto.randomUUID().slice(0,8),type:c.type,label:c.label,x:e.clientX-r.left,y:e.clientY-r.top,ref:c.ref,summary:c.summary}); paintCanvas(); log('attached ref '+c.id);}
function paintCanvas(){const cv=document.getElementById('canvas'); [...cv.querySelectorAll('.node')].forEach(n=>n.remove()); nodes.forEach(n=>{let d=document.createElement('div'); d.className='node'+(selected===n.id?' selected':''); d.style.left=n.x+'px'; d.style.top=n.y+'px'; d.innerHTML=`<div class=type>${n.type}</div><b>${escapeHtml(n.label)}</b><div class=summary>${escapeHtml((n.summary||JSON.stringify(n.ref||{})).slice(0,90))}</div>`; d.onclick=e=>selectNode(e,n.id); d.onmousedown=e=>dragNode(e,n.id); cv.appendChild(d);}); paintEdges();}
function selectNode(e,id){if(e.shiftKey){ if(wireFrom && wireFrom!==id){edges.push({id:'edge-'+crypto.randomUUID().slice(0,8),from:wireFrom,to:id,label:'manual wire'}); wireFrom=null; paintCanvas(); log('wired nodes'); return;} wireFrom=id; log('wire start '+id); return;} selected=id; const n=nodes.find(x=>x.id===id); document.getElementById('inspect').textContent=JSON.stringify(n,null,2); paintCanvas();}
function dragNode(e,id){if(e.shiftKey)return; const n=nodes.find(x=>x.id===id); const sx=e.clientX, sy=e.clientY, ox=n.x, oy=n.y; document.onmousemove=ev=>{n.x=ox+ev.clientX-sx;n.y=oy+ev.clientY-sy;paintCanvas()}; document.onmouseup=()=>{document.onmousemove=null;document.onmouseup=null};}
function paintEdges(){const s=document.getElementById('edges'); s.innerHTML=''; edges.forEach(ed=>{let a=nodes.find(n=>n.id===ed.from),b=nodes.find(n=>n.id===ed.to); if(!a||!b)return; let l=document.createElementNS('http://www.w3.org/2000/svg','line'); l.setAttribute('x1',a.x+105);l.setAttribute('y1',a.y+40);l.setAttribute('x2',b.x+105);l.setAttribute('y2',b.y+40);l.setAttribute('stroke','#7cf7c4');l.setAttribute('stroke-width','2');s.appendChild(l);});}
async function saveFlow(action){const spec={name:'Operator canvas '+new Date().toISOString(),status:'draft',nodes,edges,command_previews:['./luci flow batch --dag <flow> --eval <data> --run-id <id>']}; const r=await fetch('/api/action/'+action,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(spec)}); const p=await r.json(); log(action+' -> '+p.receipt.status+' spec='+p.spec_path+' receipt='+p.receipt_path+' db='+p.receipt.db_status);}
function escapeHtml(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
init().catch(e=>log('init error '+e));
</script>
</body></html>"""


class FlowHandler(BaseHTTPRequestHandler):
    server_version = "LuciFlow/0.1"

    def _json(self, payload: Any, status: int = 200) -> None:
        data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _text(self, text: str, content_type: str = "text/html; charset=utf-8") -> None:
        data = text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/":
            self._text(render_html())
        elif path == "/api/cards":
            self._json(build_card_index())
        elif path == "/api/flow-template":
            self._json(normalize_flow_spec(cards=build_card_index(limit_per_root=4)))
        elif path == "/api/status":
            self._json({"status": "ok", "app": "luci-flow", "root": str(ROOT), "time": utc_now()})
        else:
            self._json({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            self._json({"error": "bad_json", "detail": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/api/save":
            self._json(write_flow_action(payload, "save"))
            return
        if path.startswith("/api/action/"):
            action = path.rsplit("/", 1)[-1]
            if action not in ACTIONS:
                self._json({"error": "bad_action", "allowed": sorted(ACTIONS)}, HTTPStatus.BAD_REQUEST)
                return
            self._json(write_flow_action(payload, action))
            return
        self._json({"error": "not_found", "path": path}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: Any) -> None:  # keep console quiet for tests
        if os.environ.get("LUCI_FLOW_HTTP_LOG"):
            super().log_message(fmt, *args)


def smoke(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    cards = build_card_index(limit_per_root=8)
    flow = normalize_flow_spec(cards=cards)
    result = write_flow_action(flow, "save", output_dir)
    html_text = render_html()
    checks = {
        "left_palette": "palette" in html_text,
        "center_canvas": "canvas" in html_text,
        "right_inspector": "Inspector" in html_text,
        "bottom_console": "console" in html_text,
        "top_controls": all(token in html_text for token in ["Save Flow", "Stage", "Run", "Validate", "Promote", "Rollback"]),
        "cards_indexed": len(cards) > 0,
    }
    receipt = result["receipt"]
    return {
        "schema": "lucidota.flow_smoke.v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "cards": len(cards),
        "spec_path": result["spec_path"],
        "receipt_path": result["receipt_path"],
        "db_status": receipt.get("db_status"),
    }


def serve(host: str, port: int, *, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer((host, port), FlowHandler)
    actual = httpd.server_address[1]
    url = f"http://{host}:{actual}/"
    print(json.dumps({"status": "serving", "url": url, "app": "luci-flow"}, sort_keys=True), flush=True)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Open LUCI /flow visual canvas")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--flow", help="Optional PromptFlow DAG/ref to pre-seed as a canvas card")
    ap.add_argument("--data", help="Optional eval/data ref to pre-seed as a canvas card")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    if args.smoke:
        payload = smoke(out)
        if args.json:
            print(json.dumps(payload, sort_keys=True, default=str))
        else:
            print(f"FLOW_SMOKE={payload['status']} spec={payload['spec_path']} receipt={payload['receipt_path']} db={payload['db_status']}")
        return 0 if payload["status"] == "PASS" else 1
    return serve(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    raise SystemExit(main())
