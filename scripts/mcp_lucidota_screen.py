#!/usr/bin/env python3
"""
MCP Server: lucidota-screen — Screen capture + vision analysis agent.

Tools:
  mcp__screen__capture    — Capture screenshot to CAS storage
  mcp__screen__analyze    — Capture + analyze with BitVLA vision
  mcp__screen__watch      — Set up a watch region with callback

Stdio MCP transport. For ./luci McpServerManager.
Occult Kesha ready.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CAS_DIR = ROOT / "03_VAULT" / "cas"
SCREENSHOT_DIR = ROOT / "04_RUNTIME" / "screenshots"


def _ensure_dirs():
    CAS_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


def _capture_screenshot() -> dict[str, Any]:
    """Capture a screenshot using headless Chromium or import."""
    _ensure_dirs()

    # Try scrot first (X11)
    ts = time.strftime("%Y%m%dT%H%M%S")
    filename = f"screenshot_{ts}.png"
    filepath = SCREENSHOT_DIR / filename

    # Method 1: scrot
    try:
        subprocess.run(
            ["scrot", "-z", str(filepath)],
            capture_output=True, timeout=10,
        )
        if filepath.exists() and filepath.stat().st_size > 1000:
            return _process_screenshot(filepath)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 2: import (ImageMagick)
    try:
        subprocess.run(
            ["import", "-window", "root", str(filepath)],
            capture_output=True, timeout=10,
        )
        if filepath.exists() and filepath.stat().st_size > 1000:
            return _process_screenshot(filepath)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 3: XDG desktop portal
    try:
        result = subprocess.run(
            ["dbus-send", "--print-reply", "--dest=org.freedesktop.portal.Desktop",
             "/org/freedesktop/portal/desktop",
             "org.freedesktop.portal.Screenshot.Screenshot"],
            capture_output=True, timeout=10,
        )
        return {"status": "portal_attempted", "detail": result.stdout[:200].decode(errors="replace")}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Method 4: Try reading from /dev/fb0 (Linux framebuffer)
    try:
        fb_data = Path("/dev/fb0").read_bytes()[:100]  # Just check if it exists
        return {"status": "framebuffer_available", "detail": f"fb0 readable, {len(fb_data)} bytes"}
    except (OSError, FileNotFoundError):
        pass

    return {"status": "error", "error": "No screenshot method available (try: sudo apt install scrot)"}


def _process_screenshot(filepath: Path) -> dict[str, Any]:
    """Store screenshot in CAS and return metadata."""
    try:
        data = filepath.read_bytes()
    except OSError as e:
        return {"status": "error", "error": f"Failed to read screenshot: {e}"}
    sha256 = hashlib.sha256(data).hexdigest()

    # Store in CAS
    cas_subdir = CAS_DIR / sha256[:4]
    cas_subdir.mkdir(parents=True, exist_ok=True)
    cas_path = cas_subdir / sha256
    cas_path.write_bytes(data)

    return {
        "status": "ok",
        "filename": filepath.name,
        "sha256": sha256,
        "size_bytes": len(data),
        "path": str(filepath.relative_to(ROOT)),
        "cas_uri": f"cas://{sha256}",
    }


def _analyze_with_bitvla(image_path: str, prompt: str) -> dict[str, Any]:
    """Send image to BitVLA vision server for analysis."""
    import urllib.request

    vision_url = "http://127.0.0.1:7845/vision/analyze"

    # Resolve path and guard against directory traversal
    if image_path.startswith("/"):
        full_path = image_path
    else:
        full_path = str((ROOT / image_path).resolve())
    # Ensure the resolved path is within the project root
    if not full_path.startswith(str(ROOT.resolve())):
        return {"status": "path_error", "error": f"Image path outside project root: {image_path}"}

    try:
        import http.client
        import mimetypes

        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        try:
            with open(full_path, "rb") as f:
                file_data = f.read()
        except OSError as e:
            return {"status": "read_error", "error": str(e), "image": image_path}

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="screenshot.png"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + file_data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="prompt"\r\n\r\n'
            f"{prompt}\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            vision_url,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            return result
    except Exception as e:
        # Vision server not running — return embedding-less result
        return {"status": "vision_unavailable", "error": str(e), "image": image_path}


# ─── MCP Protocol ──────────────────────────────────────────────────────

def jsonrpc_error(id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}}


def jsonrpc_result(id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id, "result": result}


_watch_jobs: dict[str, dict[str, Any]] = {}
_watch_counter: int = 0


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    global _watch_counter
    rid = req.get("id")
    method = req.get("method", "")

    if method == "initialize":
        return jsonrpc_result(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "lucidota-screen", "version": "0.1.0"},
        })

    if method == "notifications/initialized":
        return None

    if method == "tools/list":
        tools = [
            {
                "name": "mcp__screen__capture",
                "description": "Capture a screenshot and store in CAS",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "mcp__screen__analyze",
                "description": "Capture screenshot and analyze with BitVLA vision",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "prompt": {"type": "string", "description": "Analysis prompt (default: describe the screen)"},
                    },
                },
            },
            {
                "name": "mcp__screen__watch",
                "description": "Set up a screen watch job that polls and calls back on match",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string", "description": "Text condition to watch for"},
                        "interval_s": {"type": "integer", "description": "Poll interval in seconds (default: 30)"},
                        "callback": {"type": "string", "description": "Callback URL or pipe"},
                    },
                },
            },
            {
                "name": "mcp__screen__watch_status",
                "description": "List active screen watch jobs",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
        return jsonrpc_result(rid, {"tools": tools})

    if method == "tools/call":
        params = req.get("params", {})
        name = params.get("name", "")
        arguments = params.get("arguments", {})

        if name == "mcp__screen__capture":
            result = _capture_screenshot()
            return jsonrpc_result(rid, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})

        if name == "mcp__screen__analyze":
            prompt = arguments.get("prompt", "Describe what's on this screen in detail.")
            capture = _capture_screenshot()
            if capture.get("status") == "ok":
                analysis = _analyze_with_bitvla(capture["path"], prompt)
                return jsonrpc_result(rid, {
                    "content": [{"type": "text", "text": json.dumps({
                        "capture": capture,
                        "analysis": analysis,
                    }, indent=2)}]
                })
            return jsonrpc_result(rid, {
                "content": [{"type": "text", "text": json.dumps({
                    "capture": capture,
                    "analysis": {"status": "skipped", "reason": "capture failed"},
                }, indent=2)}]
            })

        if name == "mcp__screen__watch":
            condition = arguments.get("condition", "")
            interval_s = int(arguments.get("interval_s", 30))
            interval_s = max(1, interval_s)  # Minimum 1s interval
            callback = arguments.get("callback", "")
            _watch_counter += 1
            job_id = f"watch_{_watch_counter}"
            _watch_jobs[job_id] = {
                "condition": condition,
                "interval_s": interval_s,
                "callback": callback,
                "status": "active",
                "created": time.time(),
            }
            return jsonrpc_result(rid, {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "ok",
                    "job_id": job_id,
                    "note": "Watch jobs are polled externally. Use watch_status to check matches.",
                }, indent=2)}]
            })

        if name == "mcp__screen__watch_status":
            return jsonrpc_result(rid, {
                "content": [{"type": "text", "text": json.dumps({
                    "active_jobs": len(_watch_jobs),
                    "jobs": _watch_jobs,
                }, indent=2)}]
            })

        return jsonrpc_error(rid, -32601, f"Unknown tool: {name}")

    if method == "shutdown":
        return jsonrpc_result(rid, None)

    return jsonrpc_error(rid, -32601, f"Method not found: {method}")


def main():
    """MCP stdio transport."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
        except json.JSONDecodeError:
            resp = jsonrpc_error(None, -32700, "Parse error")
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception:
            resp = jsonrpc_error(None, -32603, traceback.format_exc())
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
