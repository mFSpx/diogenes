#!/usr/bin/env python3
"""Minimal Anthropic Messages API shim over local llama.cpp OpenAI-compatible server."""
from __future__ import annotations
import argparse, json, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "http://127.0.0.1:8080/v1"
MODEL = "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"


def text_from_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    out.append(str(b.get("text", "")))
                elif b.get("type") == "tool_result":
                    out.append(str(b.get("content", "")))
        return "\n".join(x for x in out if x)
    return str(content or "")


def anthropic_to_openai(payload):
    # Local demo shim: keep prompts inside the 2k-8k llama.cpp context and ignore tool schemas.
    # Claw's full Claude system prompt + tools can exceed small local model limits.
    messages = []
    system = text_from_content(payload.get("system"))
    if system:
        system = system[-2500:]
        messages.append({"role": "system", "content": system})
    for m in payload.get("messages", [])[-6:]:
        role = m.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            role = "user"
        content = text_from_content(m.get("content"))[-2500:]
        messages.append({"role": role, "content": content})
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    requested = int(payload.get("max_tokens") or 512)
    return {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max(16, min(requested, 512)),
        "temperature": float(payload.get("temperature") or 0.2),
        "stream": False,
    }


def call_llama(payload):
    req = urllib.request.Request(
        UPSTREAM.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json", "authorization": "Bearer not-needed"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"upstream {exc.code}: {body}") from exc
    msg = data.get("choices", [{}])[0].get("message", {})
    text = msg.get("content") or msg.get("reasoning_content") or ""
    if not text.strip():
        text = "READY"
    usage = data.get("usage") or {}
    return text, usage


class Handler(BaseHTTPRequestHandler):
    server_version = "LucidotaAnthropicShim/0.1"

    def log_message(self, fmt, *args):
        return

    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse(self, event, payload):
        self.wfile.write(f"event: {event}\n".encode())
        self.wfile.write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode())
        self.wfile.flush()

    def do_GET(self):
        if self.path in ("/health", "/"):
            self._json(200, {"ok": True, "upstream": UPSTREAM, "model": MODEL})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/messages":
            self._json(404, {"error": {"type": "not_found", "message": "not found"}})
            return
        try:
            n = int(self.headers.get("content-length", "0"))
            payload = json.loads(self.rfile.read(n) or b"{}")
            text, usage = call_llama(anthropic_to_openai(payload))
            mid = "msg_lucidota_" + str(int(time.time() * 1000))
            if payload.get("stream"):
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.end_headers()
                self._sse("message_start", {"type":"message_start","message":{"id":mid,"type":"message","role":"assistant","model":MODEL,"content":[],"stop_reason":None,"stop_sequence":None,"usage":{"input_tokens":usage.get("prompt_tokens",0),"output_tokens":0}}})
                self._sse("content_block_start", {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})
                self._sse("content_block_delta", {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":text}})
                self._sse("content_block_stop", {"type":"content_block_stop","index":0})
                self._sse("message_delta", {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":None},"usage":{"input_tokens":usage.get("prompt_tokens",0),"output_tokens":usage.get("completion_tokens",0)}})
                self._sse("message_stop", {"type":"message_stop"})
            else:
                self._json(200, {"id":mid,"type":"message","role":"assistant","model":MODEL,"content":[{"type":"text","text":text}],"stop_reason":"end_turn","stop_sequence":None,"usage":{"input_tokens":usage.get("prompt_tokens",0),"output_tokens":usage.get("completion_tokens",0)}})
        except Exception as exc:
            self._json(500, {"error": {"type": "proxy_error", "message": str(exc)[:500]}})


def main():
    global UPSTREAM, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--upstream", default=UPSTREAM)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()
    UPSTREAM = args.upstream
    MODEL = args.model
    print(json.dumps({"ok": True, "proxy": f"http://{args.host}:{args.port}", "upstream": UPSTREAM, "model": MODEL}), flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
