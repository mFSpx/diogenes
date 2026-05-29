"""LUCIDOTA model router — local-first, Groq fallback, exponential backoff.

Tasks
-----
embed      list[str] -> list[list[float]]  FlagEmbedding/bge-m3 -> llama.cpp compat embed
ner_encode list[str] -> list[dict]         transformers/modernbert -> Groq llama-3.1-8b-instant
docparse   list[path] -> list[dict]        docling+smoldocling   -> Groq vision

Every call emits one JSON line to stderr:
  {"router": task, "backend": "local|groq|llamacpp", "batch_size": N, "latency_ms": N, "ok": bool}
"""
from __future__ import annotations
import json, os, sys, time, urllib.request
from pathlib import Path
from typing import Any

BGE_PATH   = os.environ.get("LUCIDOTA_BGE_MODEL_PATH",       "/home/mfspx/LUCIDOTA/04_RUNTIME/models/bge-m3")
MBERT_PATH = os.environ.get("LUCIDOTA_MODERNBERT_MODEL_PATH", "/home/mfspx/LUCIDOTA/04_RUNTIME/models/modernbert-base")
SMOL_PATH  = os.environ.get("LUCIDOTA_SMOLDOCLING_MODEL_PATH","/home/mfspx/LUCIDOTA/04_RUNTIME/models/smoldocling-256m-preview")
LLAMA_URL  = os.environ.get("LUCIDOTA_LLAMA_URL",             "http://127.0.0.1:8080/v1")
GROQ_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_URL   = "https://api.groq.com/openai/v1"
LOCAL_TO, GROQ_TO, RETRIES = 30, 60, 3


def _log(task: str, backend: str, n: int, t0: float, ok: bool) -> None:
    print(json.dumps({"router": task, "backend": backend, "batch_size": n,
                      "latency_ms": round((time.monotonic()-t0)*1000, 1), "ok": ok}),
          file=sys.stderr, flush=True)


def _sleep(attempt: int) -> None:
    time.sleep(min(2**attempt, 16))


def _post_json(url: str, payload: dict, api_key: str, timeout: int) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _retry(fn, retries: int, backoff: bool = True):
    """Run fn up to retries times; return (result, True) or (exc, False)."""
    for i in range(retries):
        try:
            return fn(), True
        except Exception as exc:
            last = exc
            if i < retries - 1 and backoff:
                _sleep(i)
    return last, False


# ---------------------------------------------------------------------------
# EMBED  (no Groq — Groq has no embed endpoint)
# ---------------------------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    t0 = time.monotonic()

    def _flag():
        from FlagEmbedding import FlagModel  # type: ignore
        return FlagModel(BGE_PATH, use_fp16=False, query_instruction_for_retrieval=None).encode(texts).tolist()

    def _st():
        from sentence_transformers import SentenceTransformer  # type: ignore
        return SentenceTransformer(BGE_PATH).encode(texts, convert_to_numpy=True).tolist()

    def _llama():
        body = _post_json(f"{LLAMA_URL.rstrip('/')}/embeddings",
                          {"model": "nomic-embed-text", "input": texts}, "", LOCAL_TO)
        return [d["embedding"] for d in sorted(body["data"], key=lambda x: x["index"])]

    for label, fn in [("local", _flag), ("local", _st), ("llamacpp", _llama)]:
        try:
            result, ok = _retry(fn, RETRIES)
            if ok:
                _log("embed", label, len(texts), t0, True)
                return result
        except TypeError:
            pass  # ImportError inside lambda → skip tier
    _log("embed", "local", len(texts), t0, False)
    raise RuntimeError("embed: all backends exhausted (no Groq embed endpoint)")


# ---------------------------------------------------------------------------
# NER / ENCODE
# ---------------------------------------------------------------------------
_NER_PROMPT = ("Extract named entities. Return ONLY valid JSON: "
               '{"entities":[{"text":str,"label":str}]}. Text: {text}')


def ner_encode(texts: list[str]) -> list[dict[str, Any]]:
    t0 = time.monotonic()

    def _local():
        from transformers import pipeline  # type: ignore
        pipe = pipeline("token-classification", model=MBERT_PATH, aggregation_strategy="simple")
        return [{"entities": [{"text": e["word"], "label": e["entity_group"],
                               "score": round(e["score"], 4)} for e in pipe(t[:512])]}
                for t in texts]

    result, ok = _retry(_local, RETRIES)
    if ok:
        _log("ner_encode", "local", len(texts), t0, True)
        return result

    if not GROQ_KEY:
        _log("ner_encode", "groq", len(texts), t0, False)
        raise RuntimeError("ner_encode: local failed and GROQ_API_KEY not set")

    out = []
    for text in texts:
        def _groq_one(t=text):
            body = _post_json(f"{GROQ_URL}/chat/completions", {
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": _NER_PROMPT.format(text=t[:4000])}],
                "temperature": 0, "response_format": {"type": "json_object"},
            }, GROQ_KEY, GROQ_TO)
            return json.loads(body["choices"][0]["message"]["content"])
        r, ok2 = _retry(_groq_one, RETRIES)
        out.append(r if ok2 else {"entities": [], "error": "groq_exhausted"})
    _log("ner_encode", "groq", len(texts), t0, True)
    return out


# ---------------------------------------------------------------------------
# DOCPARSE
# ---------------------------------------------------------------------------
_DOC_PROMPT = ('Parse this document. Return ONLY valid JSON: '
               '{"title":str,"sections":[{"heading":str,"body":str}],'
               '"tables":[],"metadata":{}}')


def docparse(paths: list[str | Path]) -> list[dict[str, Any]]:
    t0 = time.monotonic()
    ps = [Path(p) for p in paths]

    def _local():
        from docling.document_converter import DocumentConverter  # type: ignore
        smol = Path(SMOL_PATH)
        has_weights = smol.exists() and (any(smol.glob("*.bin")) or any(smol.glob("*.safetensors")))
        kw: dict = {"model_path": str(smol)} if has_weights else {}
        conv = DocumentConverter(**kw)
        results = []
        for p in ps:
            res = conv.convert(str(p))
            doc = getattr(res, "document", res)
            text = doc.export_to_markdown() if hasattr(doc, "export_to_markdown") else str(doc)
            results.append({"path": str(p), "text": text, "backend": "docling"})
        return results

    result, ok = _retry(_local, RETRIES)
    if ok:
        _log("docparse", "local", len(ps), t0, True)
        return result

    if not GROQ_KEY:
        _log("docparse", "groq", len(ps), t0, False)
        raise RuntimeError("docparse: local failed and GROQ_API_KEY not set")

    import base64
    out = []
    for p in ps:
        def _groq_one(p=p):
            suffix = p.suffix.lower().lstrip(".")
            mime = {"pdf":"application/pdf","png":"image/png",
                    "jpg":"image/jpeg","jpeg":"image/jpeg"}.get(suffix,"application/octet-stream")
            b64 = base64.b64encode(p.read_bytes()).decode()
            body = _post_json(f"{GROQ_URL}/chat/completions", {
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": _DOC_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]}],
                "temperature": 0, "response_format": {"type": "json_object"},
            }, GROQ_KEY, GROQ_TO)
            parsed = json.loads(body["choices"][0]["message"]["content"])
            parsed.update({"path": str(p), "backend": "groq"})
            return parsed
        r, ok2 = _retry(_groq_one, RETRIES)
        out.append(r if ok2 else {"path": str(p), "error": "groq_exhausted", "backend": "groq"})
    _log("docparse", "groq", len(ps), t0, True)
    return out
