#!/usr/bin/env python3
"""KRAMPUS ARCHIVE CRUSH — streaming, recursive, all-types ingest of the archive corpus.

The real corpus is ~76GB of archives in KRAMPUSCHEWING (zips, tars, phone backups,
nested archives) holding the ~90K markdowns + the legal document corpus. You do NOT
chunk a zip — you recursively unpack it.

Safety (born from the 2026-05-29 OOM freeze): NEVER read a whole archive or an
oversized member into RAM. Archives are opened by directory/stream; members are read
streamed; nested archives are spilled to a temp file in bounded chunks then recursed.
A bounded producer->queue->worker pipeline keeps memory FLAT regardless of corpus size.
Run under scripts/lucidota_capped_run.sh for a hard ceiling on top of that.

Lanes (honors the fast/slow-lane design):
  FAST  text/md/json/csv/html/...  -> chunk + embed via the BGE fleet -> corpus_chunk
  SLOW  pdf/office/image/audio/video/unknown -> deferred JSONL with provenance
        (a targeted pdftotext / OCR / transcribe pass consumes this next)

Reuses corpus_groq_extractor primitives (chunk_text, embed_text, write_chunk, sink).

Usage:
  scripts/lucidota_capped_run.sh env LUCIDOTA_BGE_FLEET="$(cat 04_RUNTIME/inference_os/bge_fleet.endpoints)" \\
    LUCIDOTA_MAX_WORKERS=6 python3 scripts/krampus_archive_crush.py --root KRAMPUSCHEWING
  ... --archives KRAMPUSCHEWING/docs_Luci-010.zip   # specific archives
  ... --limit 500                                   # cap members (validation)
"""
import os, sys, io, json, time, shutil, hashlib, logging, zipfile, tarfile, gzip, tempfile, threading, argparse
from pathlib import Path
from queue import Queue

sys.path.insert(0, str(Path(__file__).resolve().parent))
import corpus_groq_extractor as cge  # reuse: chunk_text, embed_text, write_chunk, connect_db, Counter

ROOT = Path(os.environ.get('LUCIDOTA_HOME', '/home/mfspx/LUCIDOTA'))
MEMBER_CAP = int(os.environ.get('LUCIDOTA_MEMBER_CAP_BYTES', str(64 * 1024 * 1024)))
MAX_DEPTH = int(os.environ.get('LUCIDOTA_ARCHIVE_MAX_DEPTH', '8'))
MAX_WORKERS = max(1, int(os.environ.get('LUCIDOTA_MAX_WORKERS', '2')))
STREAM_CHUNK = int(os.environ.get('LUCIDOTA_STREAM_CHUNK_BYTES', '65536'))
SLOW_LANE = Path(os.environ.get('LUCIDOTA_SLOW_LANE_FILE', str(ROOT / '04_RUNTIME/krampus_slow_lane.jsonl')))
PROGRESS = Path(os.environ.get('LUCIDOTA_PROGRESS_FILE', str(ROOT / '04_RUNTIME/krampus_archive_progress.json')))
TMP = ROOT / '04_RUNTIME/krampus_tmp'
TMP.mkdir(parents=True, exist_ok=True)

TEXT_EXT = {'.md', '.markdown', '.txt', '.text', '.json', '.jsonl', '.ndjson', '.csv', '.tsv',
            '.yaml', '.yml', '.html', '.htm', '.xml', '.log', '.rst', '.tex', '.srt', '.vtt',
            '.ini', '.cfg', '.toml', '.py', '.js', '.ts', '.rs', '.c', '.h', '.cpp', '.go', '.sh'}
ARCHIVE_EXT = {'.zip', '.tar', '.gz', '.tgz', '.bz2', '.tbz', '.xz', '.txz', '.7z', '.rar', '.zst'}
IMAGE_EXT = {'.png', '.jpg', '.jpeg', '.gif', '.tiff', '.tif', '.bmp', '.webp'}
# fast = text embedded locally; ocr = images transcribed via GROQ (cloud offload) then embedded.
LANE = os.environ.get('LUCIDOTA_LANE', 'fast').lower()

class Ctr:
    def __init__(self):
        self.lock = threading.Lock()
        self.archives = self.members = self.text = self.chunks = self.slow = self.skipped = self.errors = self.nested = 0
    def u(self, **k):
        with self.lock:
            for kk, v in k.items():
                setattr(self, kk, getattr(self, kk) + v)
    def snap(self):
        with self.lock:
            return {k: getattr(self, k) for k in
                    ('archives', 'members', 'text', 'chunks', 'slow', 'skipped', 'errors', 'nested')}

ctr = Ctr()
seen = set()
seen_lock = threading.Lock()
slow_lock = threading.Lock()
Q = Queue(maxsize=MAX_WORKERS * 8)   # bounded -> backpressure -> flat memory
LIMIT = None
_stop = threading.Event()

def stop():
    return _stop.is_set() or (LIMIT is not None and ctr.members >= LIMIT)

def load_seen():
    try:
        c = cge.connect_db(); cur = c.cursor()
        cur.execute("SELECT DISTINCT sha256 FROM lucidota_korpus.corpus_chunk")
        for (s,) in cur:
            seen.add(s)
        c.close()
        logging.info(f'resume: {len(seen)} member shas already ingested')
    except Exception as e:
        logging.warning(f'load_seen failed (continuing fresh): {e}')

def defer(provenance, ext, size):
    with slow_lock:
        with open(SLOW_LANE, 'a') as f:
            f.write(json.dumps({'src': provenance, 'ext': ext, 'size': size}) + '\n')
    ctr.u(slow=1)

def enqueue_text(data, provenance, ext):
    sha = hashlib.sha256(data).hexdigest()
    with seen_lock:
        if sha in seen:
            ctr.u(skipped=1); return
        seen.add(sha)
    try:
        text = data.decode('utf-8', 'ignore')
    except Exception:
        return
    if not text.strip():
        return
    chunks = cge.chunk_text(text)
    ctr.u(text=1)
    mime = 'text/' + (ext.lstrip('.') or 'plain')
    for i, ch in enumerate(chunks):
        Q.put((sha, i, ch, provenance, mime))

def handle_image(data, provenance, ext):
    # SLOW LANE: transcribe via GROQ vision (cloud), then embed the text locally.
    sha = hashlib.sha256(data).hexdigest()
    with seen_lock:
        if sha in seen:
            ctr.u(skipped=1); return
        seen.add(sha)
    try:
        text = cge.ocr_image(data, 'image/' + ext.lstrip('.'))
    except Exception as e:
        logging.error(f'ocr {provenance}: {e}'); ctr.u(errors=1); return
    if not text or not text.strip():
        ctr.u(skipped=1); return
    chunks = cge.chunk_text(text)
    ctr.u(text=1)
    for i, ch in enumerate(chunks):
        Q.put((sha, i, ch, provenance, 'image/' + ext.lstrip('.')))

def route_member(name, size, open_stream, prov_prefix, depth):
    if stop():
        return
    ctr.u(members=1)
    n = ctr.members
    if n % 500 == 0:
        write_progress()
    prov = f"{prov_prefix}!{name}"
    ext = os.path.splitext(name)[1].lower()
    if ext in ARCHIVE_EXT:
        ctr.u(nested=1)
        tmppath = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, dir=str(TMP)) as tmp:
                shutil.copyfileobj(open_stream(), tmp, STREAM_CHUNK)  # streamed spill, bounded RAM
                tmppath = tmp.name
            process_archive(tmppath, prov, depth + 1)
        except Exception as e:
            logging.error(f'nested {prov}: {e}'); ctr.u(errors=1)
        finally:
            if tmppath:
                try: os.unlink(tmppath)
                except OSError: pass
        return
    if LANE == 'ocr':
        # SLOW LANE: images -> GROQ vision OCR (cloud offload) -> embed. Ignore text/other.
        if ext in IMAGE_EXT:
            if size is not None and size > MEMBER_CAP:
                defer(prov, ext, size); return
            try:
                data = open_stream().read(MEMBER_CAP + 1)
            except Exception as e:
                logging.error(f'read {prov}: {e}'); ctr.u(errors=1); return
            if len(data) > MEMBER_CAP:
                defer(prov, ext, len(data)); return
            handle_image(data, prov, ext)
        return
    # FAST LANE: text embedded locally via the BGE fleet; media/unknown -> slow-lane queue.
    if ext in TEXT_EXT:
        if size is not None and size > MEMBER_CAP:
            defer(prov, ext, size); return
        try:
            data = open_stream().read(MEMBER_CAP + 1)
        except Exception as e:
            logging.error(f'read {prov}: {e}'); ctr.u(errors=1); return
        if len(data) > MEMBER_CAP:
            defer(prov, ext, len(data)); return
        enqueue_text(data, prov, ext)
        return
    defer(prov, ext, size or 0)   # pdf/office/image/audio/video/unknown -> slow lane

def process_archive(path, prov, depth):
    if depth > MAX_DEPTH:
        logging.warning(f'max depth {MAX_DEPTH} at {prov}'); return
    if stop():
        return
    ctr.u(archives=1)
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as zf:
                for info in zf.infolist():
                    if stop(): break
                    if info.is_dir(): continue
                    route_member(info.filename, info.file_size, lambda i=info: zf.open(i), prov, depth)
        elif tarfile.is_tarfile(path):
            with tarfile.open(path, 'r:*') as tf:
                for m in tf:
                    if stop(): break
                    if not m.isfile(): continue
                    route_member(m.name, m.size, lambda m=m: tf.extractfile(m), prov, depth)
        elif str(path).lower().endswith('.gz'):
            inner = os.path.basename(str(path))[:-3] or 'gz_member'
            route_member(inner, None, lambda: gzip.open(path, 'rb'), prov, depth)
        else:
            logging.warning(f'not a recognized archive: {prov}'); ctr.u(skipped=1)
    except Exception as e:
        logging.error(f'archive {prov}: {e}'); ctr.u(errors=1)

def consumer():
    while True:
        item = Q.get()
        if item is None:
            Q.task_done(); break
        sha, i, ch, prov, mime = item
        try:
            cge.write_chunk(sha, i, ch, prov, mime)
            ctr.u(chunks=1)
        except Exception as e:
            logging.error(f'write_chunk {prov}#{i}: {e}'); ctr.u(errors=1)
        Q.task_done()

def write_progress():
    try:
        with open(PROGRESS, 'w') as f:
            json.dump(ctr.snap(), f)
    except OSError:
        pass

def find_archives(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in ARCHIVE_EXT:
                out.append(os.path.join(dp, fn))
    return sorted(out)

def main():
    global LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None, help='dir to scan for top-level archives')
    ap.add_argument('--archives', nargs='*', default=None, help='explicit archive paths')
    ap.add_argument('--limit', type=int, default=None, help='cap members processed (validation)')
    ap.add_argument('--walk', default=None, help='os.walk a directory: ingest every LOOSE file by type and recurse archives (100% disk coverage)')
    args = ap.parse_args()
    LIMIT = args.limit

    if args.archives:
        archives = args.archives
    elif args.root:
        archives = find_archives(args.root)
    else:
        archives = find_archives(str(ROOT / 'KRAMPUSCHEWING'))
    print(f'KRAMPUS CRUSH: {len(archives)} top-level archives | workers={MAX_WORKERS} member_cap={MEMBER_CAP} fleet={cge.EMBED_ENDPOINTS}')
    logging.info(f'krampus crush start: {len(archives)} archives workers={MAX_WORKERS} cap={MEMBER_CAP}')

    load_seen()
    workers = [threading.Thread(target=consumer, daemon=True) for _ in range(MAX_WORKERS)]
    for w in workers:
        w.start()
    try:
        for a in archives:
            if stop(): break
            process_archive(a, os.path.basename(a), 0)
            write_progress()
            s = ctr.snap()
            print(f"  archives={s['archives']} members={s['members']} text={s['text']} chunks={s['chunks']} slow={s['slow']} nested={s['nested']} errors={s['errors']}")
    finally:
        for _ in workers:
            Q.put(None)
        Q.join()
        write_progress()
    print('DONE', ctr.snap())

if __name__ == '__main__':
    main()
