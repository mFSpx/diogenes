import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
import psycopg
import requests

def _load_groq_key():
    # Only needed for image OCR. Read from env or the secure secrets file; NEVER /tmp.
    key = os.environ.get('GROQ_API_KEY', '').strip()
    if key:
        return key
    secrets = Path.home() / '.config' / 'lucidota' / 'secrets.env'
    try:
        for line in secrets.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('GROQ_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return ''

GROQ_API_KEY = _load_groq_key()
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
BGE_FLEET = ['http://127.0.0.1:8101', 'http://127.0.0.1:8102', 'http://127.0.0.1:8103', 'http://127.0.0.1:8104']
CAS_ROOT = Path('03_VAULT/cas')
PROGRESS_FILE = Path('04_RUNTIME/corpus_extract_progress.json')
LOG_FILE = Path('04_RUNTIME/corpus_extract_log.txt')

# Safe-ops caps — advisory env vars ENFORCED here. Born from the 2026-05-29 OOM
# freeze: an unbounded read of a 10.7GB CAS file plus an uncapped 8-thread fan-out
# global-OOM'd this 7.6GB box. These ceilings make that impossible.
MAX_FILE_BYTES = int(os.environ.get('LUCIDOTA_MAX_FILE_BYTES', '1500000'))
MAX_WORKERS = max(1, int(os.environ.get('LUCIDOTA_MAX_WORKERS', '1')))
MAX_BATCH = max(1, int(os.environ.get('LUCIDOTA_MAX_BATCH', '16')))

def _norm_embed(base):
    base = base.strip().rstrip('/')
    if base.endswith('/embeddings'):
        return base
    if base.endswith('/v1'):
        return base + '/embeddings'
    return base + '/v1/embeddings'

def _embed_endpoints():
    # Horizontal embed fleet: LUCIDOTA_BGE_FLEET is a comma-separated list of base
    # URLs (per-mission scale, set by scripts/lucidota_bge_fleet.sh). Falls back to
    # a single LUCIDOTA_BGE_EMBED_URL, then the default local fleet. embed_text
    # round-robins across whatever is live, so the fleet scales up/down freely.
    fleet = os.environ.get('LUCIDOTA_BGE_FLEET', '').strip()
    if fleet:
        return [_norm_embed(u) for u in fleet.split(',') if u.strip()]
    env = os.environ.get('LUCIDOTA_BGE_EMBED_URL', '').strip()
    if env:
        return [_norm_embed(env)]
    return [_norm_embed(u) for u in BGE_FLEET]

EMBED_ENDPOINTS = _embed_endpoints()

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class Counter:
    def __init__(self):
        self.files_done = 0
        self.chunks = 0
        self.embeds = 0
        self.deferred = 0
        self.errors = 0
        self.lock = threading.Lock()

    def update(self, files_done=0, chunks=0, embeds=0, deferred=0, errors=0):
        with self.lock:
            self.files_done += files_done
            self.chunks += chunks
            self.embeds += embeds
            self.deferred += deferred
            self.errors += errors

counter = Counter()
_tls = threading.local()
_total_files = 0

def connect_db():
    return psycopg.connect('postgresql:///lucidota_storage')

def get_files(limit=None, resume=True):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute('SELECT file_uuid, sha256, cas_uri, mime, file_kind FROM lucidota_korpus.file_object')
    files = cur.fetchall()
    if resume:
        cur.execute('SELECT sha256 FROM lucidota_korpus.corpus_chunk')
        existing_shas = set(row[0] for row in cur.fetchall())
        files = [file for file in files if file[1] not in existing_shas]
    if limit:
        files = files[:limit]
    return files

def read_cas_file(sha256):
    path = CAS_ROOT / sha256[:2] / sha256[2:4] / sha256
    if not path.exists():
        return None
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        # Capability preserved as a deferred lane, not deleted: oversized files
        # need a streaming extractor; never slurp them whole on a 7.6GB box.
        logging.warning(f'deferring oversized cas file {sha256} ({size}B > {MAX_FILE_BYTES}B)')
        counter.update(deferred=1)
        return None
    return path.read_bytes()

def extract_text(file_bytes, mime):
    if mime.startswith('text/'):
        return file_bytes.decode('utf-8', errors='ignore')
    elif mime.startswith('image/'):
        return ocr_image(file_bytes, mime)
    elif mime == 'application/zip':
        return extract_zip(file_bytes)
    else:
        return None

def ocr_image(file_bytes, mime):
    if not GROQ_API_KEY:
        return ""
    try:
        b64 = base64.b64encode(file_bytes).decode('utf-8')
        payload = {'model': 'meta-llama/llama-4-scout-17b-16e-instruct', 'max_tokens': 1024, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Transcribe ALL text in this image.'}, {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{b64}'}}]}]}
        headers = {'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json', 'User-Agent': 'groq-python/0.28.0'}
        for attempt in range(4):
            response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=90)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            if response.status_code == 429:   # rate limited -> back off and retry
                time.sleep(2 * (attempt + 1)); continue
            logging.error(f'ocr_image http {response.status_code}')
            return ""
        return ""
    except Exception as e:
        logging.error(f'Error ocr_image: {e}')
        return ""

def extract_zip(file_bytes):
    import zipfile
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as zip_file:
        texts = []
        for member in zip_file.infolist():
            if member.file_size > 10 * 1024 * 1024:
                continue
            with zip_file.open(member) as f:
                file_bytes = f.read()
                mime = get_mime(member.filename)
                text = extract_text(file_bytes, mime)
                if text:
                    texts.append(text)
        return '\n'.join(texts)

def get_mime(filename):
    if filename.endswith('.txt'):
        return 'text/plain'
    elif filename.endswith('.jpg') or filename.endswith('.png'):
        return 'image/jpeg'
    elif filename.endswith('.zip'):
        return 'application/zip'
    else:
        return 'application/octet-stream'

def chunk_text(text):
    chunks = []
    for i in range(0, len(text), 1800):
        chunk = text[i:i+1800]
        chunks.append(chunk)
    return chunks

def embed_text(chunk):
    import random
    last = None
    for _ in range(4):
        url = random.choice(EMBED_ENDPOINTS)
        try:
            r = requests.post(url, json={'model': 'bge-m3', 'input': chunk},
                              headers={'User-Agent': 'groq-python/0.28.0'}, timeout=60)
            if r.status_code == 200:
                return r.json()['data'][0]['embedding']
            # "input too large to process" / transient server error -> shrink + retry elsewhere
            if r.status_code in (400, 413, 500) and len(chunk) > 400:
                chunk = chunk[: max(400, len(chunk) // 2)]
                last = f'{r.status_code}->shrink {len(chunk)}'
                continue
            last = f'http {r.status_code}'
        except Exception as e:
            last = type(e).__name__
    raise RuntimeError(f'embed failed after retries: {last}')

def write_chunk(sha256, chunk_index, chunk, source_path, mime):
    if not hasattr(_tls, 'db'):
        _tls.db = connect_db()
    cur = _tls.db.cursor()
    chunk_uuid = str(uuid5(NAMESPACE_URL, sha256 + source_path + str(chunk_index)))
    embedding = embed_text(chunk)
    cur.execute('INSERT INTO lucidota_korpus.corpus_chunk (chunk_uuid, file_uuid, sha256, source_path, mime, chunk_index, content, go25, embedding, embedding_model, extractor) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (sha256, chunk_index) DO NOTHING', (chunk_uuid, None, sha256, source_path, mime, chunk_index, chunk, '{}', '['+','.join(repr(float(x)) for x in embedding)+']', 'bge-m3', 'groq:llama-4-scout'))
    _tls.db.commit()
    counter.update(chunks=1, embeds=1)

def process_file(file_uuid, sha256, cas_uri, mime, file_kind):
    try:
        file_bytes = read_cas_file(sha256)
        if file_bytes is None:
            logging.error(f'Error reading file {file_uuid}: file not found')
            return
        text = extract_text(file_bytes, mime)
        if text:
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                write_chunk(sha256, i, chunk, 'cas://' + sha256, mime)
        counter.update(files_done=1)
        if counter.files_done % 20 == 0:
            print(f'EXTRACT files={counter.files_done} chunks={counter.chunks} embeds={counter.embeds} deferred={counter.deferred} errors={counter.errors}')
            with open(PROGRESS_FILE, 'w') as f:
                json.dump({'files_done': counter.files_done, 'chunks': counter.chunks, 'embeds': counter.embeds, 'deferred': counter.deferred, 'errors': counter.errors, 'total': _total_files, 'last_path': '', 'started_at': datetime.now().isoformat()}, f)
    except Exception as e:
        logging.error(f'Error processing file {file_uuid}: {e}')
        counter.update(errors=1)

async def _run_one(executor, semaphore, file):
    async with semaphore:
        await asyncio.get_running_loop().run_in_executor(executor, process_file, *file)

async def main(limit=None, resume=True):
    global _total_files
    files = get_files(limit, resume)
    _total_files = len(files)
    logging.info(f'corpus extract start: {_total_files} files max_workers={MAX_WORKERS} max_batch={MAX_BATCH} max_file_bytes={MAX_FILE_BYTES}')
    semaphore = asyncio.Semaphore(MAX_WORKERS)
    # Bounded fan-out: at most MAX_WORKERS files in flight, drained in MAX_BATCH
    # slices so we never hold O(all-files) futures (or their file bytes) at once.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for start in range(0, len(files), MAX_BATCH):
            batch = files[start:start + MAX_BATCH]
            await asyncio.gather(*[_run_one(executor, semaphore, f) for f in batch])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--resume', action='store_true', default=True)
    args = parser.parse_args()
    start_time = time.time()
    asyncio.run(main(args.limit, args.resume))