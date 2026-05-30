import os
import sys
import json
import uuid
import hashlib
import base64
import requests
import psycopg2
import subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def get_mime_type(file_path):
    mime_types = {
        'txt': 'text/plain',
        'md': 'text/markdown',
        'json': 'application/json',
        'csv': 'text/csv',
        'log': 'text/plain',
        'html': 'text/html',
        'xml': 'application/xml',
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'webp': 'image/webp',
        'mp4': 'video/mp4',
        'mov': 'video/quicktime',
        'mp3': 'audio/mpeg',
        'wav': 'audio/wav',
        'zip': 'application/zip',
        'gz': 'application/gzip',
        'tar': 'application/x-tar',
        '7z': 'application/x-7z-compressed',
        'gguf': 'application/gguf',
        'bin': 'application/octet-stream',
        'so': 'application/octet-stream',
        'pkl': 'application/octet-stream',
        'onnx': 'application/octet-stream'
    }
    file_ext = file_path.split('.')[-1]
    return mime_types.get(file_ext, 'application/octet-stream')

def extract_text(file_path):
    mime_type = get_mime_type(file_path)
    if mime_type in ['text/plain', 'text/markdown', 'application/json', 'text/csv', 'text/html', 'application/xml']:
        with open(file_path, 'r', errors='ignore') as f:
            text = f.read()
            if mime_type == 'text/html':
                text = text.replace('<', ' ').replace('>', ' ')
            return text
    elif mime_type in ['application/pdf', 'image/png', 'image/jpeg', 'image/webp']:
        with open(file_path, 'rb') as f:
            image_data = f.read()
            image_url = 'data:' + mime_type + ';base64,' + base64.b64encode(image_data).decode('utf-8')
            response = requests.post('https://api.groq.com/openai/v1/chat/completions', json={
                'model': 'meta-llama/llama-4-scout-17b-16e-instruct',
                'prompt': 'Transcribe ALL text',
                'image_url': image_url
            }, headers={'Authorization': f'Bearer {open("/tmp/lucidota_groq_key", "r").read().strip()}'})
            return response.json()['choices'][0]['text']
    else:
        return None

def chunk_text(text):
    chunks = []
    for i in range(0, len(text), 1800):
        chunk = text[i:i+1800]
        if len(chunk.split()) > 500:
            chunk = ' '.join(chunk.split()[:500])
        chunks.append(chunk)
    return chunks

def embed_text(chunk):
    response = requests.post('http://127.0.0.1:8101/v1/embeddings', json={
        'model': 'bge-m3',
        'input': chunk
    }, headers={'User-Agent': 'lucidota'})
    return response.json()['embeddings']

def write_to_db(component):
    try:
        conn = psycopg2.connect(
            dbname='lucidota',
            user='lucidota',
            password='lucidota',
            host='localhost',
            port='5432'
        )
        cur = conn.cursor()
        cur.execute('INSERT INTO lucidota_korpus.component (component_uuid, file_uuid, component_index, component_kind, language, title, symbol, start_line, end_line, sha256, token_count, content, go_terms, minhash_signature, entropy, river_decision, river_score, vibe_spike, river_features, embedding_model, embedding_status, embedded_at, graph_item_uuid, detail, created_at, parent_component_uuid, fts_vector, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', (
            component['component_uuid'],
            component['file_uuid'],
            component['component_index'],
            component['component_kind'],
            component['language'],
            component['title'],
            component['symbol'],
            component['start_line'],
            component['end_line'],
            component['sha256'],
            component['token_count'],
            component['content'],
            component['go_terms'],
            component['minhash_signature'],
            component['entropy'],
            component['river_decision'],
            component['river_score'],
            component['vibe_spike'],
            component['river_features'],
            component['embedding_model'],
            component['embedding_status'],
            component['embedded_at'],
            component['graph_item_uuid'],
            component['detail'],
            component['created_at'],
            component['parent_component_uuid'],
            component['fts_vector'],
            component['embedding']
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logging.error(f'Error writing to DB: {e}')

def process_file(file_path):
    try:
        file_hash = hashlib.sha256(open(file_path, 'rb').read()).hexdigest()
        file_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, file_hash)
        content_addressed_path = f'03_VAULT/cas/{file_hash[0:2]}/{file_hash[2:4]}/{file_hash}'
        if not os.path.exists(content_addressed_path):
            os.makedirs(os.path.dirname(content_addressed_path), exist_ok=True)
            with open(content_addressed_path, 'wb') as f:
                f.write(open(file_path, 'rb').read())
        text = extract_text(file_path)
        if text is None:
            logging.info(f'Skipping file {file_path} (unsupported format)')
            return
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            component = {
                'component_uuid': uuid.uuid5(uuid.NAMESPACE_DNS, f'{file_hash}{i}'),
                'file_uuid': file_uuid,
                'component_index': i,
                'component_kind': 'text',
                'language': 'en',
                'title': os.path.basename(file_path),
                'symbol': '',
                'start_line': 0,
                'end_line': 0,
                'sha256': file_hash,
                'token_count': len(chunk.split()),
                'content': chunk,
                'go_terms': {},
                'minhash_signature': {},
                'entropy': 0.0,
                'river_decision': '',
                'river_score': 0,
                'vibe_spike': False,
                'river_features': {},
                'embedding_model': 'bge-m3',
                'embedding_status': 'pending',
                'embedded_at': None,
                'graph_item_uuid': None,
                'detail': {},
                'created_at': datetime.now(),
                'parent_component_uuid': None,
                'fts_vector': '',
                'embedding': None
            }
            with ThreadPoolExecutor(max_workers=6) as executor:
                embedding = executor.submit(embed_text, chunk).result()
            component['embedding'] = embedding
            component['embedding_status'] = 'embedded'
            write_to_db(component)
    except Exception as e:
        logging.error(f'Error processing file {file_path}: {e}')

def main():
    source_dir = sys.argv[1] if len(sys.argv) > 1 else '09_STORAGE/krampuschewing_unpacked'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    resume = sys.argv[3] if len(sys.argv) > 3 else None
    progress_file = '04_RUNTIME/corpus_ingest_progress.json'
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress = json.load(f)
    else:
        progress = {
            'done': 0,
            'skipped': 0,
            'chunks': 0,
            'embeds': 0,
            'errors': 0,
            'last_path': None,
            'started_at': datetime.now().isoformat(),
            'total': 0
        }
    file_paths = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
    if limit is not None:
        file_paths = file_paths[:limit]
    progress['total'] = len(file_paths)
    for file_path in file_paths:
        if resume is not None and file_path != resume:
            continue
        process_file(file_path)
        progress['done'] += 1
        progress['last_path'] = file_path
        with open(progress_file, 'w') as f:
            json.dump(progress, f)
        if progress['done'] % 25 == 0:
            logging.info(f'INGEST done={progress["done"]} skipped={progress["skipped"]} chunks={progress["chunks"]} embeds={progress["embeds"]} errors={progress["errors"]}')

if __name__ == '__main__':
    main()