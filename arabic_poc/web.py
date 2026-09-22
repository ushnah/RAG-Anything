"""Local-only demo server: python -m arabic_poc.web."""
import argparse
import json
import mimetypes
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / 'web_assets'
DATASETS = {
    'library': {'label': 'Library', 'storage': 'rag_storage_arabic_video_search', 'description': 'All sources', 'questions': ['Show me video of Makkah', 'أرني القبة الخضراء']},
    'videos': {'label': 'All videos', 'storage': 'rag_storage_arabic_video_search',
               'description': 'Search descriptions, speech, OCR and source labels together',
               'questions': ['Show me video of Makkah', 'أرني فيديو للقبة الخضراء', 'Show me a video about Wikidata']},
    'visual': {'label': 'Images & scenes', 'storage': 'rag_storage_arabic_visual',
               'description': 'Visual descriptions and source labels; candidate matches',
               'questions': ['أرني صورة باب الملك عبد العزيز في مكة', 'Show me video of the mosque courtyard']},
    'text': {'label': 'Arabic library', 'storage': 'rag_storage_arabic_demo',
             'description': 'A short fictional library document',
             'questions': ['من يشرف على فهرسة المخطوطات؟', 'متى تفتح المكتبة أبوابها؟']},
    'spoken': {'label': 'Spoken Arabic', 'storage': 'rag_storage_arabic_spoken',
               'description': 'A Tunisian tutorial lecture and a Syrian Arabic speaker',
               'questions': ['ما هوايات المتحدث في الفيديو؟', 'ما الخدمة التي يذكرها المتحدث لاستخراج المعطيات من ويكي بيانات؟']},
    'media': {'label': 'Audio & video', 'storage': 'rag_storage_arabic_media_final',
              'description': 'Arabic narration, a greeting, and a captioned video',
              'questions': ['كيف يعرّف التسجيل الطائرة؟', 'ما المحتوى الموجود في جهاز الإنترنت في صندوق كما يوضح الفيديو؟']},
}
ACTIVE_INDEXES = ROOT / 'arabic_poc' / 'active_indexes.json'
if ACTIVE_INDEXES.exists():
    active = json.loads(ACTIVE_INDEXES.read_text())
    for key, storage in active['datasets'].items():
        DATASETS[key]['storage'] = storage
        if key in ('text', 'spoken', 'media'):
            DATASETS[key]['description'] = 'Refreshed multimodal corpus: graph + vector retrieval'
JOBS = {}
LOCK = threading.Lock()
BUSY = threading.Lock()


def catalog(dataset):
    if dataset in ('videos', 'library'):
        file = ROOT / DATASETS[dataset]['storage'] / 'index.json'
        return json.loads(file.read_text())['records'] if file.exists() else []
    if dataset == 'visual':
        file = ROOT / DATASETS[dataset]['storage'] / 'visual-index.json'
        return json.loads(file.read_text())['records'] if file.exists() else []
    return [r for p in sorted((ROOT / DATASETS[dataset]['storage'] / 'sources').glob('*.json'))
            for r in json.loads(p.read_text())['records']]


def public_record(row, dataset):
    item = dict(row)
    item.pop('source', None)
    item.pop('normalized_text', None)
    item['source_key'] = __import__('hashlib').sha256(row['source'].encode()).hexdigest()[:24]
    item['media_url'] = f'/api/media/{dataset}/{row["id"]}'
    suffix = Path(row['source']).suffix.lower()
    item['media_type'] = ('video' if suffix in {'.webm', '.mp4', '.mov', '.mkv', '.avi'} else
                          'audio' if suffix in {'.ogg', '.wav', '.mp3', '.m4a', '.flac', '.opus', '.aac'} else
                          'image' if suffix in {'.png', '.jpg', '.jpeg', '.webp'} else 'document')
    return item


def public_answer(answer, dataset, mode):
    answer['sources'] = [public_record(row, dataset) for row in answer.get('sources', [])]
    answer['mode'] = mode
    return answer


def run_question(job_id, dataset, question, media_type='all'):
    started = time.monotonic()
    try:
        env = os.environ.copy()
        manifest = ROOT / DATASETS[dataset]['storage'] / 'index-settings.json'
        if manifest.exists():
            settings = json.loads(manifest.read_text())
            for key in ('BGE_MODEL', 'RAG_TOKENIZER_MODEL'):
                value = settings.get(key)
                if value and value != 'tiktoken-default':
                    env[key] = value
        command = ([sys.executable, '-m', 'arabic_poc.visual', '--storage',
            DATASETS[dataset]['storage'], 'search', question, '--media-type', media_type]
            if dataset == 'visual' else [sys.executable, '-m', 'arabic_poc', '--storage',
            DATASETS[dataset]['storage'], 'ask', question, '--top-k', '3'])
        if dataset in ('videos', 'library'):
            command = [sys.executable, '-m', 'arabic_poc.video_search', 'search', question, '--media-type', media_type if dataset == 'library' else 'video']
        proc = subprocess.run(command,
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=480)
        if proc.returncode:
            # Avoid exposing configuration or a traceback through the browser.
            print(proc.stderr[-4000:], file=sys.stderr, flush=True)
            raise RuntimeError('The local RAG request failed. Check the configured model service and the server terminal.')
        answer = json.loads(proc.stdout)
        if dataset in ('library', 'videos', 'visual'):
            from .answers import answer_from_evidence
            answer = answer_from_evidence(question, answer, catalog(dataset))
        result = public_answer(answer, dataset, 'live')
        with LOCK:
            JOBS[job_id].update(status='done', result=result, elapsed=round(time.monotonic()-started, 1))
    except Exception as exc:
        print(f'Demo query failed: {exc}', file=sys.stderr, flush=True)
        message = ('The request timed out. Try a shorter question or inspect the local model server.'
                   if isinstance(exc, subprocess.TimeoutExpired) else
                   'The local RAG request failed. Check the configured model service and the server terminal.')
        with LOCK:
            JOBS[job_id].update(status='error', error=message)
    finally:
        BUSY.release()


class Handler(BaseHTTPRequestHandler):
    def parse_request(self):
        if not super().parse_request():
            return False
        port = self.server.server_port
        if self.headers.get('Host') not in {f'127.0.0.1:{port}', f'localhost:{port}'}:
            self.send_error(403, 'Local host required')
            return False
        return True

    def json_response(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/api/catalog':
            datasets = []
            for key, spec in DATASETS.items():
                rows = catalog(key)
                if key == 'videos':
                    rows = [r for r in rows if Path(r['source']).suffix.lower() in {'.mp4','.webm','.mov','.mkv','.avi'}]
                if key in ('videos', 'library'):
                    rows = list({r['video_id']: r for r in reversed(rows)}.values())
                datasets.append(dict(id=key, label=spec['label'], description=spec['description'],
                    questions=spec['questions'], count=len(rows),
                    sources=[public_record(r, key) for r in rows]))
            return self.json_response({'datasets': datasets})
        if path.startswith('/api/jobs/'):
            with LOCK:
                job = JOBS.get(path.rsplit('/', 1)[-1])
                return self.json_response(job or {'error': 'Job not found'}, 200 if job else 404)
        if path.startswith('/api/examples/'):
            name = path.rsplit('/', 1)[-1]
            if name not in ('audio', 'video'):
                return self.json_response({'error': 'Unknown example'}, 404)
            file = ROOT / 'arabic_poc' / 'validation' / f'media_{name}_answer.json'
            if not file.exists():
                return self.json_response({'error': 'Saved example unavailable'}, 404)
            return self.json_response(public_answer(json.loads(file.read_text()), 'media', 'saved'))
        if path.startswith('/api/media/'):
            parts = path.split('/')
            if len(parts) != 5 or parts[3] not in DATASETS:
                return self.send_error(404)
            row = next((r for r in catalog(parts[3]) if r['id'] == parts[4]), None)
            if not row:
                return self.send_error(404)
            file = Path(row['source']).resolve()
            allowed = [ROOT / 'data', ROOT / 'arabic_poc' / 'demo']
            if not any(file.is_relative_to(folder) for folder in allowed) or not file.is_file():
                return self.send_error(404)
            return self.serve_file(file)
        files = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}
        if path in files:
            return self.serve_file(ASSETS / files[path])
        self.send_error(404)

    def serve_file(self, path):
        size = path.stat().st_size
        start, end = 0, size-1
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.send_error(416)
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), end) if right else end
            else:
                start = max(0, size-int(right))
            if start > end or start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
        self.send_response(206 if requested else 200)
        self.send_header('Content-Type', mimetypes.guess_type(path)[0] or 'application/octet-stream')
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Length', str(end-start+1))
        if requested:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as file:
            file.seek(start)
            remaining = end-start+1
            while remaining > 0:
                chunk = file.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        if self.path != '/api/ask':
            return self.json_response({'error': 'Not found'}, 404)
        # Block cross-origin pages from triggering local model work.
        origin = self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':
            return self.json_response({'error': 'Origin not allowed'}, 403)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 10000:
                raise ValueError()
            body = json.loads(self.rfile.read(length))
            dataset, question = body.get('dataset'), body.get('question', '')
            media_type = body.get('media_type', 'all')
            if media_type not in ('all', 'image', 'video', 'audio', 'document'):
                raise ValueError()
            if dataset not in DATASETS or not isinstance(question, str) or not 1 <= len(question.strip()) <= 2000:
                raise ValueError()
        except (ValueError, AttributeError, TypeError):
            return self.json_response({'error': 'Choose a collection and enter a question (up to 2,000 characters).'}, 400)
        if not catalog(dataset):
            return self.json_response({'error': 'This collection has no indexed sources yet.'}, 400)
        if not BUSY.acquire(blocking=False):
            return self.json_response({'error': 'A question is already running. Please wait for it to finish.'}, 409)
        job_id = uuid4().hex
        with LOCK:
            if len(JOBS) >= 30:
                JOBS.pop(next(iter(JOBS)))
            JOBS[job_id] = {'status': 'running'}
        threading.Thread(target=run_question, args=((job_id, dataset, question.strip(), media_type) if dataset in ('visual', 'library') else (job_id, dataset, question.strip())), daemon=True).start()
        self.json_response({'job_id': job_id}, 202)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Arabic RAG demo: http://127.0.0.1:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
