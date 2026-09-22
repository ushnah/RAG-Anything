"""Refresh descriptions/ASR while retaining existing OCR and original text.

python -m arabic_poc.refresh_corpus rag_storage_arabic_remote_v2
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .__main__ import VIDEO, AUDIO, IMAGES, command, records, save_json
from .remote import RemoteASR, RemoteVision, enabled, chat
from .video_search import collect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('storage', type=Path)
    parser.add_argument('--frame-seconds', type=float, default=20)
    parser.add_argument('--local-vision-fallback', action='store_true')
    parser.add_argument('--add', nargs='*', type=Path, default=[],
                        help='Additional files or directories to add to this corpus.')
    args = parser.parse_args()
    if args.frame_seconds <= 0:
        parser.error('frame-seconds must be positive')
    if not enabled('vision') or not enabled('asr'):
        raise ValueError('Configure remote vision and ASR before refreshing')
    old = collect()
    paths = {row['source'] for row in old}
    for supplied in args.add:
        if not supplied.exists():
            raise FileNotFoundError(supplied)
        candidates = supplied.rglob('*') if supplied.is_dir() else [supplied]
        paths.update(str(path.resolve()) for path in candidates
                     if path.suffix.lower() in VIDEO | AUDIO | IMAGES | {'.txt', '.md', '.pdf'})
    paths = sorted(paths)
    vision, asr = RemoteVision(describe=True), RemoteASR()
    local_vision = None

    def cached(key, fn):
        target = args.storage / 'extraction_cache' / (hashlib.sha256(key.encode()).hexdigest() + '.json')
        if target.exists():
            return json.loads(target.read_text())
        value = fn()
        save_json(target, value)
        return value

    class Refresh:
        def extract(self, path):
            # Preserve the original OCR provenance, never relabel it as remote output.
            for row in old:
                if row['source'] == str(path) and row['anchor']['kind'] not in ('speech','visual_description','metadata'):
                    yield row['original_text'], row['anchor'], row['engine']
            suffix = path.suffix.lower()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            duration = None
            if suffix in AUDIO | VIDEO:
                probe = json.loads(command('ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)))
                duration = float(probe['format']['duration'])
                if any(stream['codec_type']=='audio' for stream in probe['streams']):
                    print(f'Transcribing {path.name}', flush=True)
                    def transcribe():
                        with tempfile.TemporaryDirectory() as folder:
                            wav = Path(folder) / 'audio.wav'
                            command('ffmpeg','-v','error','-i',str(path),'-vn','-ac','1','-ar','16000',str(wav))
                            return asr.transcribe(wav)
                    transcript = cached(digest+':asr:'+os.environ.get('REMOTE_ASR_MODEL','whisper'), transcribe)
                    for segment in transcript['segments']:
                        start, end = max(0,float(segment['start'])), min(duration,float(segment['end']))
                        if start < end:
                            yield segment['text'], {'kind':'speech','start':start,'end':end}, 'whisper:remote:'+os.environ.get('REMOTE_ASR_MODEL','whisper')
            if suffix in VIDEO | IMAGES:
                with tempfile.TemporaryDirectory() as folder:
                    frames = [(path,None)]
                    if suffix in VIDEO:
                        command('ffmpeg','-v','error','-i',str(path),'-vf',f'fps=1/{args.frame_seconds}:start_time=0:round=up,scale=640:-2',str(Path(folder)/'%06d.jpg'))
                        frames = [(frame,i*args.frame_seconds) for i,frame in enumerate(sorted(Path(folder).glob('*.jpg'))) if i*args.frame_seconds < duration]
                    for frame,start in frames:
                        print(f'Describing {path.name} at {start}', flush=True)
                        nonlocal local_vision
                        key = digest+':vision:'+os.environ['REMOTE_VISION_MODEL']+':'+str(start)
                        remote_cache = args.storage/'extraction_cache'/(hashlib.sha256(key.encode()).hexdigest()+'.json')
                        engine = 'remote-vision:generated:'+os.environ['REMOTE_VISION_MODEL']
                        if args.local_vision_fallback and not remote_cache.exists():
                            if local_vision is None:
                                from .models import QwenParser
                                local_vision = QwenParser('models/qwen2.5-vl-3b',describe=True)
                                local_vision.concise = True
                            english = cached(digest+':local-qwen-en-v1:'+str(start),lambda:local_vision.parse_image(frame)[0]['text'])
                            text = cached(digest+':local-qwen-en-ar-v3:'+str(start),lambda:chat([{'role':'system','content':'Translate the supplied image description into Arabic faithfully. Output only five concise, retrieval-friendly fields in this exact order: المشهد: ... | العناصر والعدد المرئي: ... | النص المرئي: ... | الموضع/الألوان: ... | تفاصيل معمارية: ... . Preserve a stated count only when it is explicit; otherwise use «العدد غير محسوم». Use «لا يوجد نص مقروء» when appropriate. Add no details.'},{'role':'user','content':english}]))
                            engine = 'qwen-vl:generated:models/qwen2.5-vl-3b:translated:'+os.environ.get('REMOTE_TEXT_MODEL','gpt-oss')
                        else:
                            text = cached(key,lambda:vision.parse_image(frame)[0]['text'])
                        anchor={'kind':'visual_description'}
                        if start is not None: anchor['start']=start
                        yield text,anchor,engine
    def refresh_file(name):
        path = Path(name)
        key = hashlib.sha256(name.encode()).hexdigest()[:24]
        target = args.storage/'sources'/(key+'.json')
        if target.exists():
            print(f'Cached {path.name}',flush=True)
            return
        save_json(target, {'fingerprint':{'refresh':'remote-v3','frame_seconds':args.frame_seconds,
                                          'description_schema_version': 3},'records':records(path,Refresh())})
    def guarded(name):
        try:
            refresh_file(name)
        except Exception as exc:
            print(f'Failed {Path(name).name}: {type(exc).__name__}: {exc}', flush=True)
            raise
    if args.local_vision_fallback:
        for name in paths:
            guarded(name)
    else:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(guarded, paths))
    save_json(args.storage/'extraction_complete.json', {'sources':paths, 'frame_seconds':args.frame_seconds,
                                                         'description_schema_version': 3})
    print(f'Refreshed {len(paths)} source files',flush=True)


if __name__ == '__main__':
    main()
