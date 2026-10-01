"""Refresh sampled video descriptions and entities in the active app library."""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from dotenv import load_dotenv
from .__main__ import VIDEO, Extractor, command, records, save_json
from . import video_search
from .visual import embedding_model, search_text


def main():
    load_dotenv(video_search.ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='*', type=Path, help='Additional videos')
    parser.add_argument('--refresh-library', action='store_true')
    parser.add_argument('--frame-seconds', type=float, default=20)
    parser.add_argument('--resume', type=Path, help='Resume a previous run directory')
    args = parser.parse_args()
    if not math.isfinite(args.frame_seconds) or args.frame_seconds <= 0:
        parser.error('frame-seconds must be finite and positive')
    storage = video_search.STORAGE
    combined = json.loads((storage / 'index.json').read_text())
    visual = json.loads((storage / 'visual-index.json').read_text())
    paths = {p.resolve() for p in args.paths}
    if args.refresh_library:
        paths.update(Path(r['source']) for r in combined['records'] if r['media_type'] == 'video')
    if not paths:
        parser.error('Provide video paths or --refresh-library')
    for p in paths:
        if not p.is_file() or p.suffix.lower() not in VIDEO:
            parser.error(f'Not a video file: {p}')
    run = args.resume or storage / ('video-refresh-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    run.mkdir(parents=True, exist_ok=True)
    settings = {'paths': sorted(map(str, paths)), 'frame_seconds': args.frame_seconds}
    if (run / 'settings.json').exists() and json.loads((run / 'settings.json').read_text()) != settings:
        parser.error('Resume settings do not match this run')
    save_json(run / 'settings.json', settings)
    print(f'Run directory: {run}', flush=True)
    extractor = Extractor(describe_images=True)

    class Descriptions:
        def extract(self, path):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            duration = float(json.loads(command('ffprobe', '-v', 'error', '-show_format', '-of', 'json', str(path)))['format']['duration'])
            with tempfile.TemporaryDirectory() as folder:
                command('ffmpeg', '-v', 'error', '-i', str(path), '-vf',
                        f'fps=1/{args.frame_seconds}:start_time=0:round=up,scale=1280:-2', str(Path(folder) / '%06d.jpg'))
                for i, frame in enumerate(sorted(Path(folder).glob('*.jpg'))):
                    start = i * args.frame_seconds
                    if start >= duration:
                        break
                    cache = run / 'frames' / f'{digest}-{start}.json'
                    if cache.exists():
                        result = json.loads(cache.read_text())
                    else:
                        print(f'Describing {path.name} at {start:g}s', flush=True)
                        result = extractor.describe(frame)
                        save_json(cache, result)
                    yield result['description'], {'kind': 'visual_description', 'start': start,
                        'entities': result['entities'], 'person_matches': result['person_matches']}, extractor.description_engine()

    new_rows = []
    for path in sorted(paths):
        rows = records(path, Descriptions())
        # Keep visual passage IDs separate from existing OCR/transcript IDs.
        for row in rows:
            row['id'] += '-visual'
        new_rows.extend(rows)
    replaced = set(map(str, paths))
    encoder, model = embedding_model()
    if model != visual['embedding_model'] or model != combined['embedding_model']:
        raise ValueError('Embedding model differs from existing library')
    new_visual = [dict(r, media_type='video', source_label=r.get('metadata', {}).get('visual_label', '')) for r in new_rows]
    vectors = encoder.encode([search_text(r) for r in new_visual], batch_size=4, return_dense=True)['dense_vecs']
    kept = [(r, v) for r, v in zip(visual['records'], visual['vectors']) if r['source'] not in replaced]
    visual.update(records=[r for r, _ in kept] + new_visual, vectors=[v for _, v in kept] + vectors.tolist())
    with tempfile.TemporaryDirectory(dir=storage) as folder:
        stage = Path(folder)
        shutil.copytree(storage / 'sources', stage / 'sources')
        for file in (stage / 'sources').glob('*.json'):
            doc = json.loads(file.read_text())
            doc['records'] = [r for r in doc['records'] if not (r['source'] in replaced and r['anchor']['kind'] == 'visual_description')]
            save_json(file, doc)
        for path in paths:
            key = hashlib.sha256(str(path).encode()).hexdigest()[:24]
            save_json(stage / 'sources' / (key + '-visual.json'), {'fingerprint': settings,
                'records': [r for r in new_rows if r['source'] == str(path)]})
        save_json(stage / 'visual-index.json', visual)
        video_search.INPUTS = [str(stage.resolve())]
        all_rows = video_search.collect()
        existing = {r['id']: (r['normalized_text'], v) for r, v in zip(combined['records'], combined['vectors'])}
        needed = [r for r in all_rows if r['id'] not in existing or existing[r['id']][0] != r['normalized_text']]
        if needed:
            vectors = encoder.encode([r['normalized_text'] for r in needed], batch_size=4, return_dense=True)['dense_vecs']
            existing.update({r['id']: (r['normalized_text'], v.tolist()) for r, v in zip(needed, vectors)})
        combined.update(records=all_rows, vectors=[existing[r['id']][1] for r in all_rows])
        backup = run / 'backup'
        backup.mkdir(exist_ok=True)
        for name in ['index.json', 'visual-index.json']:
            shutil.copy2(storage / name, backup / name)
        shutil.copytree(storage / 'sources', backup / 'sources', dirs_exist_ok=True)
        for file in (stage / 'sources').glob('*.json'):
            shutil.copy2(file, storage / 'sources' / file.name)
        save_json(storage / 'visual-index.json', visual)
        save_json(storage / 'index.json', combined)
    print(f'Refreshed {len(paths)} videos, {len(new_rows)} scene descriptions. Existing speech/OCR retained.', flush=True)


if __name__ == '__main__':
    main()
